"""Training-curve parsing from a run's `metrics.csv` (pulled over SSH).

The trainer appends one row per log step: loss, lr, grad_norm, t_mean,
x_t_offmanifold_frac, steps_per_s, step. We fetch + parse it and downsample to a
bounded number of points so the frontend can plot any column against `step`.
"""

from __future__ import annotations

import csv
import io
import json
import shlex

from .. import config as cfgmod
from ..cluster import ssh


def fetch_metrics(run: str, max_points: int = 1500) -> dict:
    """{columns, rows} for a run's metrics.csv (rows downsampled if huge)."""
    runs = shlex.quote(ssh.abs_remote(cfgmod.cluster_runs_dir()))
    # Wildcard the task subdir (train/eval/viz) — run names are unique.
    res = ssh.run(
        f"cat {runs}/*/*/{shlex.quote(run)}/metrics.csv 2>/dev/null", timeout=25, check=False
    )
    if not res.ok or not res.stdout.strip():
        raise FileNotFoundError(f"no metrics.csv for run {run!r}")

    reader = csv.reader(io.StringIO(res.stdout))
    header = next(reader, [])
    raw = [r for r in reader if len(r) == len(header)]

    def num(v: str):
        try:
            return float(v)
        except ValueError:
            return None

    rows = [[num(v) for v in r] for r in raw]
    rows, restarts = _prune_restarts(rows, header)
    eras = _grad_accum_eras(run, restarts)
    rows = _downsample(rows, max_points)

    return {
        "run": run,
        "columns": header,
        "rows": rows,
        "n": len(rows),
        # Where training was resumed from an earlier checkpoint. The abandoned
        # rows are already gone from `rows`; these mark the seams for the plot.
        "restarts": restarts,
        # [{from_step, grad_accum}] — the divisor that turns the logged loss back
        # into the per-sample loss over each stretch. None when we can't tell.
        "eras": eras,
    }


# A rollback shorter than this is a requeue resuming from the checkpoint it had
# just written — same job, same config, not an era boundary.
_MINOR_RESUME_STEPS = 1000


def _grad_accum_eras(run: str, restarts: list[dict]) -> list[dict] | None:
    """Per-era `grad_accum`, so the plot can undo the logging multiplier.

    `train.py` divides the loss by `grad_accum` before backward and multiplies it
    straight back out for the metrics row (`loss_metric = accum_loss *
    grad_accum`), so the logged loss is `grad_accum ×` the real per-sample loss.
    Relaunch a run with a different `grad_accum` and the curve takes a step that
    is pure bookkeeping — rmg_mid went 2 → 4 at its tf32 relaunch and the logged
    loss doubled with nothing about training having changed.

    `metrics.csv` doesn't carry `grad_accum`, but the app submitted these jobs, so
    its own job store does. A major rollback is where one launch handed over to
    the next, so N such seams ⇒ N+1 eras; we only return a mapping when that
    lines up exactly with the run's train jobs and every one of them states its
    `grad_accum`. Anything ambiguous returns None — a wrong divisor silently
    rescales a loss curve, which is worse than not offering the correction.
    """
    # Distinct rollback TARGETS, not rollback events: a handover often takes a
    # few attempts (rmg_mid rolled back to 160.1k twice — once for the guarded
    # bf16 rerun, once for tf32), and every attempt but the last is pruned, so
    # they leave a single seam in the surviving curve.
    seams = sorted({
        r["step"] for r in restarts
        if r["abandoned_to"] - r["step"] >= _MINOR_RESUME_STEPS
    })

    try:
        from ..cluster.jobs import get_manager

        jobs = [
            j for j in get_manager().list()
            if j.kind == "train" and j.run_name == run
        ]
    except Exception:
        return None

    jobs.sort(key=lambda j: j.submitted_at)
    accums = [(j.params or {}).get("grad_accum") for j in jobs]
    if not accums or len(accums) != len(seams) + 1:
        return None
    if any(not isinstance(a, (int, float)) or a <= 0 for a in accums):
        return None

    starts = [0.0] + seams
    return [{"from_step": st, "grad_accum": int(a)} for st, a in zip(starts, accums)]


def _prune_restarts(rows: list[list], header: list[str]) -> tuple[list[list], list[dict]]:
    """Drop the abandoned branches left by resumes, keeping one monotonic curve.

    A cancelled/requeued run resumes from its last checkpoint and keeps appending
    to the SAME metrics.csv, so the step column walks backwards at each seam:
    ...183700, 160200, 160300... Plotted in file order that draws a line running
    right-to-left across the whole axis, and the superseded steps sit on top of
    the ones that replaced them.

    So whenever a row's step is not ahead of the highest step kept so far, we
    treat it as a resume: every already-kept row at or beyond that step belongs
    to the branch that was thrown away, and is dropped. What survives is the
    lineage that actually produced the final checkpoint.
    """
    if "step" not in header:
        return rows, []
    si = header.index("step")

    kept: list[list] = []
    restarts: list[dict] = []
    for r in rows:
        s = r[si] if si < len(r) else None
        if s is None:
            continue  # a repeated header row, or a row without a step
        if kept and s <= kept[-1][si]:
            abandoned_to = kept[-1][si]
            n_before = len(kept)
            while kept and kept[-1][si] >= s:
                kept.pop()
            dropped = n_before - len(kept)
            # A requeue that resumes from the checkpoint it just wrote rewinds by
            # a step or two — a seam worth stitching, not worth annotating.
            if dropped > 1:
                restarts.append(
                    {"step": s, "abandoned_to": abandoned_to, "dropped": dropped}
                )
        kept.append(r)

    return kept, restarts


def _downsample(rows: list[list], max_points: int) -> list[list]:
    """Stride down to ~`max_points` rows, but never drop the storm spikes.

    Plain striding preserves the curve's gross shape yet skips the rare,
    isolated rows a gradient storm lives in — exactly the outliers the plot
    wants to flag as clip triangles. So we keep the strided backbone AND
    force-include, for every numeric column, the rows sitting beyond a robust
    outlier fence (plus each column's global extremum). A per-column cap keeps
    the point count bounded even for a pathologically spiky column.
    """
    n = len(rows)
    if n <= max_points:
        return rows
    stride = n // max_points + 1
    keep = set(range(0, n, stride))
    keep.add(0)
    keep.add(n - 1)

    ncol = max((len(r) for r in rows), default=0)
    cap = 40  # most-extreme outliers kept per column
    for c in range(ncol):
        col = [(i, r[c]) for i, r in enumerate(rows) if c < len(r) and r[c] is not None]
        if len(col) < 8:
            continue
        vals = sorted(v for _, v in col)
        q1 = vals[len(vals) // 4]
        q3 = vals[(3 * len(vals)) // 4]
        iqr = q3 - q1
        # Global extrema always survive — the peak of the storm must be in frame.
        keep.add(max(col, key=lambda t: t[1])[0])
        keep.add(min(col, key=lambda t: t[1])[0])
        if iqr <= 0:
            continue  # flat / monotonic column (e.g. step) has no spikes to keep
        hi, lo = q3 + 3 * iqr, q1 - 3 * iqr
        out = [(i, v) for i, v in col if v > hi or v < lo]
        # Keep only the most extreme `cap` so a noisy column can't defeat striding.
        out.sort(key=lambda t: -abs(t[1] - (q1 + q3) / 2))
        for i, _ in out[:cap]:
            keep.add(i)

    return [rows[i] for i in sorted(keep)]


def fetch_run_info(run: str) -> dict:
    """The run's saved config.json + an approx wall-clock duration.

    Duration ≈ newest-artifact mtime − config.json mtime (the trainer writes
    config.json at startup and checkpoints throughout). One SSH round-trip."""
    runs = shlex.quote(ssh.abs_remote(cfgmod.cluster_runs_dir()))
    rdir = f"{runs}/*/*/{shlex.quote(run)}"  # model + task subdir wildcarded
    script = (
        f"cd {rdir} 2>/dev/null || exit 3; "
        'echo "===CONFIG==="; cat config.json 2>/dev/null; '
        'echo "===TIMES==="; '
        'stat -c %Y config.json 2>/dev/null || echo ""; '
        'ls -t checkpoints/*.pt metrics.csv samples/*.pt 2>/dev/null | head -1 '
        '| xargs -r stat -c %Y 2>/dev/null || echo ""'
    )
    r = ssh.run(script, timeout=20, check=False)
    if r.returncode == 3:
        raise FileNotFoundError(f"run {run!r} not found")

    cfg_part, _, times_part = r.stdout.partition("===TIMES===")
    cfg_text = cfg_part.replace("===CONFIG===", "", 1).strip()
    config = None
    if cfg_text:
        try:
            config = json.loads(cfg_text)
        except json.JSONDecodeError:
            config = None

    times = [t.strip() for t in times_part.strip().splitlines() if t.strip()]
    started_at = duration = None
    if len(times) >= 2 and times[0].isdigit() and times[1].isdigit():
        started_at = int(times[0])
        duration = max(0, int(times[1]) - int(times[0]))

    return {"run": run, "config": config, "started_at": started_at,
            "duration_seconds": duration}
