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
        f"cat {runs}/*/{shlex.quote(run)}/metrics.csv 2>/dev/null", timeout=25, check=False
    )
    if not res.ok or not res.stdout.strip():
        raise FileNotFoundError(f"no metrics.csv for run {run!r}")

    reader = csv.reader(io.StringIO(res.stdout))
    header = next(reader, [])
    rows = [r for r in reader if len(r) == len(header)]

    # Downsample by striding so the curve shape is preserved.
    if len(rows) > max_points:
        stride = len(rows) // max_points + 1
        rows = rows[::stride]

    def num(v: str):
        try:
            return float(v)
        except ValueError:
            return None

    return {
        "run": run,
        "columns": header,
        "rows": [[num(v) for v in r] for r in rows],
        "n": len(rows),
    }


def fetch_run_info(run: str) -> dict:
    """The run's saved config.json + an approx wall-clock duration.

    Duration ≈ newest-artifact mtime − config.json mtime (the trainer writes
    config.json at startup and checkpoints throughout). One SSH round-trip."""
    runs = shlex.quote(ssh.abs_remote(cfgmod.cluster_runs_dir()))
    rdir = f"{runs}/*/{shlex.quote(run)}"  # task subdir wildcarded
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
