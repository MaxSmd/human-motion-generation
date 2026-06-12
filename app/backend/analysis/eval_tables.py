"""Eval-metric parsing + run-comparison tables.

`evaluate.py` writes `<run>/eval/results.json` as `{omega: {fid, r_precision:[r1,
r2,r3], mm_dist, diversity, diversity_real, multimodality}}`. Eval runs live
under `<project>/runs/<model>/eval/<run>` (all models share this layout).

`comparison()` picks each run's best-FID guidance and emits normalized rows + a
copy-ready booktabs LaTeX `tabular` (best per column bolded).
"""

from __future__ import annotations

import json
import re
import shlex

from .. import config as cfgmod
from ..cluster import ssh
from ..cluster.squeue import resolve_run_dir

# An eval run writes results.json incrementally (one ω at a time, overwriting),
# so a fetch can catch it truncated. Each top-level entry is `"<ω>": { …flat… }`
# with no nested braces, so we can salvage the complete entries from a partial file.
_ENTRY_RE = re.compile(r'"(\d+(?:\.\d+)?)"\s*:\s*(\{[^{}]*\})')


def _parse_results(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        out: dict[str, dict] = {}
        for w, body in _ENTRY_RE.findall(text):
            try:
                out[w] = json.loads(body)
            except json.JSONDecodeError:
                continue
        if not out:
            raise
        return out

# Columns: key, header, LaTeX header (with arrow), better = min|max
COLUMNS = [
    ("fid", "FID", r"FID$\downarrow$", "min"),
    ("r1", "R@1", r"R@1$\uparrow$", "max"),
    ("r2", "R@2", r"R@2$\uparrow$", "max"),
    ("r3", "R@3", r"R@3$\uparrow$", "max"),
    ("mm_dist", "MM-Dist", r"MM-Dist$\downarrow$", "min"),
    ("diversity", "Diversity", "Diversity", None),
    ("multimodality", "MModality", "MModality", None),
]


def list_eval_runs() -> list[dict]:
    """Runs that have an eval/results.json, across every model's eval/ subdir."""
    base = shlex.quote(ssh.abs_remote(cfgmod.cluster_runs_dir()))
    r = ssh.run(f"ls -1 {base}/*/eval/*/eval/results.json 2>/dev/null", check=False)
    out: list[dict] = []
    seen: set[str] = set()
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        run = line.split("/")[-3]
        if run in seen:
            continue
        seen.add(run)
        out.append({"run": run, "path": line})
    return sorted(out, key=lambda d: d["run"])


def fetch_results(run: str) -> dict:
    """{omega(float): metrics} for a run, from whichever root holds it.

    Distinguishes a genuine miss (`cat` exit 1) from an SSH-transport error
    (exit 255) so a transient blip under load isn't reported as "file missing";
    transport errors get one rebuild-and-retry."""
    p = f"{resolve_run_dir(run)}/eval/results.json"
    last_err: Exception | None = None
    for _ in range(2):
        r = ssh.run(f"cat {shlex.quote(p)}", check=False)  # no 2>/dev/null: keep the exit code meaningful
        if r.returncode == 0 and r.stdout.strip():
            raw = _parse_results(r.stdout)
            return {float(k): _flatten(v) for k, v in raw.items()}
        if r.returncode == 255:  # SSH transport error, not "no such file"
            last_err = ssh.SSHError(255, p, r.stderr)
            ssh.ensure_master(force=True)  # rebuild the master, then retry once
            continue
        break  # genuine miss
    if last_err:
        raise last_err
    raise FileNotFoundError(f"no eval/results.json for run {run!r}")


def _flatten(m: dict) -> dict:
    """Pull r_precision[0:3] out into r1/r2/r3; keep the scalar metrics."""
    rp = m.get("r_precision") or [None, None, None]
    return {
        "fid": m.get("fid"),
        "r1": rp[0] if len(rp) > 0 else None,
        "r2": rp[1] if len(rp) > 1 else None,
        "r3": rp[2] if len(rp) > 2 else None,
        "mm_dist": m.get("mm_dist"),
        "diversity": m.get("diversity"),
        "diversity_real": m.get("diversity_real"),
        "multimodality": m.get("multimodality"),
    }


def _best_omega_row(run: str, per_omega: dict) -> dict:
    """Row for a run at its best-FID guidance (lower FID = better)."""
    best_w = min(per_omega, key=lambda w: per_omega[w]["fid"] if per_omega[w]["fid"] is not None else float("inf"))
    row = {"run": run, "omega": best_w}
    row.update(per_omega[best_w])
    return row


def comparison(runs: list[str]) -> dict:
    """Best-ω row per run + per-column winners + LaTeX + full per-ω sweep data.

    Returns the full per-guidance metrics too (`sweeps`), so the frontend renders
    both the table and the guidance-sweep charts from ONE request — important
    since the head node serializes SSH (no parallel fan-out)."""
    rows: list[dict] = []
    errors: dict[str, str] = {}
    sweeps: dict[str, dict] = {}
    for run in runs:
        try:
            per = fetch_results(run)
            rows.append(_best_omega_row(run, per))
            sweeps[run] = {str(w): m for w, m in per.items()}
        except Exception as e:  # noqa: BLE001
            errors[run] = str(e)

    # Per-column best (for bolding).
    best: dict[str, int] = {}
    for key, _, _, direction in COLUMNS:
        if direction is None:
            continue
        vals = [(i, r[key]) for i, r in enumerate(rows) if r.get(key) is not None]
        if not vals:
            continue
        pick = min if direction == "min" else max
        best[key] = pick(vals, key=lambda t: t[1])[0]

    return {"rows": rows, "best": best, "errors": errors, "sweeps": sweeps,
            "latex": build_latex(rows, best)}


def _fmt(v) -> str:
    if v is None:
        return "—"
    return f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def build_latex(rows: list[dict], best: dict[str, int]) -> str:
    """Booktabs tabular; best per column wrapped in \\textbf{}."""
    metric_cols = [c for c in COLUMNS]
    colspec = "l" + "c" * (1 + len(metric_cols))  # run + omega + metrics
    head = " & ".join(["Run", r"$\omega$"] + [c[2] for c in metric_cols]) + r" \\"
    lines = [
        r"\begin{tabular}{" + colspec + "}",
        r"\toprule",
        head,
        r"\midrule",
    ]
    for i, r in enumerate(rows):
        cells = [_tex_escape(r["run"]), f"{r['omega']:.1f}"]
        for key, _, _, _ in metric_cols:
            txt = _fmt(r.get(key))
            if best.get(key) == i and txt != "—":
                txt = r"\textbf{" + txt + "}"
            cells.append(txt)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)
