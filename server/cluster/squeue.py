"""`squeue` / run listing over SSH."""

from __future__ import annotations

import shlex

from .. import config as cfgmod
from . import ssh

# field order must match the parser below
_FMT = "%i|%j|%T|%M|%l|%D|%R"

# Run names are unique across the per-task subdirs, so a run can be located by
# probing train/ eval/ viz/ in turn.
_RUN_KINDS = ("train", "eval", "viz")


def resolve_run_dir(run: str) -> str:
    """Abs path of a run dir, probing the per-task subdirs (train/eval/viz).
    Falls back to `<base>/<run>` if not found anywhere (keeps errors legible)."""
    base = ssh.abs_remote(cfgmod.cluster_runs_dir())
    script = (
        f"base={shlex.quote(base)}; run={shlex.quote(run)}; "
        'for k in train eval viz; do '
        'if [ -d "$base/$k/$run" ]; then printf %s "$base/$k/$run"; exit 0; fi; '
        'done; printf %s "$base/$run"'
    )
    return ssh.run(script, timeout=15, check=False).stdout.strip()


def squeue_me() -> list[dict]:
    """Current user's jobs as parsed rows."""
    res = ssh.run(f"squeue --me -h -o {shlex.quote(_FMT)}", timeout=15)
    rows = []
    for line in res.stdout.splitlines():
        parts = line.split("|")
        if len(parts) != 7:
            continue
        jobid, name, state, t, limit, nodes, reason = parts
        rows.append({
            "jobid": jobid.strip(),
            "name": name.strip(),
            "state": state.strip(),
            "time": t.strip(),
            "time_limit": limit.strip(),
            "nodes": nodes.strip(),
            "reason": reason.strip(),
        })
    return rows


def sacct_state(slurm_id: str) -> str | None:
    """Allocation-level final/again state via sacct (-X), e.g. RUNNING, COMPLETED,
    FAILED, CANCELLED, TIMEOUT, PENDING. None if sacct has no record yet."""
    sid = shlex.quote(str(slurm_id))
    res = ssh.run(
        f"sacct -j {sid} -X -n -P -o State 2>/dev/null | head -n1",
        timeout=15, check=False,
    )
    if not res.ok:
        return None
    s = res.stdout.strip().split()  # "CANCELLED by 12345" → "CANCELLED"
    return s[0] if s else None


def list_runs() -> list[dict]:
    """Runs under each per-task subdir of the runs root, flagged with what they
    contain and tagged with their task `kind` (train/eval/viz)."""
    base = shlex.quote(ssh.abs_remote(cfgmod.cluster_runs_dir()))
    # For each run dir under train/ eval/ viz/: kind, name, has checkpoints/samples/viz/metrics.
    script = (
        f'base={base}; '
        'for k in train eval viz; do '
        'cd "$base/$k" 2>/dev/null || continue; '
        'for d in */; do d="${d%/}"; [ -d "$d" ] || continue; '
        '[ -d "$d/checkpoints" ] && c=1 || c=0; '
        '[ -d "$d/samples" ] && s=1 || s=0; '
        '[ -d "$d/viz" ] && v=1 || v=0; '
        '[ -f "$d/metrics.csv" ] && m=1 || m=0; '
        'echo "$k|$d|$c|$s|$v|$m"; done; done'
    )
    res = ssh.run(script, timeout=20, check=False)
    out = []
    for line in res.stdout.splitlines():
        p = line.split("|")
        if len(p) != 6:
            continue
        out.append({
            "kind": p[0],
            "run": p[1],
            "has_checkpoints": p[2] == "1",
            "has_samples": p[3] == "1",
            "has_viz": p[4] == "1",
            "has_metrics": p[5] == "1",
        })
    return out


def list_checkpoints(run: str) -> list[str]:
    """Checkpoint filenames under a run, newest first (by name). The run's task
    subdir is wildcarded since checkpoints only ever live under train runs."""
    runs = shlex.quote(ssh.abs_remote(cfgmod.cluster_runs_dir()))
    res = ssh.run(
        f"ls -1 {runs}/*/{shlex.quote(run)}/checkpoints/*.pt 2>/dev/null",
        timeout=15, check=False,
    )
    files = [l.strip() for l in res.stdout.splitlines() if l.strip()]
    return sorted(files, reverse=True)
