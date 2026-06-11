"""`squeue` / run listing over SSH."""

from __future__ import annotations

import shlex

from .. import config as cfgmod
from . import ssh

# field order must match the parser below
_FMT = "%i|%j|%T|%M|%l|%D|%R"


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
    """Runs under the remote runs dir, flagged with what they contain."""
    runs = shlex.quote(ssh.abs_remote(cfgmod.cluster_runs_dir()))
    # For each run dir: name, has checkpoints, has samples, has viz.
    script = (
        f'cd {runs} 2>/dev/null || exit 0; '
        'for d in */; do d="${d%/}"; '
        '[ -d "$d/checkpoints" ] && c=1 || c=0; '
        '[ -d "$d/samples" ] && s=1 || s=0; '
        '[ -d "$d/viz" ] && v=1 || v=0; '
        'echo "$d|$c|$s|$v"; done'
    )
    res = ssh.run(script, timeout=20, check=False)
    out = []
    for line in res.stdout.splitlines():
        p = line.split("|")
        if len(p) != 4:
            continue
        out.append({
            "run": p[0],
            "has_checkpoints": p[1] == "1",
            "has_samples": p[2] == "1",
            "has_viz": p[3] == "1",
        })
    return out


def list_checkpoints(run: str) -> list[str]:
    """Checkpoint filenames under a run, newest first (by name)."""
    runs = ssh.abs_remote(cfgmod.cluster_runs_dir())
    path = shlex.quote(f"{runs}/{run}/checkpoints")
    res = ssh.run(f"ls -1 {path}/*.pt 2>/dev/null", timeout=15, check=False)
    files = [l.strip() for l in res.stdout.splitlines() if l.strip()]
    return sorted(files, reverse=True)
