"""Generation-forensics stats (gen-vs-real motion dynamics) for the Validation panel.

The forensics numbers compare a run's *generated* EMA samples against real packed
clips along a handful of physically-meaningful axes: quaternion norm (are the
rotations valid?), per-frame angular velocity `2·acos(|<q_t, q_{t+1}>|)`,
translation velocity, and |translation|. They are the "smoking gun that isn't" —
the generations are valid, conditioned, and diverse, just slightly jittery and
conservative. See `docs/rmg_mid_validation.md` §4 + appendix.

Computing them fresh means loading `runs/.../samples/step-*.pt` + a sample of real
packed clips on the cluster, which is heavy and out of scope for the request path.
So this reads a cached `eval/forensics.json` if the eval wrote one, and 404s
otherwise; the frontend falls back to the documented static values. A follow-up can
compute on demand.  TODO(forensics-endpoint): compute over SSH when no cache exists.

Expected `forensics.json` shape (all values are plain floats)::

    {
      "velocity": {
        "quat_norm":       {"gen": 1.0000, "real": 1.0000},
        "angular_vel":     {"gen": 0.0875, "real": 0.0645},   # rad/frame
        "translation_vel": {"gen": 0.0241, "real": 0.0171},   # m/frame
        "translation_abs": {"gen": 0.514,  "real": 0.646}     # m
      },
      "functional": {
        "r1":        {"gen": 0.45, "gt": 0.513, "published": 0.511},
        "diversity": {"gen": 8.61, "gt": 9.79,  "published": 9.50}
      }
    }
"""

from __future__ import annotations

import json
import shlex

from ..cluster import ssh
from ..cluster.squeue import resolve_run_dir


def fetch_forensics(run: str) -> dict:
    """Read a run's cached `eval/forensics.json`.

    Mirrors `eval_tables.fetch_results`: distinguishes a genuine miss from an SSH
    transport error (exit 255) so a transient blip isn't reported as "no file",
    and rebuilds the master + retries once on transport failure.
    """
    p = f"{resolve_run_dir(run)}/eval/forensics.json"
    last_err: Exception | None = None
    for _ in range(2):
        r = ssh.run(f"cat {shlex.quote(p)}", check=False)  # no 2>/dev/null: keep the exit code
        if r.returncode == 0 and r.stdout.strip():
            data = json.loads(r.stdout)
            return {"run": run, **data}
        if r.returncode == 255:  # SSH transport error, not "no such file"
            last_err = ssh.SSHError(255, p, r.stderr)
            ssh.ensure_master(force=True)
            continue
        break  # genuine miss
    if last_err:
        raise last_err
    raise FileNotFoundError(f"no eval/forensics.json for run {run!r}")
