"""Host-side analysis of pulled joint `.npy` files (T, 22, 3).

`visualize.py` dumps a `<clip>.npy` next to every rendered GIF; the job poller
rsyncs both into `media/jobs/<job>/`. These are kilobytes, so we compute cheap
smoothness/trajectory series locally (no model, no animation). Jitter uses the
same `‖Δ³x‖` jerk measure surfaced in the training-curve view.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import cache

# HumanML3D foot joints (L_Ankle, L_Foot, R_Ankle, R_Foot) — matches
# rmg.representation.skeleton.FOOT_CONTACT_IDX.
FOOT_IDX = (7, 10, 8, 11)
CONTACT_THRESHOLD = 0.05  # metres above ground counted as "planted"


def _series(values: np.ndarray, start: int = 0) -> list[list[float]]:
    """[[frame, value], …] with frame index offset by `start`."""
    return [[int(start + i), round(float(v), 5)] for i, v in enumerate(values)]


def resolve_npy(job_id: str, name: str) -> Path:
    """Safe path to a pulled .npy under media/jobs/<job>/ (no traversal)."""
    if "/" in name or ".." in name or not name.endswith(".npy"):
        raise ValueError(f"bad npy name {name!r}")
    p = cache.media_dir() / "jobs" / job_id / name
    if not p.exists():
        raise FileNotFoundError(f"{name} not found for job {job_id}")
    return p


def analyze(path: Path, fps: int = 20) -> dict:
    """Trajectory + jitter + speed + foot-height series from a (T,22,3) clip."""
    joints = np.load(path).astype(np.float64)
    if joints.ndim != 3 or joints.shape[1] < 22 or joints.shape[2] != 3:
        raise ValueError(f"expected (T,22,3) joints, got {joints.shape}")
    T = joints.shape[0]
    root_xz = joints[:, 0, [0, 2]]  # (T, 2)

    # jerk magnitude per frame (mean over joints) — the smoothness signal.
    jerk = np.linalg.norm(np.diff(joints, n=3, axis=0), axis=-1).mean(axis=-1)  # (T-3,)
    accel = np.linalg.norm(np.diff(joints, n=2, axis=0), axis=-1).mean(axis=-1)  # (T-2,)
    speed = np.linalg.norm(np.diff(root_xz, axis=0), axis=-1) * fps              # (T-1,)
    foot_h = joints[:, FOOT_IDX, 1].min(axis=1)                                  # (T,)

    return {
        "frames": int(T),
        "fps": fps,
        "jerk_mean": round(float(jerk.mean()), 5) if jerk.size else None,
        "accel_mean": round(float(accel.mean()), 5) if accel.size else None,
        "trajectory": [[round(float(x), 4), round(float(z), 4)] for x, z in root_xz],
        "jitter": _series(jerk, start=3),
        "speed": _series(speed, start=1),
        "foot_height": _series(foot_h),
        "contact_threshold": CONTACT_THRESHOLD,
    }
