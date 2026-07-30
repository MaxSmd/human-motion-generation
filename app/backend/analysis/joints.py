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
# shared.geometry.FOOT_CONTACT_IDX.
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
    """Trajectory + jitter + speed + foot-height + foot-skate + per-joint jerk
    series from a (T,22,3) clip."""
    joints = np.load(path).astype(np.float64)
    if joints.ndim != 3 or joints.shape[1] < 22 or joints.shape[2] != 3:
        raise ValueError(f"expected (T,22,3) joints, got {joints.shape}")
    T = joints.shape[0]
    root_xz = joints[:, 0, [0, 2]]  # (T, 2)

    # per-frame magnitudes (mean over joints) — the smoothness signals.
    jerk_pf = np.linalg.norm(np.diff(joints, n=3, axis=0), axis=-1)  # (T-3, J)
    accel_pf = np.linalg.norm(np.diff(joints, n=2, axis=0), axis=-1)  # (T-2, J)
    jerk = jerk_pf.mean(axis=-1)                                     # (T-3,)
    accel = accel_pf.mean(axis=-1)                                   # (T-2,)
    speed = np.linalg.norm(np.diff(root_xz, axis=0), axis=-1) * fps  # (T-1,)
    foot_h = joints[:, FOOT_IDX, 1].min(axis=1)                      # (T,)

    # Per-joint mean jerk (T-3 averaged) — which joints are roughest.
    per_joint_jerk = jerk_pf.mean(axis=0) if jerk_pf.size else np.zeros(joints.shape[1])

    # Foot-skate: horizontal drift (m/s) of a foot while it is planted
    # (height < contact threshold on both frames). A clean plant → ~0. Reported
    # both as a per-frame series (max over the four foot joints) and a scalar
    # mean over all planted foot-frames.
    foot = joints[:, FOOT_IDX, :]                                   # (T, 4, 3)
    horiz = np.linalg.norm(np.diff(foot[:, :, [0, 2]], axis=0), axis=-1) * fps  # (T-1, 4)
    planted = (foot[:-1, :, 1] < CONTACT_THRESHOLD) & (foot[1:, :, 1] < CONTACT_THRESHOLD)
    skate_masked = np.where(planted, horiz, 0.0)
    skate_pf = skate_masked.max(axis=1) if skate_masked.size else np.zeros(0)   # (T-1,)
    n_planted = int(planted.sum())
    foot_skate_mean = float(horiz[planted].mean()) if n_planted else 0.0

    return {
        "frames": int(T),
        "fps": fps,
        "jerk_mean": round(float(jerk.mean()), 5) if jerk.size else None,
        "accel_mean": round(float(accel.mean()), 5) if accel.size else None,
        "foot_skate_mean": round(foot_skate_mean, 5),
        "trajectory": [[round(float(x), 4), round(float(z), 4)] for x, z in root_xz],
        "jitter": _series(jerk, start=3),
        "accel": _series(accel, start=2),
        "speed": _series(speed, start=1),
        "foot_height": _series(foot_h),
        "foot_skate": _series(skate_pf, start=1),
        "per_joint_jerk": [round(float(v), 5) for v in per_joint_jerk],
        "contact_threshold": CONTACT_THRESHOLD,
    }


def compare_pair(real_path: Path, gen_path: Path) -> dict:
    """Per-joint positional error (metres) between a GT clip and a generated
    clip of the SAME motion (e.g. a compare job's real-<id>/gen-<id> pair).

    Both are root-aligned per frame (subtract the pelvis) before differencing so
    the metric reflects POSE error, not a global trajectory offset — the model
    can walk to a slightly different spot yet still be pose-faithful. Clips are
    truncated to the shorter length. Returns per-joint mean error + the overall
    mean (an MPJPE-style summary) and a per-frame mean-error series.
    """
    a = np.load(real_path).astype(np.float64)
    b = np.load(gen_path).astype(np.float64)
    for name, arr in (("real", a), ("gen", b)):
        if arr.ndim != 3 or arr.shape[1] < 22 or arr.shape[2] != 3:
            raise ValueError(f"expected (T,22,3) {name} joints, got {arr.shape}")
    T = min(a.shape[0], b.shape[0])
    J = min(a.shape[1], b.shape[1])
    a = a[:T, :J] - a[:T, 0:1]  # root-align (subtract pelvis)
    b = b[:T, :J] - b[:T, 0:1]
    err = np.linalg.norm(a - b, axis=-1)  # (T, J) metres
    per_joint = err.mean(axis=0)          # (J,)
    per_frame = err.mean(axis=1)          # (T,)
    return {
        "frames": int(T),
        "joints": int(J),
        "mpjpe": round(float(err.mean()), 5),
        "per_joint_error": [round(float(v), 5) for v in per_joint],
        "error_series": _series(per_frame),
    }
