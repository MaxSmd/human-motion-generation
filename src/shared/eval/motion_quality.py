"""Physical-plausibility metrics on world-space joint positions.

Reported alongside FID / R-precision whenever generation is *constrained*,
because that is where the two can disagree: a hard spatial constraint can hit
its targets exactly and still produce a body that slides across the floor. FID
is largely blind to that (it scores a learned motion embedding), so the cost of
enforcement has to be measured directly on the geometry.

Definitions follow the controllable-motion literature (GMD / OmniControl):
a foot is *in contact* when it is within `height_thr` of the floor, and it
*skates* when it is in contact and still moves more than `slide_thr`
horizontally between consecutive frames.

Kept separate from `metrics.py` (numpy, distribution-level scores) because
these are torch, per-clip and geometric. World frame = HumanML3D FK frame:
X right, Y up, Z forward, floor at y = 0.
"""

from __future__ import annotations

import torch
from torch import Tensor

# (ankle, toe) per side in the 22-joint HumanML3D skeleton.
LEFT_FOOT = (7, 10)
RIGHT_FOOT = (8, 11)


def _valid_frames(T: int, lengths: Tensor | None, B: int, device) -> Tensor:
    if lengths is None:
        return torch.ones(B, T, dtype=torch.bool, device=device)
    return torch.arange(T, device=device).view(1, T) < lengths.to(device).view(B, 1)


def foot_skate_ratio(
    joints: Tensor,
    lengths: Tensor | None = None,
    fps: float = 20.0,
    height_thr: float = 0.05,
    slide_thr: float = 0.025,
) -> float:
    """Fraction of frame-transitions in which a grounded foot slides.

    `joints` is (B, T, J, 3). A transition counts as skating when either foot
    is below `height_thr` at both ends of it and its horizontal displacement
    exceeds `slide_thr` (metres, measured per frame at the clip's own rate — the
    thresholds are the published ones for 20 fps HumanML3D).

    Returns a ratio in [0, 1]; 0 means every planted foot stayed planted.
    """
    if joints.ndim != 4:
        raise ValueError(f"joints must be (B, T, J, 3), got {tuple(joints.shape)}")
    B, T = joints.shape[0], joints.shape[1]
    if T < 2:
        return 0.0
    device = joints.device
    valid = _valid_frames(T, lengths, B, device)
    trans_valid = valid[:, :-1] & valid[:, 1:]                    # (B, T-1)
    if not bool(trans_valid.any()):
        return 0.0

    skating = torch.zeros(B, T - 1, dtype=torch.bool, device=device)
    scale = 20.0 / max(float(fps), 1e-6)   # thresholds are quoted per 20-fps frame
    for foot in (LEFT_FOOT, RIGHT_FOOT):
        p = joints[:, :, foot, :]                                  # (B,T,2,3)
        height = p[..., 1].min(dim=-1).values                       # (B,T) lowest of ankle/toe
        grounded = (height < height_thr)
        contact = grounded[:, :-1] & grounded[:, 1:]                # planted across the step
        horiz = p[:, 1:, :, ::2] - p[:, :-1, :, ::2]                # x,z displacement
        slide = torch.linalg.vector_norm(horiz, dim=-1).max(dim=-1).values   # (B,T-1)
        skating |= contact & (slide > slide_thr * scale)

    n = int(trans_valid.sum().item())
    return float((skating & trans_valid).sum().item()) / max(n, 1)


def jerk(joints: Tensor, lengths: Tensor | None = None, fps: float = 20.0) -> float:
    """Mean ‖Δ³ position‖ over joints and frames — the roughness of the motion.

    The same third-difference measure the app's per-clip analysis reports, so
    numbers here and there are comparable. Scaled to units of m/s³ using `fps`.
    """
    if joints.ndim != 4:
        raise ValueError(f"joints must be (B, T, J, 3), got {tuple(joints.shape)}")
    T = joints.shape[1]
    if T < 4:
        return 0.0
    d3 = joints[:, 3:] - 3 * joints[:, 2:-1] + 3 * joints[:, 1:-2] - joints[:, :-3]
    mag = torch.linalg.vector_norm(d3, dim=-1)                     # (B, T-3, J)
    valid = _valid_frames(T, lengths, joints.shape[0], joints.device)[:, 3:]
    w = valid.unsqueeze(-1).to(mag.dtype)
    total = float((mag * w).sum().item())
    n = float(w.sum().item() * mag.shape[-1])
    return (total / n) * (float(fps) ** 3) if n else 0.0


def root_speed(joints: Tensor, lengths: Tensor | None = None, fps: float = 20.0) -> float:
    """Mean pelvis speed (m/s) — a sanity read that a constraint has not frozen
    or catapulted the body."""
    T = joints.shape[1]
    if T < 2:
        return 0.0
    step = torch.linalg.vector_norm(joints[:, 1:, 0, :] - joints[:, :-1, 0, :], dim=-1)
    valid = _valid_frames(T, lengths, joints.shape[0], joints.device)
    w = (valid[:, :-1] & valid[:, 1:]).to(step.dtype)
    n = float(w.sum().item())
    return float((step * w).sum().item()) / n * float(fps) if n else 0.0


def motion_quality(
    joints: Tensor,
    lengths: Tensor | None = None,
    fps: float = 20.0,
) -> dict[str, float]:
    """All three physical readouts for one batch of clips."""
    return {
        "foot_skate_ratio": foot_skate_ratio(joints, lengths, fps=fps),
        "jerk": jerk(joints, lengths, fps=fps),
        "root_speed": root_speed(joints, lengths, fps=fps),
    }
