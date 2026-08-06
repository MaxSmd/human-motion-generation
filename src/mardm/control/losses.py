"""Differentiable spatial-control loss + OmniControl-style metrics for MARDM.

The control signal is a sparse set of world-frame joint targets: which
(frame, joint) cells are constrained and where they should be. The loss path
is fully differentiable — AE.decode -> denormalize -> recover_joints_from_ric
are all torch ops — so gradients flow from the joint-space error back to the
AE latents (and through the diffusion head to its condition `z`, see
`mardm.control.guidance`).

Frames here are *decoded* frames: a latent sequence of length L decodes to
T = L * ae.downsample_rate essential frames, and joints are recovered per
decoded frame. Control signals must be built at that resolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from shared.geometry import FOOT_CONTACT_IDX, recover_joints_from_ric

from ..representation import denormalize


@dataclass
class ControlSignal:
    """Sparse world-frame joint targets over decoded frames.

    targets: (B, T, J, 3) desired world-frame positions (arbitrary where unmasked)
    mask:    (B, T, J) bool — True where the (frame, joint) cell is constrained
    """

    targets: Tensor
    mask: Tensor

    def __post_init__(self) -> None:
        if self.targets.shape[:-1] != self.mask.shape:
            raise ValueError(
                f"targets {tuple(self.targets.shape)} / mask {tuple(self.mask.shape)} mismatch"
            )

    def to(self, device: torch.device) -> "ControlSignal":
        return ControlSignal(self.targets.to(device), self.mask.to(device))

    @property
    def num_constraints(self) -> int:
        return int(self.mask.sum())

    @staticmethod
    def from_gt_joints(joints: Tensor, *, joint_ids: list[int], num_keyframes: int,
                       length: int | None = None,
                       generator: torch.Generator | None = None) -> "ControlSignal":
        """Sample sparse waypoints from a GT clip (OmniControl-style protocol).

        joints: (T, J, 3) ground-truth world-frame positions for ONE clip.
        Picks `num_keyframes` frames uniformly without replacement from the
        first `length` frames (default: all) and constrains `joint_ids` there.
        Returns a batch-of-1 signal spanning the full T frames.
        """
        T, J, _ = joints.shape
        usable = T if length is None else min(length, T)
        k = min(num_keyframes, usable)
        frame_idx = torch.randperm(usable, generator=generator)[:k]
        mask = torch.zeros(1, T, J, dtype=torch.bool)
        for j in joint_ids:
            mask[0, frame_idx, j] = True
        return ControlSignal(joints.unsqueeze(0).clone(), mask)


def latents_to_joints(latents: Tensor, ae, mean: Tensor, std: Tensor) -> Tensor:
    """(B, L, ae_dim) latents -> (B, L*ds, 22, 3) world-frame joints. Differentiable.

    Layout note: takes the (B, L, ae_dim) sequence-major layout the guidance
    loop works in and permutes to AE.decode's channels-first internally.
    """
    essential = ae.decode(latents.permute(0, 2, 1))          # (B, L*ds, 67), normalized
    essential = denormalize(essential, mean, std)
    return recover_joints_from_ric(essential)


def control_loss(joints: Tensor, signal: ControlSignal) -> Tensor:
    """Mean Euclidean distance over constrained cells. Differentiable.

    joints: (B, T, J, 3); signal frames beyond T are ignored (and vice versa),
    so a control signal built on GT length T_gt >= decoded length works as-is.
    """
    T = min(joints.shape[1], signal.targets.shape[1])
    mask = signal.mask[:, :T]
    if not mask.any():
        return joints.sum() * 0.0
    dist = torch.linalg.vector_norm(joints[:, :T] - signal.targets[:, :T], dim=-1)  # (B, T, J)
    return (dist * mask).sum() / mask.sum()


def dynamics_loss(joints: Tensor, reference: Tensor, signal: ControlSignal,
                  *, root_relative: bool = True,
                  exclude: list[int] | tuple[int, ...] | None = None) -> Tensor:
    """Anchor the velocity profile to an unguided reference sample.

    ‖Δ_t J_g − Δ_t J_u‖² averaged over frames and over joints that are NEVER
    constrained (a joint pinned by a waypoint must be free to move). This is
    the term that stops the optimizer from collapsing onto a static pose: a
    frozen clip minimizes `control_loss` whenever the prior's habitual motion
    conflicts with the waypoints, but it maximizes this one.

    With `root_relative` (default) velocities are taken after subtracting the
    pelvis, so the term constrains ARTICULATION (gait) only and stays agnostic
    to where the body travels — anchoring world-frame velocities would fight
    the waypoints directly, since every joint's world position carries the root
    translation. Set False to reproduce the literal world-frame form.
    """
    T = min(joints.shape[1], reference.shape[1])
    g, u = joints[:, :T], reference[:, :T]
    if root_relative:
        g = g - g[:, :, 0:1]
        u = u - u[:, :, 0:1]
    if T < 2:
        return joints.sum() * 0.0
    dg, du = g[:, 1:] - g[:, :-1], u[:, 1:] - u[:, :-1]
    free = ~signal.mask.any(dim=1).any(dim=0)                # (J,) never constrained
    if exclude:                                              # joints an angle objective drives
        free = free.clone()                                 # must not also be anchored here
        free[list(exclude)] = False
    if not free.any():
        return joints.sum() * 0.0
    return (dg[:, :, free] - du[:, :, free]).pow(2).sum(-1).mean()


def foot_skate_loss(joints: Tensor, *, height: float = 0.05) -> Tensor:
    """Horizontal foot velocity gated on foot height. Differentiable.

    The 4 binary contact channels live in the 196 dims dropped from the 67-D
    essential group, so contact has to be inferred geometrically: a linear gate
    `(1 - h/height)+` that is 1 on the floor and 0 above `height` metres,
    multiplying the per-frame horizontal displacement of ankles and toes.

    Note this term alone cannot prevent freezing (a static pose has zero foot
    velocity and so zero skate) — it is the complement of `dynamics_loss`,
    which supplies the motion the gate then has to keep honest.
    """
    if joints.shape[1] < 2:
        return joints.sum() * 0.0
    feet = joints[:, :, list(FOOT_CONTACT_IDX)]              # (B, T, 4, 3)
    horiz = feet[:, 1:, :, [0, 2]] - feet[:, :-1, :, [0, 2]]
    speed = torch.linalg.vector_norm(horiz, dim=-1)          # (B, T-1, 4)
    gate = (1.0 - feet[:, :-1, :, 1] / height).clamp(min=0.0, max=1.0)
    return (speed * gate).mean()


# --------------------------------------------------------------------------- bend

def _bend_angle(joints: Tensor, triplet: list[int] | tuple[int, ...]) -> Tensor:
    """Interior bend angle (radians) at the middle joint of `triplet` = (a, b, c).

    RMG's bend definition (flow.constraints), evaluated on decoded positions
    instead of quaternions: the angle between the incoming bone a->b and the
    outgoing bone b->c. 0 = straight (collinear bones), larger = more flexed.
    For the knee, triplet = (hip, knee, ankle). joints (B, T, J, 3) -> (B, T).
    """
    a, b, c = int(triplet[0]), int(triplet[1]), int(triplet[2])
    u = joints[:, :, b] - joints[:, :, a]                    # incoming bone
    w = joints[:, :, c] - joints[:, :, b]                    # outgoing bone
    denom = (torch.linalg.vector_norm(u, dim=-1)
             * torch.linalg.vector_norm(w, dim=-1)).clamp_min(1e-8)
    cos = ((u * w).sum(-1) / denom).clamp(-1.0, 1.0)
    return torch.arccos(cos)                                 # (B, T)


def bend_loss(joints: Tensor, triplet: list[int] | tuple[int, ...], max_deg: float,
              window: tuple[int, int] | None = None) -> Tensor:
    """Soft one-sided limit keeping the bend at `triplet` <= `max_deg`. Differentiable.

    ReLU(bend - max)^2 averaged over frames — the soft counterpart of RMG's hard
    per-step quaternion clamp, here as an objective on decoded joint positions
    (we have no rotation coordinate to project). `window` = (start, end) restricts
    the penalty to a half-open frame range; None = whole clip.
    """
    bend = _bend_angle(joints, triplet)                      # (B, T)
    if window is not None:
        bend = bend[:, int(window[0]):int(window[1])]
    if bend.numel() == 0:
        return joints.sum() * 0.0
    return (bend - math.radians(max_deg)).clamp_min(0.0).pow(2).mean()


@torch.no_grad()
def bend_metrics(joints: Tensor, triplet: list[int] | tuple[int, ...], lengths: Tensor,
                 max_deg: float, window: tuple[int, int] | None = None) -> dict[str, float]:
    """Joint-angle satisfaction diagnostics over valid frames.

    mean_bend_deg / max_bend_deg: bend-angle stats (deg, 0 = straight)
    violation_frac: fraction of valid frames whose bend exceeds `max_deg`
    """
    bend = torch.rad2deg(_bend_angle(joints, triplet))       # (B, T)
    _, t = bend.shape
    idx = torch.arange(t, device=bend.device)
    valid = idx.unsqueeze(0) < lengths.to(bend.device).unsqueeze(1)
    if window is not None:
        w = torch.zeros_like(valid)
        w[:, int(window[0]):int(window[1])] = True
        valid = valid & w
    if not bool(valid.any()):
        return {"mean_bend_deg": 0.0, "max_bend_deg": 0.0, "violation_frac": 0.0}
    n = valid.sum().clamp(min=1)
    masked = torch.where(valid, bend, torch.zeros_like(bend))
    return {
        "mean_bend_deg": float(masked.sum() / n),
        "max_bend_deg": float(bend[valid].max()),
        "violation_frac": float(((bend > max_deg) & valid).sum() / n),
    }


@torch.no_grad()
def motion_metrics(joints: Tensor, lengths: Tensor, *, height: float = 0.05,
                   skate_speed: float = 0.0025) -> dict[str, float]:
    """Realism diagnostics FID is blind to: skating, damping, smoothness.

    foot_skate:  fraction of (frame, foot) cells that are grounded (< `height`)
                 yet slide more than `skate_speed` m/frame
    motion_mag:  mean root-relative joint speed (m/frame) — freezing detector
    jerk:        mean magnitude of the third position difference (m/frame³)
    joints: (B, T, J, 3); `lengths` (B,) gives the valid frame count per clip.
    """
    b, t, j, _ = joints.shape
    idx = torch.arange(t, device=joints.device)
    valid = idx.unsqueeze(0) < lengths.to(joints.device).unsqueeze(1)   # (B, T)

    feet = joints[:, :, list(FOOT_CONTACT_IDX)]
    horiz = torch.linalg.vector_norm(
        feet[:, 1:, :, [0, 2]] - feet[:, :-1, :, [0, 2]], dim=-1)       # (B, T-1, 4)
    grounded = feet[:, :-1, :, 1] < height
    v1 = valid[:, 1:].unsqueeze(-1)
    n_ground = (grounded & v1).sum().clamp(min=1)
    skate = ((horiz > skate_speed) & grounded & v1).sum() / n_ground

    local = joints - joints[:, :, 0:1]
    speed = torch.linalg.vector_norm(local[:, 1:] - local[:, :-1], dim=-1)  # (B, T-1, J)
    mag = (speed * v1).sum() / (v1.sum() * j).clamp(min=1)

    d3 = joints[:, 3:] - 3 * joints[:, 2:-1] + 3 * joints[:, 1:-2] - joints[:, :-3]
    v3 = valid[:, 3:].unsqueeze(-1)
    jerk = ((torch.linalg.vector_norm(d3, dim=-1) * v3).sum()
            / (v3.sum() * j).clamp(min=1)) if t > 3 else torch.zeros((), device=joints.device)
    return {
        "foot_skate": float(skate),
        "motion_mag": float(mag),
        "jerk": float(jerk),
    }


@torch.no_grad()
def control_metrics(joints: Tensor, signal: ControlSignal,
                    threshold: float = 0.5) -> dict[str, float]:
    """OmniControl protocol metrics over constrained cells.

    traj_err: fraction of sequences with ANY constrained cell off by > threshold
    loc_err:  fraction of constrained cells off by > threshold
    avg_err:  mean Euclidean distance (meters) over constrained cells
    """
    T = min(joints.shape[1], signal.targets.shape[1])
    mask = signal.mask[:, :T]
    dist = torch.linalg.vector_norm(joints[:, :T] - signal.targets[:, :T], dim=-1)
    dist = torch.where(mask, dist, torch.zeros_like(dist))
    per_seq_any = (dist > threshold).flatten(1).any(dim=1)   # (B,)
    n = mask.sum().clamp(min=1)
    return {
        "traj_err": float(per_seq_any.float().mean()),
        "loc_err": float(((dist > threshold) & mask).sum() / n),
        "avg_err": float(dist.sum() / n),
    }
