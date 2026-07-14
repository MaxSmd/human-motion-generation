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

from dataclasses import dataclass

import torch
from torch import Tensor

from shared.geometry import recover_joints_from_ric

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
