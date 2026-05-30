"""MARDM "essential" HumanML3D representation + evaluation bridge.

MARDM trains on the *essential* feature group — the first 67 dims of the
standard 263-D HumanML3D feature: root angular velocity (1), root XZ linear
velocity (2), root height (1), and the 21 local joint positions ("ric", 63).
The remaining 196 dims (6D joint rotations, joint velocities, foot contacts)
are redundant for animation and are dropped. See Meng et al. 2025 §3.1.

This module reuses `rmg`'s 263-D machinery rather than forking it:
  * `encode_essential` / `EssentialRepresentation` — packed (translation,
    quaternions) -> 67-D feature (the first 67 dims of `rmg`'s 263-D encoding);
  * `compute_essential_stats` — train-split per-dim mean/std for z-normalization;
  * `essential_to_h3d` — the eval bridge back to full 263-D for the Guo
    evaluator: de-normalize -> recover joints -> inverse kinematics -> 263-D.

The IK step lazily imports `external/HumanML3D`'s upstream `Skeleton`, the same
one `scripts/prepare_humanml3d.py` uses to build the dataset's quaternions, so
the conversion stays consistent with the packed data.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from rmg.representation.humanml3d_io import (
    recover_joints_from_ric,
    tplusr_to_h3d_features_with_quats,
)
from rmg.representation.skeleton import Skeleton

# root(1+2+1) + (J-1)*3 local joint positions = 4 + 63
ESSENTIAL_DIM = 67
# HumanML3D face joints: R_Hip, L_Hip, R_Shoulder, L_Shoulder.
FACE_JOINT_INDX = (2, 1, 17, 16)


# ---------------------------------------------------------------------------
# Encode: packed (translation, quaternions) -> 67-D essential feature
# ---------------------------------------------------------------------------


def encode_essential(translation: Tensor, quaternions: Tensor, skeleton: Skeleton) -> Tensor:
    """(T, 3) + (T, 22, 4) -> (T-1, 67). The first 67 dims of the 263-D feature."""
    feats = tplusr_to_h3d_features_with_quats(translation, quaternions, skeleton)
    return feats[:, :ESSENTIAL_DIM]


class EssentialRepresentation:
    """Adapter so `rmg`'s `HumanML3DDataset` yields 67-D essential features.

    Implements just the `encode_clip(translation, quaternions, skeleton=...)`
    method the dataset calls. When `mean`/`std` are set the features are
    z-normalized (MARDM standardizes the essential dims; `rmg`'s manifold
    normalization does not apply here).

    NB: the encoded feature has length T-1 (velocity diff), one shorter than the
    input clip — callers that track sequence length should use the encoded
    length, not the raw frame count.
    """

    def __init__(self, mean: Tensor | None = None, std: Tensor | None = None):
        self.mean = mean
        self.std = std

    def set_stats(self, mean: Tensor, std: Tensor) -> None:
        self.mean, self.std = mean, std

    def encode_clip(self, translation: Tensor, quaternions: Tensor,
                    skeleton: Skeleton | None = None) -> Tensor:
        if skeleton is None:
            raise ValueError("EssentialRepresentation.encode_clip needs a skeleton (target offsets).")
        feats = encode_essential(translation, quaternions, skeleton)
        if self.mean is not None and self.std is not None:
            feats = (feats - self.mean.to(feats)) / self.std.to(feats).clamp_min(1e-8)
        return feats


def denormalize(feats: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """Invert z-normalization: feats * std + mean."""
    return feats * std.to(feats).clamp_min(1e-8) + mean.to(feats)


# ---------------------------------------------------------------------------
# Train-split statistics for z-normalization
# ---------------------------------------------------------------------------


def compute_essential_stats(dataset, max_clips: int | None = None) -> tuple[Tensor, Tensor]:
    """Per-dim mean/std over every frame in `dataset`.

    `dataset` must yield samples whose `.x1` is the *raw* (unnormalized) 67-D
    essential feature — i.e. an `EssentialRepresentation` with no stats set.
    Accumulates in float64 for numerical stability.
    """
    count = 0
    total = torch.zeros(ESSENTIAL_DIM, dtype=torch.float64)
    total_sq = torch.zeros(ESSENTIAL_DIM, dtype=torch.float64)
    n = len(dataset) if max_clips is None else min(max_clips, len(dataset))
    for i in range(n):
        x = dataset[i].x1.double()  # (L, 67)
        total += x.sum(dim=0)
        total_sq += (x * x).sum(dim=0)
        count += x.shape[0]
    if count == 0:
        raise ValueError("compute_essential_stats: dataset yielded no frames")
    mean = total / count
    var = (total_sq / count) - mean * mean
    std = var.clamp_min(1e-12).sqrt()
    return mean.float(), std.float()


# ---------------------------------------------------------------------------
# Eval bridge: essential 67-D -> full 263-D (via recover-joints + upstream IK)
# ---------------------------------------------------------------------------


def _import_upstream_skeleton(humanml3d_repo: Path):
    """Vendor the HumanML3D `Skeleton` + offsets at runtime (mirrors
    `scripts/prepare_humanml3d.py`). Polyfills `np.float` (removed in numpy
    1.24+) and silences upstream's per-call `torch.cross` deprecation spam."""
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    warnings.filterwarnings(
        "ignore", message=r"Using torch\.cross without specifying the dim arg is deprecated.*"
    )
    repo = Path(humanml3d_repo).resolve()
    if not (repo / "common" / "skeleton.py").exists():
        raise FileNotFoundError(
            f"HumanML3D submodule not initialized at {repo}; the essential->263 eval "
            "bridge needs it for inverse kinematics. Run `git submodule update --init "
            f"{humanml3d_repo}` or point at the shared cluster checkout."
        )
    sys.path.insert(0, str(repo))
    from common.skeleton import Skeleton as UpstreamSkeleton  # type: ignore
    from paramUtil import t2m_kinematic_chain, t2m_raw_offsets  # type: ignore

    return UpstreamSkeleton, t2m_raw_offsets, t2m_kinematic_chain


def _ik_quaternions(joints: Tensor, target_offsets: Tensor, humanml3d_repo: Path) -> Tensor:
    """World joint positions (T, 22, 3) -> per-joint local quaternions (T, 22, 4)
    via the upstream HumanML3D IK (the same one that built the dataset)."""
    UpstreamSkeleton, raw_offsets, kinematic_chain = _import_upstream_skeleton(humanml3d_repo)
    skel = UpstreamSkeleton(torch.from_numpy(raw_offsets), kinematic_chain, "cpu")
    skel.set_offset(target_offsets.detach().cpu())
    quat = skel.inverse_kinematics_np(
        joints.detach().cpu().numpy(), list(FACE_JOINT_INDX), smooth_forward=False
    )
    return torch.from_numpy(quat).float()


def essential_to_h3d(
    essential: Tensor,
    skeleton: Skeleton,
    *,
    mean: Tensor | None = None,
    std: Tensor | None = None,
    humanml3d_repo: str | Path = "external/HumanML3D",
) -> Tensor:
    """Generated essential feature (L, 67) -> full (L-1, 263) for the Guo evaluator.

    Pipeline: de-normalize -> recover canonical joint positions (`rmg`'s
    `recover_joints_from_ric`, which reads only the first 67 dims) -> upstream
    IK to per-joint quaternions -> `rmg`'s 263-D featurization (FK + features),
    matching how the packed dataset was built.
    """
    if essential.dim() != 2 or essential.shape[-1] != ESSENTIAL_DIM:
        raise ValueError(f"essential must be (L, {ESSENTIAL_DIM}), got {tuple(essential.shape)}")
    if mean is not None and std is not None:
        essential = denormalize(essential, mean, std)
    joints = recover_joints_from_ric(essential)          # (L, 22, 3), canonical frame
    quats = _ik_quaternions(joints, skeleton.offsets, Path(humanml3d_repo))
    translation = joints[:, 0, :].contiguous()           # (L, 3) recovered root trajectory
    return tplusr_to_h3d_features_with_quats(translation, quats, skeleton)
