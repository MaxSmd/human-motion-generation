"""Tests for MARDM's essential (67-D) representation + eval bridge.

The full essential->263 bridge needs the `external/HumanML3D` submodule (for
inverse kinematics) and is validated on the cluster; here we cover the parts
that run without it: encoding, the 263-prefix relation, recover-joints
consistency, z-normalization, stats, and the bridge's missing-submodule guard.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from mardm.representation import (
    ESSENTIAL_DIM,
    EssentialRepresentation,
    compute_essential_stats,
    denormalize,
    encode_essential,
    essential_to_h3d,
)
from rmg.representation.humanml3d_io import tplusr_to_h3d_features_with_quats
from rmg.representation.skeleton import NUM_JOINTS, Skeleton


def _synthetic_skeleton() -> Skeleton:
    torch.manual_seed(0)
    offsets = torch.randn(NUM_JOINTS, 3) * 0.1
    # Guarantee a non-degenerate facing direction (distinct L/R hips + shoulders).
    offsets[1, 0], offsets[2, 0] = 0.10, -0.10     # L_Hip / R_Hip
    offsets[16, 0], offsets[17, 0] = 0.12, -0.12   # L_Shoulder / R_Shoulder
    return Skeleton(offsets=offsets)


def _synthetic_motion(T: int = 24) -> tuple[torch.Tensor, torch.Tensor]:
    """Root yaw-only rotation + a translation random walk; other joints identity."""
    torch.manual_seed(1)
    quats = torch.zeros(T, NUM_JOINTS, 4)
    quats[..., 0] = 1.0  # identity everywhere
    yaw = torch.cumsum(torch.randn(T) * 0.05, dim=0)
    quats[:, 0, 0] = torch.cos(yaw / 2)
    quats[:, 0, 2] = torch.sin(yaw / 2)  # pure-Y (yaw) root rotation
    translation = torch.cumsum(torch.randn(T, 3) * 0.02, dim=0)
    return translation, quats


def test_encode_shape_and_finite():
    skel = _synthetic_skeleton()
    translation, quats = _synthetic_motion(T=24)
    feats = encode_essential(translation, quats, skel)
    assert feats.shape == (23, ESSENTIAL_DIM)  # T-1 frames
    assert torch.isfinite(feats).all()


def test_essential_is_263_prefix():
    skel = _synthetic_skeleton()
    translation, quats = _synthetic_motion()
    essential = encode_essential(translation, quats, skel)
    full = tplusr_to_h3d_features_with_quats(translation, quats, skel)
    assert full.shape[-1] == 263
    torch.testing.assert_close(essential, full[:, :ESSENTIAL_DIM])


def test_recover_root_height_matches_feature():
    from rmg.representation.humanml3d_io import recover_joints_from_ric

    skel = _synthetic_skeleton()
    translation, quats = _synthetic_motion()
    essential = encode_essential(translation, quats, skel)
    joints = recover_joints_from_ric(essential)              # (T-1, 22, 3)
    assert joints.shape == (essential.shape[0], NUM_JOINTS, 3)
    # Root height is stored directly at feature dim 3 and copied back verbatim.
    torch.testing.assert_close(joints[:, 0, 1], essential[:, 3])


def test_normalization_roundtrip():
    skel = _synthetic_skeleton()
    translation, quats = _synthetic_motion()
    raw = encode_essential(translation, quats, skel)
    mean, std = raw.mean(0), raw.std(0).clamp_min(1e-6)
    rep = EssentialRepresentation(mean=mean, std=std)
    normed = rep.encode_clip(translation, quats, skeleton=skel)
    torch.testing.assert_close(denormalize(normed, mean, std), raw, atol=1e-4, rtol=1e-4)


def test_compute_essential_stats():
    @dataclass
    class _Sample:
        x1: torch.Tensor

    class _DS:
        def __init__(self, n):
            torch.manual_seed(2)
            self._x = [_Sample(torch.randn(10 + i, ESSENTIAL_DIM)) for i in range(n)]

        def __len__(self):
            return len(self._x)

        def __getitem__(self, i):
            return self._x[i]

    mean, std = compute_essential_stats(_DS(8))
    assert mean.shape == (ESSENTIAL_DIM,)
    assert std.shape == (ESSENTIAL_DIM,)
    assert (std > 0).all()


def test_encode_clip_requires_skeleton():
    rep = EssentialRepresentation()
    translation, quats = _synthetic_motion()
    with pytest.raises(ValueError):
        rep.encode_clip(translation, quats, skeleton=None)


def test_eval_bridge_missing_submodule_raises():
    skel = _synthetic_skeleton()
    essential = torch.randn(20, ESSENTIAL_DIM)
    with pytest.raises(FileNotFoundError):
        essential_to_h3d(essential, skel, humanml3d_repo="/nonexistent/HumanML3D")
