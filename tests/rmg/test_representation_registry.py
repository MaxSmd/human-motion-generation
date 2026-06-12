"""Tests for the Representation registry (paper Figure 3 ablations).

Verifies:
  * each registered name builds a valid (Manifold, prior_mu, ambient_dim) triple
  * encode_clip → on-manifold tensor of correct shape
  * to_h3d_features → (T-1, 263) shape
  * dT / dR config stubs raise a clear error from `build_representation`
"""

from __future__ import annotations

import math

import pytest
import torch

from rmg.representation import (
    H3D_FEATURE_DIM,
    NUM_JOINTS,
    Skeleton,
    TPRepresentation,
    TRPRepresentation,
    TRRepresentation,
    build_representation,
)


def _toy_offsets() -> torch.Tensor:
    o = torch.zeros(NUM_JOINTS, 3, dtype=torch.float32)
    o[1] = torch.tensor([0.10, -0.05, 0.0])
    o[2] = torch.tensor([-0.10, -0.05, 0.0])
    o[3] = torch.tensor([0.0, 0.10, 0.0])
    o[4] = torch.tensor([0.0, -0.40, 0.0])
    o[5] = torch.tensor([0.0, -0.40, 0.0])
    o[6] = torch.tensor([0.0, 0.10, 0.0])
    o[7] = torch.tensor([0.0, -0.40, 0.0])
    o[8] = torch.tensor([0.0, -0.40, 0.0])
    o[9] = torch.tensor([0.0, 0.10, 0.0])
    o[10] = torch.tensor([0.0, -0.05, 0.10])
    o[11] = torch.tensor([0.0, -0.05, 0.10])
    o[12] = torch.tensor([0.0, 0.20, 0.0])
    o[13] = torch.tensor([0.05, 0.10, 0.0])
    o[14] = torch.tensor([-0.05, 0.10, 0.0])
    o[15] = torch.tensor([0.0, 0.10, 0.0])
    o[16] = torch.tensor([0.10, 0.0, 0.0])
    o[17] = torch.tensor([-0.10, 0.0, 0.0])
    o[18] = torch.tensor([0.30, 0.0, 0.0])
    o[19] = torch.tensor([-0.30, 0.0, 0.0])
    o[20] = torch.tensor([0.30, 0.0, 0.0])
    o[21] = torch.tensor([-0.30, 0.0, 0.0])
    return o


def _toy_skeleton() -> Skeleton:
    return Skeleton(offsets=_toy_offsets())


def _identity_quats(T: int) -> torch.Tensor:
    q = torch.zeros(T, NUM_JOINTS, 4, dtype=torch.float32)
    q[..., 0] = 1.0
    return q


def _articulated_quats(T: int, seed: int = 0) -> torch.Tensor:
    """Small non-degenerate per-joint rotations. A perfectly rigid (all-identity)
    body makes upstream `process_file` hit an arccos singularity in the
    rotation-invariant features; real motion never does, so the H3D-feature
    test uses an articulated body."""
    g = torch.Generator().manual_seed(seed)
    axis = torch.randn(T, NUM_JOINTS, 3, generator=g)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    angle = 0.3 * torch.rand(T, NUM_JOINTS, 1, generator=g)
    return torch.cat([torch.cos(angle / 2), axis * torch.sin(angle / 2)], dim=-1)


def _trans(T: int) -> torch.Tensor:
    t = torch.zeros(T, 3, dtype=torch.float32)
    t[:, 0] = torch.linspace(0.0, 0.5, T)
    t[:, 1] = 0.9
    return t


# ---------------------------------------------------------------------------
# Registry resolution
# ---------------------------------------------------------------------------


def test_build_representation_known_names() -> None:
    assert isinstance(build_representation("tr"), TRRepresentation)
    assert isinstance(build_representation("tp"), TPRepresentation)
    assert isinstance(build_representation("trp"), TRPRepresentation)


def test_build_representation_unknown_raises_with_helpful_message() -> None:
    with pytest.raises(ValueError, match="dT/dR"):
        build_representation("dt_plus_r")
    with pytest.raises(ValueError, match="dT/dR"):
        build_representation("t_plus_dr")


# ---------------------------------------------------------------------------
# TR
# ---------------------------------------------------------------------------


def test_tr_ambient_and_encode() -> None:
    rep = TRRepresentation()
    assert rep.ambient_dim == 3 + 4 * 22
    M = rep.build_manifold()
    assert M.ambient_dim == rep.ambient_dim
    flat = rep.encode_clip(_trans(10), _identity_quats(10), skeleton=_toy_skeleton())
    assert flat.shape == (10, rep.ambient_dim)
    assert M.validate(flat, atol=1e-6).all()


def test_tr_to_h3d_shape() -> None:
    rep = TRRepresentation()
    sk = _toy_skeleton()
    flat = rep.encode_clip(_trans(20), _articulated_quats(20), skeleton=sk)
    feat = rep.to_h3d_features(flat, sk)
    assert feat.shape == (19, H3D_FEATURE_DIM)
    assert torch.isfinite(feat).all()


# ---------------------------------------------------------------------------
# TP
# ---------------------------------------------------------------------------


def test_tp_ambient_and_encode_lies_on_manifold() -> None:
    rep = TPRepresentation()
    sk = _toy_skeleton()
    assert rep.ambient_dim == 3 + 3 * 22
    M = rep.build_manifold()
    assert M.ambient_dim == rep.ambient_dim

    flat = rep.encode_clip(_trans(8), _identity_quats(8), skeleton=sk)
    assert flat.shape == (8, rep.ambient_dim)
    assert M.validate(flat, atol=1e-5).all()


def test_tp_prior_mu_from_skeleton_lies_on_manifold() -> None:
    rep = TPRepresentation()
    sk = _toy_skeleton()
    mu = rep.prior_mu_from_skeleton(sk, dtype=torch.float64).unsqueeze(0)
    M = rep.build_manifold()
    assert M.validate(mu, atol=1e-7).all()


def test_tp_to_h3d_shape() -> None:
    rep = TPRepresentation()
    sk = _toy_skeleton()
    flat = rep.encode_clip(_trans(16), _identity_quats(16), skeleton=sk)
    feat = rep.to_h3d_features(flat, sk)
    assert feat.shape == (15, H3D_FEATURE_DIM)
    assert torch.isfinite(feat).all()


# ---------------------------------------------------------------------------
# TRP
# ---------------------------------------------------------------------------


def test_trp_ambient_and_encode_lies_on_manifold() -> None:
    rep = TRPRepresentation()
    sk = _toy_skeleton()
    assert rep.ambient_dim == 3 + 4 * 22 + 3 * 22
    M = rep.build_manifold()
    flat = rep.encode_clip(_trans(8), _identity_quats(8), skeleton=sk)
    assert flat.shape == (8, rep.ambient_dim)
    assert M.validate(flat, atol=1e-5).all()


def test_trp_prior_mu_from_skeleton_lies_on_manifold() -> None:
    rep = TRPRepresentation()
    sk = _toy_skeleton()
    mu = rep.prior_mu_from_skeleton(sk, dtype=torch.float64).unsqueeze(0)
    M = rep.build_manifold()
    assert M.validate(mu, atol=1e-7).all()


def test_trp_decode_via_rotation_vs_preshape_differ_for_articulated_motion() -> None:
    """With non-identity per-joint rotations, the two decode_via paths must
    diverge — that's exactly the Figure 3a 'recovered by R' vs 'recovered by P'
    comparison. (For identity rotations the body is rigid and both agree.)"""
    sk = _toy_skeleton()
    T = 12

    # non-identity rotations: small Y-rotation per joint, varying with time
    quats = _identity_quats(T)
    angles = torch.linspace(0.0, 0.3, T)
    for j in range(NUM_JOINTS):
        for t in range(T):
            a = angles[t] * (1.0 + 0.05 * j)
            quats[t, j, 0] = math.cos(a / 2.0)
            quats[t, j, 1] = math.sin(a / 2.0)  # x-axis rotation per joint
    quats = quats / quats.norm(dim=-1, keepdim=True)

    flat = TRPRepresentation().encode_clip(_trans(T), quats, skeleton=sk)
    rep_r = TRPRepresentation(decode_via="rotation")
    rep_p = TRPRepresentation(decode_via="preshape")
    fr = rep_r.to_h3d_features(flat, sk)
    fp = rep_p.to_h3d_features(flat, sk)
    assert fr.shape == fp.shape == (T - 1, H3D_FEATURE_DIM)
    assert not torch.allclose(fr, fp, atol=1e-6), \
        "rotation- vs preshape-recovery should disagree for articulated motion"
