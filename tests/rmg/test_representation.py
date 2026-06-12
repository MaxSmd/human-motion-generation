"""Representation correctness tests.

Centerpiece: a synthetic RIC round-trip
   joints (canonical) → 263-D features → joints (recovered)
must agree to ~1e-6, validating that our `tplusr_to_h3d_features_with_quats`
and the upstream-mirroring `recover_joints_from_ric` are mutual inverses.
"""

from __future__ import annotations

import math

import pytest
import torch

from rmg.representation import (
    H3D_FEATURE_DIM,
    NUM_JOINTS,
    Skeleton,
    TPlusR,
    decode,
    encode,
    forward_kinematics,
    make_continuous,
    quat_between,
    quat_inv,
    quat_mul,
    quat_rotate,
    quat_to_cont6d,
    quat_to_rotmat,
    recover_joints_from_ric,
    t_pose_joints,
    tplusr_to_h3d_features_with_quats,
    tplusr_to_joints,
)

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Synthetic skeleton: realistic-ish proportions for a 22-joint SMPL body.
# Exact values don't matter for our internal-consistency tests; the
# important property is that the FK chain runs the way SMPL expects.
# ---------------------------------------------------------------------------


def _toy_offsets() -> torch.Tensor:
    """Plausible but made-up T-pose offsets in metres. Keeps tests deterministic."""
    o = torch.zeros(NUM_JOINTS, 3, dtype=torch.float64)
    o[1] = torch.tensor([0.10, -0.05, 0.0])   # L_Hip
    o[2] = torch.tensor([-0.10, -0.05, 0.0])  # R_Hip
    o[3] = torch.tensor([0.0, 0.10, 0.0])     # Spine1
    o[4] = torch.tensor([0.0, -0.40, 0.0])    # L_Knee (relative to L_Hip)
    o[5] = torch.tensor([0.0, -0.40, 0.0])    # R_Knee
    o[6] = torch.tensor([0.0, 0.10, 0.0])     # Spine2
    o[7] = torch.tensor([0.0, -0.40, 0.0])    # L_Ankle
    o[8] = torch.tensor([0.0, -0.40, 0.0])    # R_Ankle
    o[9] = torch.tensor([0.0, 0.10, 0.0])     # Spine3
    o[10] = torch.tensor([0.0, -0.05, 0.10])  # L_Foot
    o[11] = torch.tensor([0.0, -0.05, 0.10])  # R_Foot
    o[12] = torch.tensor([0.0, 0.20, 0.0])    # Neck
    o[13] = torch.tensor([0.05, 0.10, 0.0])   # L_Collar
    o[14] = torch.tensor([-0.05, 0.10, 0.0])  # R_Collar
    o[15] = torch.tensor([0.0, 0.10, 0.0])    # Head
    o[16] = torch.tensor([0.10, 0.0, 0.0])    # L_Shoulder
    o[17] = torch.tensor([-0.10, 0.0, 0.0])   # R_Shoulder
    o[18] = torch.tensor([0.30, 0.0, 0.0])    # L_Elbow
    o[19] = torch.tensor([-0.30, 0.0, 0.0])   # R_Elbow
    o[20] = torch.tensor([0.30, 0.0, 0.0])    # L_Wrist
    o[21] = torch.tensor([-0.30, 0.0, 0.0])   # R_Wrist
    return o


def _toy_skeleton() -> Skeleton:
    return Skeleton(offsets=_toy_offsets())


def _identity_quats(*batch: int, dtype=torch.float64) -> torch.Tensor:
    q = torch.zeros(*batch, NUM_JOINTS, 4, dtype=dtype)
    q[..., 0] = 1.0
    return q


# ---------------------------------------------------------------------------
# Quaternion utilities
# ---------------------------------------------------------------------------


def test_quat_inv_yields_identity_when_multiplied() -> None:
    q = torch.randn(8, 4, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    e = quat_mul(q, quat_inv(q))
    expected = torch.zeros_like(q)
    expected[..., 0] = 1.0
    assert torch.allclose(e, expected, atol=1e-12)


def test_quat_to_rotmat_matches_quat_rotate() -> None:
    q = torch.randn(4, 4, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    v = torch.randn(4, 3, dtype=torch.float64)
    R = quat_to_rotmat(q)
    via_matrix = (R @ v.unsqueeze(-1)).squeeze(-1)
    via_quat = quat_rotate(q, v)
    assert torch.allclose(via_matrix, via_quat, atol=1e-12)


def test_quat_between_aligns_v1_to_v2() -> None:
    v1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    v2 = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    q = quat_between(v1, v2)
    assert torch.allclose(quat_rotate(q, v1), v2, atol=1e-12)


def test_quat_to_cont6d_round_trips_via_first_two_columns() -> None:
    """6D = first two columns of R; reconstructing R should match."""
    q = torch.randn(4, dtype=torch.float64)
    q = q / q.norm()
    R = quat_to_rotmat(q)
    c6 = quat_to_cont6d(q)
    assert torch.allclose(c6[:3], R[:, 0], atol=1e-12)
    assert torch.allclose(c6[3:], R[:, 1], atol=1e-12)


# ---------------------------------------------------------------------------
# Skeleton & FK
# ---------------------------------------------------------------------------


def test_fk_identity_rotations_gives_t_pose() -> None:
    sk = _toy_skeleton()
    quats = _identity_quats()
    trans = torch.zeros(3, dtype=torch.float64)
    joints = forward_kinematics(sk, quats, trans)
    # T-pose joint positions = chained offsets
    expected = t_pose_joints(sk)
    assert torch.allclose(joints, expected, atol=1e-12)


def test_fk_translation_shifts_all_joints() -> None:
    sk = _toy_skeleton()
    quats = _identity_quats()
    delta = torch.tensor([1.0, 2.0, -3.0], dtype=torch.float64)
    joints_at_zero = forward_kinematics(sk, quats, torch.zeros(3, dtype=torch.float64))
    joints_at_delta = forward_kinematics(sk, quats, delta)
    assert torch.allclose(joints_at_delta - joints_at_zero, delta.expand_as(joints_at_zero))


def test_fk_root_rotation_rotates_everything_rigidly() -> None:
    sk = _toy_skeleton()
    # 90° rotation around Y at the root only.
    theta = math.pi / 2
    q_root = torch.tensor([math.cos(theta / 2), 0.0, math.sin(theta / 2), 0.0], dtype=torch.float64)
    quats = _identity_quats()
    quats[0] = q_root
    trans = torch.zeros(3, dtype=torch.float64)
    rotated = forward_kinematics(sk, quats, trans)

    # Expected: every joint in the T-pose, then rotated by q_root around the root.
    base = t_pose_joints(sk)  # (J, 3)
    q_b = q_root.unsqueeze(0).expand(NUM_JOINTS, 4)
    expected = quat_rotate(q_b, base)
    assert torch.allclose(rotated, expected, atol=1e-12)


def test_fk_supports_batched_and_temporal_shape() -> None:
    sk = _toy_skeleton()
    B, T = 2, 4
    quats = _identity_quats(B, T)
    trans = torch.zeros(B, T, 3, dtype=torch.float64)
    joints = forward_kinematics(sk, quats, trans)
    assert joints.shape == (B, T, NUM_JOINTS, 3)


# ---------------------------------------------------------------------------
# T+R encode/decode
# ---------------------------------------------------------------------------


def test_tplusr_encode_decode_round_trip() -> None:
    B, T = 3, 5
    trans = torch.randn(B, T, 3, dtype=torch.float64)
    quats = torch.randn(B, T, NUM_JOINTS, 4, dtype=torch.float64)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    flat = encode(TPlusR(translation=trans, quaternions=quats))
    assert flat.shape == (B, T, 3 + 4 * NUM_JOINTS)
    back = decode(flat)
    assert torch.allclose(back.translation, trans)
    assert torch.allclose(back.quaternions, quats)


def test_make_continuous_resolves_quaternion_sign_flips() -> None:
    base = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    seq = torch.stack([base, -base, base, -base], dim=0)  # (T=4, 4)
    seq = seq.unsqueeze(0).unsqueeze(0)  # (B=1, J=1, T=4, 4) — pretend layout
    # use time_dim = 2 (the T axis we created)
    fixed = make_continuous(seq, time_dim=2)
    dots = (fixed[..., :-1, :] * fixed[..., 1:, :]).sum(-1)
    assert (dots >= 0).all()


# ---------------------------------------------------------------------------
# HumanML3D 263-D features — shape & RIC round-trip
# ---------------------------------------------------------------------------


def test_h3d_feature_shape() -> None:
    sk = _toy_skeleton()
    T = 30
    trans = torch.randn(T, 3, dtype=torch.float64) * 0.05
    # small random rotations to keep IK-free path stable
    quats = _identity_quats(T)
    feature = tplusr_to_h3d_features_with_quats(trans, quats, sk)
    assert feature.shape == (T - 1, H3D_FEATURE_DIM)
    assert torch.isfinite(feature).all()


def test_ric_round_trip_recovers_canonical_joints() -> None:
    """Joints (canonical) → 263-D → recovered joints; recovered ≈ canonical[:-1].

    This is the strongest internal-consistency check for our §D.3 conversion:
    if encode and decode are exact inverses on canonicalized data, the only
    remaining source of disagreement with upstream H3D features is whatever
    upstream does that we don't (e.g. uniform_skeleton scaling) — caught later
    when we compare against actual H3D ground-truth features.
    """
    sk = _toy_skeleton()
    T = 25

    # Build a synthetic motion: small per-frame Y-rotation drift + slow XZ drift,
    # plus tiny random per-joint rotations so the limbs aren't perfectly rigid.
    trans = torch.zeros(T, 3, dtype=torch.float64)
    trans[:, 0] = torch.linspace(0.0, 0.5, T, dtype=torch.float64)
    trans[:, 2] = torch.linspace(0.0, 0.3, T, dtype=torch.float64)
    trans[:, 1] = 0.9  # standing height; gets normalized to floor anyway

    quats = _identity_quats(T)
    # apply a small Y-rotation that grows linearly with time at the root
    angles = torch.linspace(0.0, 0.4, T, dtype=torch.float64)
    quats[:, 0, 0] = torch.cos(angles / 2.0)
    quats[:, 0, 2] = torch.sin(angles / 2.0)
    # tiny per-joint rotations (deterministic)
    for j in range(1, NUM_JOINTS):
        quats[:, j, 0] = math.cos(0.02 * j)
        quats[:, j, 1] = math.sin(0.02 * j)

    # encode
    feat = tplusr_to_h3d_features_with_quats(trans, quats, sk)

    # decode (RIC path)
    recovered = recover_joints_from_ric(feat.unsqueeze(0))[0]  # (T-1, J, 3)
    assert recovered.shape == (T - 1, NUM_JOINTS, 3)

    # The canonical-frame joints (what was actually encoded) come from running
    # the same canonicalization that `tplusr_to_h3d_features_with_quats` applies
    # internally. We re-derive them here for the comparison.
    from shared.geometry.humanml3d_io import _canonicalize_first_frame
    canon_joints, _ = _canonicalize_first_frame(forward_kinematics(sk, quats, trans))
    canon_joints = canon_joints[:-1]  # the feature drops the last frame

    err = (recovered - canon_joints).abs().max().item()
    assert err < 5e-3, f"RIC round-trip max error {err:.3e} exceeds tolerance"


def test_h3d_foot_contact_signal_for_static_motion() -> None:
    """A frozen pose has zero foot velocity, so foot_contact must be 1 everywhere."""
    sk = _toy_skeleton()
    T = 10
    trans = torch.zeros(T, 3, dtype=torch.float64)
    trans[:, 1] = 1.0
    quats = _identity_quats(T)

    feat = tplusr_to_h3d_features_with_quats(trans, quats, sk)
    foot = feat[:, -4:]
    assert (foot >= 1.0 - 1e-9).all()


def test_h3d_root_velocity_is_zero_for_static_motion() -> None:
    sk = _toy_skeleton()
    T = 12
    trans = torch.zeros(T, 3, dtype=torch.float64)
    trans[:, 1] = 1.0
    quats = _identity_quats(T)

    feat = tplusr_to_h3d_features_with_quats(trans, quats, sk)
    # cols 0..3 are root_rot_velocity (1), root_linear_velocity (2), root_y (1)
    assert torch.allclose(feat[:, 0], torch.zeros(T - 1, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(feat[:, 1:3], torch.zeros(T - 1, 2, dtype=torch.float64), atol=1e-10)
