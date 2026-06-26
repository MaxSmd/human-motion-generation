"""Bend-angle constraint tests.

The bend at a joint is the angle between its incoming and outgoing bones
(0° = straight). These tests check the bend_clamp projection (exact + range,
twist preservation) and that, driven through the real per-chain FK, clamping a
joint's *controller* quaternion actually changes the joint's interior angle.
"""

from __future__ import annotations

import math

import pytest
import torch

from rmg.flow import BendConstraint, bend_clamp, build_bend_projector, parse_bends
from shared.geometry.skeleton import NUM_JOINTS, Skeleton, forward_kinematics

torch.manual_seed(0)

# straight rest arm: upper arm and forearm both along +x
U = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)  # incoming bone
V = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)  # outgoing bone
Z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)


def _quat_axis_angle(axis, deg):
    h = math.radians(deg) / 2
    a = axis / axis.norm()
    return torch.cat([torch.tensor([math.cos(h)], dtype=torch.float64), math.sin(h) * a])


def _bend_of(q):
    """Bend angle (deg) implied by controller quat q for bones U, V."""
    from shared.geometry.skeleton import quat_rotate
    d = quat_rotate(q, V)
    return math.degrees(math.acos(float((d @ U).clamp(-1, 1))))


def test_bend_clamp_exact_sets_angle() -> None:
    q = _quat_axis_angle(Z, 120.0)          # forearm bent 120° in the xy-plane
    out = bend_clamp(q, U, V, math.radians(90), math.radians(90))
    assert _bend_of(out) == pytest.approx(90.0, abs=1e-3)
    assert torch.allclose(out.norm(), torch.tensor(1.0, dtype=torch.float64), atol=1e-6)


def test_bend_clamp_range() -> None:
    above = _quat_axis_angle(Z, 120.0)
    assert _bend_of(bend_clamp(above, U, V, 0.0, math.radians(90))) == pytest.approx(90.0, abs=1e-3)
    inside = _quat_axis_angle(Z, 30.0)
    assert _bend_of(bend_clamp(inside, U, V, 0.0, math.radians(90))) == pytest.approx(30.0, abs=1e-3)
    below = _quat_axis_angle(Z, 20.0)
    assert _bend_of(bend_clamp(below, U, V, math.radians(45), math.radians(90))) == pytest.approx(45.0, abs=1e-3)


def test_bend_clamp_preserves_twist() -> None:
    # q = bend(about z) then twist(about the forearm axis v=x). Clamping the bend
    # must leave the twist-about-v component unchanged.
    from shared.geometry.skeleton import quat_mul
    bend = _quat_axis_angle(Z, 120.0)
    twist = _quat_axis_angle(V, 50.0)
    q = quat_mul(bend, twist)
    out = bend_clamp(q, U, V, math.radians(90), math.radians(90))

    def twist_about_v(qq):
        d = (qq[1:] * (V / V.norm())).sum()
        return math.degrees(2 * math.atan2(float(d), float(qq[0])))

    assert _bend_of(out) == pytest.approx(90.0, abs=1e-3)
    assert twist_about_v(out) == pytest.approx(twist_about_v(q), abs=1e-2)


def _arm_skeleton():
    off = torch.zeros(NUM_JOINTS, 3, dtype=torch.float64)
    off[3] = off[6] = off[9] = torch.tensor([0, 0.1, 0])
    off[13] = torch.tensor([0.08, 0.05, 0]); off[16] = torch.tensor([0.10, 0, 0])
    off[18] = torch.tensor([0.28, 0, 0])   # upper arm (shoulder→elbow)
    off[20] = torch.tensor([0.25, 0, 0])   # forearm   (elbow→wrist)
    return Skeleton(off)


def test_bend_projector_changes_interior_angle_via_fk() -> None:
    """Clamping the L_Elbow bend must move the elbow's interior FK angle to the
    target — proving the projector acts on the right (controller) quaternion."""
    sk = _arm_skeleton()
    T = 4
    flat = torch.zeros(1, T, 3 + 4 * NUM_JOINTS, dtype=torch.float64)
    quats = torch.zeros(1, T, NUM_JOINTS, 4, dtype=torch.float64)
    quats[..., 0] = 1.0
    # bend the elbow hard by rotating the WRIST controller (index 20) 120° about z
    quats[:, :, 20] = _quat_axis_angle(Z, 120.0)
    flat[..., 3:] = quats.reshape(1, T, -1)

    proj = build_bend_projector([BendConstraint(joint="L_Elbow", min_deg=30, max_deg=30)],
                                sk, num_frames=T, dtype=torch.float64)
    assert proj is not None
    out = proj(flat)

    q_out = out[..., 3:].reshape(1, T, NUM_JOINTS, 4)
    j = forward_kinematics(sk, q_out[0], torch.zeros(T, 3, dtype=torch.float64))
    v1 = j[:, 16] - j[:, 18]   # elbow→shoulder
    v2 = j[:, 20] - j[:, 18]   # elbow→wrist
    interior = torch.rad2deg(torch.acos(
        ((v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1))).clamp(-1, 1)))
    # interior = 180 − bend; target bend 30 → interior 150
    assert torch.allclose(interior, torch.full((T,), 150.0, dtype=torch.float64), atol=1e-2)


def test_empty_bends_no_projector() -> None:
    assert build_bend_projector(parse_bends(None), _arm_skeleton(), num_frames=10) is None
