"""Joint-angle constraint plumbing tests.

Covers the anatomical joint → bend-controller remap (the per-chain FK
off-by-one), and the soft-hold refinements on the bend projector: partial
`strength` and `ease_frames` window ramps. The core bend geometry lives in
test_bend.py.
"""

from __future__ import annotations

import math

import pytest
import torch

from rmg.flow import BendConstraint, bend_clamp, bend_controller_index, build_bend_projector, parse_bends
from rmg.flow.constraints import _ease_ramp
from shared.geometry.skeleton import NUM_JOINTS, Skeleton

torch.manual_seed(0)


def _arm_skeleton() -> Skeleton:
    # Minimal offsets with a real left arm so the elbow has incoming/outgoing
    # bones (shoulder 16 → elbow 18 → wrist 20).
    off = torch.zeros(NUM_JOINTS, 3, dtype=torch.float64)
    off[18] = torch.tensor([0.28, 0, 0])   # upper arm (shoulder→elbow)
    off[20] = torch.tensor([0.25, 0, 0])   # forearm   (elbow→wrist)
    return Skeleton(offsets=off)


def test_bend_controller_index_mapping() -> None:
    # bend at joint J is controlled by the next joint along its kinematic chain
    assert bend_controller_index("L_Elbow") == 20  # L_Wrist
    assert bend_controller_index("R_Elbow") == 21  # R_Wrist
    assert bend_controller_index("L_Knee") == 7    # L_Ankle
    assert bend_controller_index("R_Knee") == 8    # R_Ankle
    assert bend_controller_index("L_Shoulder") == 18  # L_Elbow
    assert bend_controller_index("Neck") == 15     # Head
    assert bend_controller_index("Spine3") == 12   # Neck
    assert bend_controller_index("L_Ankle") == 10  # L_Foot (toe) — ankle has a child
    # true end-effectors have no outgoing bone → no representable bend
    for ee in ("L_Wrist", "R_Wrist", "L_Foot", "R_Foot", "Head"):
        with pytest.raises(ValueError):
            bend_controller_index(ee)
    # root quaternion is global orientation, not a bend
    with pytest.raises(ValueError):
        bend_controller_index("pelvis")


def test_from_dict_accepts_bend_and_legacy_keys() -> None:
    # preferred exact / range keys
    assert (BendConstraint.from_dict({"joint": "L_Elbow", "bend_deg": 90}).min_deg,) == (90.0,)
    r = BendConstraint.from_dict({"joint": "L_Knee", "bend_min": 10, "bend_max": 80})
    assert (r.min_deg, r.max_deg) == (10.0, 80.0)
    # legacy axis-era keys map onto bend semantics
    leg = BendConstraint.from_dict({"joint": "L_Elbow", "angle_deg": 45})
    assert leg.min_deg == leg.max_deg == 45.0
    # soft-hold refinements
    s = BendConstraint.from_dict({"joint": "L_Elbow", "bend_deg": 90, "strength": 0.5, "ease_frames": 4})
    assert s.strength == 0.5 and s.ease_frames == 4


def test_ease_ramp_shape_and_endpoints() -> None:
    r = _ease_ramp(10, 3, device=None, dtype=torch.float64)
    assert r.shape == (10,)
    assert r.max() <= 1.0 + 1e-9 and r.min() > 0.0
    assert float(r[0]) < float(r[2]) <= 1.0           # ramps up at the start
    assert float(r[-1]) < float(r[-3]) <= 1.0          # ramps down at the end
    assert torch.allclose(r[4:6], torch.ones(2, dtype=torch.float64))  # flat in the middle
    # ease_frames=0 ⇒ all ones (hard on/off)
    assert torch.allclose(_ease_ramp(5, 0, None, torch.float64), torch.ones(5, dtype=torch.float64))


def _bend_of(joints_q, u, v):
    from shared.geometry.skeleton import quat_rotate
    d = quat_rotate(joints_q, v)
    cos_a = (d * u).sum(-1).clamp(-1, 1)
    return math.degrees(float(torch.acos(cos_a)))


def test_strength_zero_is_a_noop() -> None:
    # strength=0 applies none of the correction: the quaternion is unchanged.
    q = torch.tensor([math.cos(math.radians(60)), 0.0, 0.0, math.sin(math.radians(60))], dtype=torch.float64)
    u = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    v = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    out = bend_clamp(q, u, v, 0.0, 0.0, strength=0.0)
    assert torch.allclose(out, q, atol=1e-6)


def test_partial_strength_moves_part_way() -> None:
    # forearm bent 120°, target 0°: strength 0.5 should roughly halve the bend.
    q = torch.tensor([math.cos(math.radians(60)), 0.0, 0.0, math.sin(math.radians(60))], dtype=torch.float64)
    u = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    v = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    out = bend_clamp(q, u, v, 0.0, 0.0, strength=0.5)
    assert _bend_of(out, u, v) == pytest.approx(60.0, abs=1e-2)


def test_projector_respects_strength() -> None:
    # A strength<1 hold lands between the free pose and the exact target.
    skel = _arm_skeleton()
    T = 6
    c_soft = parse_bends([{"joint": "L_Elbow", "bend_deg": 0, "strength": 0.5}])
    proj = build_bend_projector(c_soft, skel, num_frames=T, num_joints=NUM_JOINTS, dtype=torch.float64)
    assert proj is not None
    x = torch.zeros(1, T, 3 + 4 * NUM_JOINTS, dtype=torch.float64)
    x[..., 0] = 1.0
    for j in range(NUM_JOINTS):
        x[..., 3 + 4 * j] = 1.0
    # bend the elbow 120° via its controller (L_Wrist, idx 20) about z
    lo = 3 + 4 * 20
    x[..., lo] = math.cos(math.radians(60))
    x[..., lo + 3] = math.sin(math.radians(60))
    out = proj(x)
    # half-corrected: still a unit quaternion, and not the full clamp
    q = out[0, 0, lo:lo + 4]
    assert torch.allclose(q.norm(), torch.tensor(1.0, dtype=torch.float64), atol=1e-6)
    assert not torch.allclose(q, x[0, 0, lo:lo + 4], atol=1e-3)


def test_empty_bends_no_projector() -> None:
    assert build_bend_projector(parse_bends(None), _arm_skeleton(), num_frames=10) is None
