"""Room / euclidean constraint tests: SDFs, exact spawn placement, energy, and
that sampler guidance reduces the scene penalty vs. an unguided run."""

from __future__ import annotations

import math

import pytest
import torch

from rmg.flow import (
    OracleVelocity,
    RiemannianEulerSampler,
    SamplerCfg,
    WrappedGaussianPrior,
    build_room_energy_fn,
    parse_scene,
    place_joints,
    place_motion,
    rest_pose_mu,
    rmg_manifold,
    scene_energy,
)
from rmg.flow import contact_energy, foot_skate_energy, parse_contacts
from rmg.flow.scene import _sdf_box, _sdf_cylinder, _sdf_sphere
from shared.geometry.skeleton import FOOT_CONTACT_IDX, NUM_JOINTS, Skeleton, forward_kinematics

torch.manual_seed(0)


def _toy_skeleton() -> Skeleton:
    # Simple non-degenerate offsets (unit spacing); enough for FK shape/placement.
    offs = torch.zeros(NUM_JOINTS, 3)
    offs[:, 1] = 0.1  # each joint 0.1 above its parent → a vertical-ish stack
    return Skeleton(offsets=offs)


def test_sdf_signs() -> None:
    p = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert _sdf_sphere(p, (0, 0, 0), 1.0)[0] < 0  # inside
    assert _sdf_sphere(p, (0, 0, 0), 1.0)[1] > 0  # outside
    assert _sdf_box(p, (0, 0, 0), (1, 1, 1))[0] < 0
    assert _sdf_box(p, (0, 0, 0), (1, 1, 1))[1] > 0
    # cylinder radius 1 about y, height 2 centred at y=0
    assert _sdf_cylinder(p, (0, 0), 1.0, cy=0.0, half_h=1.0)[0] < 0
    assert _sdf_cylinder(p, (0, 0), 1.0, cy=0.0, half_h=1.0)[1] > 0


def test_box_yaw_rotation() -> None:
    # Box wide in x, thin in z. A point off the thin (z) face is outside when
    # axis-aligned, but inside once the box is yawed 90° (x/z extents swap).
    p = torch.tensor([[0.0, 0.0, 0.5]])
    half = (1.0, 1.0, 0.2)
    assert _sdf_box(p, (0, 0, 0), half, yaw_rad=0.0)[0] > 0
    assert _sdf_box(p, (0, 0, 0), half, yaw_rad=math.radians(90))[0] < 0


def test_place_motion_puts_pelvis_at_spawn() -> None:
    T = 5
    translation = torch.randn(T, 3)
    quats = torch.zeros(T, NUM_JOINTS, 4)
    quats[..., 0] = 1.0
    spawn = (1.5, -0.5, 90.0)
    t2, q2 = place_motion(translation, quats, spawn)
    # frame-0 pelvis xz lands exactly on spawn; y preserved
    assert torch.allclose(t2[0, ::2], torch.tensor([1.5, -0.5]), atol=1e-5)
    assert torch.allclose(t2[0, 1], translation[0, 1], atol=1e-5)


def test_spawn_rotation_is_absolute_walk_direction() -> None:
    # A clip that walks +X canonically must, at spawn rotation 0, walk +Z (the
    # spawn arrow); at rotation 90° it must walk +X.
    T = 10
    quats = torch.zeros(T, NUM_JOINTS, 4)
    quats[..., 0] = 1.0
    translation = torch.zeros(T, 3)
    translation[:, 0] = torch.linspace(0, 2.0, T)  # walk +X

    t0, _ = place_motion(translation, quats, (0.0, 0.0, 0.0))
    disp0 = t0[-1] - t0[0]
    assert disp0[2] > 0.5 and abs(disp0[0]) < 1e-4, f"rot 0 should walk +Z, got {disp0}"

    t90, _ = place_motion(translation, quats, (0.0, 0.0, 90.0))
    disp90 = t90[-1] - t90[0]
    assert disp90[0] > 0.5 and abs(disp90[2]) < 1e-4, f"rot 90 should walk +X, got {disp90}"


def test_place_motion_matches_place_joints() -> None:
    skel = _toy_skeleton()
    T = 4
    translation = torch.randn(T, 3) * 0.3
    quats = torch.zeros(T, NUM_JOINTS, 4)
    quats[..., 0] = 1.0
    spawn = (0.7, 0.2, 35.0)
    # placement on representation → FK
    t2, q2 = place_motion(translation, quats, spawn)
    j_repr = forward_kinematics(skel, q2, t2)
    # FK → placement on joints
    j_fk = forward_kinematics(skel, quats, translation)
    j_joints = place_joints(j_fk.unsqueeze(0), spawn)[0]
    assert torch.allclose(j_repr, j_joints, atol=1e-4)


def test_energy_zero_inside_positive_outside() -> None:
    scene = parse_scene({"room": {"width": 4, "depth": 4, "height": 2.5},
                         "objects": [{"kind": "sphere", "x": 0, "y": 1, "z": 0, "radius": 0.5}],
                         "spawn": {"x": 0, "z": 0, "rotation": 0}})
    inside = torch.tensor([[[[1.0, 1.5, 1.0]]]])    # in room, clear of sphere
    assert float(scene_energy(inside, scene)) == 0.0
    outside = torch.tensor([[[[5.0, 1.5, 0.0]]]])   # past the wall
    assert float(scene_energy(outside, scene)) > 0.0
    in_sphere = torch.tensor([[[[0.0, 1.0, 0.0]]]])  # inside the sphere
    assert float(scene_energy(in_sphere, scene)) > 0.0


def test_padding_brakes_before_contact() -> None:
    # A joint 0.1 m outside a sphere: no violation without padding, but a
    # standoff of 0.2 m flags it (guidance brakes before contact).
    base = {"room": {"width": 4, "depth": 4, "height": 2.5},
            "objects": [{"kind": "sphere", "x": 0, "y": 1, "z": 0, "radius": 0.5}],
            "spawn": {"x": 0, "z": 0, "rotation": 0}}
    near = torch.tensor([[[[0.6, 1.0, 0.0]]]])  # 0.1 m outside the r=0.5 sphere
    assert float(scene_energy(near, parse_scene(base))) == 0.0
    assert float(scene_energy(near, parse_scene({**base, "padding": 0.2}))) > 0.0


def test_guidance_reduces_energy() -> None:
    """A guided sample should land with lower scene energy than the unguided one
    drawn from the same noise (oracle field pulls toward a fixed x1)."""
    J, T = NUM_JOINTS, 6
    M = rmg_manifold(num_joints=J)
    mu = rest_pose_mu(num_joints=J, dtype=torch.float64).to(torch.float64)
    prior = WrappedGaussianPrior(M, mu, sigma=0.6)
    x0 = prior.sample((1, T), dtype=torch.float64)
    x1 = prior.sample((1, T), dtype=torch.float64)
    model = OracleVelocity(M, x0, x1).double()
    sampler = RiemannianEulerSampler(M, prior, SamplerCfg(num_steps=30, guidance_scale=1.0))

    skel = _toy_skeleton()
    # A small room + a big obstacle at the origin so the unguided clip violates it.
    scene = parse_scene({"room": {"width": 2, "depth": 2, "height": 2},
                         "objects": [{"kind": "sphere", "x": 0, "y": 0.3, "z": 0, "radius": 0.6}],
                         "spawn": {"x": 0, "z": 0, "rotation": 0}})
    energy_fn = build_room_energy_fn(scene, skel, num_joints=J)

    def run(weight):
        out = sampler.sample(model, shape=(1, T), num_steps=30, dtype=torch.float64,
                             energy_fn=(energy_fn if weight else None), guidance_weight=weight)
        from rmg.flow.scene import place_joints as pj
        tr = out[..., :3]
        q = out[..., 3:3 + 4 * J].reshape(1, T, J, 4)
        q = q / q.norm(dim=-1, keepdim=True)
        joints = pj(forward_kinematics(skel, q, tr), scene.spawn)
        return float(scene_energy(joints, scene))

    e_unguided = run(0.0)
    e_guided = run(1.0)
    # Guidance should cut the violation energy by a large margin.
    assert e_guided < 0.5 * e_unguided, f"guided {e_guided} !<< unguided {e_unguided}"


# --------------------------------------------------------------------------- contacts


def _box_scene(contacts=None, foot_skate_weight=0.0):
    return parse_scene({
        "room": {"width": 6, "depth": 6, "height": 3},
        "objects": [{"id": "box1", "kind": "box", "x": 1.0, "y": 0.25, "z": 0.0,
                     "w": 1.0, "h": 0.5, "d": 1.0, "rotation": 0}],
        "spawn": {"x": 0, "z": 0, "rotation": 0},
        "contacts": contacts or [],
        "foot_skate_weight": foot_skate_weight,
    })


def _joints(B, T, pos):
    # All joints at `pos`; pos may be (3,) or (B,T,3).
    j = torch.zeros(B, T, NUM_JOINTS, 3, dtype=torch.float64)
    p = torch.as_tensor(pos, dtype=torch.float64)
    j[...] = p.view(*([1] * (j.dim() - 1)), 3) if p.dim() == 1 else p.unsqueeze(-2)
    return j


def test_contact_zero_when_joint_on_target() -> None:
    scene = _box_scene([{"joint": "pelvis", "target": "obstacle_top", "object_id": "box1", "tol": 0.05}])
    c = parse_contacts(scene.contacts)[0]
    # pelvis sitting exactly on the box top (y = 0.25 + 0.25 = 0.5), over its footprint
    on_top = _joints(1, 4, [1.0, 0.5, 0.0])
    assert float(contact_energy(on_top, c, scene)) == pytest.approx(0.0, abs=1e-9)
    # pelvis well above the top → positive energy that pulls it down
    above = _joints(1, 4, [1.0, 1.5, 0.0])
    assert float(contact_energy(above, c, scene)) > 0.0


def test_contact_floor_pulls_to_ground() -> None:
    scene = _box_scene([{"joint": "L_Foot", "target": "floor", "tol": 0.02}])
    c = parse_contacts(scene.contacts)[0]
    grounded = _joints(1, 3, [0.3, 0.0, 0.2])
    assert float(contact_energy(grounded, c, scene)) == pytest.approx(0.0, abs=1e-9)
    floating = _joints(1, 3, [0.3, 0.4, 0.2])
    assert float(contact_energy(floating, c, scene)) > 0.0


def test_contact_window_limits_frames() -> None:
    scene = _box_scene([{"joint": "L_Foot", "target": "floor", "frame_start": 0, "frame_end": 2}])
    c = parse_contacts(scene.contacts)[0]
    j = _joints(1, 5, [0.0, 0.5, 0.0])     # floating everywhere
    e_win = float(contact_energy(j, c, scene))
    c_all = parse_contacts([{"joint": "L_Foot", "target": "floor"}])[0]
    e_all = float(contact_energy(j, c_all, scene))
    assert 0.0 < e_win < e_all              # only the 2-frame window contributes


def test_foot_skate_penalises_sliding_plant() -> None:
    scene = _box_scene(foot_skate_weight=1.0)
    T = 6
    # feet glued to the floor but sliding in x → should be penalised
    sliding = torch.zeros(1, T, NUM_JOINTS, 3, dtype=torch.float64)
    sliding[..., 0] = torch.linspace(0, 1.0, T).view(1, T, 1)   # x drifts
    e_slide = float(foot_skate_energy(sliding, scene, FOOT_CONTACT_IDX, 1.0))
    # same feet but lifted high (not planted) → little/no penalty
    lifted = sliding.clone()
    lifted[..., 1] = 1.0
    e_lift = float(foot_skate_energy(lifted, scene, FOOT_CONTACT_IDX, 1.0))
    assert e_slide > 0.0
    assert e_lift < 0.05 * e_slide


def test_unknown_contact_target_raises() -> None:
    import pytest as _pt
    with _pt.raises(ValueError):
        parse_contacts([{"joint": "pelvis", "target": "nonsense"}])
