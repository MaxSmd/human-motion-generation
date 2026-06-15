"""Room / euclidean constraint tests: SDFs, exact spawn placement, energy, and
that sampler guidance reduces the scene penalty vs. an unguided run."""

from __future__ import annotations

import math

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
from rmg.flow.scene import _sdf_box, _sdf_cylinder, _sdf_sphere
from shared.geometry.skeleton import NUM_JOINTS, Skeleton, forward_kinematics

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
    e_guided = run(3.0)
    # Guidance should cut the violation energy by a large margin.
    assert e_guided < 0.5 * e_unguided, f"guided {e_guided} !<< unguided {e_unguided}"
