from __future__ import annotations

import torch
from torch import nn

from momask.constraints import LatentRefinementConfig, refine_motion_latents
from momask.scene_constraints import (
    RoomGeometryConstraint,
    SceneObstacle,
    build_scene_transform,
    place_joints_in_scene,
    sample_body_points,
    scene_clearance_violations,
    scene_geometry_loss,
)
from momask.scripts.evaluate_momask_constraints import (
    accumulate_scene_statistics,
    summarize_scene_statistics,
)
from shared.geometry import H3D_FEATURE_DIM, NUM_JOINTS


def _joints(batch: int = 1, time: int = 2) -> torch.Tensor:
    joints = torch.zeros(batch, time, NUM_JOINTS, 3)
    joints[..., 1] = 1.0
    return joints


def test_scene_transform_places_root_and_aligns_travel_with_spawn() -> None:
    joints = _joints()
    joints[:, 1, :, 0] = 2.0
    scene = RoomGeometryConstraint(room_size=(8.0, 8.0, 3.0), spawn=(1.0, -2.0, 0.0))

    transform = build_scene_transform(joints, scene)
    placed = place_joints_in_scene(joints, transform)

    assert torch.allclose(placed[0, 0, 0, [0, 2]], torch.tensor([1.0, -2.0]), atol=1e-6)
    displacement = placed[0, 1, 0] - placed[0, 0, 0]
    assert abs(float(displacement[0])) < 1e-5
    assert float(displacement[2]) > 1.9


def test_scene_loss_is_zero_for_clear_motion_and_positive_in_obstacle() -> None:
    joints = _joints()
    obstacle = SceneObstacle.box(center=(0.0, 1.0, 0.0), size=(0.5, 0.5, 0.5))
    scene = RoomGeometryConstraint(
        room_size=(6.0, 6.0, 3.0), obstacles=(obstacle,), body_radius=0.0, bone_samples=0
    )

    assert scene_geometry_loss(joints + torch.tensor([2.0, 0.0, 0.0]), scene) == 0.0
    assert scene_geometry_loss(joints, scene) > 0.0


def test_bone_samples_detect_collision_between_clear_joint_endpoints() -> None:
    joints = _joints(time=1)
    joints[..., 0] = 2.0
    joints[0, 0, 0] = torch.tensor([-1.0, 1.0, 0.0])
    joints[0, 0, 1] = torch.tensor([1.0, 1.0, 0.0])
    obstacle = SceneObstacle.box(center=(0.0, 1.0, 0.0), size=(0.2, 0.2, 0.2))
    endpoints_only = RoomGeometryConstraint(
        room_size=(6.0, 6.0, 3.0), obstacles=(obstacle,), body_radius=0.0, bone_samples=0
    )
    sampled = RoomGeometryConstraint(
        room_size=(6.0, 6.0, 3.0), obstacles=(obstacle,), body_radius=0.0, bone_samples=1
    )

    assert scene_clearance_violations(joints, endpoints_only).max() == 0.0
    assert scene_clearance_violations(joints, sampled).max() > 0.0
    assert sample_body_points(joints, 1).shape[-2] == NUM_JOINTS + NUM_JOINTS - 1


def test_scene_loss_ignores_padded_frames() -> None:
    joints = _joints()
    joints[:, 1, :, 0] = 10.0
    scene = RoomGeometryConstraint(room_size=(4.0, 4.0, 3.0), body_radius=0.0)
    mask = torch.tensor([[True, False]])

    assert scene_geometry_loss(joints, scene, mask) == 0.0


def test_cylinder_center_has_a_finite_escape_gradient() -> None:
    joints = _joints(time=1).requires_grad_(True)
    scene = RoomGeometryConstraint(
        room_size=(6.0, 6.0, 3.0),
        obstacles=(
            SceneObstacle.cylinder(center=(0.0, 1.0, 0.0), radius=0.5, height=0.5),
        ),
        body_radius=0.0,
        bone_samples=0,
    )

    loss = scene_geometry_loss(joints, scene)
    loss.backward()

    assert joints.grad is not None
    assert torch.isfinite(joints.grad).all()
    assert joints.grad.abs().sum() > 0.0


def test_dataset_scene_statistics_are_weighted_by_frames_and_body_points() -> None:
    scene = RoomGeometryConstraint(room_size=(4.0, 4.0, 3.0), body_radius=0.0, bone_samples=0)
    colliding = _joints(time=1)
    colliding[0, 0, 0, 0] = 3.0
    clear = _joints(time=3)
    total: dict[str, float] = {}

    accumulate_scene_statistics(
        total,
        joints_world=colliding,
        constraint=scene,
        frame_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    accumulate_scene_statistics(
        total,
        joints_world=clear,
        constraint=scene,
        frame_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    summary = summarize_scene_statistics(total)

    assert summary["scene_max_violation_m"] == 1.0
    assert summary["scene_mean_violation_m"] == 1.0
    assert summary["scene_violating_point_fraction"] == 1.0 / (4 * NUM_JOINTS)
    assert summary["scene_colliding_frame_fraction"] == 0.25


class _IdentityDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return latents * self.scale


def test_latent_refinement_reduces_scene_penetration() -> None:
    model = _IdentityDecoder()
    initial = torch.zeros(1, 1, H3D_FEATURE_DIM)
    initial[0, 0, 4] = 0.95  # L_Hip x coordinate in HumanML3D RIC features.
    scene = RoomGeometryConstraint(
        room_size=(6.0, 6.0, 3.0),
        obstacles=(
            SceneObstacle.box(center=(1.0, 0.0, 0.0), size=(0.4, 0.4, 0.4)),
        ),
        body_radius=0.0,
        bone_samples=0,
    )

    result = refine_motion_latents(
        model,
        initial,
        mean=torch.zeros(H3D_FEATURE_DIM),
        std=torch.ones(H3D_FEATURE_DIM),
        target_len=1,
        scene_constraint=scene,
        config=LatentRefinementConfig(
            steps=40,
            learning_rate=0.1,
            scene_weight=1.0,
            latent_weight=0.001,
            dynamics_weight=0.0,
            root_weight=0.0,
            bone_weight=0.0,
            max_delta_norm=2.0,
            grad_clip_norm=10.0,
        ),
    )

    assert result.scene_joints is not None
    assert result.metrics["scene_final_max_violation_m"] < result.metrics[
        "scene_initial_max_violation_m"
    ]
    assert model.scale.grad is None
