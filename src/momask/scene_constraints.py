"""Differentiable room containment and obstacle avoidance for MoMask.

Generated HumanML3D clips live in a canonical coordinate frame. A fixed SE(2)
transform aligns the original clip with a scene spawn marker, then signed-
distance penalties are evaluated in scene coordinates. Collision checks use
joints plus samples along every kinematic edge so an obstacle cannot slip
between two collision-free joint endpoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from shared.geometry import NUM_JOINTS, PARENTS


@dataclass(frozen=True)
class SceneObstacle:
    """A box, sphere, or upright capped cylinder in scene coordinates."""

    kind: str
    center: tuple[float, float, float]
    size: tuple[float, float, float] | None = None
    radius: float | None = None
    height: float | None = None
    yaw_degrees: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in {"box", "sphere", "cylinder"}:
            raise ValueError("obstacle kind must be box, sphere, or cylinder")
        if len(self.center) != 3 or not all(math.isfinite(v) for v in self.center):
            raise ValueError("obstacle center must contain three finite values")
        if self.kind == "box":
            if (
                self.size is None
                or len(self.size) != 3
                or any(not math.isfinite(v) or v <= 0.0 for v in self.size)
            ):
                raise ValueError("box obstacles require a positive (width, height, depth) size")
        elif self.kind == "sphere":
            if self.radius is None or not math.isfinite(self.radius) or self.radius <= 0.0:
                raise ValueError("sphere obstacles require a positive radius")
        elif (
            self.radius is None
            or not math.isfinite(self.radius)
            or self.radius <= 0.0
            or self.height is None
            or not math.isfinite(self.height)
            or self.height <= 0.0
        ):
            raise ValueError("cylinder obstacles require positive radius and height")
        if not math.isfinite(self.yaw_degrees):
            raise ValueError("obstacle yaw must be finite")

    @classmethod
    def box(
        cls,
        *,
        center: tuple[float, float, float],
        size: tuple[float, float, float],
        yaw_degrees: float = 0.0,
    ) -> "SceneObstacle":
        return cls("box", center, size=size, yaw_degrees=yaw_degrees)

    @classmethod
    def sphere(
        cls, *, center: tuple[float, float, float], radius: float
    ) -> "SceneObstacle":
        return cls("sphere", center, radius=radius)

    @classmethod
    def cylinder(
        cls,
        *,
        center: tuple[float, float, float],
        radius: float,
        height: float,
    ) -> "SceneObstacle":
        return cls("cylinder", center, radius=radius, height=height)


@dataclass(frozen=True)
class RoomGeometryConstraint:
    """Room bounds and obstacles used by inference-time latent refinement.

    ``room_size`` is ``(width, depth, height)``. The room is centered at the
    scene XZ origin with its floor at Y=0. ``spawn`` is
    ``(x, z, yaw_degrees)`` and yaw zero points along +Z. ``body_radius``
    expands obstacles and the walls/ceiling; the floor remains a contact surface.
    """

    room_size: tuple[float, float, float]
    obstacles: tuple[SceneObstacle, ...] = ()
    spawn: tuple[float, float, float] = (0.0, 0.0, 0.0)
    padding: float = 0.0
    body_radius: float = 0.05
    bone_samples: int = 2
    swept_samples: int = 3
    heading_epsilon: float = 0.12

    def __post_init__(self) -> None:
        if len(self.room_size) != 3 or any(
            not math.isfinite(v) or v <= 0.0 for v in self.room_size
        ):
            raise ValueError("room_size must be a positive (width, depth, height) tuple")
        if len(self.spawn) != 3 or not all(math.isfinite(v) for v in self.spawn):
            raise ValueError("spawn must contain finite (x, z, yaw_degrees) values")
        if (
            not math.isfinite(self.padding)
            or not math.isfinite(self.body_radius)
            or self.padding < 0.0
            or self.body_radius < 0.0
        ):
            raise ValueError("scene padding and body radius must be non-negative")
        if not isinstance(self.bone_samples, int) or isinstance(self.bone_samples, bool):
            raise TypeError("bone_samples must be an integer")
        if self.bone_samples < 0:
            raise ValueError("bone_samples must be non-negative")
        if not isinstance(self.swept_samples, int) or isinstance(self.swept_samples, bool):
            raise TypeError("swept_samples must be an integer")
        if self.swept_samples < 0:
            raise ValueError("swept_samples must be non-negative")
        if not math.isfinite(self.heading_epsilon) or self.heading_epsilon <= 0.0:
            raise ValueError("heading_epsilon must be positive")
        if not isinstance(self.obstacles, tuple):
            object.__setattr__(self, "obstacles", tuple(self.obstacles))
        if any(not isinstance(obstacle, SceneObstacle) for obstacle in self.obstacles):
            raise TypeError("obstacles must contain SceneObstacle instances")

    def as_dict(self) -> dict:
        return {
            "room_size": list(self.room_size),
            "spawn": list(self.spawn),
            "padding": self.padding,
            "body_radius": self.body_radius,
            "bone_samples": self.bone_samples,
            "swept_samples": self.swept_samples,
            "obstacles": [
                {
                    "kind": obstacle.kind,
                    "center": list(obstacle.center),
                    "size": None if obstacle.size is None else list(obstacle.size),
                    "radius": obstacle.radius,
                    "height": obstacle.height,
                    "yaw_degrees": obstacle.yaw_degrees,
                }
                for obstacle in self.obstacles
            ],
        }


@dataclass(frozen=True)
class SceneTransform:
    """Fixed canonical-to-scene placement, one yaw and origin per sample."""

    yaw_radians: Tensor
    canonical_origin_xz: Tensor
    spawn_xz: Tensor


def _rotate_y(points: Tensor, yaw: Tensor) -> Tensor:
    while yaw.ndim < points.ndim - 1:
        yaw = yaw.unsqueeze(-1)
    c, s = torch.cos(yaw), torch.sin(yaw)
    x, y, z = points.unbind(dim=-1)
    return torch.stack((c * x + s * z, y, -s * x + c * z), dim=-1)


def build_scene_transform(
    reference_joints: Tensor,
    constraint: RoomGeometryConstraint,
    frame_mask: Tensor | None = None,
) -> SceneTransform:
    """Freeze placement from the original sample so optimization cannot move it."""

    if reference_joints.ndim != 4 or reference_joints.shape[-2:] != (NUM_JOINTS, 3):
        raise ValueError(f"reference_joints must be (B, T, {NUM_JOINTS}, 3)")
    batch, time = reference_joints.shape[:2]
    valid = (
        torch.ones(batch, time, dtype=torch.bool, device=reference_joints.device)
        if frame_mask is None
        else frame_mask.to(device=reference_joints.device, dtype=torch.bool)
    )
    if valid.shape != (batch, time) or not bool(valid.any(dim=1).all()):
        raise ValueError("frame_mask must contain at least one valid frame per sample")
    lengths = valid.sum(dim=1)
    prefix = torch.arange(time, device=valid.device).unsqueeze(0) < lengths.unsqueeze(1)
    if not bool((valid == prefix).all()):
        raise ValueError("scene frame_mask must contain one contiguous valid prefix per sample")

    pelvis = reference_joints[:, :, 0]
    first = pelvis[:, 0]
    last = pelvis[torch.arange(batch, device=pelvis.device), lengths - 1]
    displacement = last - first
    distance = torch.linalg.vector_norm(displacement[:, [0, 2]], dim=-1)
    heading = torch.atan2(displacement[:, 0], displacement[:, 2])
    desired = reference_joints.new_full((batch,), math.radians(constraint.spawn[2]))
    yaw = torch.where(distance > constraint.heading_epsilon, desired - heading, desired)
    return SceneTransform(
        yaw_radians=yaw.detach(),
        canonical_origin_xz=first[:, [0, 2]].detach(),
        spawn_xz=reference_joints.new_tensor(constraint.spawn[:2]).expand(batch, 2).detach(),
    )


def place_joints_in_scene(joints: Tensor, transform: SceneTransform) -> Tensor:
    """Apply a frozen scene transform to ``(B,T,J,3)`` joints."""

    if joints.ndim != 4 or joints.shape[-2:] != (NUM_JOINTS, 3):
        raise ValueError(f"joints must be (B, T, {NUM_JOINTS}, 3)")
    batch = joints.shape[0]
    if transform.yaw_radians.shape != (batch,):
        raise ValueError(f"scene yaw must be ({batch},)")
    if transform.canonical_origin_xz.shape != (batch, 2) or transform.spawn_xz.shape != (
        batch,
        2,
    ):
        raise ValueError(f"scene origins must be ({batch}, 2)")
    origin = joints.new_zeros(joints.shape[0], 1, 1, 3)
    origin[..., 0] = transform.canonical_origin_xz[:, 0].view(-1, 1, 1)
    origin[..., 2] = transform.canonical_origin_xz[:, 1].view(-1, 1, 1)
    placed = _rotate_y(joints - origin, transform.yaw_radians)
    translation = joints.new_zeros(joints.shape[0], 1, 1, 3)
    translation[..., 0] = transform.spawn_xz[:, 0].view(-1, 1, 1)
    translation[..., 2] = transform.spawn_xz[:, 1].view(-1, 1, 1)
    return placed + translation


def sample_body_points(joints: Tensor, bone_samples: int) -> Tensor:
    """Return joints and evenly spaced interior samples along all 21 bones."""

    if joints.ndim != 4 or joints.shape[-2:] != (NUM_JOINTS, 3):
        raise ValueError(f"joints must be (B, T, {NUM_JOINTS}, 3)")
    if not isinstance(bone_samples, int) or isinstance(bone_samples, bool):
        raise TypeError("bone_samples must be an integer")
    if bone_samples < 0:
        raise ValueError("bone_samples must be non-negative")
    if bone_samples == 0:
        return joints
    children = torch.arange(1, NUM_JOINTS, device=joints.device)
    parents = torch.tensor(PARENTS[1:], device=joints.device)
    start = joints.index_select(-2, parents)
    end = joints.index_select(-2, children)
    fractions = torch.linspace(
        0.0,
        1.0,
        bone_samples + 2,
        device=joints.device,
        dtype=joints.dtype,
    )[1:-1]
    samples = start.unsqueeze(-2) + fractions.view(1, 1, 1, -1, 1) * (
        end - start
    ).unsqueeze(-2)
    return torch.cat((joints, samples.flatten(-3, -2)), dim=-2)


def sample_swept_body_points(
    joints: Tensor,
    bone_samples: int,
    swept_samples: int,
) -> Tensor:
    """Interpolate body samples between frames as ``(B,T-1,S*P,3)``."""

    if not isinstance(swept_samples, int) or isinstance(swept_samples, bool):
        raise TypeError("swept_samples must be an integer")
    if swept_samples < 1:
        raise ValueError("swept_samples must be positive")
    body = sample_body_points(joints, bone_samples)
    if body.shape[1] < 2:
        return body.new_empty(body.shape[0], 0, body.shape[2] * swept_samples, 3)
    fractions = torch.linspace(
        0.0,
        1.0,
        swept_samples + 2,
        device=body.device,
        dtype=body.dtype,
    )[1:-1]
    swept = body[:, :-1].unsqueeze(-2) + fractions.view(1, 1, 1, -1, 1) * (
        body[:, 1:] - body[:, :-1]
    ).unsqueeze(-2)
    return swept.flatten(-3, -2)


def _sdf_box(points: Tensor, obstacle: SceneObstacle) -> Tensor:
    center = points.new_tensor(obstacle.center)
    local = points - center
    if obstacle.yaw_degrees:
        local = _rotate_y(local, points.new_tensor(math.radians(-obstacle.yaw_degrees)))
    assert obstacle.size is not None
    # torch.abs has a zero derivative at the exact obstacle centre. Selecting
    # the positive branch at zero gives optimization a deterministic escape.
    absolute = torch.where(local >= 0.0, local, -local)
    q = absolute - points.new_tensor(obstacle.size) / 2.0
    outside = torch.linalg.vector_norm(q.clamp_min(0.0), dim=-1)
    return outside + q.amax(dim=-1).clamp_max(0.0)


def _obstacle_sdf(points: Tensor, obstacle: SceneObstacle) -> Tensor:
    if obstacle.kind == "box":
        return _sdf_box(points, obstacle)
    center = points.new_tensor(obstacle.center)
    if obstacle.kind == "sphere":
        assert obstacle.radius is not None
        delta = points - center
        stuck = torch.linalg.vector_norm(delta, dim=-1, keepdim=True) < 1e-8
        delta = delta + stuck.to(delta.dtype) * delta.new_tensor([1e-6, 0.0, 0.0])
        return torch.linalg.vector_norm(delta, dim=-1) - obstacle.radius
    assert obstacle.radius is not None and obstacle.height is not None
    delta_xz = (points - center)[..., [0, 2]]
    stuck = torch.linalg.vector_norm(delta_xz, dim=-1, keepdim=True) < 1e-8
    delta_xz = delta_xz + stuck.to(delta_xz.dtype) * delta_xz.new_tensor([1e-6, 0.0])
    radial = torch.linalg.vector_norm(delta_xz, dim=-1) - obstacle.radius
    delta_y = points[..., 1] - center[1]
    absolute_y = torch.where(delta_y >= 0.0, delta_y, -delta_y)
    vertical = absolute_y - obstacle.height / 2.0
    outside = torch.linalg.vector_norm(
        torch.stack((radial.clamp_min(0.0), vertical.clamp_min(0.0)), dim=-1), dim=-1
    )
    return outside + torch.maximum(radial, vertical).clamp_max(0.0)


def _clearance_violation_components_from_points(
    points: Tensor,
    constraint: RoomGeometryConstraint,
) -> dict[str, Tensor]:
    width, depth, height = constraint.room_size
    clearance = constraint.padding + constraint.body_radius
    wall_x = (points[..., 0].abs() + clearance - width / 2.0).clamp_min(0.0)
    wall_z = (points[..., 2].abs() + clearance - depth / 2.0).clamp_min(0.0)
    wall = torch.linalg.vector_norm(torch.stack((wall_x, wall_z), dim=-1), dim=-1)
    ceiling = (points[..., 1] + clearance - height).clamp_min(0.0)
    # The floor is a contact surface, so body-radius clearance is not applied.
    floor = (-points[..., 1]).clamp_min(0.0)
    room = torch.linalg.vector_norm(
        torch.stack((wall_x, wall_z, ceiling, floor), dim=-1), dim=-1
    )
    obstacle_violation = torch.zeros_like(room)
    for obstacle in constraint.obstacles:
        current = (clearance - _obstacle_sdf(points, obstacle)).clamp_min(0.0)
        obstacle_violation = torch.maximum(obstacle_violation, current)
    return {
        "combined": torch.maximum(room, obstacle_violation),
        "obstacle": obstacle_violation,
        "floor": floor,
        "wall": wall,
        "ceiling": ceiling,
    }


def scene_clearance_violation_components(
    joints_world: Tensor,
    constraint: RoomGeometryConstraint,
) -> dict[str, Tensor]:
    """Return combined and source-specific body-point violations in metres."""

    points = sample_body_points(joints_world, constraint.bone_samples)
    return _clearance_violation_components_from_points(points, constraint)


def scene_clearance_violations(
    joints_world: Tensor,
    constraint: RoomGeometryConstraint,
) -> Tensor:
    """Return per-body-point clearance violation in metres."""

    return scene_clearance_violation_components(joints_world, constraint)["combined"]


def scene_swept_clearance_violations(
    joints_world: Tensor,
    constraint: RoomGeometryConstraint,
) -> Tensor:
    """Return clearance violations at interpolated points between frames."""

    if constraint.swept_samples < 1 or joints_world.shape[1] < 2:
        return joints_world.new_empty(
            joints_world.shape[0], max(joints_world.shape[1] - 1, 0), 0
        )
    swept_points = sample_swept_body_points(
        joints_world, constraint.bone_samples, constraint.swept_samples
    )
    return _clearance_violation_components_from_points(swept_points, constraint)[
        "combined"
    ]


def scene_geometry_loss(
    joints_world: Tensor,
    constraint: RoomGeometryConstraint,
    frame_mask: Tensor | None = None,
) -> Tensor:
    """Squared violation, summed over body samples and mean over valid frames."""

    violation = scene_clearance_violations(joints_world, constraint)
    valid = (
        torch.ones(violation.shape[:2], dtype=torch.bool, device=violation.device)
        if frame_mask is None
        else frame_mask.to(device=violation.device, dtype=torch.bool)
    )
    if valid.shape != violation.shape[:2]:
        raise ValueError("scene frame_mask must match the joint batch/time dimensions")
    weights = valid.to(violation.dtype)
    loss = (violation.pow(2).sum(dim=-1) * weights).sum() / weights.sum().clamp_min(1.0)
    if constraint.swept_samples > 0 and joints_world.shape[1] > 1:
        swept_violation = scene_swept_clearance_violations(joints_world, constraint)
        segment_valid = valid[:, :-1] & valid[:, 1:]
        segment_weights = segment_valid.to(swept_violation.dtype)
        swept_loss = (
            swept_violation.pow(2).sum(dim=-1) * segment_weights
        ).sum() / segment_weights.sum().clamp_min(1.0)
        loss = loss + swept_loss
    return loss


def scene_peak_violation_loss(
    joints_world: Tensor,
    constraint: RoomGeometryConstraint,
    frame_mask: Tensor | None = None,
) -> Tensor:
    """Squared worst penetration per sample, including swept-frame points."""

    violation = scene_clearance_violations(joints_world, constraint)
    valid = (
        torch.ones(violation.shape[:2], dtype=torch.bool, device=violation.device)
        if frame_mask is None
        else frame_mask.to(device=violation.device, dtype=torch.bool)
    )
    if valid.shape != violation.shape[:2] or not bool(valid.any(dim=1).all()):
        raise ValueError("scene frame mask must match and contain a valid frame")
    frame_peak = violation.masked_fill(~valid.unsqueeze(-1), -1.0).amax(dim=(1, 2))
    if constraint.swept_samples > 0 and joints_world.shape[1] > 1:
        swept_violation = scene_swept_clearance_violations(joints_world, constraint)
        segment_valid = valid[:, :-1] & valid[:, 1:]
        swept_peak = swept_violation.masked_fill(
            ~segment_valid.unsqueeze(-1), -1.0
        ).amax(dim=(1, 2))
        swept_peak = torch.where(
            segment_valid.any(dim=1), swept_peak, torch.zeros_like(swept_peak)
        )
        frame_peak = torch.maximum(frame_peak, swept_peak)
    return frame_peak.clamp_min(0.0).pow(2).mean()


@torch.no_grad()
def scene_geometry_metrics(
    joints_world: Tensor,
    constraint: RoomGeometryConstraint,
    frame_mask: Tensor | None = None,
) -> dict[str, float]:
    """Clearance magnitude and collision rates for one batch."""

    components = scene_clearance_violation_components(joints_world, constraint)
    violation = components["combined"]
    valid_frames = (
        torch.ones(violation.shape[:2], dtype=torch.bool, device=violation.device)
        if frame_mask is None
        else frame_mask.to(device=violation.device, dtype=torch.bool)
    )
    if valid_frames.shape != violation.shape[:2]:
        raise ValueError("scene frame_mask must match the joint batch/time dimensions")
    valid_points = valid_frames.unsqueeze(-1).expand_as(violation)

    def summarize(values: Tensor) -> dict[str, float]:
        active = values[valid_points]
        positive = active[active > 0.0]
        colliding_frames = (values > 0.0).any(dim=-1) & valid_frames
        return {
            "max_violation_m": float(active.max().cpu()) if active.numel() else 0.0,
            "mean_violation_m": float(positive.mean().cpu()) if positive.numel() else 0.0,
            "violating_point_fraction": float((active > 0.0).float().mean().cpu())
            if active.numel()
            else 0.0,
            "colliding_frame_fraction": float(
                colliding_frames.sum().float().cpu()
                / valid_frames.sum().clamp_min(1).cpu()
            ),
        }

    result = summarize(violation)
    if constraint.swept_samples > 0 and joints_world.shape[1] > 1:
        swept = scene_swept_clearance_violations(joints_world, constraint)
        valid_segments = valid_frames[:, :-1] & valid_frames[:, 1:]
        valid_swept_points = valid_segments.unsqueeze(-1).expand_as(swept)
        active = swept[valid_swept_points]
        positive = active[active > 0.0]
        colliding_segments = (swept > 0.0).any(dim=-1) & valid_segments
        result.update(
            {
                "swept_max_violation_m": float(active.max().cpu())
                if active.numel()
                else 0.0,
                "swept_mean_violation_m": float(positive.mean().cpu())
                if positive.numel()
                else 0.0,
                "swept_violating_point_fraction": float(
                    (active > 0.0).float().mean().cpu()
                )
                if active.numel()
                else 0.0,
                "swept_colliding_segment_fraction": float(
                    colliding_segments.sum().float().cpu()
                    / valid_segments.sum().clamp_min(1).cpu()
                ),
            }
        )
    for source in ("obstacle", "floor", "wall", "ceiling"):
        result.update(
            {
                f"{source}_{key}": value
                for key, value in summarize(components[source]).items()
            }
        )
    return result
