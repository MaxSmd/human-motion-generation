"""Room / euclidean constraints for RMG sampling.

Two parts, matching the geometry:

* **Spawn placement** is *exact* — a rigid SE(2) transform (yaw about the up
  axis + xz translation) applied to the root so the clip starts at the spawn
  marker. `place_motion` operates on the (translation, root-quaternion)
  representation; `place_joints` is the equivalent transform on FK output (used
  inside the guidance energy). The two are algebraically identical.

* **Room containment + obstacle avoidance** is *soft* — there is no closed-form
  projection of FK joint positions (a nonlinear function of all 22 quaternions +
  the root) onto "inside the room, outside every obstacle". So we use
  differentiable **energy guidance**: a penalty (ReLU(violation)² via signed
  distance fields) evaluated on the clean-sample estimate x̂₁ each ODE step,
  backpropagated through FK, nudging the sample toward feasibility.

World frame = HumanML3D FK frame: X right, **Y up**, Z forward; floor at y=0.
The room is centred at the x/z origin and spans y∈[0, height] — matching the
frontend RoomEditor's convention (and so a "low ceiling" is just a small height).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from shared.geometry.skeleton import NUM_JOINTS, Skeleton, forward_kinematics, quat_mul

from .constraints import CONSTRAINABLE_REPRESENTATIONS  # noqa: F401 (re-exported use)


# --------------------------------------------------------------------------- math helpers


def _rot_y(p: Tensor, angle_rad: float) -> Tensor:
    """Rotate points `p` (..., 3) about the up (Y) axis by `angle_rad`."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    return torch.stack([c * x + s * z, y, -s * x + c * z], dim=-1)


def _sdf_box(p: Tensor, center, half, yaw_rad: float = 0.0) -> Tensor:
    """Signed distance to an (optionally yaw-rotated) axis box. <0 inside."""
    center = p.new_tensor(center)
    half = p.new_tensor(half)
    local = _rot_y(p - center, -yaw_rad) if yaw_rad else (p - center)
    q = local.abs() - half
    outside = torch.linalg.vector_norm(q.clamp_min(0.0), dim=-1)
    inside = q.max(dim=-1).values.clamp_max(0.0)
    return outside + inside


def _sdf_sphere(p: Tensor, center, r: float) -> Tensor:
    center = p.new_tensor(center)
    return torch.linalg.vector_norm(p - center, dim=-1) - r


def _sdf_cylinder(p: Tensor, center_xz, r: float, cy: float, half_h: float) -> Tensor:
    """Signed distance to a Y-axis capped cylinder. <0 inside."""
    cxz = p.new_tensor(center_xz)
    d_xz = torch.linalg.vector_norm(p[..., ::2] - cxz, dim=-1) - r  # x,z columns
    d_y = (p[..., 1] - cy).abs() - half_h
    outside = torch.sqrt(d_xz.clamp_min(0.0) ** 2 + d_y.clamp_min(0.0) ** 2 + 1e-12)
    inside = torch.maximum(d_xz, d_y).clamp_max(0.0)
    return outside + inside


# --------------------------------------------------------------------------- scene model


@dataclass
class Scene:
    room: tuple[float, float, float]      # (width, depth, height)
    objects: list[dict]                   # normalized obstacle dicts
    spawn: tuple[float, float, float]     # (x, z, rotation_deg)

    @property
    def spawn_rad(self) -> float:
        return math.radians(self.spawn[2])


def parse_scene(d: dict | None) -> Scene | None:
    """Parse the frontend scene JSON into a Scene (None when empty/absent)."""
    if not d:
        return None
    room = d.get("room", {})
    spawn = d.get("spawn", {})
    return Scene(
        room=(float(room.get("width", 4.0)), float(room.get("depth", 4.0)), float(room.get("height", 2.5))),
        objects=list(d.get("objects", []) or []),
        spawn=(float(spawn.get("x", 0.0)), float(spawn.get("z", 0.0)), float(spawn.get("rotation", 0.0))),
    )


# --------------------------------------------------------------------------- placement (exact)


def place_joints(joints: Tensor, spawn: tuple[float, float, float]) -> Tensor:
    """Rigidly place FK joints (B, T, J, 3) so the frame-0 pelvis sits at the
    spawn xz and the clip is yawed by the spawn rotation. Differentiable."""
    sx, sz, rot = spawn
    p0 = joints[:, 0:1, 0:1, :]                      # (B,1,1,3) frame-0 pelvis
    recenter = joints.new_tensor([1.0, 0.0, 1.0])    # drop xz only, keep y
    centered = joints - p0 * recenter
    placed = _rot_y(centered, math.radians(rot))
    return placed + joints.new_tensor([sx, 0.0, sz])


def place_motion(translation: Tensor, quats: Tensor, spawn: tuple[float, float, float]) -> tuple[Tensor, Tensor]:
    """Exact spawn placement on the representation: yaw + xz-translate the root
    trajectory and pre-rotate the root orientation. (T, 3), (T, J, 4) → placed.

    Equivalent (verified) to `place_joints(FK(...))`; used to place the *final*
    sample before decode/render so the returned clip actually starts at spawn.
    """
    sx, sz, rot = spawn
    rad = math.radians(rot)
    p0 = translation[0]
    centered = translation - translation.new_tensor([p0[0], 0.0, p0[2]])
    trans2 = _rot_y(centered, rad) + translation.new_tensor([sx, 0.0, sz])
    q_yaw = quats.new_tensor([math.cos(rad / 2), 0.0, math.sin(rad / 2), 0.0])
    quats2 = quats.clone()
    quats2[..., 0, :] = quat_mul(q_yaw.expand_as(quats[..., 0, :]), quats[..., 0, :])
    return trans2, quats2


# --------------------------------------------------------------------------- energy (soft)


def scene_energy(joints_world: Tensor, scene: Scene) -> Tensor:
    """Mean squared violation: outside-the-room + inside-any-obstacle. >=0."""
    w, d, h = scene.room
    # room containment — penalise leaving the box (incl. floor y<0 and ceiling y>h)
    sdf_room = _sdf_box(joints_world, center=(0.0, h / 2, 0.0), half=(w / 2, h / 2, d / 2))
    energy = sdf_room.clamp_min(0.0).pow(2).mean()

    for o in scene.objects:
        kind = o.get("kind")
        if kind == "sphere":
            sdf = _sdf_sphere(joints_world, (o["x"], o["y"], o["z"]), float(o["radius"]))
        elif kind == "cylinder":
            sdf = _sdf_cylinder(joints_world, (o["x"], o["z"]), float(o["radius"]),
                                cy=float(o["y"]), half_h=float(o["height"]) / 2)
        elif kind == "box":
            sdf = _sdf_box(joints_world, center=(o["x"], o["y"], o["z"]),
                           half=(float(o["w"]) / 2, float(o["h"]) / 2, float(o["d"]) / 2),
                           yaw_rad=math.radians(float(o.get("rotation", 0.0))))
        else:
            continue
        energy = energy + sdf.mul(-1.0).clamp_min(0.0).pow(2).mean()  # penetration depth²
    return energy


def build_room_energy_fn(scene: Scene, skeleton: Skeleton, num_joints: int = NUM_JOINTS):
    """Build `energy_fn(xhat) -> scalar` for the sampler's guidance hook.

    xhat is the flat clean-sample estimate (B, T, 3+4J) of a quaternion
    representation (tr/trp); decode → FK → place into the room → scene penalty.
    """
    qdim = 3 + 4 * num_joints

    def energy_fn(xhat: Tensor) -> Tensor:
        translation = xhat[..., :3]
        quats = xhat[..., 3:qdim].reshape(*xhat.shape[:-1], num_joints, 4)
        quats = quats / torch.linalg.vector_norm(quats, dim=-1, keepdim=True).clamp_min(1e-8)
        joints = forward_kinematics(skeleton, quats, translation)  # (B,T,J,3)
        return scene_energy(place_joints(joints, scene.spawn), scene)

    return energy_fn
