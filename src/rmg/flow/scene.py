"""Room / euclidean constraints for RMG sampling.

Two parts, matching the geometry:

* **Spawn placement** is *exact* — a rigid SE(2) transform (yaw about the up
  axis + xz translation) applied to the root so the clip starts at the spawn
  marker. `place_motion` operates on the (translation, root-quaternion)
  representation; `place_joints` is the equivalent transform on FK output (used
  inside the guidance energy). The two are algebraically identical.

* Everything else is *soft* — there is no closed-form projection of FK joint
  positions (a nonlinear function of all 22 quaternions + the root) onto the
  feasible set. So we use differentiable **energy guidance**: a penalty
  evaluated on the clean-sample estimate x̂₁ each ODE step, backpropagated
  through FK, nudging the sample toward feasibility. The soft channel composes:
    - **room containment + obstacle avoidance** — ReLU(violation)² via SDFs,
    - **contacts** — pull a joint onto a target (obstacle top / floor / surface
      / point) so it actually rests there (sit, step on, hand on a wall),
    - **foot anti-skate** — penalise horizontal sliding of planted feet.
  `build_room_energy_fn` composes all of these into one `energy_fn`; hard
  joint-angle limits stay on the projection hook (see flow.constraints).

World frame = HumanML3D FK frame: X right, **Y up**, Z forward; floor at y=0.
The room is centred at the x/z origin and spans y∈[0, height] — matching the
frontend RoomEditor's convention (and so a "low ceiling" is just a small height).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from shared.geometry.skeleton import (
    FOOT_CONTACT_IDX,
    NUM_JOINTS,
    Skeleton,
    forward_kinematics,
    quat_mul,
)

from .constraints import (  # noqa: F401 (CONSTRAINABLE_REPRESENTATIONS re-exported)
    CONSTRAINABLE_REPRESENTATIONS,
    _clamp_window,
    _resolve_joint,
)


# --------------------------------------------------------------------------- math helpers


def _rot_y(p: Tensor, angle_rad: float) -> Tensor:
    """Rotate points `p` (..., 3) about the up (Y) axis by `angle_rad` (float)."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    return torch.stack([c * x + s * z, y, -s * x + c * z], dim=-1)


def _rot_y_t(p: Tensor, ang: Tensor) -> Tensor:
    """Rotate points `p` (..., 3) about Y by a tensor angle `ang` (broadcasts)."""
    c, s = torch.cos(ang), torch.sin(ang)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    return torch.stack([c * x + s * z, y, -s * x + c * z], dim=-1)


# Heading measured so 0 = +Z (forward) and +90° = +X (right) — matches the
# RoomEditor spawn arrow (cone points +Z at rotation 0; rotating the marker +90°
# about Y sends it to +X). atan2(x, z) gives exactly that.
def _heading(dx: Tensor, dz: Tensor) -> Tensor:
    return torch.atan2(dx, dz)


_MOVE_EPS = 0.12  # m of horizontal travel below which heading is ill-defined


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
    padding: float = 0.0                  # standoff margin (m): brake this far out
    contacts: list[dict] | None = None    # contact targets (joint → surface/point)
    foot_skate_weight: float = 0.0        # >0 ⇒ penalise sliding planted feet
    fps: float = 20.0                     # frame rate (for foot velocity / skate)

    @property
    def spawn_rad(self) -> float:
        return math.radians(self.spawn[2])

    def object_by_id(self, oid) -> dict | None:
        for o in self.objects:
            if o.get("id") == oid:
                return o
        return None


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
        padding=float(d.get("padding", 0.0)),
        contacts=list(d.get("contacts", []) or []),
        foot_skate_weight=float(d.get("foot_skate_weight", 0.0)),
        fps=float(d.get("fps", 20.0)),
    )


# --------------------------------------------------------------------------- placement (exact)


def _alignment_yaw(traj: Tensor, desired_rad: Tensor | float) -> Tensor:
    """Yaw that rotates the clip's *walk direction* to face `desired_rad`.

    `traj` is the pelvis trajectory (..., T, 3). We take its net horizontal
    displacement as the clip's heading and return desired − heading, so the spawn
    rotation is ABSOLUTE (0° ⇒ walks toward +Z, the spawn arrow) regardless of
    the model's canonical facing. Falls back to `desired` when the clip barely
    moves (heading undefined — e.g. waving in place). Per leading batch dim.
    """
    disp = traj[..., -1, :] - traj[..., 0, :]            # (..., 3)
    dx, dz = disp[..., 0], disp[..., 2]
    horiz = torch.sqrt(dx * dx + dz * dz + 1e-12)         # +eps ⇒ finite grad at 0
    heading = _heading(dx, dz)
    desired = traj.new_tensor(desired_rad) if not torch.is_tensor(desired_rad) else desired_rad
    return torch.where(horiz > _MOVE_EPS, desired - heading, torch.broadcast_to(desired, heading.shape))


def place_joints(joints: Tensor, spawn: tuple[float, float, float]) -> Tensor:
    """Rigidly place FK joints (B, T, J, 3): frame-0 pelvis at the spawn xz, clip
    yawed so its walk direction faces the spawn rotation. Differentiable."""
    sx, sz, rot = spawn
    pelvis = joints[:, :, 0, :]                       # (B,T,3)
    # Detach the heading: orientation is fixed by the spawn, not something the
    # avoidance gradient should drive. Backprop through atan2 of a tiny/noisy
    # early-step trajectory otherwise produces huge gradients (sampler diverges).
    yaw = _alignment_yaw(pelvis, math.radians(rot)).detach()   # (B,)
    p0 = joints[:, 0:1, 0:1, :]                       # (B,1,1,3) frame-0 pelvis
    recenter = joints.new_tensor([1.0, 0.0, 1.0])     # drop xz only, keep y
    centered = joints - p0 * recenter
    placed = _rot_y_t(centered, yaw.view(-1, 1, 1))  # (B,1,1) ⇒ broadcasts over T,J
    return placed + joints.new_tensor([sx, 0.0, sz])


def place_motion(translation: Tensor, quats: Tensor, spawn: tuple[float, float, float]) -> tuple[Tensor, Tensor]:
    """Exact spawn placement on the representation: align the walk direction to
    the spawn rotation, then xz-translate the root to spawn and pre-rotate the
    root orientation. (T, 3), (T, J, 4) → placed.

    Equivalent (verified) to `place_joints(FK(...))`; used to place the *final*
    sample before decode/render so the returned clip actually starts at spawn.
    """
    sx, sz, rot = spawn
    yaw = _alignment_yaw(translation, math.radians(rot))   # scalar tensor
    p0 = translation[0]
    centered = translation - translation.new_tensor([p0[0], 0.0, p0[2]])
    trans2 = _rot_y_t(centered, yaw) + translation.new_tensor([sx, 0.0, sz])
    half = 0.5 * yaw
    z = torch.zeros_like(half)
    q_yaw = torch.stack([torch.cos(half), z, torch.sin(half), z])  # (4,)
    quats2 = quats.clone()
    quats2[..., 0, :] = quat_mul(q_yaw.expand_as(quats[..., 0, :]), quats[..., 0, :])
    return trans2, quats2


# --------------------------------------------------------------------------- energy (soft)


def _obstacle_sdf(joints_world: Tensor, o: dict) -> Tensor | None:
    """Signed distance from every joint to obstacle `o` (<0 inside). None for an
    unknown kind. Single dispatch shared by avoidance and contact."""
    kind = o.get("kind")
    if kind == "sphere":
        return _sdf_sphere(joints_world, (o["x"], o["y"], o["z"]), float(o["radius"]))
    if kind == "cylinder":
        return _sdf_cylinder(joints_world, (o["x"], o["z"]), float(o["radius"]),
                             cy=float(o["y"]), half_h=float(o["height"]) / 2)
    if kind == "box":
        return _sdf_box(joints_world, center=(o["x"], o["y"], o["z"]),
                        half=(float(o["w"]) / 2, float(o["h"]) / 2, float(o["d"]) / 2),
                        yaw_rad=math.radians(float(o.get("rotation", 0.0))))
    return None


def scene_energy(joints_world: Tensor, scene: Scene) -> Tensor:
    """Squared violation energy: outside-the-room + inside-any-obstacle. >=0.

    Aggregation is **sum over joints, mean over frames/batch** — NOT a flat mean
    over all joints. Averaging across all 22 joints would dilute a few penetrating
    feet by the ~20 clear joints, making the gradient vanishingly small (the
    reason high guidance weights felt inert). Summing keeps the penalty
    proportional to total penetration so the gradient actually pushes joints out.
    """
    m = max(0.0, float(scene.padding))               # standoff margin
    w, d, h = scene.room
    # Room containment — penalise leaving the box, and (with margin) coming within
    # `m` of a wall/ceiling/floor, so the body keeps clear: ReLU(sdf + m).
    sdf_room = _sdf_box(joints_world, center=(0.0, h / 2, 0.0), half=(w / 2, h / 2, d / 2))
    energy = _agg((sdf_room + m).clamp_min(0.0))

    for o in scene.objects:
        sdf = _obstacle_sdf(joints_world, o)
        if sdf is None:
            continue
        # Penalise inside-the-obstacle AND within `m` of its surface: ReLU(m − sdf).
        energy = energy + _agg((m - sdf).clamp_min(0.0))
    return energy


def _agg(violation: Tensor) -> Tensor:               # (B,T,J) -> scalar
    """Sum over joints, mean over frames/batch (see scene_energy)."""
    return violation.pow(2).sum(dim=-1).mean()


# ----- contact targets (attraction) ---------------------------------------------------
#
# A contact PULLS one joint onto a target each frame of its window: the top face
# of an obstacle (sit / step on), the floor (plant a foot), the nearest surface
# of an obstacle (hand on a wall), or a fixed point. The target is a
# differentiable point so the guidance gradient can drag the joint to it; the
# energy is ReLU(‖joint − target‖ − tol)² so anything within `tol` is satisfied.


_CONTACT_TARGETS = ("floor", "point", "obstacle_top", "obstacle")


@dataclass
class ContactConstraint:
    """Pull one joint onto a target over a frame window.

    Attributes:
        joint: joint index/name whose position is driven to the target.
        target: "floor" | "point" | "obstacle_top" | "obstacle".
        object_id: obstacle id (for "obstacle_top"/"obstacle").
        x, y, z: target point (for "point").
        tol: contact tolerance (m) — within this distance counts as satisfied.
        weight: relative strength of this contact term in the energy.
        frame_start / frame_end: half-open window; end None/≤0 ⇒ to last frame.
    """

    joint: int | str
    target: str = "floor"
    object_id: str | None = None
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    tol: float = 0.05
    weight: float = 1.0
    frame_start: int = 0
    frame_end: int | None = None

    def __post_init__(self) -> None:
        if self.target not in _CONTACT_TARGETS:
            raise ValueError(f"unknown contact target {self.target!r} — expected one of {_CONTACT_TARGETS}")
        self.joint_idx = _resolve_joint(self.joint)

    @classmethod
    def from_dict(cls, d: dict) -> "ContactConstraint":
        return cls(
            joint=d["joint"],
            target=str(d.get("target", "floor")),
            object_id=d.get("object_id"),
            x=float(d.get("x", 0.0)), y=float(d.get("y", 0.0)), z=float(d.get("z", 0.0)),
            tol=float(d.get("tol", 0.05)),
            weight=float(d.get("weight", 1.0)),
            frame_start=int(d.get("frame_start", 0)),
            frame_end=(int(d["frame_end"]) if d.get("frame_end") not in (None, "", -1) else None),
        )

    def frame_window(self, num_frames: int) -> tuple[int, int]:
        return _clamp_window(self.frame_start, self.frame_end, num_frames)


def parse_contacts(specs: list[dict] | None) -> list[ContactConstraint]:
    """Parse a list of JSON-ish contact dicts into objects (skips empties)."""
    if not specs:
        return []
    return [ContactConstraint.from_dict(s) for s in specs]


def _contact_target(jp: Tensor, contact: "ContactConstraint", scene: Scene) -> Tensor:
    """Per-frame target point (B,T,3) for the contact joint position `jp` (B,T,3)."""
    kind = contact.target
    if kind == "floor":
        # touch y=0; xz free (target tracks the joint horizontally).
        return torch.stack([jp[..., 0], torch.zeros_like(jp[..., 1]), jp[..., 2]], dim=-1)
    if kind == "point":
        return jp.new_tensor([contact.x, contact.y, contact.z]).expand_as(jp)

    o = scene.object_by_id(contact.object_id)
    if o is None:
        return jp  # unknown object → no pull (zero energy)
    okind = o.get("kind")

    if kind == "obstacle_top":
        if okind == "box":
            cx, cy, cz = float(o["x"]), float(o["y"]), float(o["z"])
            hw, hh, hd = float(o["w"]) / 2, float(o["h"]) / 2, float(o["d"]) / 2
            yaw = math.radians(float(o.get("rotation", 0.0)))
            local = _rot_y(jp - jp.new_tensor([cx, cy, cz]), -yaw)
            lx = local[..., 0].clamp(-hw, hw)
            lz = local[..., 2].clamp(-hd, hd)
            top_local = torch.stack([lx, torch.full_like(lx, hh), lz], dim=-1)
            return _rot_y(top_local, yaw) + jp.new_tensor([cx, cy, cz])
        if okind == "cylinder":
            cx, cz = float(o["x"]), float(o["z"])
            r = float(o["radius"]); top = float(o["y"]) + float(o["height"]) / 2
            dx, dz = jp[..., 0] - cx, jp[..., 2] - cz
            dist = torch.sqrt(dx * dx + dz * dz + 1e-12)
            scale = (dist.clamp_max(r)) / dist
            return torch.stack([cx + dx * scale, torch.full_like(dx, top), cz + dz * scale], dim=-1)
        if okind == "sphere":
            return jp.new_tensor([float(o["x"]), float(o["y"]) + float(o["radius"]), float(o["z"])]).expand_as(jp)
        return jp

    # "obstacle" — nearest point on the surface: step off the SDF by its value
    # along the surface normal (∇sdf), so the joint is pulled onto sdf≈0.
    sdf = _obstacle_sdf(jp.unsqueeze(-2), o)  # (B,T,1)
    if sdf is None:
        return jp
    sdf = sdf.squeeze(-1)
    with torch.no_grad():
        # finite-difference normal would need grad; cheap approximation: move
        # toward the obstacle centre when outside, away when inside.
        centre = jp.new_tensor([float(o.get("x", 0.0)), float(o.get("y", 0.0)), float(o.get("z", 0.0))])
        to_centre = centre - jp
        n = to_centre / torch.linalg.vector_norm(to_centre, dim=-1, keepdim=True).clamp_min(1e-6)
    return jp + n * sdf.unsqueeze(-1)


def contact_energy(joints_world: Tensor, contact: "ContactConstraint", scene: Scene) -> Tensor:
    """Squared shortfall pulling `contact.joint` to its target over its window."""
    T = joints_world.shape[-3]
    fs, fe = contact.frame_window(T)
    if fe <= fs:
        return joints_world.new_zeros(())
    jp = joints_world[:, fs:fe, contact.joint_idx, :]        # (B,Tw,3)
    tgt = _contact_target(jp, contact, scene)
    dist = torch.linalg.vector_norm(jp - tgt, dim=-1)         # (B,Tw)
    short = (dist - float(contact.tol)).clamp_min(0.0)
    return float(contact.weight) * short.pow(2).mean()


def foot_skate_energy(joints_world: Tensor, scene: Scene, foot_idx, weight: float) -> Tensor:
    """Penalise horizontal sliding of feet while they are planted near the floor.

    A foot is "planted" by a soft indicator (sigmoid of how far below `band` it
    sits); the energy is plant·‖horizontal velocity‖². Differentiable through FK,
    so the guidance gradient discourages foot-skate without hard-pinning feet.
    """
    feet = joints_world[..., list(foot_idx), :]               # (B,T,F,3)
    height = feet[..., 1]                                      # (B,T,F)
    band = 0.05                                                # m: "on the floor"
    plant = torch.sigmoid((band - height) / 0.02)             # ~1 when grounded
    vel = (feet[:, 1:] - feet[:, :-1]) * float(scene.fps)     # (B,T-1,F,3)
    horiz2 = vel[..., 0] ** 2 + vel[..., 2] ** 2              # (B,T-1,F)
    plant_pair = 0.5 * (plant[:, 1:] + plant[:, :-1])
    return float(weight) * (plant_pair * horiz2).sum(dim=-1).mean()


def build_room_energy_fn(scene: Scene, skeleton: Skeleton, num_joints: int = NUM_JOINTS):
    """Build `energy_fn(xhat) -> scalar` for the sampler's guidance hook.

    xhat is the flat clean-sample estimate (B, T, 3+4J) of a quaternion
    representation (tr/trp); decode → FK → place into the room → composite
    penalty: room containment + obstacle avoidance + every contact target +
    (optionally) foot anti-skate. Hard joint-angle limits are handled separately
    by the projection hook (see flow.constraints); this is the soft channel.
    """
    qdim = 3 + 4 * num_joints
    contacts = parse_contacts(scene.contacts)

    def energy_fn(xhat: Tensor) -> Tensor:
        translation = xhat[..., :3]
        quats = xhat[..., 3:qdim].reshape(*xhat.shape[:-1], num_joints, 4)
        quats = quats / torch.linalg.vector_norm(quats, dim=-1, keepdim=True).clamp_min(1e-8)
        joints = forward_kinematics(skeleton, quats, translation)  # (B,T,J,3)
        placed = place_joints(joints, scene.spawn)
        energy = scene_energy(placed, scene)
        for c in contacts:
            energy = energy + contact_energy(placed, c, scene)
        if scene.foot_skate_weight > 0.0:
            energy = energy + foot_skate_energy(placed, scene, FOOT_CONTACT_IDX, scene.foot_skate_weight)
        return energy

    return energy_fn
