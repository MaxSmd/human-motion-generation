"""Sampling-time joint-angle constraints for RMG (no fine-tuning).

RMG motion lives on the product manifold R^3 × (S^3)^J: each joint is an
*independent* unit quaternion (paper §3.1). Because the factors are independent,
"fix a joint to a target angle" is just **inpainting on that S^3 factor** — we
overwrite the joint's quaternion with a fixed target after every ODE step of the
Riemannian Euler sampler. The target is a unit quaternion, so it stays on the
manifold; the rest of the body adapts because the network sees the pinned value
at each step. This is sampling-only — the trained weights are untouched.

A "fixed joint angle" is specified as **axis + angle** (a 1-DOF hinge-style
rotation about a chosen axis, relative to the joint's parent), optionally over a
sub-range of frames. axis-angle → quaternion is the standard
q = [cos(θ/2), sin(θ/2)·axiŝ].

The flat layout (shared by the `tr` and `trp` representations) is:
    flat[..., 0:3]                    = translation
    flat[..., 3+4j : 3+4(j+1)]        = quaternion for joint j  (j=0 is the root)
matching ProductManifold([Euclidean(3)] + [Sphere(3)]·J).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from shared.geometry.quaternions import quat_to_upper_hemisphere
from shared.geometry.skeleton import JOINT_NAMES, NUM_JOINTS

# Layout constants for the quaternion-on-S^3 representations (tr / trp). The
# translation occupies the first 3 dims; each joint quaternion is 4 contiguous
# dims thereafter. Representations without this prefix layout aren't supported.
CONSTRAINABLE_REPRESENTATIONS = ("tr", "trp")

# Named axes for the common hinge cases (axis-angle, joint-local frame).
NAMED_AXES: dict[str, tuple[float, float, float]] = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}


def axis_angle_to_quat(
    axis: str | tuple[float, float, float] | Tensor,
    angle_rad: float,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Axis-angle → unit quaternion [w, x, y, z] (upper hemisphere, w ≥ 0).

    `axis` may be a named axis ("x"/"y"/"z") or an explicit 3-vector.
    """
    if isinstance(axis, str):
        axis = NAMED_AXES[axis.lower()] if axis.lower() in NAMED_AXES else axis
        if isinstance(axis, str):
            raise ValueError(f"unknown axis {axis!r} — expected one of {list(NAMED_AXES)} or a 3-vector")
    a = torch.as_tensor(axis, dtype=dtype)
    norm = torch.linalg.vector_norm(a)
    if float(norm) < 1e-8:
        raise ValueError("constraint axis must be non-zero")
    a = a / norm
    half = 0.5 * float(angle_rad)
    q = torch.cat([torch.tensor([math.cos(half)], dtype=dtype), math.sin(half) * a])
    return quat_to_upper_hemisphere(q)


def _resolve_joint(joint: int | str) -> int:
    if isinstance(joint, int):
        idx = joint
    else:
        try:
            idx = JOINT_NAMES.index(joint)
        except ValueError as e:
            raise ValueError(
                f"unknown joint {joint!r} — expected one of {JOINT_NAMES}"
            ) from e
    if not 0 <= idx < NUM_JOINTS:
        raise ValueError(f"joint index {idx} out of range [0, {NUM_JOINTS})")
    return idx


def _resolve_axis(axis: str | tuple[float, float, float] | Tensor):
    if isinstance(axis, str):
        key = axis.lower()
        if key not in NAMED_AXES:
            raise ValueError(f"unknown axis {axis!r} — expected one of {list(NAMED_AXES)} or a 3-vector")
        return NAMED_AXES[key]
    return axis


@dataclass
class JointAngleConstraint:
    """Pin one joint to a fixed axis-angle rotation over a frame range.

    Attributes:
        joint: joint index (0..J-1) or SMPL name (e.g. "L_Elbow").
        axis: "x"/"y"/"z" or an explicit 3-vector — the rotation axis.
        angle_deg: rotation magnitude in degrees.
        frame_start / frame_end: half-open frame window [start, end). `end=None`
            (or ≤ 0) means "to the last frame". Defaults pin all frames.
    """

    joint: int | str
    axis: str | tuple[float, float, float] | Tensor = "z"
    angle_deg: float = 90.0
    frame_start: int = 0
    frame_end: int | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "JointAngleConstraint":
        return cls(
            joint=d["joint"],
            axis=d.get("axis", "z"),
            angle_deg=float(d.get("angle_deg", 90.0)),
            frame_start=int(d.get("frame_start", 0)),
            frame_end=(int(d["frame_end"]) if d.get("frame_end") not in (None, "", -1) else None),
        )

    def quat(self, dtype: torch.dtype = torch.float32) -> Tensor:
        return axis_angle_to_quat(_resolve_axis(self.axis), math.radians(self.angle_deg), dtype=dtype)

    def frame_window(self, num_frames: int) -> tuple[int, int]:
        start = max(0, min(int(self.frame_start), num_frames))
        end = num_frames if self.frame_end in (None,) or int(self.frame_end) <= 0 else int(self.frame_end)
        end = max(start, min(end, num_frames))
        return start, end


def build_inpaint_targets(
    constraints: list[JointAngleConstraint],
    num_frames: int,
    num_joints: int = NUM_JOINTS,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Build (values, mask) tensors of shape (T, 3 + 4*num_joints).

    `mask` is True on the 4 quaternion dims of each constrained joint over its
    frame window; `values` holds the target quaternion there (zeros elsewhere,
    unused). Pass both to `RiemannianEulerSampler.sample(fixed_values=, fixed_mask=)`.
    Returns empty tensors with all-False mask when `constraints` is empty.
    """
    ambient = 3 + 4 * num_joints
    values = torch.zeros(num_frames, ambient, device=device, dtype=dtype)
    mask = torch.zeros(num_frames, ambient, device=device, dtype=torch.bool)
    for c in constraints:
        j = _resolve_joint(c.joint)
        lo, hi = 3 + 4 * j, 3 + 4 * (j + 1)
        fs, fe = c.frame_window(num_frames)
        if fe <= fs:
            continue
        q = c.quat(dtype=dtype).to(device=device)
        values[fs:fe, lo:hi] = q
        mask[fs:fe, lo:hi] = True
    return values, mask


def parse_constraints(specs: list[dict] | None) -> list[JointAngleConstraint]:
    """Parse a list of JSON-ish constraint dicts (from the API/CLI) into objects."""
    if not specs:
        return []
    return [JointAngleConstraint.from_dict(s) for s in specs]
