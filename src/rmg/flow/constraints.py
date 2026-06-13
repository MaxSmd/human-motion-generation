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
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from shared.geometry.quaternions import normalize_quaternions, quat_to_upper_hemisphere
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


def _clamp_window(frame_start: int, frame_end: int | None, num_frames: int) -> tuple[int, int]:
    """Half-open [start, end) frame window; end None/≤0 ⇒ to the last frame."""
    start = max(0, min(int(frame_start), num_frames))
    end = num_frames if frame_end in (None,) or int(frame_end) <= 0 else int(frame_end)
    end = max(start, min(end, num_frames))
    return start, end


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
        return _clamp_window(self.frame_start, self.frame_end, num_frames)


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


# ---------------------------------------------------------------------------
# Joint RANGES (hinge limits) via swing-twist decomposition + projection.
#
# A range can't be inpainted (there's no single value to fix). Instead we
# *project* each constrained joint onto its feasible per-joint submanifold after
# every ODE step. A hinge (e.g. a knee) decomposes its quaternion about the
# hinge axis â into q = q_swing · q_twist (twist = rotation about â, swing =
# the off-axis remainder). We clamp the signed twist angle into [min, max] and
# shrink the swing toward identity (anatomically: the joint only flexes about â),
# then recompose and renormalize onto S^3. This is an exact hard limit, unlike a
# soft barrier penalty.
# ---------------------------------------------------------------------------

_QUAT_CONJ = torch.tensor([1.0, -1.0, -1.0, -1.0])


def _quat_mul(q1: Tensor, q2: Tensor) -> Tensor:
    """Hamilton product of [w, x, y, z] quaternions (broadcasting over leading dims)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def swing_twist_clamp(
    q: Tensor,
    axis: Tensor,
    min_rad: float,
    max_rad: float,
    swing_max_rad: float = 0.0,
    eps: float = 1e-6,
) -> Tensor:
    """Project unit quaternions `q` (..., 4) onto a hinge about `axis` (3,).

    Decompose q = swing · twist (twist about `axis`), clamp the signed twist
    angle into [min_rad, max_rad], clamp the swing angle into [0, swing_max_rad]
    (swing_max_rad=0 ⇒ pure hinge — swing forced to identity), recompose and
    renormalize. Returns unit quaternions on the upper hemisphere.
    """
    conj = _QUAT_CONJ.to(device=q.device, dtype=q.dtype)
    q = normalize_quaternions(q)  # unit + w ≥ 0
    a = axis.to(device=q.device, dtype=q.dtype)

    w = q[..., :1]
    v = q[..., 1:]
    d = (v * a).sum(dim=-1, keepdim=True)  # axial component of the vector part

    # original twist (rotation about a), normalized; norm² = w² + d²
    twist_norm = torch.sqrt(w * w + d * d).clamp_min(eps)
    twist_orig = torch.cat([w, d * a], dim=-1) / twist_norm

    # swing = q · twist⁻¹, lifted to the upper hemisphere
    swing = quat_to_upper_hemisphere(_quat_mul(q, twist_orig * conj))

    # signed twist angle θ = 2·atan2(d, w) ∈ (−π, π]; clamp into the range
    theta = 2.0 * torch.atan2(d, w)
    th = 0.5 * theta.clamp(min_rad, max_rad)
    twist_new = torch.cat([torch.cos(th), torch.sin(th) * a], dim=-1)

    # swing angle φ ∈ [0, π]; shrink toward identity, keeping the swing axis
    sw = swing[..., :1]
    sv = swing[..., 1:]
    sv_norm = torch.linalg.vector_norm(sv, dim=-1, keepdim=True)
    phi = 2.0 * torch.atan2(sv_norm, sw)
    ph = 0.5 * phi.clamp(0.0, swing_max_rad)
    axis_s = sv / sv_norm.clamp_min(eps)
    swing_new = torch.cat([torch.cos(ph), torch.sin(ph) * axis_s], dim=-1)
    identity = torch.zeros_like(swing_new)
    identity[..., 0] = 1.0
    swing_new = torch.where(sv_norm < eps, identity, swing_new)

    q_new = normalize_quaternions(_quat_mul(swing_new, twist_new))
    # Degenerate twist (≈180° about a perpendicular axis): leave q untouched.
    return torch.where(twist_norm < 1e-4, q, q_new)


@dataclass
class HingeConstraint:
    """Clamp one joint to a hinge: twist angle ∈ [min_deg, max_deg] about `axis`,
    swing shrunk toward identity. Applied as a projection each sampling step.

    Attributes:
        joint: joint index (0..J-1) or SMPL name (e.g. "L_Knee").
        axis: "x"/"y"/"z" or a 3-vector — the hinge (twist) axis.
        min_deg / max_deg: allowed signed twist range (e.g. 0..10 for a knee).
        swing_max_deg: max off-axis swing (0 = pure hinge).
        frame_start / frame_end: half-open window; end None/≤0 ⇒ to last frame.
    """

    joint: int | str
    axis: str | tuple[float, float, float] | Tensor = "x"
    min_deg: float = 0.0
    max_deg: float = 10.0
    swing_max_deg: float = 0.0
    frame_start: int = 0
    frame_end: int | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "HingeConstraint":
        return cls(
            joint=d["joint"],
            axis=d.get("axis", "x"),
            min_deg=float(d.get("min_deg", 0.0)),
            max_deg=float(d.get("max_deg", 10.0)),
            swing_max_deg=float(d.get("swing_max_deg", 0.0)),
            frame_start=int(d.get("frame_start", 0)),
            frame_end=(int(d["frame_end"]) if d.get("frame_end") not in (None, "", -1) else None),
        )

    def frame_window(self, num_frames: int) -> tuple[int, int]:
        return _clamp_window(self.frame_start, self.frame_end, num_frames)


def parse_ranges(specs: list[dict] | None) -> list[HingeConstraint]:
    """Parse a list of JSON-ish hinge-limit dicts into objects."""
    if not specs:
        return []
    return [HingeConstraint.from_dict(s) for s in specs]


def build_hinge_projector(
    constraints: list[HingeConstraint],
    num_frames: int,
    num_joints: int = NUM_JOINTS,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Callable[[Tensor], Tensor] | None:
    """Compile hinge limits into a projector `fn(x) -> x` for the sampler.

    The returned callable clamps each constrained joint's quaternion (over its
    frame window) onto its hinge submanifold; pass it as
    `RiemannianEulerSampler.sample(project_fn=...)`. Returns None when there are
    no effective constraints (so the sampler skips the hook entirely).
    """
    specs = []
    for c in constraints:
        j = _resolve_joint(c.joint)
        lo, hi = 3 + 4 * j, 3 + 4 * (j + 1)
        fs, fe = c.frame_window(num_frames)
        if fe <= fs:
            continue
        a = torch.as_tensor(_resolve_axis(c.axis), dtype=dtype, device=device)
        a = a / torch.linalg.vector_norm(a).clamp_min(1e-8)
        specs.append((lo, hi, fs, fe, a,
                      math.radians(c.min_deg), math.radians(c.max_deg),
                      math.radians(c.swing_max_deg)))
    if not specs:
        return None

    def project(x: Tensor) -> Tensor:
        x = x.clone()
        for lo, hi, fs, fe, a, mn, mx, sm in specs:
            x[:, fs:fe, lo:hi] = swing_twist_clamp(x[:, fs:fe, lo:hi], a, mn, mx, sm)
        return x

    return project
