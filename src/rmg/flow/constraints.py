"""Sampling-time joint-angle constraints for RMG (no fine-tuning).

RMG motion lives on the product manifold R^3 × (S^3)^J: each joint is an
*independent* unit quaternion (paper §3.1). We constrain the **bend angle** at a
joint — the angle between its incoming bone (parent→joint) and its outgoing bone
(joint→child): 0° = straight, larger = more flexed. This is the intuitive,
axis-free quantity ("elbow at 90°") and is exactly what the FK interior-angle
plots measure.

A constraint is applied as a per-step **projection**: after every ODE step of
the Riemannian Euler sampler we clamp the controlling quaternion so the bend
lands in [min, max]. A fixed angle is the degenerate range min == max, so "pin
to θ" and "limit to [lo, hi]" use ONE projector. The projection is closed-form
(bend is a function of a single quaternion), so this is an exact hard limit —
not a soft barrier penalty. Sampling-only: the trained weights are untouched.

Two refinements make the hold feel natural rather than a hard teleport:

* **strength** (0..1) — apply only that fraction of the correction each step, so
  the joint is *biased* toward the target instead of snapped. 1.0 = hard clamp.
* **ease_frames** — ramp the strength up/down over this many frames at the edges
  of a partial frame window, so a windowed constraint doesn't snap on/off.

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


from shared.geometry.skeleton import (
    JOINT_NAMES,
    NUM_JOINTS,
    ROOT_JOINT,
    T2M_KINEMATIC_CHAINS,
    quat_mul,
    quat_rotate,
)

# Layout constants for the quaternion-on-S^3 representations (tr / trp). The
# translation occupies the first 3 dims; each joint quaternion is 4 contiguous
# dims thereafter. Representations without this prefix layout aren't supported.
CONSTRAINABLE_REPRESENTATIONS = ("tr", "trp")


def _unit(q: Tensor, eps: float = 1e-8) -> Tensor:
    """Unit-normalise quaternions without touching their sign.

    Distinct from `normalize_quaternions`, which additionally canonicalises onto
    the q_w >= 0 hemisphere. That canonicalisation is right for a stored pose and
    wrong inside the sampler, where q and -q are antipodal points of the ambient
    sphere the ODE is integrating on.
    """
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


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


# ---------------------------------------------------------------------------
# Anatomical joint → controlling quaternion (per-chain FK off-by-one).
#
# RMG/HumanML3D forward kinematics (skeleton.forward_kinematics) uses a
# per-chain convention: the offset of `chain[i]` is rotated by the running
# rotation *after* multiplying in `quats[chain[i]]`. As a result, `quats[j]`
# orients the bone *leading into* joint j (the parent→j segment), NOT the bend
# at j. The anatomical bend at joint j — the angle between its incoming and
# outgoing bones — is governed by the quaternion of the NEXT joint along its
# kinematic chain (its child). Concretely the L_Elbow bend is controlled by
# quats[L_Wrist], the L_Knee bend by quats[L_Ankle], etc.
#
# Constraints are authored anatomically ("hold/limit the elbow"), so we remap
# joint → controller before touching the state. Without this, pinning quats[j]
# only re-aims the segment *into* j and leaves the bend free (the bug the
# elbow demo exposed). End-effectors have no outgoing bone and so no
# representable bend; the root quaternion is global orientation, not a bend.
# ---------------------------------------------------------------------------


def bend_controller_index(joint: int | str) -> int:
    """Index of the quaternion that controls the anatomical bend at `joint`.

    See the module note on the per-chain FK convention. Raises if `joint` is the
    root (global orientation, not a bend) or a kinematic end-effector (wrists,
    ankles, feet, head — no outgoing bone in the 22-joint skeleton).
    """
    j = _resolve_joint(joint)
    if j == ROOT_JOINT:
        raise ValueError(
            "the root (pelvis) quaternion is global orientation, not a bend — "
            "it has no controllable joint angle"
        )
    for chain in T2M_KINEMATIC_CHAINS:
        if j in chain:
            pos = chain.index(j)
            if pos + 1 < len(chain):
                return chain[pos + 1]
            break  # j is the last joint on its chain → end-effector
    raise ValueError(
        f"{JOINT_NAMES[j]!r} is a kinematic end-effector — its bend is not "
        "representable in the 22-joint skeleton; constrain its parent instead"
    )


def _clamp_window(frame_start: int, frame_end: int | None, num_frames: int) -> tuple[int, int]:
    """Half-open [start, end) frame window; end None/≤0 ⇒ to the last frame."""
    start = max(0, min(int(frame_start), num_frames))
    end = num_frames if frame_end in (None,) or int(frame_end) <= 0 else int(frame_end)
    end = max(start, min(end, num_frames))
    return start, end


def _ease_ramp(n: int, ease_frames: int, device, dtype) -> Tensor:
    """Per-frame strength multiplier in [0, 1] over an `n`-frame window.

    Ramps linearly 0→1 across the first `ease_frames` and 1→0 across the last
    `ease_frames`, flat at 1 in the middle. `ease_frames=0` ⇒ all ones (no
    ramp). Keeps a windowed/partial constraint from snapping on and off.
    """
    r = torch.ones(n, device=device, dtype=dtype)
    e = max(0, int(ease_frames))
    if e == 0 or n == 0:
        return r
    e = min(e, (n + 1) // 2)
    ramp = (torch.arange(1, e + 1, device=device, dtype=dtype)) / (e + 1)
    r[:e] = ramp
    r[n - e:] = torch.minimum(r[n - e:], ramp.flip(0))
    return r


# ---------------------------------------------------------------------------
# BEND ANGLE projection.
#
# The bend at joint j is governed by quats[child(j)] =: q (the per-chain
# off-by-one, see bend_controller_index). With u = incoming rest bone and
# v = outgoing rest bone (both unit), the current outgoing direction is d = q·v
# and the bend is α = angle(u, d) — independent of all other joints and of the
# global pose. We clamp α into [min, max] by rotating d within the (u, d) plane
# (axis n = u×d) by `strength·(β−α)`, i.e. q ← Δ ⊗ q with Δ = quat(n, s(β−α)).
# Because Δ acts only on the outgoing direction, the joint's TWIST about its own
# bone and the bend DIRECTION are preserved — the model keeps everything except
# the bend magnitude.
# ---------------------------------------------------------------------------


def bend_clamp(
    q: Tensor,
    u: Tensor,
    v: Tensor,
    min_rad: float,
    max_rad: float,
    strength: float | Tensor = 1.0,
    eps: float = 1e-6,
) -> Tensor:
    """Project controller quaternion(s) `q` (..., 4) so the bend angle between
    incoming bone `u` (3,) and rotated outgoing bone `q·v` (v: (3,)) is pulled
    toward [min_rad, max_rad].

    `strength` (scalar or (...,1) tensor in [0, 1]) is the fraction of the
    correction applied: 1.0 = exact hard clamp, <1.0 = soft bias toward the
    band. Preserves bend direction and twist about v. Unit out.

    The ambient SIGN of `q` is preserved. `normalize_quaternions` would also map
    q to the q_w >= 0 hemisphere; q and -q are the same rotation, but they are
    antipodal points of S^3, and this projector runs inside the ODE loop on the
    ambient state. Canonicalising the sign there teleports the state across the
    sphere between integration steps, which changes the trajectory even when the
    angular correction is zero. With a plain unit normalisation a constraint that
    is already satisfied leaves the state untouched.
    """
    q = _unit(q)
    u = u.to(device=q.device, dtype=q.dtype)
    v = v.to(device=q.device, dtype=q.dtype)
    u = u / torch.linalg.vector_norm(u).clamp_min(eps)
    v = v / torch.linalg.vector_norm(v).clamp_min(eps)

    ushape = u.expand(*q.shape[:-1], 3)
    d = quat_rotate(q, v.expand(*q.shape[:-1], 3))           # current outgoing dir
    cos_a = (d * ushape).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    alpha = torch.acos(cos_a)                                # current bend (...,1)
    beta = alpha.clamp(min_rad, max_rad)
    s = strength if torch.is_tensor(strength) else q.new_tensor(float(strength))

    n = torch.cross(ushape, d, dim=-1)                       # bend-plane normal = u×d
    n_norm = torch.linalg.vector_norm(n, dim=-1, keepdim=True)
    axis = n / n_norm.clamp_min(eps)
    half = 0.5 * s * (beta - alpha)                          # partial correction
    delta = torch.cat([torch.cos(half), torch.sin(half) * axis], dim=-1)
    q_new = _unit(quat_mul(delta, q))
    # Keep the representative on the same side as the input: delta is a rotation
    # by (beta - alpha), so the product is already the near representative, but
    # a large correction must not be allowed to hand back the antipode.
    q_new = torch.where((q_new * q).sum(-1, keepdim=True) < 0, -q_new, q_new)
    # bone (anti)parallel to u ⇒ bend direction undefined: leave q untouched.
    return torch.where(n_norm < eps, q, q_new)


def bend_angles_deg(x: Tensor, joint: int | str, skeleton: "Skeleton") -> Tensor:
    """Per-frame bend angle in degrees at `joint`, read off a flat state.

    The inverse of what `bend_clamp` writes: takes `x` (..., D) in the tr/trp
    layout and returns (...,) the angle between the incoming rest bone and the
    rotated outgoing bone. Used to verify a constraint held, from the same
    quantity the projector acted on rather than from decoded joint positions.
    """
    j = _resolve_joint(joint)
    ctrl = bend_controller_index(j)
    lo, hi = 3 + 4 * ctrl, 3 + 4 * (ctrl + 1)
    q = _unit(x[..., lo:hi])
    offsets = skeleton.offsets.to(device=x.device, dtype=x.dtype)
    u = offsets[j]
    v = offsets[ctrl]
    u = u / torch.linalg.vector_norm(u).clamp_min(1e-6)
    v = v / torch.linalg.vector_norm(v).clamp_min(1e-6)
    d = quat_rotate(q, v.expand(*q.shape[:-1], 3))
    cos = (d * u.expand_as(d)).sum(-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


@dataclass
class BendConstraint:
    """Constrain the bend angle at one joint into [min_deg, max_deg] over a frame
    window. Fixed angle = min_deg == max_deg. Applied as a per-step projection.

    Attributes:
        joint: joint index (0..J-1) or SMPL name (e.g. "L_Elbow").
        min_deg / max_deg: allowed bend range (0 = straight). For an exact pin,
            set both equal (or use `from_dict` with `bend_deg`).
        frame_start / frame_end: half-open window; end None/≤0 ⇒ to last frame.
        strength: fraction of the correction applied each step (0..1). 1.0 = hard
            hold, <1.0 = soft bias that lets the model fight back.
        ease_frames: ramp `strength` up/down over this many frames at the window
            edges (0 = snap on/off). Smooths the onset of a windowed constraint.
    """

    joint: int | str
    min_deg: float = 0.0
    max_deg: float = 0.0
    frame_start: int = 0
    frame_end: int | None = None
    strength: float = 1.0
    ease_frames: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "BendConstraint":
        # Preferred keys: `bend_deg` (exact) or `bend_min`/`bend_max` (range).
        # Legacy keys are also accepted so an older axis-based frontend keeps
        # working with bend semantics: `angle_deg` → exact bend, `min_deg`/
        # `max_deg` → bend range. `axis` is ignored (a bend angle is axis-free).
        if d.get("bend_deg") is not None:
            lo = hi = float(d["bend_deg"])
        elif d.get("angle_deg") is not None:
            lo = hi = float(d["angle_deg"])
        elif d.get("bend_min") is not None or d.get("bend_max") is not None:
            lo = float(d.get("bend_min", 0.0))
            hi = float(d.get("bend_max", lo))
        else:
            lo = float(d.get("min_deg", 0.0))
            hi = float(d.get("max_deg", lo))
        return cls(
            joint=d["joint"],
            min_deg=lo,
            max_deg=hi,
            frame_start=int(d.get("frame_start", 0)),
            frame_end=(int(d["frame_end"]) if d.get("frame_end") not in (None, "", -1) else None),
            strength=float(d.get("strength", 1.0)),
            ease_frames=int(d.get("ease_frames", 0)),
        )

    def frame_window(self, num_frames: int) -> tuple[int, int]:
        return _clamp_window(self.frame_start, self.frame_end, num_frames)


def parse_bends(specs: list[dict] | None) -> list[BendConstraint]:
    """Parse a list of JSON-ish bend-angle dicts into objects."""
    if not specs:
        return []
    return [BendConstraint.from_dict(s) for s in specs]


def build_bend_projector(
    constraints: list[BendConstraint],
    skeleton: "Skeleton",
    num_frames: int,
    num_joints: int = NUM_JOINTS,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Callable[[Tensor], Tensor] | None:
    """Compile bend-angle constraints into a projector `fn(x) -> x`.

    Needs the `skeleton` for the rest-bone offsets that define each joint's bend.
    For each constraint, pulls the bend at the joint (acting on its controller
    quaternion — `bend_controller_index`) toward [min, max] every ODE step, with
    a per-frame strength = constraint.strength · ease-ramp. Pass the result as
    `RiemannianEulerSampler.sample(project_fn=...)`. Returns None when there are
    no effective constraints.
    """
    offsets = skeleton.offsets.to(device=device, dtype=dtype)
    specs = []
    for c in constraints:
        j = _resolve_joint(c.joint)
        ctrl = bend_controller_index(j)
        lo, hi = 3 + 4 * ctrl, 3 + 4 * (ctrl + 1)
        fs, fe = c.frame_window(num_frames)
        if fe <= fs:
            continue
        u = offsets[j]        # incoming bone: parent(j) → j
        v = offsets[ctrl]     # outgoing bone: j → child(j) == ctrl
        # Per-frame strength over the window (stiffness · ease-ramp), stored 1D.
        strength = float(c.strength) * _ease_ramp(fe - fs, c.ease_frames, device, dtype)
        specs.append((lo, hi, fs, fe, u, v,
                      math.radians(c.min_deg), math.radians(c.max_deg), strength))
    if not specs:
        return None

    def project(x: Tensor) -> Tensor:
        x = x.clone()
        for lo, hi, fs, fe, u, v, mn, mx, strength in specs:
            # x[:, fs:fe, lo:hi] is (B, Tw, 4); broadcast strength as (1, Tw, 1).
            s = strength.to(device=x.device, dtype=x.dtype).view(1, fe - fs, 1)
            x[:, fs:fe, lo:hi] = bend_clamp(x[:, fs:fe, lo:hi], u, v, mn, mx, strength=s)
        return x

    return project
