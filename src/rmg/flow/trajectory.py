"""Sampling-time TRAJECTORY (spatial) constraints for RMG — no fine-tuning.

The task, as posed by the mask-control line of work (OmniControl, GMD,
PriorMDM, MaskControl/ControlMM): condition generation on a **sparse spatial
control signal** — a set of world-space positions for chosen joints at chosen
frames, described by a binary (frame × joint) mask — and hit those positions
while staying on the text prompt and on the motion manifold. The canonical
report is (Traj. err, Loc. err, Avg. err) at a 50 cm threshold plus FID /
R-precision / foot-skate, swept over keyframe densities {1, 2, 5, 49, all} and
over the "cross-combination" joint set {pelvis, feet, head, wrists}.

Those methods all condition an **HumanML3D-263-D** model, where the root is
stored as a *velocity* (position is a cumulative sum over frames). A position
constraint there is a global, non-local function of the whole prefix, so it can
only be imposed softly — by backpropagating a positional loss into the sample
each denoising step ("spatial guidance"), which is why their trajectory error
is nonzero even for the pelvis.

RMG is different in a way that matters here. Motion lives on
R^3 × (S^3)^J and forward kinematics reads

    p_j = τ + f_j(q)          with     p_root = τ                       (FK)

— the R^3 factor *is* the world-space pelvis position at that frame, and every
other joint is that same τ plus a pose term. Two consequences we exploit:

* **Pelvis control is an exact projection.** Writing τ ← target hits the
  target with zero error and zero iterations. It is the Euclidean factor of the
  product manifold, so unlike a quaternion pin it does not fight the geometry.
* **Any single joint is exact too, via the root.** For a target on joint j,
  τ ← target − f_j(q) gives p_j = target exactly, leaving the *pose* q
  untouched — the shift is absorbed by the root. Only when several joints are
  controlled in the same frame does a residual remain (their mutual offsets are
  a pose property), and that residual is what needs gradient guidance.

So this module offers three enforcement modes:

    "project"  exact root absorption each ODE step (residual left alone)
    "guide"    gradient guidance only — the literature's mechanism, our baseline
    "hybrid"   project, then guide the residual (default)

The catch, and the thing the evaluation has to measure: writing τ at scattered
keyframes and leaving neighbouring frames where the model put them injects a
**velocity discontinuity** at each keyframe — the trajectory error goes to zero
but the motion can foot-skate or jerk. That trade is controlled by how the
per-frame root correction δ_t is spread over the clip (`blend`):

    "interp"  (default) linearly interpolate δ between keyframes, constant
              outside them — exact AT the keyframes and C0 everywhere, i.e. the
              whole root path is *warped* to pass through the control points.
    "contact" as interp, but the correction advances in proportion to how FREE
              each frame is (both feet airborne) instead of uniformly in time.
              Same exactness, but the ground is covered during swing rather than
              dragged out from under a planted foot — the fix for "the feet slide
              because the gait was never re-planned".
    "local"   raised-cosine falloff to zero within `blend_frames` of a keyframe
              — keeps the rest of the clip where the model put it, at the cost
              of a sharper local edit.
    "none"    write δ only at controlled frames (the naive inpainting baseline;
              lowest error, worst smoothness).

Flat layout (tr / trp representations), shared with `flow.constraints`:
    flat[..., 0:3]                 = translation  (= world pelvis position)
    flat[..., 3+4j : 3+4(j+1)]     = quaternion for joint j
World frame = HumanML3D FK frame: X right, Y up, Z forward, floor at y=0.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor

from shared.geometry.skeleton import (
    JOINT_NAMES,
    NUM_JOINTS,
    ROOT_JOINT,
    Skeleton,
    forward_kinematics,
    quat_mul,
)

from .constraints import CONSTRAINABLE_REPRESENTATIONS, _resolve_joint, _unit  # noqa: F401

_EPS = 1e-8

# Joint sets used by the mask-control evaluation protocol. "cross" is the
# cross-combination set those papers sweep; per clip ONE joint is drawn from it
# (see `sample_control_signal`), which is also what they do.
CONTROL_JOINT_SETS: dict[str, tuple[int, ...]] = {
    "pelvis": (0,),
    "left_foot": (10,),
    "right_foot": (11,),
    "head": (15,),
    "left_wrist": (20,),
    "right_wrist": (21,),
    "cross": (0, 10, 11, 15, 20, 21),
}

# Keyframe densities swept by the protocol. "all" ⇒ every valid frame.
CONTROL_DENSITIES: tuple[int | str, ...] = (1, 2, 5, 49, "all")

ENFORCEMENT_MODES = ("project", "guide", "hybrid")
BLEND_MODES = ("interp", "local", "none", "contact")

# Foot joints and the height band under which one counts as planted, for
# contact-aware blending. Matches `shared.eval.motion_quality`.
_FOOT_IDX = (7, 10, 8, 11)          # L_Ankle, L_Foot, R_Ankle, R_Foot
_PLANT_BAND = 0.05                  # m
_PLANT_SOFT = 0.02                  # m — sigmoid width, keeps the weight smooth

# Error thresholds (metres) for the success-rate metrics. 0.5 m is the number
# the papers headline; 0.2 m is the stricter cut we also report.
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.5, 0.2)


def _axis_mask(axes: str, device=None, dtype=torch.float32) -> Tensor:
    """(3,) 0/1 selector for which world axes a constraint controls.

    A hand-drawn floor path constrains x and z but says nothing about height,
    so `axes="xz"` leaves y to the model instead of pinning the body to
    whatever height the author's 2-D sketch implied.
    """
    a = str(axes or "xyz").lower()
    bad = set(a) - set("xyz")
    if bad:
        raise ValueError(f"axes must be a subset of 'xyz', got {axes!r}")
    if not a:
        raise ValueError("axes must name at least one of x/y/z")
    return torch.tensor(
        [1.0 if "x" in a else 0.0, 1.0 if "y" in a else 0.0, 1.0 if "z" in a else 0.0],
        device=device, dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Control signal
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryConstraint:
    """World-space position targets for ONE joint over a frame mask.

    Shapes are either **shared** — positions (T, 3), mask (T,), the same signal
    for every sample in the batch, which is what the interactive app sends — or
    **per-row** — positions (B, T, 3), mask (B, T), one signal per clip, which
    is what the evaluation protocol needs (every test clip contributes its own
    reference trajectory). Both broadcast against a (B, T, D) state.

    Attributes:
        joint: joint index (0..J-1) or SMPL name (e.g. "pelvis", "L_Wrist").
        positions: (T, 3) or (B, T, 3) target positions. Entries where `mask` is
            False are ignored (they may hold anything).
        mask: (T,) or (B, T) bool — the controlled frames. This is the
            mask-control "spatial mask" for this joint.
        axes: which world axes are controlled ("xyz" | "xz" | "y" | …).
        weight: relative strength when several joints are controlled at once.
        tol: metres of slack — an error below `tol` contributes no energy.
    """

    joint: int | str
    positions: Tensor
    mask: Tensor
    axes: str = "xyz"
    weight: float = 1.0
    tol: float = 0.0

    def __post_init__(self) -> None:
        self.joint_idx = _resolve_joint(self.joint)
        self.positions = torch.as_tensor(self.positions, dtype=torch.float32)
        if self.positions.ndim not in (2, 3) or self.positions.shape[-1] != 3:
            raise ValueError(
                f"positions must be (T, 3) or (B, T, 3), got {tuple(self.positions.shape)}"
            )
        # Normalise to a leading batch axis of 1 (shared) or B (per-row) so
        # everything downstream is one broadcast rather than two code paths.
        if self.positions.ndim == 2:
            self.positions = self.positions.unsqueeze(0)
        self.mask = torch.as_tensor(self.mask).to(torch.bool)
        if self.mask.ndim == 1:
            self.mask = self.mask.unsqueeze(0)
        if self.mask.ndim != 2:
            raise ValueError(f"mask must be (T,) or (B, T), got {tuple(self.mask.shape)}")
        if self.mask.shape[-1] != self.positions.shape[-2]:
            raise ValueError(
                f"mask length {self.mask.shape[-1]} != positions length "
                f"{self.positions.shape[-2]}"
            )
        if self.mask.shape[0] != self.positions.shape[0] and 1 not in (
            self.mask.shape[0], self.positions.shape[0]
        ):
            raise ValueError(
                f"mask batch {self.mask.shape[0]} and positions batch "
                f"{self.positions.shape[0]} do not broadcast"
            )
        if self.weight <= 0:
            raise ValueError(f"weight must be > 0, got {self.weight}")
        _axis_mask(self.axes)  # validate early

    @property
    def num_frames(self) -> int:
        return int(self.positions.shape[-2])

    @property
    def batch_size(self) -> int:
        """1 when the signal is shared across the batch, else the row count."""
        return max(int(self.positions.shape[0]), int(self.mask.shape[0]))

    @property
    def num_controlled(self) -> int:
        """Controlled slots per clip (the max over rows, for a per-row signal)."""
        return int(self.mask.sum(dim=-1).max().item())

    def to(self, device=None, dtype=None) -> "TrajectoryConstraint":
        c = TrajectoryConstraint(
            joint=self.joint_idx,
            positions=self.positions.to(device=device, dtype=dtype),
            mask=self.mask.to(device=device),
            axes=self.axes, weight=self.weight, tol=self.tol,
        )
        return c


@dataclass
class TrajectoryControl:
    """A full spatial control signal: several per-joint constraints + how to
    enforce them.

    Attributes:
        constraints: per-joint position targets.
        mode: "project" | "guide" | "hybrid" — see the module docstring.
        blend: how the exact root correction is spread over frames
            ("interp" | "local" | "none").
        blend_frames: falloff radius in frames for blend="local".
        guidance_weight: strength of the gradient channel (mode "guide" /
            "hybrid"), in the sampler's velocity-relative units.
        face_path: also turn the body to face along the target path. Moving the
            root's POSITION without its ORIENTATION drags the body sideways or
            backwards along the path — measured at 125° between facing and
            travel on an L-path, against 13° for an unconstrained walk. Heading
            is exactly projectable too (a yaw on the root quaternion), so this
            closes the gap rather than papering over it. Root-driven control
            only; a wrist target says nothing about which way the body faces.
        face_strength: fraction of the yaw correction applied per step (0..1).
        retime: follow the path's GEOMETRY on the model's own schedule, instead
            of pinning point p to frame f. A drawn path is normally resampled at
            constant speed, which demands full walking speed from frame 0 — but
            a generated clip stands still for ~1.9 s first, then accelerates and
            decelerates (measured). Forcing constant speed drags a standing body
            and flattens the whole speed profile. With `retime`, each frame's
            target is taken from how far along the path the body has ALREADY
            travelled, so standing frames stay put and the correction becomes a
            small sideways snap onto the path rather than a shove along it.

            This changes the contract from *trajectory* control ("be here at
            frame f") to *path* control ("go this way"), so it is for authored
            paths — NOT for the benchmark protocol, which scores position at
            specific frames. Root-driven control only.
        face_smooth: frames to average the target heading over. The target path
            is chord-linear between keyframes, so its raw tangent is piecewise
            CONSTANT and snaps — at a right-angled corner, 90° in a single frame.
            Following that literally spikes the jerk 30× (measured). Averaging
            the heading over ~half a second turns the corner the way a body
            actually can. 0/1 disables it.
    """

    constraints: list[TrajectoryConstraint] = field(default_factory=list)
    mode: str = "hybrid"
    blend: str = "interp"
    blend_frames: int = 10
    guidance_weight: float = 1.0
    face_path: bool = False
    face_strength: float = 1.0
    face_smooth: int = 9
    retime: bool = False
    foot_lock: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ENFORCEMENT_MODES:
            raise ValueError(f"mode must be one of {ENFORCEMENT_MODES}, got {self.mode!r}")
        if self.blend not in BLEND_MODES:
            raise ValueError(f"blend must be one of {BLEND_MODES}, got {self.blend!r}")
        self.constraints = [c for c in self.constraints if c.num_controlled > 0]

    def __bool__(self) -> bool:
        return bool(self.constraints)

    @property
    def num_frames(self) -> int:
        return max((c.num_frames for c in self.constraints), default=0)

    @property
    def uses_projection(self) -> bool:
        return self.mode in ("project", "hybrid")

    @property
    def uses_guidance(self) -> bool:
        return self.mode in ("guide", "hybrid") and self.guidance_weight != 0.0

    def to(self, device=None, dtype=None) -> "TrajectoryControl":
        return TrajectoryControl(
            constraints=[c.to(device=device, dtype=dtype) for c in self.constraints],
            mode=self.mode, blend=self.blend, blend_frames=self.blend_frames,
            guidance_weight=self.guidance_weight,
            face_path=self.face_path, face_strength=self.face_strength,
            face_smooth=self.face_smooth, retime=self.retime,
            foot_lock=self.foot_lock,
        )


# ---------------------------------------------------------------------------
# Wire format (frontend / config JSON) → TrajectoryControl
# ---------------------------------------------------------------------------


def resample_path(points, num_frames: int, closed: bool = False) -> Tensor:
    """Arc-length-resample a polyline to exactly `num_frames` points.

    Turns a path the user drew (a handful of clicks) into a per-frame target so
    the body traverses it at constant speed. `points` is (K, 3) or (K, 2)
    (interpreted as xz, y=0). Returns (num_frames, 3).
    """
    p = torch.as_tensor(points, dtype=torch.float32)
    if p.ndim != 2 or p.shape[-1] not in (2, 3):
        raise ValueError(f"path points must be (K, 2) or (K, 3), got {tuple(p.shape)}")
    if p.shape[-1] == 2:  # xz sketch → world xz, floor height
        p = torch.stack([p[:, 0], torch.zeros_like(p[:, 0]), p[:, 1]], dim=-1)
    if closed and p.shape[0] > 1:
        p = torch.cat([p, p[:1]], dim=0)
    n = max(1, int(num_frames))
    if p.shape[0] == 1:
        return p.expand(n, 3).clone()

    seg = torch.linalg.vector_norm(p[1:] - p[:-1], dim=-1)
    cum = torch.cat([torch.zeros(1), torch.cumsum(seg, dim=0)])
    total = float(cum[-1])
    if total < _EPS:  # degenerate (all points coincide)
        return p[:1].expand(n, 3).clone()
    want = torch.linspace(0.0, total, n)
    # Piecewise-linear lookup: for each target arc length find its segment.
    idx = torch.clamp(torch.searchsorted(cum, want, right=True) - 1, 0, p.shape[0] - 2)
    seg_len = (cum[idx + 1] - cum[idx]).clamp_min(_EPS)
    t = ((want - cum[idx]) / seg_len).clamp(0.0, 1.0).unsqueeze(-1)
    return p[idx] * (1 - t) + p[idx + 1] * t


def _constraint_from_dict(d: dict, num_frames: int) -> TrajectoryConstraint | None:
    """Parse one joint's control signal. Three accepted shapes:

      sparse   {"joint":…, "frames":[i,…], "points":[[x,y,z],…]}
      dense    {"joint":…, "positions":[[x,y,z]×T], "mask":[bool×T]}
      path     {"joint":…, "path":[[x,z],…], "frame_start":a, "frame_end":b}
               — arc-length-resampled across [a, b) so the joint traverses it.
    """
    T = int(num_frames)
    joint = d["joint"]
    axes = str(d.get("axes", "xyz"))
    weight = float(d.get("weight", 1.0))
    tol = float(d.get("tol", 0.0))

    positions = torch.zeros(T, 3)
    mask = torch.zeros(T, dtype=torch.bool)

    if d.get("path"):
        fs = max(0, int(d.get("frame_start", 0)))
        fe_raw = d.get("frame_end")
        fe = T if fe_raw in (None, "", -1) else int(fe_raw)
        fe = max(fs, min(fe, T))
        if fe <= fs:
            return None
        positions[fs:fe] = resample_path(d["path"], fe - fs, closed=bool(d.get("closed", False)))
        mask[fs:fe] = True
        # A path can be sub-sampled to a sparse keyframe set with "every".
        every = int(d.get("every", 1) or 1)
        if every > 1:
            keep = torch.zeros(T, dtype=torch.bool)
            keep[fs:fe:every] = True
            keep[fe - 1] = True  # always keep the endpoint
            mask &= keep
    elif d.get("positions") is not None:
        pos = torch.as_tensor(d["positions"], dtype=torch.float32)
        if pos.ndim != 2 or pos.shape[-1] != 3:
            raise ValueError(f"positions must be (T, 3), got {tuple(pos.shape)}")
        n = min(T, pos.shape[0])
        positions[:n] = pos[:n]
        m = d.get("mask")
        if m is None:
            mask[:n] = True
        else:
            mt = torch.as_tensor(m).to(torch.bool).reshape(-1)
            mask[:min(T, mt.shape[0])] = mt[:min(T, mt.shape[0])]
    else:
        frames = [int(f) for f in (d.get("frames") or [])]
        pts = torch.as_tensor(d.get("points") or [], dtype=torch.float32)
        if pts.numel() == 0 or not frames:
            return None
        if pts.ndim != 2 or pts.shape[-1] != 3:
            raise ValueError(f"points must be (K, 3), got {tuple(pts.shape)}")
        if len(frames) != pts.shape[0]:
            raise ValueError(f"frames ({len(frames)}) and points ({pts.shape[0]}) must match")
        for f, p in zip(frames, pts):
            if 0 <= f < T:
                positions[f] = p
                mask[f] = True

    if not bool(mask.any()):
        return None
    return TrajectoryConstraint(
        joint=joint, positions=positions, mask=mask, axes=axes, weight=weight, tol=tol,
    )


def parse_trajectory(d: dict | None, num_frames: int) -> TrajectoryControl | None:
    """Parse the frontend/config trajectory JSON into a TrajectoryControl.

    Shape: {"mode":…, "blend":…, "blend_frames":…, "guidance_weight":…,
            "constraints":[ …see `_constraint_from_dict`… ]}
    Returns None when there is no effective control signal.
    """
    if not d:
        return None
    specs = d.get("constraints") or d.get("joints") or []
    cons = [c for c in (_constraint_from_dict(s, num_frames) for s in specs) if c is not None]
    if not cons:
        return None
    ctrl = TrajectoryControl(
        constraints=cons,
        mode=str(d.get("mode", "hybrid")),
        blend=str(d.get("blend", "interp")),
        blend_frames=int(d.get("blend_frames", 10)),
        guidance_weight=float(d.get("guidance_weight", 1.0)),
        face_path=bool(d.get("face_path", False)),
        face_strength=float(d.get("face_strength", 1.0)),
        face_smooth=int(d.get("face_smooth", 9)),
        retime=bool(d.get("retime", False)),
        foot_lock=bool(d.get("foot_lock", False)),
    )
    return ctrl or None


# ---------------------------------------------------------------------------
# FK on the flat state
# ---------------------------------------------------------------------------


def flat_to_joints(x: Tensor, skeleton: Skeleton, num_joints: int = NUM_JOINTS) -> Tensor:
    """(B, T, D) flat tr/trp state → (B, T, J, 3) world joint positions.

    Differentiable (used inside the guidance energy) and sign-safe: quaternions
    are unit-normalised without hemisphere canonicalisation, since this runs on
    a state the ODE is still integrating (see `constraints._unit`).
    """
    qd = 3 + 4 * num_joints
    if x.shape[-1] < qd:
        raise ValueError(
            f"state has {x.shape[-1]} dims, need ≥ {qd} for {num_joints} joints — "
            f"trajectory constraints require a {CONSTRAINABLE_REPRESENTATIONS} representation"
        )
    trans = x[..., :3]
    quats = _unit(x[..., 3:qd].reshape(*x.shape[:-1], num_joints, 4))
    return forward_kinematics(skeleton, quats, trans)


# ---------------------------------------------------------------------------
# The exact channel: root absorption + how the correction is spread
# ---------------------------------------------------------------------------


def _neighbour_keys(has: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """For every frame, the nearest controlled frame at or before it and at or
    after it. `has` is (B, T) bool. Returns (prev, next, has_prev, has_next)
    with the index tensors already clamped into range.

    A running cummax over `arange` masked by `has` gives "last True index ≤ t";
    the mirrored cummin gives "first True index ≥ t". This is what lets each row
    of the batch carry its OWN keyframe set — which the evaluation protocol
    needs, since every test clip contributes a different reference trajectory.
    """
    B, T = has.shape
    device = has.device
    idx = torch.arange(T, device=device).view(1, T).expand(B, T)
    prev = torch.where(has, idx, torch.full_like(idx, -1)).cummax(dim=1).values
    nxt = torch.where(has, idx, torch.full_like(idx, T)).flip(1).cummin(dim=1).values.flip(1)
    has_prev, has_next = prev >= 0, nxt <= T - 1
    return prev.clamp(0, T - 1), nxt.clamp(0, T - 1), has_prev, has_next


def swing_weight(joints_world: Tensor, floor: float = _PLANT_BAND) -> Tensor:
    """Per-frame freedom to move the root, in [0, 1]. (B, T, J, 3) -> (B, T).

    ~0 while a foot is planted, ~1 while both feet are airborne. A soft sigmoid
    rather than a hard test so the resulting warp stays differentiable-ish and
    doesn't step.
    """
    h = joints_world[:, :, _FOOT_IDX, 1].min(dim=-1).values          # (B,T) lowest foot
    return torch.sigmoid((h - floor) / _PLANT_SOFT)


def _spread(delta: Tensor, has: Tensor, blend: str, radius: int,
            free: Tensor | None = None) -> Tensor:
    """Spread the per-frame root correction `delta` (B, T, 3), defined only where
    `has` (B, T) is True, across all frames according to `blend`.

    Exact at the controlled frames in every mode — the modes differ only in what
    happens *between* and *outside* them:
      "none"    zero elsewhere (keyframes teleport the root);
      "local"   raised-cosine decay to zero within `radius` frames;
      "interp"  linear between neighbouring keyframes, constant outside the
                span, i.e. the whole root path is warped through the targets.
    """
    B, T, _ = delta.shape
    dtype = delta.dtype
    if blend == "none":
        return delta * has.unsqueeze(-1).to(dtype)

    prev, nxt, has_prev, has_next = _neighbour_keys(has)
    gp = prev.unsqueeze(-1).expand(B, T, 3)
    gn = nxt.unsqueeze(-1).expand(B, T, 3)
    v_prev = delta.gather(1, gp)
    v_next = delta.gather(1, gn)
    t = torch.arange(T, device=delta.device).view(1, T)
    d_prev = (t - prev).to(dtype)
    d_next = (nxt - t).to(dtype)
    empty = (~has.any(dim=1)).view(B, 1, 1)          # a row with no control at all

    if blend in ("interp", "contact"):
        if blend == "contact" and free is not None:
            # Advance the correction in proportion to how FREE each frame is,
            # instead of uniformly in time. Between two keyframes the total
            # displacement is unchanged — so the keyframes are still hit exactly
            # — but it is spent while the feet are in the air rather than dragged
            # out from under a planted foot. A monotone reparametrisation of the
            # same warp: the body covers the ground during swing, like walking.
            W = torch.cumsum(free.clamp_min(1e-3), dim=1)            # (B,T)
            Wp = W.gather(1, prev)
            Wn = W.gather(1, nxt)
            u = ((W - Wp) / (Wn - Wp).clamp_min(_EPS)).clamp(0.0, 1.0).unsqueeze(-1)
        else:
            # d_prev + d_next is the gap between the bracketing keyframes; at a
            # keyframe both are 0 and u=0 picks v_prev — the exact correction.
            u = (d_prev / (d_prev + d_next).clamp_min(1.0)).unsqueeze(-1)
        out = v_prev * (1 - u) + v_next * u
        # Outside the keyframe span hold the nearest value constant, so the
        # clip's ends are not dragged off by an extrapolated trend.
        out = torch.where(has_prev.unsqueeze(-1), out, v_next)
        out = torch.where(has_next.unsqueeze(-1), out, v_prev)
        return torch.where(empty, torch.zeros_like(out), out)

    # "local" — nearest keyframe only, raised cosine falling to 0 at `radius`.
    r = float(max(1, int(radius)))
    inf = torch.full_like(d_prev, float("inf"))
    dp = torch.where(has_prev, d_prev, inf)
    dn = torch.where(has_next, d_next, inf)
    use_prev = dp <= dn
    d = torch.minimum(dp, dn)
    v = torch.where(use_prev.unsqueeze(-1), v_prev, v_next)
    w = 0.5 * (1.0 + torch.cos(math.pi * (d / r).clamp(max=1.0)))
    w = torch.where(d > r, torch.zeros_like(w), w)
    return torch.where(empty, torch.zeros_like(v), v * w.unsqueeze(-1))


# Upstream's FACE_JOINT_INDX = (r_hip, l_hip, r_shoulder, l_shoulder) — the
# joints HumanML3D's own canonicalisation uses to define which way a body faces.
FACE_JOINT_IDX = (2, 1, 17, 16)

_MOVE_EPS = 1e-3      # m/frame of target motion below which a heading is undefined


def body_forward(joints_world: Tensor) -> Tensor:
    """Horizontal facing direction of the body, (..., J, 3) -> (..., 3).

    Built exactly as `humanml3d_io._canonicalize_first_frame` builds it: an
    "across" vector from hips and shoulders, then forward = up × across. Using
    the dataset's own construction means the heading this projector corrects is
    the same heading the evaluation and the viewer measure.
    """
    r_hip, l_hip, sdr_r, sdr_l = FACE_JOINT_IDX
    across = (joints_world[..., r_hip, :] - joints_world[..., l_hip, :]) + (
        joints_world[..., sdr_r, :] - joints_world[..., sdr_l, :]
    )
    across = across / torch.linalg.vector_norm(across, dim=-1, keepdim=True).clamp_min(_EPS)
    # forward = up × across with up = +Y  ⇒  (across_z, 0, -across_x)
    fwd = torch.stack(
        [across[..., 2], torch.zeros_like(across[..., 0]), -across[..., 0]], dim=-1
    )
    return fwd / torch.linalg.vector_norm(fwd, dim=-1, keepdim=True).clamp_min(_EPS)


def _smooth_time(v: Tensor, win: int) -> Tensor:
    """Centred moving average of `v` (B, T, C) over `win` frames, edge-padded.

    Box filter via a cumulative sum — O(T) and no dependency on conv layers.
    """
    w = int(win)
    if w <= 1 or v.shape[1] < 2:
        return v
    pad = w // 2
    w = 2 * pad + 1                                   # force odd ⇒ symmetric
    vp = torch.cat([v[:, :1].expand(-1, pad, -1), v, v[:, -1:].expand(-1, pad, -1)], dim=1)
    cs = torch.cumsum(vp, dim=1)
    cs = torch.cat([torch.zeros_like(cs[:, :1]), cs], dim=1)
    return (cs[:, w:] - cs[:, :-w]) / float(w)


def _yaw_between(f: Tensor, g: Tensor) -> Tensor:
    """Signed yaw about +Y taking horizontal direction `f` onto `g`. (...,3)->(...,)"""
    cross_y = f[..., 2] * g[..., 0] - f[..., 0] * g[..., 2]
    dot = f[..., 0] * g[..., 0] + f[..., 2] * g[..., 2]
    return torch.atan2(cross_y, dot)


def _apply_root_yaw(q_root: Tensor, yaw: Tensor) -> Tensor:
    """Pre-multiply the root quaternion by a yaw about +Y. (...,4), (...) -> (...,4).

    Pre-multiplication rotates the body in the WORLD frame about the vertical
    axis through its own pelvis (translation is a separate factor), which is
    exactly the same operation `scene.place_motion` applies for spawn placement.
    """
    half = 0.5 * yaw
    z = torch.zeros_like(half)
    q_yaw = torch.stack([torch.cos(half), z, torch.sin(half), z], dim=-1)
    return _unit(quat_mul(q_yaw, q_root))


def _polyline_from_control(pos: Tensor, m: Tensor) -> tuple[Tensor, Tensor]:
    """Control points of one row, in frame order, as a polyline + its cumulative
    arc length. `pos` (1|B,T,3), `m` (1|B,T) — row 0 is used (retiming is for an
    authored path, which is shared across the batch). Returns ((K,3), (K,))."""
    idx = torch.nonzero(m[0], as_tuple=False).squeeze(-1)
    pts = pos[0].index_select(0, idx)                              # (K,3)
    if pts.shape[0] < 2:
        return pts, torch.zeros(pts.shape[0], device=pts.device, dtype=pts.dtype)
    seg = torch.linalg.vector_norm(
        (pts[1:] - pts[:-1]) * torch.tensor([1.0, 0.0, 1.0], device=pts.device, dtype=pts.dtype),
        dim=-1,
    )
    cum = torch.cat([torch.zeros(1, device=pts.device, dtype=pts.dtype), torch.cumsum(seg, 0)])
    return pts, cum


def _point_at_arclen(pts: Tensor, cum: Tensor, s: Tensor) -> Tensor:
    """Point on the polyline at arc length `s` (B,T) -> (B,T,3)."""
    K = pts.shape[0]
    if K == 1:
        return pts[0].view(1, 1, 3).expand(*s.shape, 3)
    i = torch.clamp(torch.searchsorted(cum.contiguous(), s.contiguous().reshape(-1)) - 1, 0, K - 2)
    lo, hi = cum[i], cum[i + 1]
    u = ((s.reshape(-1) - lo) / (hi - lo).clamp_min(_EPS)).clamp(0.0, 1.0).unsqueeze(-1)
    p = pts[i] * (1 - u) + pts[i + 1] * u
    return p.reshape(*s.shape, 3)


def _retimed_targets(joints: Tensor, pts: Tensor, cum: Tensor) -> Tensor:
    """Per-frame target from the body's OWN progress along the path. (B,T,3).

    Progress is the clip's cumulative horizontal root travel, normalised to the
    path's arc length. A frame where the body has not moved maps to the start of
    the path, so a standing lead-in is preserved instead of being dragged.
    """
    root = joints[:, :, ROOT_JOINT, :]
    step = torch.linalg.vector_norm(
        (root[:, 1:] - root[:, :-1]) * torch.tensor([1.0, 0.0, 1.0],
                                                    device=root.device, dtype=root.dtype),
        dim=-1,
    )                                                              # (B,T-1)
    travel = torch.cat([torch.zeros_like(step[:, :1]), torch.cumsum(step, dim=1)], dim=1)
    total = travel[:, -1:].clamp_min(_EPS)
    return _point_at_arclen(pts, cum, (travel / total) * cum[-1])


def _prepare_specs(
    control: TrajectoryControl,
    num_frames: int,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> list[tuple]:
    """Normalise every constraint onto a common T, device and dtype.

    Returns `(joint_idx, positions (1|B,T,3), mask (1|B,T), axis_mask (3,),
    weight, tol)` tuples. Constraints shorter than T are zero-padded and masked off
    past their end; longer ones are truncated — the batch's frame count varies
    with the clips it holds, and the control signal must follow it rather than
    silently indexing out of range.
    """
    T = int(num_frames)
    specs: list[tuple] = []
    for c in control.constraints:
        n = min(T, c.num_frames)
        Bm, Bp = c.mask.shape[0], c.positions.shape[0]
        m = torch.zeros(Bm, T, dtype=torch.bool, device=device)
        m[:, :n] = c.mask[:, :n].to(device=device)
        if not bool(m.any()):
            continue
        pos = torch.zeros(Bp, T, 3, device=device, dtype=dtype)
        pos[:, :n] = c.positions[:, :n].to(device=device, dtype=dtype)
        specs.append((c.joint_idx, pos, m, _axis_mask(c.axes, device, dtype),
                      float(c.weight), float(c.tol)))
    return specs


def build_trajectory_projector(
    control: TrajectoryControl,
    skeleton: Skeleton,
    num_frames: int,
    num_joints: int = NUM_JOINTS,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Callable[[Tensor], Tensor] | None:
    """Compile the exact (root-absorption) channel into `fn(x) -> x`.

    Each ODE step: run FK, form the per-frame root shift δ_t that best moves the
    controlled joints onto their targets (a weighted mean of their residuals —
    *exactly* the residual when one joint is controlled in that frame), spread δ
    over the clip per `control.blend`, and add it to the translation.

    Pass as `RiemannianEulerSampler.sample(project_fn=…)`. Composes with the
    bend projector via `compose_projectors`. Returns None if the control signal
    is empty or the mode is guidance-only.
    """
    if not control or not control.uses_projection:
        return None
    T = int(num_frames)
    skel = Skeleton(offsets=skeleton.offsets.to(device=device, dtype=dtype),
                    parents=skeleton.parents)
    specs = _prepare_specs(control, T, device=device, dtype=dtype)
    if not specs:
        return None
    blend, radius = control.blend, int(control.blend_frames)

    # Retiming needs the path as geometry, so it is prepared once here.
    root_spec = next((s for s in specs if s[0] == ROOT_JOINT), None)
    do_retime = bool(control.retime) and root_spec is not None
    retime_poly = _polyline_from_control(root_spec[1], root_spec[2]) if do_retime else None
    if do_retime and retime_poly[0].shape[0] < 2:
        do_retime = False                                            # a single point is no path

    def project(x: Tensor) -> Tensor:
        joints = flat_to_joints(x, skel, num_joints)                 # (B,T,J,3)
        B = x.shape[0]

        if do_retime:
            # PATH control: every frame's target is read off how far the body has
            # already travelled, so the model keeps its own speed profile — the
            # standing lead-in stays standing — and the correction becomes a
            # sideways snap onto the path instead of a shove along it.
            pts, cum = retime_poly
            tgt = _retimed_targets(joints, pts, cum)                 # (B,T,3)
            axv = root_spec[3].view(1, 1, 3).to(x.dtype)
            out = x.clone()
            out[..., :3] = x[..., :3] + (tgt - joints[:, :, ROOT_JOINT, :]) * axv
            return out

        num = torch.zeros_like(x[..., :3])                           # (B,T,3)
        # Per-AXIS denominator: an axis nobody controls in this frame keeps a
        # zero numerator and must stay zero, not be divided by another axis's
        # weight. A frame with one controlled joint therefore gets exactly its
        # own residual back — the root absorbs it in full.
        den = torch.zeros(1, T, 3, device=x.device, dtype=x.dtype)
        has = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        for j, pos, m, ax, w, _tol in specs:
            mf = m.unsqueeze(-1).to(x.dtype)                         # (1|B,T,1)
            axv = ax.view(1, 1, 3).to(x.dtype)
            # Residual restricted to the controlled axes: an "xz" path must not
            # drag the body's height around.
            r = (pos.to(x.dtype) - joints[:, :, j, :]) * axv
            num = num + w * r * mf
            den = den + w * mf * axv
            has = has | m.expand(B, T)
        delta = num / den.clamp_min(_EPS)                            # (B,T,3)
        free = swing_weight(joints) if blend == "contact" else None
        field = _spread(delta, has, blend, radius, free=free)        # (B,T,3)
        out = x.clone()
        out[..., :3] = x[..., :3] + field
        return out

    return project


def apply_path_facing(
    x: Tensor,
    control: TrajectoryControl,
    skeleton: Skeleton,
    num_joints: int = NUM_JOINTS,
    project_fn: Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    """Turn the body to face along its target path. A POST-PASS, not an ODE hook.

    Position control writes the root's translation but not its orientation, so
    the body keeps the heading the prompt produced while being dragged along a
    path pointing elsewhere — measured 125° between facing and travel on an
    L-path, against 13° for a free walk. Heading is exactly projectable (a yaw
    on the root quaternion, the same operation `scene.place_motion` applies for
    spawn), so the gap closes cleanly.

    **Why this runs after sampling rather than every ODE step.** Translation is
    the free Euclidean factor of the manifold: rewriting it each step costs the
    integrator nothing. The root *quaternion* is a point on S^3 that the ODE is
    actively integrating, and rotating it inside the loop fights the learned
    dynamics — every joint hangs off the root through FK, so the disturbance
    lands on the limbs. Measured on the trained model, in-loop yaw gave 26.9°
    alignment at a jerk of 3707, while this post-pass gives **2.4° at a jerk of
    121** — the same jerk as position-only control, i.e. free.

    A yaw about the pelvis leaves the pelvis itself fixed, so root position
    targets stay exact. `project_fn` is re-applied afterwards anyway, which
    restores exactness for any other controlled joint the rotation moved.
    """
    if not control or not control.face_path:
        return x
    specs = _prepare_specs(control, int(x.shape[1]), device=x.device, dtype=x.dtype)
    root_spec = next((s for s in specs if s[0] == ROOT_JOINT), None)
    if root_spec is None:                       # nothing root-driven ⇒ no heading
        return x

    _j, pos, m, _ax, _w, _tol = root_spec
    B, T = x.shape[0], x.shape[1]
    has = m.expand(B, T)
    # Dense target path (control points interpolated across frames), then its
    # tangent — the direction the body is meant to be travelling.
    p_star = _spread(pos.expand(B, T, 3) * m.unsqueeze(-1).to(x.dtype), has, "interp", 1)
    tang = torch.zeros_like(p_star)
    if T > 1:
        tang[:, 1:-1] = p_star[:, 2:] - p_star[:, :-2]
        tang[:, 0] = p_star[:, 1] - p_star[:, 0]
        tang[:, -1] = p_star[:, -1] - p_star[:, -2]
    tang = tang * torch.tensor([1.0, 0.0, 1.0], device=x.device, dtype=x.dtype)
    speed = torch.linalg.vector_norm(tang, dim=-1, keepdim=True)
    moving = speed.squeeze(-1) > _MOVE_EPS
    goal = tang / speed.clamp_min(_EPS)
    # Beyond the keyframe span the target is constant-extrapolated (zero tangent)
    # while the clip keeps moving; hold the nearest known heading there instead
    # of leaving the body pointing where it started.
    goal = _spread(goal * moving.unsqueeze(-1).to(x.dtype), moving, "interp", 1)
    # The path is chord-linear between keyframes, so its raw tangent is piecewise
    # constant and steps at corners; average it so the turn is one a body could make.
    goal = _smooth_time(goal, int(control.face_smooth))
    gnorm = torch.linalg.vector_norm(goal, dim=-1, keepdim=True)
    have_goal = (gnorm.squeeze(-1) > _EPS) & moving.any(dim=1, keepdim=True)
    goal = goal / gnorm.clamp_min(_EPS)

    joints = flat_to_joints(x, skeleton, num_joints)
    yaw = float(control.face_strength) * _yaw_between(body_forward(joints), goal)
    yaw = torch.where(have_goal, yaw, torch.zeros_like(yaw))

    lo, hi = 3 + 4 * ROOT_JOINT, 3 + 4 * (ROOT_JOINT + 1)
    out = x.clone()
    out[..., lo:hi] = _apply_root_yaw(_unit(x[..., lo:hi]), yaw)
    return project_fn(out) if project_fn is not None else out


def compose_projectors(*fns: Callable[[Tensor], Tensor] | None) -> Callable[[Tensor], Tensor] | None:
    """Chain projection hooks left to right, skipping Nones.

    Order matters: bend clamps touch quaternions and so move FK joint positions,
    which is why the trajectory projector (which reads those positions) is
    applied *after* them — it then absorbs whatever the bend clamp just did.
    """
    active = [f for f in fns if f is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]

    def project(x: Tensor) -> Tensor:
        for f in active:
            x = f(x)
        return x

    return project


# ---------------------------------------------------------------------------
# The soft channel: FK gradient guidance (the literature's mechanism)
# ---------------------------------------------------------------------------


def trajectory_energy(joints_world: Tensor, control: TrajectoryControl) -> Tensor:
    """Squared positional violation of the control signal. >= 0, differentiable.

    `joints_world` is (B, T, J, 3). Aggregation is **sum over controlled
    (frame, joint) entries, mean over batch** — matching `scene_energy`'s
    reasoning: averaging over all T·J slots would divide a handful of controlled
    keyframes by ~thousands of free ones and flatten the gradient to nothing.
    """
    device, dtype = joints_world.device, joints_world.dtype
    T = joints_world.shape[1]
    energy = joints_world.new_zeros(())
    for j, pos, m, ax, w, tol in _prepare_specs(control, T, device=device, dtype=dtype):
        d = (joints_world[:, :, j, :] - pos) * ax.view(1, 1, 3)
        dist = torch.sqrt((d * d).sum(-1) + 1e-12)                  # (B,T)
        viol = (dist - tol).clamp_min(0.0)
        energy = energy + w * (viol.pow(2) * m.to(dtype)).sum(dim=1).mean()
    return energy


def build_trajectory_energy_fn(
    control: TrajectoryControl,
    skeleton: Skeleton,
    num_joints: int = NUM_JOINTS,
) -> Callable[[Tensor], Tensor] | None:
    """Compile the guidance channel into `energy_fn(x) -> scalar` for the
    sampler's `energy_fn=` / `guidance_weight=` hook. None when unused."""
    if not control or not control.uses_guidance:
        return None

    def energy_fn(x: Tensor) -> Tensor:
        return trajectory_energy(flat_to_joints(x, skeleton, num_joints), control)

    return energy_fn


def compose_energies(*fns: Callable[[Tensor], Tensor] | None) -> Callable[[Tensor], Tensor] | None:
    """Sum several energy terms into one `energy_fn` (the sampler takes one).

    Used when a scene (room / obstacles / contacts) and a trajectory control
    signal are active at the same time.
    """
    active = [f for f in fns if f is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]

    def energy_fn(x: Tensor) -> Tensor:
        total = active[0](x)
        for f in active[1:]:
            total = total + f(x)
        return total

    return energy_fn


# ---------------------------------------------------------------------------
# Metrics — the mask-control report
# ---------------------------------------------------------------------------


def control_errors(
    joints_world: Tensor,
    control: TrajectoryControl,
    lengths: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Per-(sample, frame, joint) L2 error at the controlled slots.

    Returns (B, T, C) metres where C = len(control.constraints), and a matching
    bool mask of which entries are actually controlled, as `(errors, mask)`.
    Uncontrolled entries are zero in both.
    """
    device, dtype = joints_world.device, joints_world.dtype
    B, T = joints_world.shape[0], joints_world.shape[1]
    specs = _prepare_specs(control, T, device=device, dtype=dtype)
    C = len(specs)
    errs = joints_world.new_zeros(B, T, C)
    msk = torch.zeros(B, T, C, dtype=torch.bool, device=device)
    valid = None
    if lengths is not None:
        valid = torch.arange(T, device=device).view(1, T) < lengths.to(device).view(B, 1)
    for k, (j, pos, m, ax, _w, _tol) in enumerate(specs):
        d = (joints_world[:, :, j, :] - pos) * ax.view(1, 1, 3)
        errs[:, :, k] = torch.sqrt((d * d).sum(-1) + 1e-12)
        mm = m.expand(B, T)
        msk[:, :, k] = mm & valid if valid is not None else mm
    errs = errs * msk.to(dtype)
    return errs, msk


def trajectory_metrics(
    joints_world: Tensor,
    control: TrajectoryControl,
    lengths: Tensor | None = None,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> dict:
    """The mask-control control-fidelity report for one batch.

    * **avg_err** — mean L2 distance (m) between a controlled joint and its
      target, over every controlled (frame, joint) slot.
    * **loc_err@X** — fraction of controlled *slots* missed by more than X m.
    * **traj_err@X** — fraction of *clips* with at least one slot missed by
      more than X m (a clip is a failure if any keyframe is).

    Definitions follow OmniControl Table 1 so the numbers are comparable to the
    published MDM/OmniControl/MaskControl rows.
    """
    errs, msk = control_errors(joints_world, control, lengths)
    n_slots = int(msk.sum().item())
    out: dict = {"n_locations": n_slots, "n_clips": int(joints_world.shape[0])}
    if n_slots == 0:
        out["avg_err"] = 0.0
        for thr in thresholds:
            out[f"loc_err_{thr}"] = 0.0
            out[f"traj_err_{thr}"] = 0.0
        return out
    sel = errs[msk]
    out["avg_err"] = float(sel.mean().item())
    out["max_err"] = float(sel.max().item())
    for thr in thresholds:
        bad = (errs > thr) & msk
        out[f"loc_err_{thr}"] = float(bad.sum().item()) / n_slots
        # A clip with no controlled slot at all cannot fail; exclude it.
        clip_has = msk.flatten(1).any(dim=1)
        clip_bad = bad.flatten(1).any(dim=1) & clip_has
        n_clips = int(clip_has.sum().item())
        out[f"traj_err_{thr}"] = (float(clip_bad.sum().item()) / n_clips) if n_clips else 0.0
    return out


def merge_metrics(chunks: list[dict]) -> dict:
    """Combine per-batch `trajectory_metrics` into one slot-weighted report."""
    chunks = [c for c in chunks if c.get("n_locations")]
    if not chunks:
        return {"n_locations": 0, "n_clips": 0, "avg_err": 0.0}
    n = sum(c["n_locations"] for c in chunks)
    nc = sum(c["n_clips"] for c in chunks)
    out = {"n_locations": n, "n_clips": nc}
    out["avg_err"] = sum(c["avg_err"] * c["n_locations"] for c in chunks) / n
    out["max_err"] = max(c.get("max_err", 0.0) for c in chunks)
    for key in chunks[0]:
        if key.startswith("loc_err_"):
            out[key] = sum(c[key] * c["n_locations"] for c in chunks) / n
        elif key.startswith("traj_err_"):
            out[key] = sum(c[key] * c["n_clips"] for c in chunks) / max(1, nc)
    return out


# ---------------------------------------------------------------------------
# Control-signal sampling (the evaluation protocol)
# ---------------------------------------------------------------------------


def sample_keyframes(length: int, density: int | str, rng) -> Tensor:
    """Which frames carry a control target, per the protocol.

    `density` is a keyframe count (1, 2, 5, 49, …) or "all". Frames are drawn
    uniformly without replacement from [0, length) and returned sorted; a
    density at or above `length` degenerates to every frame. Frame 0 is NOT
    forced — forcing it would make the pelvis task trivially easy on a
    first-frame-canonicalised dataset.
    """
    L = int(length)
    if L <= 0:
        return torch.zeros(0, dtype=torch.long)
    if isinstance(density, str):
        if density.lower() not in ("all", "dense", "-1"):
            raise ValueError(f"density must be an int or 'all', got {density!r}")
        return torch.arange(L)
    k = int(density)
    if k <= 0:
        return torch.zeros(0, dtype=torch.long)
    if k >= L:
        return torch.arange(L)
    idx = rng.choice(L, size=k, replace=False)
    return torch.as_tensor(sorted(int(i) for i in idx), dtype=torch.long)


def sample_control_signal(
    joints_gt: Tensor,
    length: int,
    density: int | str,
    joint_set: str | tuple[int, ...] = "pelvis",
    rng=None,
    mode: str = "hybrid",
    blend: str = "interp",
    blend_frames: int = 10,
    guidance_weight: float = 1.0,
    pick_one: bool | None = None,
    axes: str = "xyz",
    face_path: bool = False,
    face_strength: float = 1.0,
    face_smooth: int = 9,
) -> TrajectoryControl:
    """Build a control signal by reading ground-truth joint positions.

    This is the evaluation protocol: the target for a clip is what the *real*
    motion did, so a perfect method reproduces the reference trajectory and the
    error is directly comparable across methods.

    Args:
        joints_gt: (T, J, 3) world joint positions of the reference clip.
        length: valid frames of the clip (padding beyond is never controlled).
        density: keyframe count or "all".
        joint_set: a `CONTROL_JOINT_SETS` name or an explicit index tuple.
        pick_one: draw ONE joint from the set for this clip (the papers'
            cross-combination protocol). Defaults True for "cross", else False.
        axes: which world axes the targets constrain.
    """
    import numpy as np

    rng = rng if rng is not None else np.random.default_rng(0)
    if isinstance(joint_set, str):
        if joint_set not in CONTROL_JOINT_SETS:
            raise ValueError(
                f"unknown joint set {joint_set!r} — expected one of "
                f"{tuple(CONTROL_JOINT_SETS)} or an explicit index tuple"
            )
        if pick_one is None:
            pick_one = joint_set == "cross"
        joints = CONTROL_JOINT_SETS[joint_set]
    else:
        joints = tuple(int(j) for j in joint_set)
        pick_one = bool(pick_one)
    if pick_one and len(joints) > 1:
        joints = (int(joints[rng.integers(len(joints))]),)

    T = int(joints_gt.shape[0])
    L = max(0, min(int(length), T))
    keys = sample_keyframes(L, density, rng)
    cons = []
    for j in joints:
        mask = torch.zeros(T, dtype=torch.bool)
        if keys.numel():
            mask[keys] = True
        cons.append(TrajectoryConstraint(
            joint=int(j), positions=joints_gt[:, int(j), :].detach().cpu().float(),
            mask=mask, axes=axes,
        ))
    return TrajectoryControl(
        constraints=cons, mode=mode, blend=blend,
        blend_frames=blend_frames, guidance_weight=guidance_weight,
        face_path=face_path, face_strength=face_strength,
        face_smooth=face_smooth,
    )


def stack_controls(controls: list[TrajectoryControl]) -> TrajectoryControl | None:
    """Merge one control signal per clip into a single batched signal.

    The evaluation protocol draws a fresh control signal for every test clip —
    and under the cross-combination protocol, a *different joint* for each. So
    the merge is by joint: every joint that any clip controls becomes one
    constraint whose (B, T) mask is True only in the rows that chose it. Rows
    that chose another joint contribute nothing to it, which is exactly what
    the per-row mask means everywhere else.

    Enforcement settings are taken from the first signal (they are a property of
    the run, not of a clip). Returns None if nothing is controlled.
    """
    controls = [c for c in controls if c]
    if not controls:
        return None
    B = len(controls)
    T = max(c.num_frames for c in controls)
    # Preserve first-seen joint order so the report columns are stable.
    meta: dict[int, tuple[str, float, float]] = {}
    for ctrl in controls:
        for c in ctrl.constraints:
            meta.setdefault(c.joint_idx, (c.axes, c.weight, c.tol))

    cons = []
    for j, (axes, weight, tol) in meta.items():
        pos = torch.zeros(B, T, 3)
        mask = torch.zeros(B, T, dtype=torch.bool)
        for b, ctrl in enumerate(controls):
            for c in ctrl.constraints:
                if c.joint_idx != j:
                    continue
                n = min(T, c.num_frames)
                # A per-clip signal is stored with a leading axis of 1.
                pos[b, :n] = c.positions[0, :n]
                mask[b, :n] = c.mask[0, :n]
        cons.append(TrajectoryConstraint(joint=j, positions=pos, mask=mask,
                                         axes=axes, weight=weight, tol=tol))
    head = controls[0]
    out = TrajectoryControl(constraints=cons, mode=head.mode, blend=head.blend,
                            blend_frames=head.blend_frames,
                            guidance_weight=head.guidance_weight,
                            face_path=head.face_path, face_strength=head.face_strength,
                            face_smooth=head.face_smooth)
    return out or None


def describe_control(control: TrajectoryControl) -> str:
    """One-line human summary, for logs and the job UI."""
    if not control:
        return "no trajectory control"
    parts = [
        f"{JOINT_NAMES[c.joint_idx]}×{c.num_controlled}"
        + (f"[{c.axes}]" if c.axes != "xyz" else "")
        for c in control.constraints
    ]
    blend = control.blend if control.uses_projection else "—"
    return (
        f"{', '.join(parts)} | mode={control.mode} blend={blend}"
        + (f" face_path({control.face_strength:g})" if control.face_path else "")
        + (f" ω_g={control.guidance_weight}" if control.uses_guidance else "")
    )
