"""Euclidean **obstacle courses** — one canonical spec for a scene experiment.

A course is a room + obstacles + a spawn + an authored pelvis path + the prompts
that describe the task, bundled so that *the same object* drives

  1. what the sampler is given (`scene_payload` / `trajectory_payload`), and
  2. what the metrics score against (`analysis.scene_eval`).

Keeping those two in one place is not tidiness — it is the only way the reported
"did it climb the stairs" can be trusted. If the frontend authored the geometry
and the analyser re-declared it, a drifting copy would silently grade clips
against a staircase that was never sampled.

Geometry conventions (HumanML3D FK frame, shared with `flow.scene`):
  X right, **Y up**, Z forward; floor at y = 0; the room box is centred on the
  xz origin and spans y ∈ [0, height]. A box obstacle's `y` is its **centre**,
  so a slab of height h resting on the floor has y = h/2.

Two measured constants set the physical budget of every course:

* `PELVIS_BASE = 0.95 m` — the model's natural pelvis height (median over 40
  generated clips), so an authored path that should mean "standing on a 0.18 m
  slab" is written as y = 0.95 + 0.18 rather than guessed.
* `WALK_SPEED ≈ 0.5 m/s` — its natural gait. Every course path is kept at
  **≈ 0.33 m/s** requested speed. This is deliberate: foot-skate under
  trajectory control is monotone in requested speed, so a path the gait cannot
  walk manufactures the very artefact the experiment is trying to measure.
  `check_speed()` re-derives it from the spec so the budget cannot rot.

**Padding is 0 on every course, on purpose.** `scene_energy` applies the
standoff margin to the *floor* as well as to walls, so a positive padding
penalises a foot at y = 0 and levitates the body ~8-17 cm — which in turn makes
the foot-skate gate never fire and the skate metric read a flattering 0.000. A
course whose whole point is *standing on* a slab cannot use a term that pushes
the body off surfaces. Obstacles stay solid regardless: at padding 0 the
obstacle energy is still ReLU(-sdf)², i.e. a penalty for being *inside*.
"""

from __future__ import annotations

import math

import numpy as np

# Measured on generated clips (see module docstring).
PELVIS_BASE = 0.95          # m — pelvis height of a normally standing body
FPS = 20
TARGET_SPEED = 0.35         # m/s — requested path speed we design toward

# Every course shares a room and a keyframe density so the arms differ only in
# the thing under test.
ROOM = {"width": 6.0, "depth": 6.0, "height": 3.0}
KEYFRAME_EVERY = 4          # control point every 4 frames (35 points @ 140)


def _box(oid: str, x: float, y: float, z: float, w: float, h: float, d: float,
         rotation: float = 0.0) -> dict:
    """A box obstacle in the wire format `flow.scene._obstacle_sdf` expects."""
    return {"id": oid, "kind": "box", "x": x, "y": y, "z": z,
            "w": w, "h": h, "d": d, "rotation": rotation}


def _slab(oid: str, z0: float, z1: float, top: float, width: float = 1.4) -> dict:
    """A box resting ON THE FLOOR spanning z ∈ [z0, z1] with its top at `top`.

    Stairs are built from these rather than from floating treads: a solid block
    from the floor up means the SDF says "solid" everywhere under the tread, so
    a foot cannot be pushed into the space beneath a step.
    """
    return _box(oid, x=0.0, y=top / 2, z=(z0 + z1) / 2,
                w=width, h=top, d=(z1 - z0))


# --------------------------------------------------------------------------- courses


def _course_slab() -> dict:
    """**Step up onto a low platform.** The minimal height task: one 0.18 m rise.

    0.18 m is a real step riser. It is small enough that the model's own gait
    could plausibly clear it, which is what makes it the control condition for
    the staircase: if the body fails *here*, the failure is not "the stairs were
    too steep".
    """
    top = 0.18
    return {
        "key": "slab",
        "label": "A · Slab",
        "title": "Step up onto a low platform",
        "blurb": "One 0.18 m riser, 0.9 m deep. The minimal height task.",
        "num_frames": 140,
        "spawn": {"x": 0.0, "z": -1.6, "rotation": 0.0},
        "objects": [_slab("slab", z0=0.0, z1=0.9, top=top, width=1.3)],
        # Pelvis path: approach on the floor, rise over the front edge, stand on top.
        "path": [
            (0.0, PELVIS_BASE, -1.60),
            (0.0, PELVIS_BASE, -0.15),
            (0.0, PELVIS_BASE + top, 0.25),
            (0.0, PELVIS_BASE + top, 0.70),
        ],
        "axes": "xyz",          # height IS the task — do not leave y to the model
        "face_path": False,     # the path is straight; the model's own heading is +Z
        "prompts": [
            "a person walks forward and steps up onto a low platform",
            "a person walks forward and steps onto a raised block",
            "a person walks straight ahead and then steps up onto a step",
        ],
        "success": {"kind": "mount", "object_ids": ["slab"], "surface_y": top},
    }


def _course_stairs() -> dict:
    """**Climb a flight of stairs.** Three 0.15 m risers with a 0.35 m run, then
    a landing. 0.15 / 0.35 is an ordinary domestic staircase.

    Each tread is a solid block from the floor (see `_slab`), and the landing
    exists so the top of the flight is somewhere to *stand* rather than an edge
    to stop on.
    """
    rise, run = 0.15, 0.35
    objects = [
        _slab(f"tread{k + 1}", z0=k * run, z1=(k + 1) * run, top=(k + 1) * rise)
        for k in range(3)
    ]
    objects.append(_slab("landing", z0=3 * run, z1=3 * run + 0.8, top=3 * rise))
    return {
        "key": "stairs",
        "label": "B · Stairs",
        "title": "Climb a flight of stairs",
        "blurb": "Three 0.15 m risers on a 0.35 m run, then a landing.",
        "num_frames": 140,
        "spawn": {"x": 0.0, "z": -1.0, "rotation": 0.0},
        "objects": objects,
        "path": [
            (0.0, PELVIS_BASE, -1.00),
            (0.0, PELVIS_BASE, -0.10),
            (0.0, PELVIS_BASE + 1 * rise, 0.175),
            (0.0, PELVIS_BASE + 2 * rise, 0.525),
            (0.0, PELVIS_BASE + 3 * rise, 0.875),
            (0.0, PELVIS_BASE + 3 * rise, 1.25),
        ],
        "axes": "xyz",
        "face_path": False,
        "prompts": [
            "a person walks forward and climbs up a flight of stairs",
            "a person walks forward and walks up the steps",
            "a person walks ahead and climbs the staircase",
        ],
        "success": {
            "kind": "climb",
            # Ordered: reaching tread k only counts once tread k-1 was reached.
            "object_ids": ["tread1", "tread2", "tread3", "landing"],
            "surface_y": 3 * rise,
        },
    }


def _course_corner() -> dict:
    """**Turn the only way the walls allow.** An L-corridor, 1.4 m wide: a wall
    straight ahead and a wall on the right, so the sole way on is left.

    This is the course where the *euclidean* constraint has something to say on
    its own. Slab and stairs need the body to go somewhere the repulsive room
    energy cannot invent (nothing rewards standing on a surface); here the
    feasible set really is "turn left", and the question is whether repulsion
    alone finds it or whether the trajectory channel has to say so.
    """
    hw = 0.7            # corridor half-width
    t = 0.15            # wall thickness
    wall_h = 1.8
    o = hw + t / 2      # wall centreline offset
    back = -1.75        # corridor mouth
    reach = -1.75       # how far the left leg runs
    objects = [
        # Right-hand wall of the approach, continuing past the junction: this is
        # the one that forbids a right turn.
        _box("wall_right", x=o, y=wall_h / 2, z=(back + (hw + t)) / 2,
             w=t, h=wall_h, d=(hw + t) - back),
        # Wall straight ahead at the junction, doubling as the far wall of the
        # left leg.
        _box("wall_front", x=(reach + (hw + t)) / 2, y=wall_h / 2, z=o,
             w=(hw + t) - reach, h=wall_h, d=t),
        # Left-hand wall of the approach — stops short of the junction, and that
        # gap IS the only exit.
        _box("wall_left", x=-o, y=wall_h / 2, z=(back - hw) / 2,
             w=t, h=wall_h, d=(-hw) - back),
        # Near wall of the left leg.
        _box("wall_exit", x=(reach - hw) / 2, y=wall_h / 2, z=-o,
             w=(-hw) - reach, h=wall_h, d=t),
    ]
    return {
        "key": "corner",
        "label": "C · Corner",
        "title": "Round a corner the walls allow only to the left",
        "blurb": "A 1.4 m L-corridor: wall ahead, wall right, the only way is left.",
        "num_frames": 140,
        "spawn": {"x": 0.0, "z": -1.3, "rotation": 0.0},
        "objects": objects,
        # Slightly rounded corner: a chord-linear right angle asks for 90° of
        # heading change in one frame, which no body can do.
        "path": [
            (0.0, PELVIS_BASE, -1.30),
            (0.0, PELVIS_BASE, -0.30),
            (-0.30, PELVIS_BASE, 0.00),
            (-1.10, PELVIS_BASE, 0.00),
        ],
        "axes": "xz",           # flat course — leave height to the model
        "face_path": True,      # position control alone does not turn the body
        "prompts": [
            "a person walks forward and then turns left",
            "a person walks forward and turns to the left at the corner",
            "a person walks straight ahead then makes a left turn",
        ],
        "success": {
            "kind": "turn",
            "sign": "left",
            # Union of the two corridor legs, as xz rectangles (x0, x1, z0, z1).
            "legal": [(-hw, hw, back, hw), (reach, hw, -hw, hw)],
            # Made the corner iff the root reaches here.
            "exit": {"x_max": -0.85, "z_min": -hw, "z_max": hw},
        },
    }


COURSES: dict[str, dict] = {
    c["key"]: c for c in (_course_slab(), _course_stairs(), _course_corner())
}

# The three arms every course is run under. `scene` toggles the euclidean room
# energy, `trajectory` toggles the spatial control channel — so the pair
# (room, room+traj) isolates the trajectory constraint, and `free` is the
# unconstrained floor without which the physical cost of either gets
# misattributed to the constraint instead of to the model's own roughness.
ARMS: dict[str, dict] = {
    "free": {
        "label": "free",
        "title": "prompt only",
        "blurb": "No room, no targets — the model's own answer to the prompt. "
                 "The reference floor for every physical metric.",
        "scene": False, "trajectory": False,
    },
    "room": {
        "label": "room",
        "title": "euclidean room",
        "blurb": "Room containment + obstacle repulsion as sampling guidance. "
                 "Spawn placement is exact.",
        "scene": True, "trajectory": False,
    },
    "room+traj": {
        "label": "room+traj",
        "title": "euclidean room + trajectory control",
        "blurb": "The same room energy, plus the authored pelvis path enforced "
                 "by exact root absorption.",
        "scene": True, "trajectory": True,
    },
}


# --------------------------------------------------------------------------- path maths


def resample_path(points, n: int) -> np.ndarray:
    """Arc-length-resample a polyline to exactly `n` points. (K,3) -> (n,3).

    A numpy mirror of `flow.trajectory.resample_path`; the sampler re-derives
    nothing from this, since we send explicit frames+points on the wire.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = max(1, int(n))
    if p.shape[0] == 1:
        return np.repeat(p, n, axis=0)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=-1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < 1e-9:
        return np.repeat(p[:1], n, axis=0)
    want = np.linspace(0.0, cum[-1], n)
    idx = np.clip(np.searchsorted(cum, want, side="right") - 1, 0, p.shape[0] - 2)
    t = ((want - cum[idx]) / np.maximum(cum[idx + 1] - cum[idx], 1e-12))[:, None]
    return p[idx] * (1 - t) + p[idx + 1] * t


def path_length(course: dict) -> float:
    p = np.asarray(course["path"], dtype=np.float64)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=-1).sum())


def check_speed(course: dict, fps: int = FPS) -> dict:
    """Requested speed of a course's path, against the model's natural gait.

    Reported (and asserted in the tests) because the single biggest cause of
    "the constrained clip slides" is asking for a path the gait cannot walk —
    a diagnosis that must be ruled out *before* any of it is blamed on the
    solver.
    """
    L = path_length(course)
    dur = course["num_frames"] / float(fps)
    return {"length_m": round(L, 3), "duration_s": round(dur, 2),
            "speed_mps": round(L / dur, 3), "target_mps": TARGET_SPEED,
            "ok": bool(L / dur <= 0.45)}


# --------------------------------------------------------------------------- payloads


def scene_payload(course: dict, foot_skate_weight: float = 0.0) -> dict:
    """The `scene` dict sent to `visualize.py` (→ `flow.scene.parse_scene`)."""
    return {
        "room": dict(ROOM),
        "objects": [dict(o) for o in course["objects"]],
        "spawn": dict(course["spawn"]),
        "padding": 0.0,                     # see module docstring
        "contacts": [],
        "foot_skate_weight": float(foot_skate_weight),
        "fps": FPS,
    }


def trajectory_payload(course: dict, every: int = KEYFRAME_EVERY) -> dict:
    """The `trajectory` dict sent to `visualize.py` (→ `parse_trajectory`).

    Explicit `frames` + `points` rather than the `path` wire format, so the
    array that is *enforced* is byte-identical to the array the metrics score
    against — the same reason `TrajectoryTab` sends them explicitly.

    `retime=True` makes this **path** control, not trajectory control: the user
    picks where, the model picks when. The model stands still for ~1.9 s before
    walking, and a constant-speed per-frame target drags that standing body.
    The consequence for the report is that per-keyframe error is the wrong
    primary metric here — `scene_eval` headlines *path deviation* instead.
    """
    T = int(course["num_frames"])
    dense = resample_path(course["path"], T)
    frames = list(range(0, T, max(1, int(every))))
    if frames[-1] != T - 1:
        frames.append(T - 1)
    return {
        "mode": "project",          # exact root absorption; guidance diverges here
        "blend": "contact",         # spend the correction while the feet are airborne
        "blend_frames": 10,
        "guidance_weight": 0.0,
        "retime": True,
        "face_path": bool(course["face_path"]),
        "face_strength": 1.0,
        "face_smooth": 9,
        "foot_lock": True,
        "constraints": [{
            "joint": "pelvis",
            "axes": course["axes"],
            "frames": frames,
            "points": [[round(float(v), 5) for v in dense[f]] for f in frames],
        }],
    }


def dense_path(course: dict) -> np.ndarray:
    """The authored path at one point per frame — the metric's reference curve."""
    return resample_path(course["path"], int(course["num_frames"]))


def job_params(course: dict, arm: str, seed: int, checkpoint: str,
               model_preset: str, train_preset: str,
               num_steps: int = 800, guidance: float = 6.5,
               room_guidance: float = 0.75) -> dict:
    """One `/cluster/jobs/viz` body: a whole (course, arm, seed) cell.

    All three of a course's prompts ride in ONE job. That is not just fewer
    submissions: `visualize.py` groups renders by
    (constraints, ranges, scene, room_guidance, num_frames, seed) and samples a
    group in a single batched ODE, seeding once per chunk. Same prompt count,
    same seed and same frame count in every arm therefore means position i of
    each arm's batch draws the *same* noise — the arms are a matched pair rather
    than independent draws.
    """
    spec = ARMS[arm]
    body = {
        "mode": "prompt",
        "checkpoint": checkpoint,
        "model_preset": model_preset,
        "train_preset": train_preset,
        "prompts": " | ".join(course["prompts"]),
        "num_frames": int(course["num_frames"]),
        "num_steps": int(num_steps),
        "guidance": float(guidance),
        "seed": int(seed),
        "use_ema": True,
        "ablation": {"study": "course", "course": course["key"], "arm": arm, "seed": int(seed)},
    }
    if spec["scene"]:
        body["scene"] = scene_payload(course)
        body["room_guidance"] = float(room_guidance)
    if spec["trajectory"]:
        body["trajectory"] = trajectory_payload(course)
    return body


def describe(course: dict) -> str:
    sp = check_speed(course)
    return (f"{course['label']}: {course['title']} — {len(course['objects'])} object(s), "
            f"path {sp['length_m']} m over {sp['duration_s']} s = {sp['speed_mps']} m/s")


def catalogue() -> list[dict]:
    """JSON-safe course list for the frontend (geometry + payloads + budget)."""
    out = []
    for c in COURSES.values():
        out.append({
            "key": c["key"], "label": c["label"], "title": c["title"],
            "blurb": c["blurb"], "num_frames": c["num_frames"],
            "room": dict(ROOM), "objects": [dict(o) for o in c["objects"]],
            "spawn": dict(c["spawn"]),
            "path": [list(map(float, p)) for p in c["path"]],
            "axes": c["axes"], "face_path": c["face_path"],
            "prompts": list(c["prompts"]),
            "success": c["success"],
            "speed": check_speed(c),
        })
    return out
