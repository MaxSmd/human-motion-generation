"""Scene-aware evaluation of a clip generated against an obstacle course.

The metrics the app already computes (`analysis.joints`) assume the world is a
flat floor at y = 0. On a course that assumption breaks in a way that does not
merely add noise — it *inverts* the reading:

* `foot_skate_mean` gates on `foot_y < 0.05`. A body standing on a 0.18 m slab
  never satisfies it, so `n_planted == 0` and the function returns its 0.0
  fallback: **the best possible score, awarded for never touching the ground.**
* Nothing at all measures whether the body went *through* the staircase rather
  than up it, which is the single thing a course experiment exists to decide.

So this module re-derives contact against the **support surface actually under
each foot** — the floor, or the top of whichever obstacle's footprint the foot
is standing in — and adds the geometric and task metrics a course needs:

  containment   penetration depth into obstacles / out of the room
  contact       support-relative foot height, plant detection, scene foot-skate
  control       deviation from the authored path, per-keyframe error, progress
  task          did it mount the slab / climb the treads / turn the legal way
  quality       jerk, acceleration, root speed, slide-per-step

Every clip is scored in **room coordinates**. Clips sampled with a scene are
already placed there by `visualize.py`; a `free` clip (no scene) is placed here
with the same rigid transform `flow.scene.place_joints` applies, so the three
arms are compared in one frame rather than three.

Input is the `(T, 22, 3)` world-joint `.npy` that `visualize.py` dumps beside
every render.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .contacts import _in_footprint, _obstacle_top_y
from .courses import COURSES, dense_path, path_length

# (ankle, toe) per side — matches analysis.joints.FOOT_IDX / FOOT_CONTACT_IDX.
FOOT_IDX = (7, 10, 8, 11)
LEFT_FOOT, RIGHT_FOOT = (7, 10), (8, 11)
ROOT = 0
FPS = 20

CONTACT_BAND = 0.05      # m above its support at which a foot counts as planted
SLIDE_THR = 0.025        # m/frame of horizontal drift that counts as a skate
PENETRATION_THR = 0.02   # m inside a solid before it counts as a real intrusion
MOUNT_MIN_FRAMES = 5     # 0.25 s of contact with a surface = "stood on it"
# How far a foot may sit BELOW a surface and still be held up by it. This is the
# few-mm interpenetration every contact has — deliberately much tighter than
# CONTACT_BAND, and one-sided. Symmetric slack was a real bug: a swing foot
# passing 3 cm under the nose of a 15 cm tread scored as *standing on* it, which
# handed a clip that walked straight through the flight a climbed tread.
SUPPORT_SLACK = 0.02
# Planted transitions below which a foot-skate number means nothing. Measured on
# the pilot: an arm that floats through the course produced a skate of 0.273 m/s
# from **2** transitions, against 0.688 m/s from 125 for the arm that actually
# walked — and read naively, the levitating arm "wins". A skate averaged over a
# handful of frames is not a smaller skate, it is an absent measurement, so it is
# reported as None. `contact_frame_frac` and `n_planted_transitions` carry the
# denominator so the reason is always visible next to the gap.
MIN_SKATE_SAMPLES = 10


# --------------------------------------------------------------------------- geometry


def _box_sdf(p: np.ndarray, o: dict) -> np.ndarray:
    """Signed distance from points `p` (..., 3) to a box obstacle. <0 inside.

    Mirrors `flow.scene._sdf_box` exactly (including the yaw), so a penetration
    reported here is a penetration the sampler's energy also saw.
    """
    c = np.array([float(o["x"]), float(o["y"]), float(o["z"])])
    half = np.array([float(o["w"]) / 2, float(o["h"]) / 2, float(o["d"]) / 2])
    d = p - c
    yaw = math.radians(float(o.get("rotation", 0.0)))
    if yaw:
        cs, sn = math.cos(-yaw), math.sin(-yaw)
        d = np.stack([cs * d[..., 0] + sn * d[..., 2], d[..., 1],
                      -sn * d[..., 0] + cs * d[..., 2]], axis=-1)
    q = np.abs(d) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return outside + inside


def _sphere_sdf(p, o):
    c = np.array([float(o["x"]), float(o["y"]), float(o["z"])])
    return np.linalg.norm(p - c, axis=-1) - float(o["radius"])


def _cylinder_sdf(p, o):
    cx, cz = float(o["x"]), float(o["z"])
    r, cy, hh = float(o["radius"]), float(o["y"]), float(o["height"]) / 2
    d_xz = np.sqrt((p[..., 0] - cx) ** 2 + (p[..., 2] - cz) ** 2) - r
    d_y = np.abs(p[..., 1] - cy) - hh
    outside = np.sqrt(np.maximum(d_xz, 0.0) ** 2 + np.maximum(d_y, 0.0) ** 2)
    return outside + np.minimum(np.maximum(d_xz, d_y), 0.0)


def obstacle_sdf(p: np.ndarray, o: dict) -> np.ndarray | None:
    kind = o.get("kind")
    if kind == "box":
        return _box_sdf(p, o)
    if kind == "sphere":
        return _sphere_sdf(p, o)
    if kind == "cylinder":
        return _cylinder_sdf(p, o)
    return None


def _room_sdf(p: np.ndarray, room: dict) -> np.ndarray:
    return _box_sdf(p, {"x": 0.0, "y": room["height"] / 2, "z": 0.0,
                        "w": room["width"], "h": room["height"], "d": room["depth"]})


def place_free(joints: np.ndarray, spawn: dict) -> np.ndarray:
    """Rigidly place an UNPLACED clip into room coordinates.

    The numpy mirror of `flow.scene.place_joints`: drop the frame-0 pelvis to the
    xz origin, yaw the clip so its net horizontal displacement points along the
    spawn rotation, translate to the spawn. Only the `free` arm needs this — the
    other two are placed cluster-side — but without it a prompt-only clip sits in
    the model's canonical frame and every task metric would score it against a
    room it was never in.
    """
    j = np.asarray(joints, dtype=np.float64).copy()
    pel = j[:, ROOT, :]
    disp = pel[-1] - pel[0]
    dx, dz = float(disp[0]), float(disp[2])
    desired = math.radians(float(spawn.get("rotation", 0.0)))
    # heading = atan2(x, z): 0 is +Z, +90° is +X — the spawn-arrow convention.
    yaw = desired - math.atan2(dx, dz) if math.hypot(dx, dz) > 0.12 else desired
    j[..., 0] -= pel[0, 0]
    j[..., 2] -= pel[0, 2]
    c, s = math.cos(yaw), math.sin(yaw)
    x, z = j[..., 0].copy(), j[..., 2].copy()
    j[..., 0] = c * x + s * z
    j[..., 2] = -s * x + c * z
    j[..., 0] += float(spawn.get("x", 0.0))
    j[..., 2] += float(spawn.get("z", 0.0))
    return j


# --------------------------------------------------------------------------- support


def support_surface(x: float, z: float, y: float, objects: list[dict]) -> tuple[float, str | None]:
    """Height and id of the surface supporting a point at (x, z) standing near y.

    The highest obstacle top whose top-view footprint contains (x, z) and which
    is not *above* the point — a tread you have not climbed yet cannot be holding
    you up — else the floor. `SUPPORT_SLACK` of one-sided tolerance lets a foot
    resting a few mm into a surface still be supported by it.

    Pass `y=float("inf")` to ask the different question "what is underneath this
    xz at all", which is what the pelvis-height and final-surface reports want.
    """
    best, best_id = 0.0, None
    for o in objects:
        if not _in_footprint(x, z, o):
            continue
        top = _obstacle_top_y(o)
        if top <= y + SUPPORT_SLACK and top > best:
            best, best_id = top, o.get("id")
    return best, best_id


def ground_under(x: float, z: float, objects: list[dict]) -> tuple[float, str | None]:
    """Highest surface under an xz, ignoring how high the body currently is."""
    return support_surface(x, z, float("inf"), objects)


def _on_surface(above: np.ndarray) -> np.ndarray:
    """Is a foot in contact with its support? One-sided: from a hair inside it up
    to `CONTACT_BAND` above it."""
    return (above >= -SUPPORT_SLACK) & (above < CONTACT_BAND)


def support_grid(joints: np.ndarray, objects: list[dict], idx=FOOT_IDX):
    """Per-(frame, foot) support height and supporting-object id. (T,F), (T,F)."""
    T = joints.shape[0]
    F = len(idx)
    heights = np.zeros((T, F))
    ids: list[list[str | None]] = [[None] * F for _ in range(T)]
    for t in range(T):
        for k, j in enumerate(idx):
            x, y, z = joints[t, j, 0], joints[t, j, 1], joints[t, j, 2]
            h, oid = support_surface(float(x), float(z), float(y), objects)
            heights[t, k] = h
            ids[t][k] = oid
    return heights, ids


# --------------------------------------------------------------------------- metric blocks


def containment_metrics(joints: np.ndarray, room: dict, objects: list[dict]) -> dict:
    """How much of the body is inside solid matter, or outside the room.

    `penetration_max` is the headline: a clip that walks *through* a staircase
    instead of up it registers ~the depth of the treads, while a clip that
    climbs registers a few millimetres of foot-on-surface contact slop.
    """
    depth = np.zeros(joints.shape[:2])                     # (T, J)
    for o in objects:
        sdf = obstacle_sdf(joints, o)
        if sdf is not None:
            depth = np.maximum(depth, np.maximum(-sdf, 0.0))
    out = np.maximum(_room_sdf(joints, room), 0.0)         # (T, J)
    per_frame = depth.max(axis=1)
    return {
        "penetration_max": round(float(depth.max()), 4),
        "penetration_mean": round(float(depth.mean()), 5),
        "penetration_frame_frac": round(float((per_frame > PENETRATION_THR).mean()), 4),
        "outside_room_max": round(float(out.max()), 4),
        "outside_room_frac": round(float((out.max(axis=1) > PENETRATION_THR).mean()), 4),
        "penetration_series": [round(float(v), 4) for v in per_frame],
    }


def contact_metrics(joints: np.ndarray, objects: list[dict], fps: int = FPS) -> dict:
    """Support-relative foot contact — the fix for "0.000 skate because it flew".

    A foot is planted when it is within `CONTACT_BAND` of *its own* support
    surface, which on a course is usually not the floor. Everything downstream
    (skate, clearance, plant count) is then measured the way it would be on flat
    ground, and stays comparable to the `free` arm's numbers.

    `n_planted == 0` returns **None**, never 0.0 — a clip that never made contact
    has no foot-skate, and reporting the best possible score for it is how a
    levitating demo passes as clean.
    """
    heights, sup_ids = support_grid(joints, objects)
    foot = joints[:, FOOT_IDX, :]                              # (T,F,3)
    above = foot[:, :, 1] - heights                            # (T,F) height over support
    planted = _on_surface(above)                               # (T,F)

    horiz = np.linalg.norm(np.diff(foot[:, :, [0, 2]], axis=0), axis=-1) * fps   # (T-1,F)
    # A transition counts only if the foot was planted at BOTH ends of it — and
    # on the SAME surface, so stepping from a tread to the next is a step, not a
    # 35 cm slide.
    same = np.array([[sup_ids[t][k] == sup_ids[t + 1][k] for k in range(len(FOOT_IDX))]
                     for t in range(len(sup_ids) - 1)])
    trans = planted[:-1] & planted[1:] & same
    n = int(trans.sum())
    measurable = n >= MIN_SKATE_SAMPLES                        # see MIN_SKATE_SAMPLES

    root_step = np.linalg.norm(np.diff(joints[:, ROOT, [0, 2]], axis=0), axis=-1)
    mean_step = float(root_step.mean()) if root_step.size else 0.0
    slide = float(horiz[trans].mean() / fps) if measurable else None    # m/frame

    # Swing clearance: how high a foot gets above its support while airborne. A
    # gait that has stopped lifting its feet (the classic symptom of a root
    # dragged along a path) collapses this toward zero.
    airborne = ~planted
    clearance = float(above[airborne].mean()) if bool(airborne.any()) else None

    return {
        "n_planted_transitions": n,
        "contact_frame_frac": round(float(planted.any(axis=1).mean()), 4),
        "foot_skate_mean": round(float(horiz[trans].mean()), 5) if measurable else None,
        "foot_skate_ratio": round(float((horiz[trans] / fps > SLIDE_THR).mean()), 4) if measurable else None,
        "slide_per_root_step": round(slide / mean_step, 4) if (slide and mean_step > 1e-6) else None,
        "support_clearance_mean": round(clearance, 4) if clearance is not None else None,
        "min_height_over_support": round(float(above.min()), 4),
        "mean_height_over_support": round(float(above.mean()), 4),
        # Naive flat-floor reading, kept for contrast with the support-aware one.
        "flat_floor_planted": int(((foot[:-1, :, 1] < CONTACT_BAND)
                                   & (foot[1:, :, 1] < CONTACT_BAND)).sum()),
    }


def _polyline_distance(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Shortest distance from each of `pts` (N,D) to the polyline `poly` (K,D)."""
    a, b = poly[:-1], poly[1:]                                  # (K-1,D)
    ab = b - a
    denom = np.maximum((ab * ab).sum(-1), 1e-12)
    ap = pts[:, None, :] - a[None, :, :]                        # (N,K-1,D)
    t = np.clip((ap * ab[None]).sum(-1) / denom[None], 0.0, 1.0)
    proj = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(pts[:, None, :] - proj, axis=-1).min(axis=1)


def control_metrics(joints: np.ndarray, course: dict) -> dict:
    """Did the body go where the course asked — as a **path**, and per keyframe.

    Two different questions, and on this study the first is the right one. The
    trajectory arm runs with `retime=True`, i.e. path control: the author picks
    where, the model picks when. Per-keyframe error then charges the clip for a
    standing lead-in it was *designed* to keep, so `path_dev_*` (distance to the
    authored curve, timing-free) is the headline and `keyframe_err_*` — the
    mask-control number — rides along beside it.
    """
    ref = dense_path(course)                                    # (T,3)
    axes = course["axes"]
    cols = [i for i, a in enumerate("xyz") if a in axes]
    root = joints[:, ROOT, :]
    T = min(root.shape[0], ref.shape[0])

    dev = _polyline_distance(root[:T][:, cols], ref[:, cols])
    kf_err = np.linalg.norm(root[:T][:, cols] - ref[:T][:, cols], axis=-1)

    # Progress: how far along the path the body actually got, as a fraction.
    seg = np.linalg.norm(np.diff(ref, axis=0), axis=-1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    end_i = int(np.argmin(np.linalg.norm(ref[:, cols] - root[T - 1, cols], axis=-1)))
    total = float(cum[-1]) or 1.0

    return {
        "path_dev_mean": round(float(dev.mean()), 4),
        "path_dev_max": round(float(dev.max()), 4),
        "path_dev_final": round(float(dev[-1]), 4),
        "keyframe_err_mean": round(float(kf_err.mean()), 4),
        "keyframe_err_max": round(float(kf_err.max()), 4),
        "progress_frac": round(float(cum[end_i] / total), 4),
        "path_length_m": round(float(total), 3),
        "travel_m": round(float(np.linalg.norm(np.diff(root[:, [0, 2]], axis=0), axis=-1).sum()), 3),
        "dev_series": [round(float(v), 4) for v in dev],
    }


def quality_metrics(joints: np.ndarray, fps: int = FPS) -> dict:
    """Smoothness and gait, on the same definitions `analysis.joints` uses so the
    numbers stay comparable to everything else in the app."""
    jerk = np.linalg.norm(np.diff(joints, n=3, axis=0), axis=-1).mean(axis=-1)
    accel = np.linalg.norm(np.diff(joints, n=2, axis=0), axis=-1).mean(axis=-1)
    speed = np.linalg.norm(np.diff(joints[:, ROOT, [0, 2]], axis=0), axis=-1) * fps
    return {
        "jerk_mean": round(float(jerk.mean()), 5) if jerk.size else None,
        "accel_mean": round(float(accel.mean()), 5) if accel.size else None,
        "root_speed_mean": round(float(speed.mean()), 4),
        "root_speed_max": round(float(speed.max()), 4),
        "pelvis_y_start": round(float(joints[0, ROOT, 1]), 4),
        "pelvis_y_end": round(float(joints[-1, ROOT, 1]), 4),
        "pelvis_rise": round(float(joints[-1, ROOT, 1] - joints[0, ROOT, 1]), 4),
        "speed_series": [round(float(v), 4) for v in speed],
    }


# --------------------------------------------------------------------------- task success


def _mount_success(joints, objects, spec) -> dict:
    """Slab: did the body actually stand on the platform?

    "Both feet on simultaneously" is the wrong test — a walking body always has
    a swing leg, so a clip that strides cleanly across the slab would score
    zero. The test is instead that **each leg** touched the surface at some
    point and that contact lasted: you cannot mount a platform having only ever
    put one foot on it, and a single frame of grazing is not standing.
    """
    heights, ids = support_grid(joints, objects)
    above = joints[:, FOOT_IDX, 1] - heights
    on = _on_surface(above) & np.array(
        [[ids[t][k] in spec["object_ids"] for k in range(len(FOOT_IDX))]
         for t in range(joints.shape[0])])
    # Indices 0,1 are the left (ankle, toe); 2,3 the right.
    left, right = on[:, 0] | on[:, 1], on[:, 2] | on[:, 3]
    frames_on = int((left | right).sum())
    rx, rz = float(joints[-1, ROOT, 0]), float(joints[-1, ROOT, 2])
    return {
        "kind": "mount",
        "frames_supported_on": frames_on,
        "frames_left_foot_on": int(left.sum()),
        "frames_right_foot_on": int(right.sum()),
        "both_legs_touched": bool(left.any() and right.any()),
        "mounted": bool(left.any() and right.any() and frames_on >= MOUNT_MIN_FRAMES),
        "final_ground_y": round(float(ground_under(rx, rz, objects)[0]), 3),
        "target_surface_y": round(float(spec["surface_y"]), 3),
    }


def _climb_success(joints, objects, spec) -> dict:
    """Stairs: which treads got a foot, **in order**.

    Order matters. A clip that clips through the flight and lands a foot on the
    landing has not climbed three steps, and an unordered count would say it
    did. `treads_climbed` therefore stops at the first tread that was skipped.
    """
    heights, ids = support_grid(joints, objects)
    supported = _on_surface(joints[:, FOOT_IDX, 1] - heights)
    first: dict[str, int | None] = {}
    for oid in spec["object_ids"]:
        hit = [t for t in range(joints.shape[0])
               if any(supported[t, k] and ids[t][k] == oid for k in range(len(FOOT_IDX)))]
        first[oid] = int(min(hit)) if hit else None

    climbed, last_t = 0, -1
    for oid in spec["object_ids"]:
        t = first[oid]
        if t is None or t < last_t:
            break
        climbed += 1
        last_t = t
    return {
        "kind": "climb",
        "treads_climbed": climbed,
        "treads_total": len(spec["object_ids"]),
        "first_contact_frame": first,
        "top_reached": bool(first[spec["object_ids"][-1]] is not None),
        "final_ground_y": round(float(ground_under(
            float(joints[-1, ROOT, 0]), float(joints[-1, ROOT, 2]), objects)[0]), 3),
        "target_surface_y": round(float(spec["surface_y"]), 3),
    }


def _turn_success(joints, objects, spec) -> dict:
    """Corner: how far it turned, which way, and whether it stayed in the corridor.

    Heading is taken from the *travel* direction averaged over the first and last
    quarter of the clip (a single-frame tangent is noise). Positive = left, i.e.
    toward −X from +Z, which is the only direction the walls leave open.
    """
    root = joints[:, ROOT, :][:, [0, 2]]
    T = root.shape[0]
    q = max(2, T // 4)
    d0 = root[q] - root[0]
    d1 = root[-1] - root[-q]
    h0, h1 = math.atan2(d0[0], d0[1]), math.atan2(d1[0], d1[1])
    turn = math.degrees((h1 - h0 + math.pi) % (2 * math.pi) - math.pi)

    inside = np.zeros(T, dtype=bool)
    for x0, x1, z0, z1 in spec["legal"]:
        inside |= (root[:, 0] >= x0) & (root[:, 0] <= x1) & (root[:, 1] >= z0) & (root[:, 1] <= z1)
    ex = spec["exit"]
    reached = bool(((root[:, 0] <= ex["x_max"]) &
                    (root[:, 1] >= ex["z_min"]) & (root[:, 1] <= ex["z_max"])).any())
    return {
        "kind": "turn",
        # Which way the body was pointing as it entered the corridor, relative to
        # +Z (the spawn arrow). This is NOT redundant with the turn: exact spawn
        # placement aligns the clip's *net* displacement to the spawn rotation,
        # so an L-shaped walk gets rotated until its start→end diagonal points
        # +Z — which aims the approach leg ~45° into the right-hand wall. Without
        # this number a poor corridor score on the `room` arm reads as "the room
        # energy failed to turn" when the cause is where placement pointed it.
        "entry_heading_deg": round(math.degrees(h0), 1),
        # Report left as positive: heading measured atan2(x, z) decreases going left.
        "turn_left_deg": round(-turn, 1),
        "turned_left": bool(-turn > 30.0),
        "turned_right": bool(-turn < -30.0),
        "inside_corridor_frac": round(float(inside.mean()), 4),
        "exit_reached": reached,
    }


_SUCCESS = {"mount": _mount_success, "climb": _climb_success, "turn": _turn_success}


# --------------------------------------------------------------------------- entry points


def evaluate(joints: np.ndarray, course: dict, arm: str, fps: int = FPS) -> dict:
    """Full scene-aware report for one clip of one (course, arm) cell."""
    j = np.asarray(joints, dtype=np.float64)
    if j.ndim != 3 or j.shape[1] < 22 or j.shape[2] != 3:
        raise ValueError(f"expected (T,22,3) joints, got {j.shape}")
    # The `free` arm was sampled with no scene, so `visualize.py` never placed
    # it; place it here or it would be scored against a room it never saw.
    if arm == "free":
        j = place_free(j, course["spawn"])

    room = {"width": 6.0, "depth": 6.0, "height": 3.0}
    objects = course["objects"]
    out = {
        "course": course["key"], "arm": arm, "frames": int(j.shape[0]),
        **containment_metrics(j, room, objects),
        **contact_metrics(j, objects, fps),
        **control_metrics(j, course),
        **quality_metrics(j, fps),
    }
    out["success"] = _SUCCESS[course["success"]["kind"]](j, objects, course["success"])
    out["root_xz"] = [[round(float(x), 3), round(float(z), 3)]
                      for x, z in j[:, ROOT, :][:, [0, 2]]]
    out["root_y"] = [round(float(y), 3) for y in j[:, ROOT, 1]]
    return out


def evaluate_file(path: str | Path, course_key: str, arm: str, fps: int = FPS) -> dict:
    course = COURSES[course_key]
    return evaluate(np.load(Path(path)), course, arm, fps)


# Series are for plots, not for tables — strip them before aggregating.
_SERIES = ("penetration_series", "dev_series", "speed_series", "root_xz", "root_y")


def _mean_sd(vals: list[float]) -> dict | None:
    v = [x for x in vals if x is not None]
    if not v:
        return None
    a = np.asarray(v, dtype=np.float64)
    return {"mean": round(float(a.mean()), 5),
            "sd": round(float(a.std(ddof=1)), 5) if a.size > 1 else 0.0,
            "n": int(a.size)}


def aggregate(rows: list[dict]) -> list[dict]:
    """Group per-clip reports into one row per (course, arm), mean ± sd.

    Success is aggregated as a *rate* over clips (mounted / top_reached /
    turned_left) plus the mean of its continuous parts, because "3 of 6 clips
    climbed" is the honest summary of a task metric — an average of a boolean
    dressed up as a score is not.
    """
    cells: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        cells.setdefault((r["course"], r["arm"]), []).append(r)

    out = []
    for (course, arm), group in cells.items():
        numeric: dict[str, list] = {}
        for r in group:
            for k, v in r.items():
                if k in _SERIES or k == "success" or not isinstance(v, (int, float)) or isinstance(v, bool):
                    continue
                numeric.setdefault(k, []).append(v)
        stats = {k: s for k, s in ((k, _mean_sd(v)) for k, v in numeric.items()) if s}

        succ = [r["success"] for r in group]
        kind = succ[0]["kind"]
        srep: dict = {"kind": kind, "n": len(group)}
        if kind == "mount":
            srep["mounted_rate"] = round(sum(s["mounted"] for s in succ) / len(succ), 3)
            srep["frames_supported_on"] = _mean_sd([s["frames_supported_on"] for s in succ])
            srep["final_ground_y"] = _mean_sd([s["final_ground_y"] for s in succ])
        elif kind == "climb":
            srep["treads_climbed"] = _mean_sd([s["treads_climbed"] for s in succ])
            srep["top_reached_rate"] = round(sum(s["top_reached"] for s in succ) / len(succ), 3)
            srep["final_ground_y"] = _mean_sd([s["final_ground_y"] for s in succ])
        else:
            srep["turned_left_rate"] = round(sum(s["turned_left"] for s in succ) / len(succ), 3)
            srep["turn_left_deg"] = _mean_sd([s["turn_left_deg"] for s in succ])
            srep["entry_heading_deg"] = _mean_sd([s["entry_heading_deg"] for s in succ])
            srep["inside_corridor_frac"] = _mean_sd([s["inside_corridor_frac"] for s in succ])
            srep["exit_reached_rate"] = round(sum(s["exit_reached"] for s in succ) / len(succ), 3)
        out.append({"course": course, "arm": arm, "n": len(group),
                    "metrics": stats, "success": srep})

    order = {"free": 0, "room": 1, "room+traj": 2}
    out.sort(key=lambda r: (list(COURSES).index(r["course"]), order.get(r["arm"], 9)))
    return out
