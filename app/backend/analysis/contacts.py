"""Auto-extract contact constraints from a *draft* clip.

The room energy is pure repulsion, so an unconstrained (or lightly guided) draft
keeps a comfortable standoff and never actually rests a limb on anything. The
only attractive term is a `ContactConstraint`, but it needs a
`(joint, frame_window, target)` tuple you can't author without seeing the
motion.

This module closes that loop: draft freely (Pass A, `room_guidance≈0`), read the
foot plants straight off the dumped `(T,22,3)` room-coordinate clip, and turn
each plant into a contact — `floor` by default, or `obstacle_top` when the plant
lands inside an obstacle's top-view footprint. The result drops straight into the
scene for a constrained resample (Pass B).

A staircase (one box per tread at increasing y) falls out for free: a flat
forward walk's successive plants land in successive tread footprints, so each
plant is lifted onto its own step.

Detection is *speed-based* (a foot is in stance when it stops moving), not purely
height-based, so it also works on clips whose feet already rest on a tread.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# HumanML3D foot joints, split per leg: (ankle, toe). Matches joints.FOOT_IDX /
# shared.geometry.FOOT_CONTACT_IDX. We DETECT on the toe (the contact point) and
# EMIT the contact on the ankle (the conventional IK contact joint) — by NAME so
# it slots straight into the frontend's joint picker.
_LEGS = {
    "left": {"ankle": 7, "toe": 10, "name": "L_Ankle"},
    "right": {"ankle": 8, "toe": 11, "name": "R_Ankle"},
}

# Stance = the foot has (nearly) stopped moving horizontally. 0.25 m/s is a
# generous walk threshold; a planted foot sits well below it, a swing foot well
# above. Vertical motion is ignored so a foot settling onto a tread still reads
# as stance.
STANCE_SPEED = 0.25          # m/s horizontal, below which a foot is "planted"
MIN_STANCE_FRAMES = 3        # ignore <0.15s (@20fps) flickers — not a real plant
MERGE_GAP_FRAMES = 5         # bridge <=0.25s speed spikes inside one plant (a real
                             # swing is longer) so a plant isn't split into shards
FOOTPRINT_MARGIN = 0.10      # m of slop when testing xz-in-obstacle footprint
DEFAULT_TOL = 0.06           # contact tolerance (m) written on emitted contacts


# --------------------------------------------------------------------------- geometry


def _in_footprint(x: float, z: float, o: dict) -> bool:
    """Is the xz point inside obstacle `o`'s top-view footprint (+margin)?"""
    m = FOOTPRINT_MARGIN
    kind = o.get("kind")
    if kind == "box":
        cx, cz = float(o["x"]), float(o["z"])
        hw, hd = float(o["w"]) / 2 + m, float(o["d"]) / 2 + m
        yaw = math.radians(float(o.get("rotation", 0.0)))
        c, s = math.cos(-yaw), math.sin(-yaw)
        dx, dz = x - cx, z - cz
        lx, lz = c * dx + s * dz, -s * dx + c * dz     # rotate into box-local
        return abs(lx) <= hw and abs(lz) <= hd
    if kind in ("cylinder", "sphere"):
        cx, cz = float(o["x"]), float(o["z"])
        r = float(o["radius"]) + m
        return (x - cx) ** 2 + (z - cz) ** 2 <= r * r
    return False


def _obstacle_top_y(o: dict) -> float:
    """World-y of the obstacle's top face / apex (for reporting the lift height)."""
    kind = o.get("kind")
    if kind == "box":
        return float(o["y"]) + float(o["h"]) / 2
    if kind == "cylinder":
        return float(o["y"]) + float(o["height"]) / 2
    if kind == "sphere":
        return float(o["y"]) + float(o["radius"])
    return 0.0


# --------------------------------------------------------------------------- detection


def _close_gaps(mask: np.ndarray, gap: int) -> np.ndarray:
    """Fill False-runs of length <= `gap` that sit between two True runs.

    Morphological closing on the boolean stance mask: a brief speed spike inside a
    single plant flips a few frames to swing; bridging them keeps the plant whole.
    Leading/trailing False runs (before the first / after the last plant) are left
    untouched — those are real swing, not gaps.
    """
    if gap <= 0:
        return mask
    out = mask.copy()
    n = len(out)
    t = 0
    while t < n:
        if out[t]:
            t += 1
            continue
        start = t
        while t < n and not out[t]:
            t += 1
        # [start, t) is a False run; fill it only if bounded by True on both sides.
        if start > 0 and t < n and (t - start) <= gap:
            out[start:t] = True
    return out


def _stance_segments(foot_xyz: np.ndarray, fps: float) -> list[tuple[int, int]]:
    """Runs of consecutive stance frames for one leg's foot track (T,3).

    Stance = horizontal speed below STANCE_SPEED. Returns half-open [start, end)
    windows of length >= MIN_STANCE_FRAMES.
    """
    horiz = np.linalg.norm(np.diff(foot_xyz[:, [0, 2]], axis=0), axis=-1) * fps  # (T-1,)
    stance = _close_gaps(horiz < STANCE_SPEED, MERGE_GAP_FRAMES)                 # (T-1,)
    segs: list[tuple[int, int]] = []
    t = 0
    n = len(stance)
    while t < n:
        if not stance[t]:
            t += 1
            continue
        start = t
        while t < n and stance[t]:
            t += 1
        # stance[t] covers the step t->t+1, so the plant spans frames [start, t].
        end = t + 1
        if end - start >= MIN_STANCE_FRAMES:
            segs.append((start, end))
    return segs


def detect_plants(joints: np.ndarray, fps: float = 20.0) -> list[dict]:
    """Every foot-plant segment in a placed (T,22,3) clip, per leg.

    Each plant: {leg, joint (ankle idx), frame_start, frame_end, x, z, y} where
    (x,z) is the median resting position and y the lowest foot height in the run.
    """
    plants: list[dict] = []
    for leg, idx in _LEGS.items():
        # Track the toe (the actual contact point); fall back to min height of the
        # ankle/toe pair for the resting y.
        toe = joints[:, idx["toe"], :]
        pair = joints[:, [idx["ankle"], idx["toe"]], :]        # (T,2,3)
        for start, end in _stance_segments(toe, fps):
            seg = toe[start:end]
            x = float(np.median(seg[:, 0]))
            z = float(np.median(seg[:, 2]))
            y = float(pair[start:end, :, 1].min())
            plants.append({
                "leg": leg,
                "joint": idx["name"],
                "frame_start": int(start),
                "frame_end": int(end),
                "x": round(x, 4),
                "z": round(z, 4),
                "y": round(y, 4),
            })
    plants.sort(key=lambda p: p["frame_start"])
    return plants


# --------------------------------------------------------------------------- assignment


def assign_targets(plants: list[dict], scene: dict | None) -> list[dict]:
    """Turn plants into contact dicts, choosing floor vs. obstacle_top by xz.

    A plant whose (x,z) lands inside an obstacle footprint is pulled onto that
    obstacle's top (steps onto a tread); otherwise it is pinned to the floor
    (grounds it, kills skate). Emitted dicts match `ContactConstraint.from_dict`.
    """
    objects = list((scene or {}).get("objects") or [])
    contacts: list[dict] = []
    for p in plants:
        hit = next((o for o in objects if _in_footprint(p["x"], p["z"], o)), None)
        c = {
            "joint": p["joint"],
            "tol": DEFAULT_TOL,
            "weight": 1.0,
            "frame_start": p["frame_start"],
            "frame_end": p["frame_end"],
        }
        if hit is not None:
            c["target"] = "obstacle_top"
            c["object_id"] = hit.get("id")
            p["target"] = "obstacle_top"
            p["object_id"] = hit.get("id")
            p["target_y"] = round(_obstacle_top_y(hit), 4)
        else:
            c["target"] = "floor"
            p["target"] = "floor"
            p["target_y"] = 0.0
        contacts.append(c)
    return contacts


def auto_contacts(path: Path, scene: dict | None, fps: float = 20.0) -> dict:
    """Detect plants in a draft clip and emit ready-to-use contact constraints.

    Returns {contacts, plants, count} — `contacts` drops straight into the scene
    for a constrained resample; `plants` is the diagnostic breakdown (which foot,
    which frames, floor vs. which obstacle) for the UI.
    """
    joints = np.load(path).astype(np.float64)
    if joints.ndim != 3 or joints.shape[1] < 22 or joints.shape[2] != 3:
        raise ValueError(f"expected (T,22,3) joints, got {joints.shape}")
    plants = detect_plants(joints, fps)
    contacts = assign_targets(plants, scene)
    n_obst = sum(1 for p in plants if p.get("target") == "obstacle_top")
    return {
        "count": len(contacts),
        "on_obstacle": n_obst,
        "on_floor": len(contacts) - n_obst,
        "contacts": contacts,
        "plants": plants,
        "fps": fps,
    }
