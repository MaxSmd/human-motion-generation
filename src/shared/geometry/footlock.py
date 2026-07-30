"""Foot locking: hold planted feet still, by IK rather than by moving the root.

Every sampling-time trajectory constraint in `rmg.flow.trajectory` works by
moving the ROOT — exactly, cheaply, and without touching the pose. That is what
makes control exact, and also what makes the feet slide: relocating the body
does not re-plan where it puts its feet, so a planted foot is carried along with
the hips. Measured, the planted-foot slide is ~1.1x the root step against ~0.42
for an unconstrained walk, and no amount of redistributing the ROOT correction
in time fixes it — the fix has to edit the LEG.

This does that, as a post-process on world joint positions:

  1. segment each foot's stance phases (low, slow),
  2. pick one anchor position per phase,
  3. pin the ankle there and re-solve the knee by closed-form two-bone IK,
     preserving bone lengths exactly and keeping the knee in its original
     bend plane,
  4. carry the toe rigidly with the ankle.

It runs on joint positions, not on the manifold state, because that is what is
rendered, exported and measured — and because a positional edit cannot be
expressed as a per-joint quaternion change without also re-deriving the whole
chain. The hips (and therefore the controlled root trajectory) are never
touched, so **foot locking cannot break a position constraint**.

Joint indices are the 22-joint HumanML3D skeleton; world frame is Y-up with the
floor at y = 0.
"""

from __future__ import annotations

import numpy as np

# hip, knee, ankle, toe per leg.
LEGS: tuple[tuple[int, int, int, int], ...] = ((1, 4, 7, 10), (2, 5, 8, 11))

DEFAULT_HEIGHT = 0.06      # m — a foot below this is a contact candidate
DEFAULT_SPEED = 0.45       # m/s — …and slower than this is actually planted
MIN_STANCE = 3             # frames; shorter runs are noise, not a footstep


def _stance_mask(
    joints: np.ndarray, ankle: int, toe: int, fps: float,
    height_thr: float, speed_thr: float,
) -> np.ndarray:
    """(T,) bool — frames where this foot is planted.

    Height alone is not enough: a shuffling gait never lifts its feet, so every
    frame would count as stance and nothing could be anchored. Requiring the
    foot to also be SLOW is what separates a real plant from a slide.
    """
    h = np.minimum(joints[:, ankle, 1], joints[:, toe, 1])
    v = np.zeros(len(joints))
    step = np.linalg.norm(joints[1:, ankle][:, [0, 2]] - joints[:-1, ankle][:, [0, 2]], axis=-1) * fps
    v[:-1] = step
    v[1:] = np.maximum(v[1:], step)                 # a frame is slow only if both its edges are
    return (h < height_thr) & (v < speed_thr)


def _segments(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Contiguous True runs of `mask` as [start, end) pairs, dropping short ones."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_len:
        out.append((start, len(mask)))
    return out


def _two_bone_ik(hip: np.ndarray, knee: np.ndarray, ankle_target: np.ndarray,
                 l1: float, l2: float) -> np.ndarray:
    """Knee position that puts the ankle at `ankle_target` with bones intact.

    Closed form: the knee lies on the circle where the two bone spheres meet.
    Which point on that circle is chosen is fixed by keeping the ORIGINAL bend
    plane — otherwise the leg would pop to an arbitrary (often backwards) knee.
    """
    d_vec = ankle_target - hip
    d = float(np.linalg.norm(d_vec))
    # Unreachable targets would make the sqrt imaginary; clamp just inside the
    # annulus so a slightly-too-far anchor straightens the leg instead of failing.
    d = float(np.clip(d, abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4))
    n = d_vec / max(np.linalg.norm(d_vec), 1e-8)
    a = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    h = float(np.sqrt(max(l1 * l1 - a * a, 0.0)))
    # Original knee offset, projected off the hip→ankle axis = the bend plane.
    rel = knee - hip
    perp = rel - np.dot(rel, n) * n
    pn = float(np.linalg.norm(perp))
    if pn < 1e-6:                                   # perfectly straight leg: pick any normal
        tmp = np.array([0.0, 0.0, 1.0]) if abs(n[1]) > 0.9 else np.array([0.0, 1.0, 0.0])
        perp = np.cross(n, tmp)
        pn = float(np.linalg.norm(perp))
    return hip + a * n + h * (perp / max(pn, 1e-8))


def _offset_field(
    ankle_xyz: np.ndarray, stances: list[tuple[int, int]], ground_y: list[float],
    airborne: np.ndarray | None = None,
) -> np.ndarray:
    """Per-frame ankle correction (T, 3): cancels drift during each stance and is
    paid back across the following swing.

    Easing the lock on and off at the stance BOUNDARIES is self-defeating: that
    easing motion happens while the foot is on the floor, so it is itself
    skating — the first version of this made foot-skate worse, not better. The
    offset instead stays exactly constant through the stance (so the foot truly
    does not move) and unwinds to zero during the swing, when the foot is in the
    air and free to travel.
    """
    T = len(ankle_xyz)
    off = np.zeros((T, 3))
    for k, (s, e) in enumerate(stances):
        anchor = ankle_xyz[s].copy()
        anchor[1] = ground_y[k]
        off[s:e] = anchor - ankle_xyz[s:e]              # hold: zero motion in stance
        # Unwind over the swing that follows, up to the next stance (or the end).
        nxt = stances[k + 1][0] if k + 1 < len(stances) else T
        gap = nxt - e
        if gap > 0:
            tail = off[e - 1]
            if airborne is None:
                ramp = np.linspace(1.0, 0.0, gap + 1)[1:]
            else:
                # Unwind ONLY on frames where the foot is actually off the floor.
                # Spreading it evenly in time pays the correction back through
                # frames that are still grounded, which is itself skating — that
                # is why the naive version made foot-skate worse on every clip.
                w = airborne[e:nxt].astype(np.float64)
                tot = w.sum()
                if tot <= 0:
                    # No airborne frame before the next stance: there is nowhere
                    # to pay this correction back without sliding, so decline to
                    # lock at all rather than smear the debt across the floor.
                    off[s:e] = 0.0
                    continue
                ramp = 1.0 - np.cumsum(w) / tot
            off[e:nxt] = tail[None, :] * ramp[:, None]
    return off


def lock_feet(
    joints: np.ndarray,
    fps: float = 20.0,
    height_thr: float = DEFAULT_HEIGHT,
    speed_thr: float = DEFAULT_SPEED,
    min_stance: int = MIN_STANCE,
    ground: bool = False,
    metric_floor: float = 0.05,
) -> tuple[np.ndarray, dict]:
    """Hold planted feet still. `joints` (T, J, 3) world positions -> (edited, info).

    The ankle follows a corrected trajectory (see `_offset_field`) and the knee is
    re-solved by two-bone IK, so bone lengths are preserved and the hips — and
    therefore any root/position constraint — are never touched.

    `ground` would drop each anchor to the floor, which sounds right but measures
    worse: pushing feet down puts MORE frames under the contact height, so more
    of the clip is scored for skating. Off by default (baseline clip: slide
    0.42 -> 0.18, skate 0.059 -> 0.050, jerk 135 -> 126; with `ground` those
    become 0.35 / 0.118 / 127).

    Locking only helps a clip that HAS a gait. On motion whose swing phase has
    collapsed into a shuffle there is no airborne frame to unwind the correction
    into, and those stances are skipped rather than smeared across the floor.
    """
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"joints must be (T, J, 3), got {joints.shape}")
    out = joints.copy().astype(np.float64)
    info = {"stances": 0, "frames_locked": 0, "max_shift": 0.0}

    for hip, knee, ankle, toe in LEGS:
        l1 = float(np.linalg.norm(joints[0, knee] - joints[0, hip]))
        l2 = float(np.linalg.norm(joints[0, ankle] - joints[0, knee]))
        mask = _stance_mask(joints, ankle, toe, fps, height_thr, speed_thr)
        stances = _segments(mask, min_stance)
        if not stances:
            continue
        gy = [float(np.min(joints[s:e, [ankle, toe], 1])) if ground else float(joints[s, ankle, 1])
              for s, e in stances]
        # "Airborne" by the same height rule foot-skate is scored with, so the
        # correction is only unwound where it cannot register as sliding.
        air = np.minimum(joints[:, ankle, 1], joints[:, toe, 1]) >= metric_floor
        off = _offset_field(joints[:, ankle], stances, gy, airborne=air)
        info["stances"] += len(stances)

        for t in range(len(out)):
            shift = off[t]
            n = float(np.linalg.norm(shift))
            if n < 1e-9:
                continue
            target = out[t, ankle] + shift
            # A target beyond the leg's reach would stretch the bones, so clamp it
            # onto the reachable sphere and move the foot only as far as it can go.
            rel = target - out[t, hip]
            d = float(np.linalg.norm(rel))
            reach = l1 + l2 - 1e-4
            if d > reach:
                target = out[t, hip] + rel * (reach / max(d, 1e-8))
                shift = target - out[t, ankle]
            info["frames_locked"] += 1
            info["max_shift"] = max(info["max_shift"], float(np.linalg.norm(shift)))
            out[t, knee] = _two_bone_ik(out[t, hip], out[t, knee], target, l1, l2)
            out[t, toe] = out[t, toe] + shift          # foot carried rigidly
            out[t, ankle] = target
    return out.astype(joints.dtype), info
