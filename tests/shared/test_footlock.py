"""Foot locking: hold planted feet by IK, without touching the root."""

from __future__ import annotations

import numpy as np
import pytest

from shared.geometry.footlock import (
    DEFAULT_HEIGHT, DEFAULT_SPEED, LEGS, MIN_STANCE,
    _segments, _stance_mask, _two_bone_ik, lock_feet,
)
from shared.geometry.skeleton import NUM_JOINTS

J = NUM_JOINTS


L1, L2, LFOOT = 0.46, 0.48, 0.12


def _walk(T=60, drift=0.004, lift=0.12):
    """A toy walk with a SELF-CONSISTENT skeleton: hips advance, each foot
    alternates a planted phase (which drifts, i.e. skates) with an airborne
    swing, and the knee is always solved so the bones keep their length. A
    fixture with inconsistent bones cannot test bone preservation."""
    j = np.zeros((T, J, 3))
    j[:, :, 1] = 1.0
    j[:, 0, 2] = np.arange(T) * 0.02                      # root advances
    for leg, (hip, knee, ankle, toe) in enumerate(LEGS):
        side = 0.1 if leg == 0 else -0.1
        guess = np.array([side, 0.5, -0.05])
        for t in range(T):
            phase = (t + leg * 10) % 20
            air = phase >= 10
            h = np.array([side, 0.9, j[t, 0, 2]])
            # Keep the ankle comfortably inside the leg's reach (< L1 + L2).
            a = np.array([side,
                          lift if air else 0.02 + drift * (phase % 10),
                          j[t, 0, 2] - 0.15])
            k = _two_bone_ik(h, guess, a, L1, L2)
            guess = k - h + np.array([0.0, 0.0, 0.0])
            guess = k
            j[t, hip], j[t, knee], j[t, ankle] = h, k, a
            j[t, toe] = a + np.array([0.0, 0.0, LFOOT])
    return j


def _bone_std(j):
    return max(
        float(np.linalg.norm(j[:, b] - j[:, a], axis=-1).std())
        for hip, knee, ankle, toe in LEGS
        for a, b in ((hip, knee), (knee, ankle), (ankle, toe))
    )


def test_two_bone_ik_hits_the_target_and_keeps_bones() -> None:
    hip = np.array([0.0, 1.0, 0.0])
    knee = np.array([0.0, 0.6, 0.05])
    l1, l2 = 0.4, 0.45
    # Comfortably inside reach (|hip-target| < l1 + l2 = 0.85); the boundary
    # case is covered by the clamping test below.
    for target in ([0.0, 0.35, 0.1], [0.2, 0.4, -0.1], [0.05, 0.3, 0.05]):
        t = np.array(target, dtype=float)
        k = _two_bone_ik(hip, knee, t, l1, l2)
        assert np.linalg.norm(k - hip) == pytest.approx(l1, abs=1e-6)
        assert np.linalg.norm(t - k) == pytest.approx(l2, abs=1e-6)


def test_two_bone_ik_clamps_unreachable_targets() -> None:
    """An out-of-reach target must straighten the leg, never stretch a bone."""
    hip = np.array([0.0, 1.0, 0.0])
    knee = np.array([0.0, 0.6, 0.05])
    l1, l2 = 0.4, 0.45
    k = _two_bone_ik(hip, knee, np.array([0.0, -5.0, 0.0]), l1, l2)
    assert np.linalg.norm(k - hip) == pytest.approx(l1, abs=1e-6)


def test_lock_feet_never_moves_the_root() -> None:
    """Position constraints act on the root, so locking must not touch it."""
    j = _walk()
    out, _ = lock_feet(j)
    np.testing.assert_allclose(out[:, 0], j[:, 0], atol=1e-12)


def test_lock_feet_preserves_bone_lengths() -> None:
    j = _walk()
    out, info = lock_feet(j)
    assert info["stances"] > 0, "toy walk should have detectable stances"
    assert _bone_std(out) < 1e-6, f"bones stretched: {_bone_std(out):.6f}"


def test_lock_feet_removes_stance_drift() -> None:
    """The whole point: within a detected stance the foot stops creeping.

    Asserted on the module's OWN stance segmentation rather than through the
    foot-skate metric's height rule — the two use different definitions of
    "planted", and testing the algorithm through someone else's threshold
    measures the thresholds, not the fix.
    """
    j = _walk(drift=0.01)
    out, info = lock_feet(j)
    assert info["stances"] > 0

    def worst_drift(clip):
        worst = 0.0
        for hip, knee, ankle, toe in LEGS:
            mask = _stance_mask(clip, ankle, toe, 20.0, DEFAULT_HEIGHT, DEFAULT_SPEED)
            for s_, e_ in _segments(mask, MIN_STANCE):
                xz = clip[s_:e_, ankle][:, [0, 2]]
                worst = max(worst, float(np.ptp(xz, axis=0).max()))
        return worst

    before, after = worst_drift(j), worst_drift(out)
    assert before > 0.02, "fixture should actually drift"
    assert after < 0.1 * before, f"stance drift not removed: {before:.4f} -> {after:.4f}"


def test_lock_feet_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        lock_feet(np.zeros((10, 3)))
