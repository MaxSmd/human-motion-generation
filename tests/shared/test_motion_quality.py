"""Physical-plausibility metrics: foot skate, jerk, root speed.

These qualify a *constrained* generation result — a hard spatial constraint can
hit its targets exactly and still slide the body across the floor — so they need
to be right in the obvious cases before any number reported next to FID means
anything.
"""

from __future__ import annotations

import pytest
import torch

from shared.eval import foot_skate_ratio, jerk, motion_quality, root_speed
from shared.eval.motion_quality import LEFT_FOOT, RIGHT_FOOT
from shared.geometry.skeleton import NUM_JOINTS

J = NUM_JOINTS


def _clip(T: int, B: int = 1) -> torch.Tensor:
    """A body standing still with both feet planted on the floor."""
    j = torch.zeros(B, T, J, 3)
    j[..., 1] = 1.0                       # everything a metre up …
    for f in (*LEFT_FOOT, *RIGHT_FOOT):
        j[:, :, f, 1] = 0.0               # … except the feet, on the ground
    return j


def test_planted_feet_do_not_skate() -> None:
    assert foot_skate_ratio(_clip(20)) == 0.0


def test_sliding_planted_foot_is_caught() -> None:
    j = _clip(20)
    # Drag the left foot 10 cm/frame along x while it stays on the floor.
    slide = torch.arange(20, dtype=torch.float32) * 0.10
    for f in LEFT_FOOT:
        j[0, :, f, 0] = slide
    assert foot_skate_ratio(j) == pytest.approx(1.0)


def test_lifted_foot_may_move_freely() -> None:
    """A foot in the air is a step, not a skate."""
    j = _clip(20)
    for f in LEFT_FOOT:
        j[0, :, f, 1] = 0.5                       # lifted well above the contact band
        j[0, :, f, 0] = torch.arange(20, dtype=torch.float32) * 0.10
    assert foot_skate_ratio(j) == 0.0


def test_skate_ratio_respects_lengths() -> None:
    """Padding frames past a clip's length must not be scored."""
    j = _clip(20)
    for f in LEFT_FOOT:
        j[0, 10:, f, 0] = torch.arange(10, dtype=torch.float32) * 0.5   # skate late
    assert foot_skate_ratio(j, lengths=torch.tensor([10])) == 0.0
    assert foot_skate_ratio(j) > 0.0


def test_skate_threshold_scales_with_fps() -> None:
    """The published 2.5 cm cut is per 20-fps frame; at 40 fps the same physical
    speed covers half the distance per frame and must still count."""
    j = _clip(10)
    for f in LEFT_FOOT:
        j[0, :, f, 0] = torch.arange(10, dtype=torch.float32) * 0.02   # 2 cm/frame
    assert foot_skate_ratio(j, fps=20.0) == 0.0        # below the 2.5 cm cut
    assert foot_skate_ratio(j, fps=40.0) > 0.0         # 1.25 cm cut at 40 fps


def test_jerk_zero_for_constant_velocity() -> None:
    j = _clip(20)
    j[..., 0] += torch.arange(20, dtype=torch.float32).view(1, 20, 1) * 0.05
    # Not exactly 0: the third difference of float32 positions leaves ~1e-7 of
    # rounding, which the m/s³ conversion multiplies by fps³ = 8000.
    assert jerk(j) < 1e-3


def test_jerk_positive_for_a_kink() -> None:
    j = _clip(20)
    j[0, 10:, :, 0] += 1.0                 # a teleport partway through
    assert jerk(j) > 1.0, "a 1 m teleport must dwarf the float32 noise floor"


def test_root_speed() -> None:
    j = _clip(21)
    j[..., 0, 0] = torch.arange(21, dtype=torch.float32) * 0.05   # 0.05 m/frame
    assert root_speed(j, fps=20.0) == pytest.approx(1.0, abs=1e-4)


def test_motion_quality_bundle_and_guards() -> None:
    q = motion_quality(_clip(20), fps=20.0)
    assert set(q) == {"foot_skate_ratio", "jerk", "root_speed"}
    # Degenerate clip lengths return zeros rather than blowing up.
    assert foot_skate_ratio(_clip(1)) == 0.0
    assert jerk(_clip(2)) == 0.0
    assert root_speed(_clip(1)) == 0.0
    with pytest.raises(ValueError):
        foot_skate_ratio(torch.zeros(4, J, 3))
