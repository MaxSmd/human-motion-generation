"""Swing-twist hinge-limit projection tests.

Validates the swing-twist decomposition/clamp and that the sampler's project_fn
keeps constrained joints inside their hinge range and on S^3.
"""

from __future__ import annotations

import math

import torch

from rmg.flow import (
    HingeConstraint,
    OracleVelocity,
    RiemannianEulerSampler,
    SamplerCfg,
    WrappedGaussianPrior,
    axis_angle_to_quat,
    build_hinge_projector,
    parse_ranges,
    rest_pose_mu,
    rmg_manifold,
    swing_twist_clamp,
)

torch.manual_seed(0)

Z = torch.tensor([0.0, 0.0, 1.0])
X = torch.tensor([1.0, 0.0, 0.0])


def _twist_angle(q: torch.Tensor, axis: torch.Tensor) -> float:
    """Signed rotation angle of q about `axis` (the twist component)."""
    w = q[0]
    d = (q[1:] * axis).sum()
    return float(2.0 * torch.atan2(d, w))


def test_pure_twist_in_range_is_unchanged() -> None:
    # 5° about z, range [0,10°] → unchanged (swing already 0).
    q = axis_angle_to_quat("z", math.radians(5.0), dtype=torch.float64)
    out = swing_twist_clamp(q, Z.double(), 0.0, math.radians(10.0), 0.0)
    assert math.degrees(_twist_angle(out, Z.double())) == \
        __import__("pytest").approx(5.0, abs=1e-3)


def test_twist_above_max_is_clamped() -> None:
    # 50° about z, range [0,10°] → clamped to 10°.
    q = axis_angle_to_quat("z", math.radians(50.0), dtype=torch.float64)
    out = swing_twist_clamp(q, Z.double(), 0.0, math.radians(10.0), 0.0)
    assert math.degrees(_twist_angle(out, Z.double())) == \
        __import__("pytest").approx(10.0, abs=1e-2)
    assert torch.allclose(out.norm(), torch.tensor(1.0, dtype=torch.float64), atol=1e-6)


def test_pure_swing_removed_with_zero_swing_budget() -> None:
    # A rotation purely about x (perpendicular to the z hinge) is all swing;
    # with swing_max=0 and twist range including 0 it collapses to identity.
    q = axis_angle_to_quat("x", math.radians(40.0), dtype=torch.float64)
    out = swing_twist_clamp(q, Z.double(), 0.0, math.radians(10.0), 0.0)
    assert torch.allclose(out, torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64), atol=1e-5)


def test_negative_twist_clamped_to_min() -> None:
    # -30° about z, range [0,10°] → clamped up to 0° (no hyperextension).
    q = axis_angle_to_quat("z", math.radians(-30.0), dtype=torch.float64)
    out = swing_twist_clamp(q, Z.double(), 0.0, math.radians(10.0), 0.0)
    assert math.degrees(_twist_angle(out, Z.double())) == \
        __import__("pytest").approx(0.0, abs=1e-2)


def test_sampler_project_fn_keeps_joint_in_range() -> None:
    J, T = 5, 8
    M = rmg_manifold(num_joints=J)
    mu = rest_pose_mu(num_joints=J, dtype=torch.float64).to(torch.float64)
    prior = WrappedGaussianPrior(M, mu, sigma=0.7)
    x0 = prior.sample((1, T), dtype=torch.float64)
    x1 = prior.sample((1, T), dtype=torch.float64)
    model = OracleVelocity(M, x0, x1).double()
    sampler = RiemannianEulerSampler(M, prior, SamplerCfg(num_steps=20, guidance_scale=1.0))

    joint = 3
    ranges = parse_ranges([{"joint": joint, "axis": "z", "min_deg": 0, "max_deg": 10}])
    # toy 5-joint skeleton: address the raw quaternion index (no SMPL chain remap)
    proj = build_hinge_projector(ranges, num_frames=T, num_joints=J, dtype=torch.float64,
                                 remap_to_controller=False)
    assert proj is not None

    out = sampler.sample(model, shape=(1, T), num_steps=20, dtype=torch.float64, project_fn=proj)

    lo = 3 + 4 * joint
    qj = out[0, :, lo:lo + 4]
    assert M.validate(out, atol=1e-4).all()  # still on the manifold
    for t in range(T):
        ang = math.degrees(_twist_angle(qj[t], Z.double()))
        assert -1e-2 <= ang <= 10.0 + 1e-2, f"frame {t}: twist {ang}° out of [0,10]"


def test_empty_ranges_no_projector() -> None:
    assert build_hinge_projector(parse_ranges(None), num_frames=10, num_joints=22) is None
