"""Sampling-time joint-angle constraint tests.

Covers axis-angle → quaternion, the inpaint-target layout, and that the
Riemannian sampler actually pins constrained joints to their target quaternion
(while leaving unconstrained joints free).
"""

from __future__ import annotations

import math

import torch

from rmg.flow import (
    JointAngleConstraint,
    OracleVelocity,
    RiemannianEulerSampler,
    SamplerCfg,
    WrappedGaussianPrior,
    axis_angle_to_quat,
    build_inpaint_targets,
    parse_constraints,
    rest_pose_mu,
    rmg_manifold,
)

torch.manual_seed(0)


def test_axis_angle_identity_and_known_rotation() -> None:
    # zero angle → identity quaternion
    q0 = axis_angle_to_quat("z", 0.0)
    assert torch.allclose(q0, torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-6)
    # 90° about z → [cos45, 0, 0, sin45]
    q = axis_angle_to_quat((0.0, 0.0, 1.0), math.radians(90.0))
    expect = torch.tensor([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])
    assert torch.allclose(q, expect, atol=1e-6)
    assert torch.allclose(q.norm(), torch.tensor(1.0), atol=1e-6)
    assert q[0] >= 0  # upper hemisphere


def test_build_inpaint_targets_layout() -> None:
    J, T = 22, 30
    c = JointAngleConstraint(joint="L_Elbow", axis="z", angle_deg=90.0, frame_start=5, frame_end=20)
    values, mask = build_inpaint_targets([c], num_frames=T, num_joints=J)
    assert values.shape == (T, 3 + 4 * J)
    assert mask.dtype == torch.bool
    j = 18  # L_Elbow index
    lo, hi = 3 + 4 * j, 3 + 4 * (j + 1)
    # masked exactly on the joint's 4 quat dims over [5, 20)
    assert mask[5:20, lo:hi].all()
    assert not mask[:5, lo:hi].any()
    assert not mask[20:, lo:hi].any()
    # no other dims touched
    other = torch.ones_like(mask)
    other[5:20, lo:hi] = False
    assert not mask[other].any()
    # target value matches the quaternion
    assert torch.allclose(values[10, lo:hi], c.quat(), atol=1e-6)


def test_frame_window_defaults_to_all_frames() -> None:
    c = JointAngleConstraint(joint=0)
    assert c.frame_window(50) == (0, 50)
    c2 = JointAngleConstraint.from_dict({"joint": "R_Knee", "angle_deg": 30, "frame_end": -1})
    assert c2.frame_window(50) == (0, 50)


def test_empty_constraints_mask_all_false() -> None:
    values, mask = build_inpaint_targets(parse_constraints(None), num_frames=10, num_joints=22)
    assert not mask.any()
    assert values.shape == (10, 3 + 4 * 22)


def test_sampler_pins_constrained_joint() -> None:
    """With a constraint, the chosen joint's quaternion must equal the target in
    the output, while an unconstrained joint must NOT match it (still free)."""
    J, T = 5, 8
    M = rmg_manifold(num_joints=J)
    mu = rest_pose_mu(num_joints=J, dtype=torch.float64).to(torch.float64)
    prior = WrappedGaussianPrior(M, mu, sigma=0.5)

    # Oracle field flows from x0 to a random x1 on the manifold; without the
    # constraint the output is x1, so we can check the constraint overrides it.
    x0 = prior.sample((1, T), dtype=torch.float64)
    x1 = prior.sample((1, T), dtype=torch.float64)
    model = OracleVelocity(M, x0, x1).double()

    sampler = RiemannianEulerSampler(M, prior, SamplerCfg(num_steps=20, guidance_scale=1.0))

    pin_joint = 2
    c = JointAngleConstraint(joint=pin_joint, axis="x", angle_deg=90.0)
    values, mask = build_inpaint_targets([c], num_frames=T, num_joints=J, dtype=torch.float64)

    out = sampler.sample(model, shape=(1, T), num_steps=20, dtype=torch.float64,
                         fixed_values=values, fixed_mask=mask)

    lo, hi = 3 + 4 * pin_joint, 3 + 4 * (pin_joint + 1)
    q_target = c.quat(dtype=torch.float64)
    # pinned joint sits exactly on the target across all frames
    assert torch.allclose(out[0, :, lo:hi], q_target.expand(T, 4), atol=1e-5)
    # output still lies on the manifold (target is a unit quaternion)
    assert M.validate(out, atol=1e-4).all()
    # an unconstrained joint is generally NOT the target (it tracked the oracle)
    free_lo = 3 + 4 * 1
    assert not torch.allclose(out[0, :, free_lo:free_lo + 4], q_target.expand(T, 4), atol=1e-3)
