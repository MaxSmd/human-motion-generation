from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from momask.constraints import (
    BendAngleConstraint,
    JointPositionConstraint,
    LatentRefinementConfig,
    bend_angle_loss,
    bend_angle_violation,
    bend_angles_from_joints,
    decode_latents_to_joints,
    joint_position_error,
    joint_position_loss,
    refine_motion_latents,
)
from momask.models import MotionRVQVAE
from momask.tasks import generate_h3d263_constrained
from shared.geometry import H3D_FEATURE_DIM, NUM_JOINTS


def _empty_joints(batch: int = 1, time: int = 1) -> torch.Tensor:
    return torch.zeros(batch, time, NUM_JOINTS, 3)


def _position_constraint(
    *, batch: int, time: int, frame: int, joint: int, target: tuple[float, float, float]
) -> JointPositionConstraint:
    targets = torch.zeros(batch, time, NUM_JOINTS, 3)
    mask = torch.zeros(batch, time, NUM_JOINTS, dtype=torch.bool)
    targets[:, frame, joint] = torch.tensor(target)
    mask[:, frame, joint] = True
    return JointPositionConstraint(targets, mask)


def test_bend_angles_use_zero_for_straight_and_pi_over_two_for_right_angle() -> None:
    joints = _empty_joints(time=2)
    joints[0, 0, 1] = torch.tensor([1.0, 0.0, 0.0])
    joints[0, 0, 4] = torch.tensor([2.0, 0.0, 0.0])
    joints[0, 1, 1] = torch.tensor([1.0, 0.0, 0.0])
    joints[0, 1, 4] = torch.tensor([1.0, 1.0, 0.0])

    angles = bend_angles_from_joints(joints, torch.tensor([[0, 1, 4]]))

    assert torch.allclose(angles[0, 0], torch.tensor([0.0]), atol=1e-6)
    assert torch.allclose(angles[0, 1], torch.tensor([math.pi / 2]), atol=1e-6)


def test_position_loss_respects_axis_and_frame_masks() -> None:
    joints = _empty_joints(time=2)
    targets = joints.clone()
    targets[0, 0, 1] = torch.tensor([2.0, 5.0, 0.0])
    targets[0, 1, 1] = torch.tensor([10.0, 0.0, 0.0])
    mask = torch.zeros(1, 2, NUM_JOINTS, dtype=torch.bool)
    mask[0, :, 1] = True
    axis_mask = torch.zeros_like(targets, dtype=torch.bool)
    axis_mask[0, :, 1, 0] = True
    constraint = JointPositionConstraint(targets, mask, axis_mask)
    frame_mask = torch.tensor([[True, False]])

    assert torch.allclose(joint_position_error(joints, constraint, frame_mask=frame_mask), torch.tensor(2.0))
    expected = torch.nn.functional.smooth_l1_loss(
        torch.tensor(2.0), torch.tensor(0.0), beta=0.05
    )
    assert torch.allclose(joint_position_loss(joints, constraint, frame_mask=frame_mask), expected)


def test_angle_range_loss_and_violation_ignore_values_inside_range() -> None:
    joints = _empty_joints()
    joints[0, 0, 1] = torch.tensor([1.0, 0.0, 0.0])
    joints[0, 0, 4] = torch.tensor([1.0, 1.0, 0.0])
    mask = torch.ones(1, 1, 1, dtype=torch.bool)
    lower = torch.full((1, 1, 1), math.radians(80.0))
    upper = torch.full((1, 1, 1), math.radians(100.0))
    constraint = BendAngleConstraint(
        triplets=torch.tensor([[0, 1, 4]]),
        mask=mask,
        min_radians=lower,
        max_radians=upper,
    )

    assert torch.allclose(bend_angle_loss(joints, constraint), torch.tensor(0.0), atol=1e-7)
    assert torch.allclose(bend_angle_violation(joints, constraint), torch.tensor(0.0), atol=1e-7)


def test_angle_loss_rejects_degenerate_constrained_bones() -> None:
    constraint = BendAngleConstraint(
        triplets=torch.tensor([[0, 1, 4]]),
        mask=torch.ones(1, 1, 1, dtype=torch.bool),
        target_radians=torch.full((1, 1, 1), math.pi / 2),
    )

    with pytest.raises(ValueError, match="degenerate bones"):
        bend_angle_loss(_empty_joints(), constraint)


def test_angle_loss_has_finite_nonzero_gradient_away_from_exact_collinearity() -> None:
    joints = _empty_joints().requires_grad_(True)
    with torch.no_grad():
        joints[0, 0, 1] = torch.tensor([1.0, 0.0, 0.0])
        joints[0, 0, 4] = torch.tensor([2.0, 0.1, 0.0])
    constraint = BendAngleConstraint(
        triplets=torch.tensor([[0, 1, 4]]),
        mask=torch.ones(1, 1, 1, dtype=torch.bool),
        target_radians=torch.full((1, 1, 1), math.pi / 2),
    )

    loss = bend_angle_loss(joints, constraint)
    loss.backward()

    assert joints.grad is not None
    assert torch.isfinite(joints.grad).all()
    assert joints.grad.abs().sum() > 0


class _IdentityDecoder(nn.Module):
    downsample = 1

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def decode_latents(self, latents: torch.Tensor, target_len: int | None = None) -> torch.Tensor:
        decoded = latents * self.scale
        if target_len is None:
            return decoded
        if decoded.shape[1] >= target_len:
            return decoded[:, :target_len]
        padding = decoded.new_zeros(decoded.shape[0], target_len - decoded.shape[1], decoded.shape[2])
        return torch.cat([decoded, padding], dim=1)


class _ZeroBaseGenerator:
    def generate(self, *, cond: torch.Tensor, seq_len: int, **_kwargs) -> torch.Tensor:
        return torch.zeros(cond.shape[0], seq_len, dtype=torch.long, device=cond.device)


class _ZeroResidualGenerator:
    def generate_residuals(self, base: torch.Tensor, **_kwargs) -> torch.Tensor:
        return base.unsqueeze(1)


def test_latent_refinement_reduces_joint_error_without_model_gradients() -> None:
    model = _IdentityDecoder()
    initial = torch.zeros(1, 4, H3D_FEATURE_DIM)
    constraint = _position_constraint(batch=1, time=4, frame=1, joint=1, target=(1.0, 0.0, 0.0))
    config = LatentRefinementConfig(
        steps=40,
        learning_rate=0.1,
        latent_weight=0.001,
        dynamics_weight=0.0,
        root_weight=0.0,
        bone_weight=0.0,
        max_delta_norm=2.0,
        grad_clip_norm=10.0,
        history_interval=1,
    )

    result = refine_motion_latents(
        model,
        initial,
        mean=torch.zeros(H3D_FEATURE_DIM),
        std=torch.ones(H3D_FEATURE_DIM),
        target_len=4,
        position_constraint=constraint,
        config=config,
    )

    before = joint_position_error(result.initial_joints, constraint)
    after = joint_position_error(result.joints, constraint)
    assert after < before * 0.2
    assert model.scale.grad is None
    assert result.history[-1]["position"] < result.history[0]["position"]
    assert result.metrics["position_error_final_m"] < result.metrics["position_error_initial_m"]


def test_latent_refinement_reduces_bend_angle_violation() -> None:
    model = _IdentityDecoder()
    initial = torch.zeros(1, 1, H3D_FEATURE_DIM)
    initial[0, 0, 4:7] = torch.tensor([1.0, 0.0, 0.0])
    joint_four_start = 4 + (4 - 1) * 3
    initial[0, 0, joint_four_start : joint_four_start + 3] = torch.tensor([2.0, 0.1, 0.0])
    constraint = BendAngleConstraint(
        triplets=torch.tensor([[0, 1, 4]]),
        mask=torch.ones(1, 1, 1, dtype=torch.bool),
        target_radians=torch.full((1, 1, 1), math.pi / 2),
    )
    config = LatentRefinementConfig(
        steps=60,
        learning_rate=0.05,
        latent_weight=0.001,
        dynamics_weight=0.0,
        root_weight=0.0,
        bone_weight=0.0,
        max_delta_norm=2.0,
        grad_clip_norm=10.0,
    )

    result = refine_motion_latents(
        model,
        initial,
        mean=torch.zeros(H3D_FEATURE_DIM),
        std=torch.ones(H3D_FEATURE_DIM),
        target_len=1,
        angle_constraint=constraint,
        config=config,
    )

    before = bend_angle_violation(result.initial_joints, constraint)
    after = bend_angle_violation(result.joints, constraint)
    assert after < before * 0.25
    assert result.metrics["angle_violation_final_rad"] < result.metrics["angle_violation_initial_rad"]


def test_latent_refinement_rejects_constraints_on_zero_padded_tail() -> None:
    model = _IdentityDecoder()
    initial = torch.zeros(1, 2, H3D_FEATURE_DIM)
    constraint = _position_constraint(batch=1, time=3, frame=2, joint=1, target=(1.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="padded or invalid"):
        refine_motion_latents(
            model,
            initial,
            mean=torch.zeros(H3D_FEATURE_DIM),
            std=torch.ones(H3D_FEATURE_DIM),
            target_len=3,
            position_constraint=constraint,
            config=LatentRefinementConfig(steps=1),
        )
    assert model.training


def test_latent_refinement_rejects_constraints_after_valid_token_prefix() -> None:
    model = _IdentityDecoder()
    initial = torch.zeros(1, 3, H3D_FEATURE_DIM)
    token_mask = torch.tensor([[True, True, False]])
    constraint = _position_constraint(batch=1, time=3, frame=2, joint=1, target=(1.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="padded or invalid"):
        refine_motion_latents(
            model,
            initial,
            mean=torch.zeros(H3D_FEATURE_DIM),
            std=torch.ones(H3D_FEATURE_DIM),
            target_len=3,
            token_mask=token_mask,
            position_constraint=constraint,
            config=LatentRefinementConfig(steps=1),
        )


def test_latent_refinement_rejects_empty_rows_in_token_mask() -> None:
    model = _IdentityDecoder()
    initial = torch.zeros(2, 2, H3D_FEATURE_DIM)
    token_mask = torch.tensor([[True, True], [False, False]])
    constraint = _position_constraint(batch=2, time=2, frame=1, joint=1, target=(1.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="every sample"):
        refine_motion_latents(
            model,
            initial,
            mean=torch.zeros(H3D_FEATURE_DIM),
            std=torch.ones(H3D_FEATURE_DIM),
            target_len=2,
            token_mask=token_mask,
            position_constraint=constraint,
            config=LatentRefinementConfig(steps=1),
        )


def test_latent_refinement_rejects_nonzero_frame_zero_root_xz() -> None:
    model = _IdentityDecoder()
    initial = torch.zeros(1, 2, H3D_FEATURE_DIM)
    constraint = _position_constraint(batch=1, time=2, frame=0, joint=0, target=(1.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="canonical origin"):
        refine_motion_latents(
            model,
            initial,
            mean=torch.zeros(H3D_FEATURE_DIM),
            std=torch.ones(H3D_FEATURE_DIM),
            target_len=2,
            position_constraint=constraint,
            config=LatentRefinementConfig(steps=1),
        )


def test_latent_refinement_runs_through_real_downsampled_decoder() -> None:
    torch.manual_seed(0)
    model = MotionRVQVAE(
        input_dim=H3D_FEATURE_DIM,
        hidden_dim=16,
        latent_dim=8,
        num_quantizers=2,
        codebook_size=8,
        downsample=4,
        num_res_blocks=1,
    )
    initial = torch.randn(1, 2, 8)
    mean = torch.zeros(H3D_FEATURE_DIM)
    std = torch.ones(H3D_FEATURE_DIM)
    with torch.no_grad():
        _, _, initial_joints = decode_latents_to_joints(
            model, initial, mean=mean, std=std, target_len=7
        )
    targets = initial_joints.clone()
    targets[0, 5, 20, 0] += 0.05
    mask = torch.zeros(1, 7, NUM_JOINTS, dtype=torch.bool)
    mask[0, 5, 20] = True

    result = refine_motion_latents(
        model,
        initial,
        mean=mean,
        std=std,
        target_len=7,
        position_constraint=JointPositionConstraint(targets, mask),
        config=LatentRefinementConfig(steps=2, learning_rate=0.01),
    )

    assert result.motion.shape == (1, 7, H3D_FEATURE_DIM)
    assert result.joints.shape == (1, 7, NUM_JOINTS, 3)
    assert torch.isfinite(result.motion).all()
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())


def test_constrained_generation_enables_grad_after_no_grad_token_sampling() -> None:
    torch.manual_seed(0)
    model = MotionRVQVAE(
        input_dim=H3D_FEATURE_DIM,
        hidden_dim=16,
        latent_dim=8,
        num_quantizers=1,
        codebook_size=4,
        downsample=4,
        num_res_blocks=1,
    )
    mean = torch.zeros(H3D_FEATURE_DIM)
    std = torch.ones(H3D_FEATURE_DIM)
    tokens = torch.zeros(1, 1, 2, dtype=torch.long)
    with torch.no_grad():
        latents = model.quantizer.decode(tokens)
        _, _, initial_joints = decode_latents_to_joints(
            model, latents, mean=mean, std=std, target_len=8
        )
    targets = initial_joints.clone()
    targets[0, 4, 20, 0] += 0.02
    mask = torch.zeros(1, 8, NUM_JOINTS, dtype=torch.bool)
    mask[0, 4, 20] = True

    result = generate_h3d263_constrained(
        vqvae=model,
        masked_transformer=_ZeroBaseGenerator(),
        residual_transformer=_ZeroResidualGenerator(),
        cond=torch.zeros(1, 4),
        seq_len=2,
        target_len=8,
        mean=mean,
        std=std,
        position_constraint=JointPositionConstraint(targets, mask),
        refinement=LatentRefinementConfig(steps=2, history_interval=1),
    )

    assert result.tokens is not None
    assert result.tokens.shape == (1, 1, 2)
    assert len(result.history) == 2
