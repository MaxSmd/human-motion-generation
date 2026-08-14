"""Differentiable inference-time constraints for MoMask motion latents.

MoMask samples discrete RVQ tokens. Those token ids are not differentiable, so
constraint guidance starts from their summed codebook embedding and refines that
continuous latent through the frozen RVQ decoder. The resulting motion is close
to, but not guaranteed to remain exactly on, the discrete codebook manifold.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from shared.geometry import FOOT_CONTACT_IDX, NUM_JOINTS, PARENTS, recover_joints_from_ric


def _as_frame_mask(frame_mask: Tensor | None, batch: int, time: int, device: torch.device) -> Tensor:
    if frame_mask is None:
        return torch.ones(batch, time, dtype=torch.bool, device=device)
    if frame_mask.shape != (batch, time):
        raise ValueError(f"frame_mask must be {(batch, time)}, got {tuple(frame_mask.shape)}")
    return frame_mask.to(device=device, dtype=torch.bool)


def _zero_loss(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    if values.shape != mask.shape:
        raise ValueError(f"values and mask must have the same shape, got {values.shape} and {mask.shape}")
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


@dataclass(frozen=True)
class JointPositionConstraint:
    """Sparse clip-canonical world-space targets for HumanML3D's 22 joints.

    ``targets`` has shape ``(B, T, 22, 3)`` and ``mask`` has shape
    ``(B, T, 22)``. ``axis_mask`` can optionally constrain only selected XYZ
    axes and has the same shape as ``targets``.
    """

    targets: Tensor
    mask: Tensor
    axis_mask: Tensor | None = None

    def __post_init__(self) -> None:
        if self.targets.ndim != 4 or self.targets.shape[-2:] != (NUM_JOINTS, 3):
            raise ValueError(
                f"targets must be (B, T, {NUM_JOINTS}, 3), got {tuple(self.targets.shape)}"
            )
        if self.mask.shape != self.targets.shape[:-1]:
            raise ValueError(
                f"mask must be {tuple(self.targets.shape[:-1])}, got {tuple(self.mask.shape)}"
            )
        if self.axis_mask is not None and self.axis_mask.shape != self.targets.shape:
            raise ValueError(
                f"axis_mask must be {tuple(self.targets.shape)}, got {tuple(self.axis_mask.shape)}"
            )
        active_axes = self.mask.unsqueeze(-1).bool().expand_as(self.targets)
        if self.axis_mask is not None:
            active_axes = active_axes & self.axis_mask.bool()
        if bool(active_axes.any()) and not bool(torch.isfinite(self.targets[active_axes]).all()):
            raise ValueError("active joint-position targets must be finite")

    def to(self, device: torch.device, dtype: torch.dtype) -> "JointPositionConstraint":
        return JointPositionConstraint(
            targets=self.targets.to(device=device, dtype=dtype),
            mask=self.mask.to(device=device, dtype=torch.bool),
            axis_mask=None
            if self.axis_mask is None
            else self.axis_mask.to(device=device, dtype=torch.bool),
        )


@dataclass(frozen=True)
class BendAngleConstraint:
    """Exact or ranged bend-angle targets over joint triplets.

    ``triplets[c]`` is ``(parent, joint, child)``. Angles use the bend
    convention: zero radians is straight, and pi radians is fully folded back.
    ``mask`` has shape ``(B, T, C)``. Supply either ``target_radians`` for an
    exact target, or both ``min_radians`` and ``max_radians`` for a range.
    """

    triplets: Tensor
    mask: Tensor
    target_radians: Tensor | None = None
    min_radians: Tensor | None = None
    max_radians: Tensor | None = None

    def __post_init__(self) -> None:
        if self.triplets.ndim != 2 or self.triplets.shape[-1] != 3:
            raise ValueError(f"triplets must be (C, 3), got {tuple(self.triplets.shape)}")
        if self.triplets.shape[0] == 0:
            raise ValueError("at least one bend-angle triplet is required")
        if self.mask.ndim != 3 or self.mask.shape[-1] != self.triplets.shape[0]:
            raise ValueError(
                f"mask must be (B, T, {self.triplets.shape[0]}), got {tuple(self.mask.shape)}"
            )
        if bool(((self.triplets < 0) | (self.triplets >= NUM_JOINTS)).any()):
            raise ValueError(f"joint indices must be in [0, {NUM_JOINTS - 1}]")
        if bool(
            (
                (self.triplets[:, 0] == self.triplets[:, 1])
                | (self.triplets[:, 0] == self.triplets[:, 2])
                | (self.triplets[:, 1] == self.triplets[:, 2])
            ).any()
        ):
            raise ValueError("each angle triplet must contain three distinct joints")
        for parent, joint, child in self.triplets.detach().cpu().tolist():
            if PARENTS[joint] != parent or PARENTS[child] != joint:
                raise ValueError(
                    f"angle triplet {(parent, joint, child)} is not a connected parent-joint-child chain"
                )
        exact = self.target_radians is not None
        ranged = self.min_radians is not None or self.max_radians is not None
        if exact == ranged:
            raise ValueError("supply either target_radians or a min/max range")
        expected = self.mask.shape
        if exact and self.target_radians is not None and self.target_radians.shape != expected:
            raise ValueError(f"target_radians must be {tuple(expected)}")
        if ranged:
            if self.min_radians is None or self.max_radians is None:
                raise ValueError("both min_radians and max_radians are required for a range")
            if self.min_radians.shape != expected or self.max_radians.shape != expected:
                raise ValueError(f"angle bounds must be {tuple(expected)}")
            if bool((self.min_radians > self.max_radians).any()):
                raise ValueError("min_radians must not exceed max_radians")

        for value, name in (
            (self.target_radians, "target_radians"),
            (self.min_radians, "min_radians"),
            (self.max_radians, "max_radians"),
        ):
            if value is not None:
                active_values = value[self.mask.bool()]
                if bool(active_values.numel()) and not bool(torch.isfinite(active_values).all()):
                    raise ValueError(f"active {name} values must be finite")
                if bool(((value < 0.0) | (value > math.pi)).any()):
                    raise ValueError(f"{name} must lie in [0, pi]")

    def to(self, device: torch.device, dtype: torch.dtype) -> "BendAngleConstraint":
        def move(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.to(device=device, dtype=dtype)

        return BendAngleConstraint(
            triplets=self.triplets.to(device=device, dtype=torch.long),
            mask=self.mask.to(device=device, dtype=torch.bool),
            target_radians=move(self.target_radians),
            min_radians=move(self.min_radians),
            max_radians=move(self.max_radians),
        )


def bend_angles_from_joints(
    joints: Tensor,
    triplets: Tensor,
    eps: float = 1e-7,
) -> Tensor:
    """Return bend angles ``(B, T, C)`` in radians.

    The vectors are ``parent -> joint`` and ``joint -> child``. Consequently,
    a straight chain has angle zero. Exactly collinear chains have no preferred
    bend direction, so unsigned optimization can stall there unless the
    reference pose already provides a slight bend or a position constraint
    supplies a direction.
    """

    angles, _ = _bend_angles_and_validity(joints, triplets, eps)
    return angles


def _bend_angles_and_validity(
    joints: Tensor,
    triplets: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    if joints.ndim != 4 or joints.shape[-2:] != (NUM_JOINTS, 3):
        raise ValueError(f"joints must be (B, T, {NUM_JOINTS}, 3), got {tuple(joints.shape)}")
    if triplets.ndim != 2 or triplets.shape[-1] != 3:
        raise ValueError(f"triplets must be (C, 3), got {tuple(triplets.shape)}")
    triplets = triplets.to(device=joints.device, dtype=torch.long)
    parent = joints[:, :, triplets[:, 0]]
    center = joints[:, :, triplets[:, 1]]
    child = joints[:, :, triplets[:, 2]]
    incoming = center - parent
    outgoing = child - center
    incoming_norm = torch.linalg.vector_norm(incoming, dim=-1)
    outgoing_norm = torch.linalg.vector_norm(outgoing, dim=-1)
    incoming_unit = incoming / incoming_norm.clamp_min(eps).unsqueeze(-1)
    outgoing_unit = outgoing / outgoing_norm.clamp_min(eps).unsqueeze(-1)
    cross = torch.cross(incoming_unit, outgoing_unit, dim=-1)
    cosine = (incoming_unit * outgoing_unit).sum(dim=-1).clamp(-1.0, 1.0)
    sine = torch.linalg.vector_norm(cross, dim=-1)
    valid = (incoming_norm > eps) & (outgoing_norm > eps)
    angles = torch.atan2(sine, cosine)
    return angles, valid


def joint_position_loss(
    joints: Tensor,
    constraint: JointPositionConstraint,
    *,
    frame_mask: Tensor | None = None,
    beta: float = 0.05,
) -> Tensor:
    """Smooth-L1 loss of Euclidean position error over constrained cells."""

    if joints.shape != constraint.targets.shape:
        raise ValueError(f"joints and targets must match, got {joints.shape} and {constraint.targets.shape}")
    valid_frames = _as_frame_mask(frame_mask, joints.shape[0], joints.shape[1], joints.device)
    active = constraint.mask.to(joints.device, torch.bool) & valid_frames.unsqueeze(-1)
    delta = joints - constraint.targets.to(joints.device, joints.dtype)
    if constraint.axis_mask is not None:
        axis_mask = constraint.axis_mask.to(joints.device, torch.bool)
        delta = delta * axis_mask.to(delta.dtype)
        active = active & axis_mask.any(dim=-1)
    if not bool(active.any()):
        return _zero_loss(joints)
    distance = torch.linalg.vector_norm(delta, dim=-1)
    errors = F.smooth_l1_loss(distance, torch.zeros_like(distance), beta=beta, reduction="none")
    return _masked_mean(errors, active)


def bend_angle_loss(
    joints: Tensor,
    constraint: BendAngleConstraint,
    *,
    frame_mask: Tensor | None = None,
    beta: float = math.radians(5.0),
    eps: float = 1e-7,
) -> Tensor:
    """Smooth-L1 exact-angle or range-violation loss."""

    angles, valid_bones = _bend_angles_and_validity(joints, constraint.triplets, eps)
    if constraint.mask.shape[:2] != joints.shape[:2]:
        raise ValueError("angle constraint batch/time dimensions must match joints")
    valid_frames = _as_frame_mask(frame_mask, joints.shape[0], joints.shape[1], joints.device)
    requested = constraint.mask.to(joints.device, torch.bool) & valid_frames.unsqueeze(-1)
    if bool((requested & ~valid_bones).any()):
        raise ValueError("active angle constraints contain degenerate bones")
    active = requested
    if not bool(active.any()):
        return _zero_loss(joints)

    if constraint.target_radians is not None:
        target = constraint.target_radians.to(joints.device, joints.dtype)
        violation = (angles - target).abs()
    else:
        assert constraint.min_radians is not None and constraint.max_radians is not None
        lower = constraint.min_radians.to(joints.device, joints.dtype)
        upper = constraint.max_radians.to(joints.device, joints.dtype)
        violation = F.relu(lower - angles) + F.relu(angles - upper)
    errors = F.smooth_l1_loss(violation, torch.zeros_like(violation), beta=beta, reduction="none")
    return _masked_mean(errors, active)


def joint_position_error(
    joints: Tensor,
    constraint: JointPositionConstraint,
    *,
    frame_mask: Tensor | None = None,
) -> Tensor:
    """Mean Euclidean position error in metres over active constraints."""

    if joints.shape != constraint.targets.shape:
        raise ValueError(f"joints and targets must match, got {joints.shape} and {constraint.targets.shape}")
    valid_frames = _as_frame_mask(frame_mask, joints.shape[0], joints.shape[1], joints.device)
    active = constraint.mask.to(joints.device, torch.bool) & valid_frames.unsqueeze(-1)
    delta = joints - constraint.targets.to(joints.device, joints.dtype)
    if constraint.axis_mask is not None:
        axis_mask = constraint.axis_mask.to(joints.device, torch.bool)
        delta = delta * axis_mask.to(delta.dtype)
        active = active & axis_mask.any(dim=-1)
    if not bool(active.any()):
        return _zero_loss(joints)
    return _masked_mean(torch.linalg.vector_norm(delta, dim=-1), active)


def bend_angle_violation(
    joints: Tensor,
    constraint: BendAngleConstraint,
    *,
    frame_mask: Tensor | None = None,
    eps: float = 1e-7,
) -> Tensor:
    """Mean absolute exact/range violation in radians."""

    angles, valid_bones = _bend_angles_and_validity(joints, constraint.triplets, eps)
    valid_frames = _as_frame_mask(frame_mask, joints.shape[0], joints.shape[1], joints.device)
    requested = constraint.mask.to(joints.device, torch.bool) & valid_frames.unsqueeze(-1)
    if bool((requested & ~valid_bones).any()):
        raise ValueError("active angle constraints contain degenerate bones")
    active = requested
    if not bool(active.any()):
        return _zero_loss(joints)
    if constraint.target_radians is not None:
        violation = (angles - constraint.target_radians.to(joints.device, joints.dtype)).abs()
    else:
        assert constraint.min_radians is not None and constraint.max_radians is not None
        lower = constraint.min_radians.to(joints.device, joints.dtype)
        upper = constraint.max_radians.to(joints.device, joints.dtype)
        violation = F.relu(lower - angles) + F.relu(angles - upper)
    return _masked_mean(violation, active)


def decode_latents_to_joints(
    vqvae,
    latents: Tensor,
    *,
    mean: Tensor,
    std: Tensor,
    target_len: int,
    token_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Decode latents into normalized features, raw features, and world joints."""

    if latents.ndim != 3:
        raise ValueError(f"latents must be (B, L, D), got {tuple(latents.shape)}")
    if target_len < 1:
        raise ValueError("target_len must be positive")
    if token_mask is not None:
        if token_mask.shape != latents.shape[:2]:
            raise ValueError(f"token_mask must be {tuple(latents.shape[:2])}")
        latents = latents.masked_fill(~token_mask.to(latents.device, torch.bool).unsqueeze(-1), 0.0)
    normalized = vqvae.decode_latents(latents, target_len=target_len)
    prepared_mean, prepared_std = _prepare_normalization(
        mean,
        std,
        feature_dim=normalized.shape[-1],
        device=normalized.device,
        dtype=normalized.dtype,
    )
    raw = normalized * prepared_std + prepared_mean
    return normalized, raw, recover_joints_from_ric(raw.float())


def _prepare_normalization(
    mean: Tensor,
    std: Tensor,
    *,
    feature_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape or mean.shape[0] != feature_dim:
        raise ValueError("mean/std must be one-dimensional and match the decoded feature dimension")
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()) or bool((std <= 0).any()):
        raise ValueError("mean/std must be finite and std must be strictly positive")
    return mean.to(device=device, dtype=dtype), std.to(device=device, dtype=dtype)


def _decode_latents_to_joints_prepared(
    vqvae,
    latents: Tensor,
    *,
    mean: Tensor,
    std: Tensor,
    target_len: int,
    token_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    latents = latents.masked_fill(~token_mask.unsqueeze(-1), 0.0)
    normalized = vqvae.decode_latents(latents, target_len=target_len)
    raw = normalized * std + mean
    return normalized, raw, recover_joints_from_ric(raw.float())


def _dynamics_preservation_loss(joints: Tensor, reference: Tensor, frame_mask: Tensor) -> Tensor:
    if joints.shape[1] < 2:
        return _zero_loss(joints)
    current_local = joints - joints[:, :, :1]
    reference_local = reference - reference[:, :, :1]
    current_velocity = current_local[:, 1:] - current_local[:, :-1]
    reference_velocity = reference_local[:, 1:] - reference_local[:, :-1]
    valid = frame_mask[:, 1:] & frame_mask[:, :-1]
    error = (current_velocity - reference_velocity).pow(2).sum(dim=-1)
    return _masked_mean(error, valid.unsqueeze(-1).expand_as(error))


def _root_trajectory_preservation_loss(joints: Tensor, reference: Tensor, frame_mask: Tensor) -> Tensor:
    error = (joints[:, :, 0] - reference[:, :, 0]).pow(2).sum(dim=-1)
    return _masked_mean(error, frame_mask)


def _bone_length_preservation_loss(joints: Tensor, reference: Tensor, frame_mask: Tensor) -> Tensor:
    children = torch.arange(1, NUM_JOINTS, device=joints.device)
    parents = torch.tensor(PARENTS[1:], device=joints.device)
    current = torch.linalg.vector_norm(joints[:, :, children] - joints[:, :, parents], dim=-1)
    original = torch.linalg.vector_norm(reference[:, :, children] - reference[:, :, parents], dim=-1)
    error = (current - original).pow(2)
    return _masked_mean(error, frame_mask.unsqueeze(-1).expand_as(error))


def _foot_skate_loss(joints: Tensor, frame_mask: Tensor, height: float) -> Tensor:
    if joints.shape[1] < 2:
        return _zero_loss(joints)
    feet = joints[:, :, list(FOOT_CONTACT_IDX)]
    horizontal = feet[:, 1:, :, [0, 2]] - feet[:, :-1, :, [0, 2]]
    speed = torch.linalg.vector_norm(horizontal, dim=-1)
    gate = (1.0 - feet[:, :-1, :, 1] / height).clamp(0.0, 1.0)
    valid = (frame_mask[:, 1:] & frame_mask[:, :-1]).unsqueeze(-1).expand_as(speed)
    return _masked_mean(speed * gate, valid)


def _jerk_loss(joints: Tensor, frame_mask: Tensor) -> Tensor:
    if joints.shape[1] < 4:
        return _zero_loss(joints)
    jerk = joints[:, 3:] - 3 * joints[:, 2:-1] + 3 * joints[:, 1:-2] - joints[:, :-3]
    magnitude = torch.linalg.vector_norm(jerk, dim=-1)
    valid = frame_mask[:, 3:] & frame_mask[:, 2:-1] & frame_mask[:, 1:-2] & frame_mask[:, :-3]
    return _masked_mean(magnitude, valid.unsqueeze(-1).expand_as(magnitude))


@dataclass(frozen=True)
class LatentRefinementConfig:
    """Optimization weights and safeguards.

    ``history_interval=0`` disables per-step CPU synchronization. Positive
    values record the first, final, and every Nth pre-update loss.
    """

    steps: int = 50
    learning_rate: float = 0.01
    position_weight: float = 1.0
    angle_weight: float = 1.0
    latent_weight: float = 0.01
    dynamics_weight: float = 0.1
    root_weight: float = 0.1
    bone_weight: float = 0.1
    foot_skate_weight: float = 0.0
    jerk_weight: float = 0.0
    position_huber_beta: float = 0.05
    angle_huber_beta: float = math.radians(5.0)
    foot_height: float = 0.05
    max_delta_norm: float | None = 1.0
    grad_clip_norm: float | None = 1.0
    history_interval: int = 0

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        for name in (
            "position_weight",
            "angle_weight",
            "latent_weight",
            "dynamics_weight",
            "root_weight",
            "bone_weight",
            "foot_skate_weight",
            "jerk_weight",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.position_huber_beta <= 0.0 or self.angle_huber_beta <= 0.0:
            raise ValueError("Huber beta values must be positive")
        if self.foot_height <= 0.0:
            raise ValueError("foot_height must be positive")
        if self.max_delta_norm is not None and self.max_delta_norm <= 0.0:
            raise ValueError("max_delta_norm must be positive when supplied")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive when supplied")
        if self.history_interval < 0:
            raise ValueError("history_interval must be non-negative")


@dataclass
class LatentRefinementResult:
    latents: Tensor
    normalized_motion: Tensor
    motion: Tensor
    joints: Tensor
    initial_motion: Tensor
    initial_joints: Tensor
    metrics: dict[str, float]
    history: list[dict[str, float]]
    tokens: Tensor | None = None


def refine_motion_latents(
    vqvae,
    initial_latents: Tensor,
    *,
    mean: Tensor,
    std: Tensor,
    target_len: int,
    token_mask: Tensor | None = None,
    frame_mask: Tensor | None = None,
    position_constraint: JointPositionConstraint | None = None,
    angle_constraint: BendAngleConstraint | None = None,
    config: LatentRefinementConfig | None = None,
) -> LatentRefinementResult:
    """Refine continuous RVQ latents while leaving all model weights frozen."""

    if position_constraint is None and angle_constraint is None:
        raise ValueError("at least one position or angle constraint is required")
    config = config or LatentRefinementConfig()
    if initial_latents.ndim != 3 or not initial_latents.is_floating_point():
        raise ValueError("initial_latents must be a floating-point (B, L, D) tensor")
    z0 = initial_latents.detach()
    batch, latent_len, _ = z0.shape
    token_valid = (
        torch.ones(batch, latent_len, dtype=torch.bool, device=z0.device)
        if token_mask is None
        else token_mask.to(device=z0.device, dtype=torch.bool)
    )
    if token_valid.shape != (batch, latent_len):
        raise ValueError(f"token_mask must be {(batch, latent_len)}, got {tuple(token_valid.shape)}")
    if not bool(token_valid.any(dim=1).all()):
        raise ValueError("every sample must contain at least one valid latent position")

    original_mode = bool(vqvae.training)
    vqvae.eval()
    try:
        with torch.no_grad():
            natural_normalized = vqvae.decode_latents(z0.masked_fill(~token_valid.unsqueeze(-1), 0.0))
            natural_len = natural_normalized.shape[1]
            prepared_mean, prepared_std = _prepare_normalization(
                mean,
                std,
                feature_dim=natural_normalized.shape[-1],
                device=z0.device,
                dtype=natural_normalized.dtype,
            )
            del natural_normalized
            valid_frames = _as_frame_mask(frame_mask, batch, target_len, z0.device)
            token_lengths = token_valid.sum(dim=1)
            prefix_mask = torch.arange(latent_len, device=z0.device).unsqueeze(0) < token_lengths.unsqueeze(1)
            if not bool((token_valid == prefix_mask).all()):
                raise ValueError("token_mask must contain one contiguous valid prefix per sample")
            if natural_len % latent_len != 0:
                raise ValueError("decoder output length must be an integer multiple of latent length")
            frames_per_token = natural_len // latent_len
            decoded_lengths = token_lengths * frames_per_token
            valid_frames = valid_frames & (
                torch.arange(target_len, device=z0.device).unsqueeze(0) < decoded_lengths.unsqueeze(1)
            )

            if position_constraint is not None:
                position_constraint = position_constraint.to(z0.device, z0.dtype)
                if position_constraint.targets.shape[:2] != (batch, target_len):
                    raise ValueError("position constraint batch/time dimensions must match target_len")
                impossible = position_constraint.mask & ~valid_frames.unsqueeze(-1)
                if bool(impossible.any()):
                    raise ValueError("position constraints target padded or invalid decoded frames")
                root_frame_zero = position_constraint.mask[:, 0, 0]
                root_xz_axes = (
                    torch.ones(batch, 2, dtype=torch.bool, device=z0.device)
                    if position_constraint.axis_mask is None
                    else position_constraint.axis_mask[:, 0, 0][:, [0, 2]]
                )
                nonzero_root_xz = position_constraint.targets[:, 0, 0][:, [0, 2]].abs() > 1e-6
                if bool((root_frame_zero.unsqueeze(-1) & root_xz_axes & nonzero_root_xz).any()):
                    raise ValueError(
                        "frame-0 root XZ is fixed at the canonical origin; apply an external scene transform"
                    )
            if angle_constraint is not None:
                angle_constraint = angle_constraint.to(z0.device, z0.dtype)
                if angle_constraint.mask.shape[:2] != (batch, target_len):
                    raise ValueError("angle constraint batch/time dimensions must match target_len")
                impossible = angle_constraint.mask & ~valid_frames.unsqueeze(-1)
                if bool(impossible.any()):
                    raise ValueError("angle constraints target padded or invalid decoded frames")

            active_position = False
            if position_constraint is not None:
                position_active = position_constraint.mask & valid_frames.unsqueeze(-1)
                if position_constraint.axis_mask is not None:
                    position_active = position_active & position_constraint.axis_mask.any(dim=-1)
                active_position = bool(position_active.any())
            active_angle = bool(
                angle_constraint is not None
                and (angle_constraint.mask & valid_frames.unsqueeze(-1)).any()
            )
            weighted_position = active_position and config.position_weight > 0.0
            weighted_angle = active_angle and config.angle_weight > 0.0
            if not weighted_position and not weighted_angle:
                raise ValueError("no positively weighted constraints remain after applying masks")

            _, initial_motion, initial_joints = _decode_latents_to_joints_prepared(
                vqvae,
                z0,
                mean=prepared_mean,
                std=prepared_std,
                target_len=target_len,
                token_mask=token_valid,
            )
            if angle_constraint is not None:
                initial_angles, angle_valid = _bend_angles_and_validity(
                    initial_joints,
                    angle_constraint.triplets,
                    1e-7,
                )
                invalid_angles = angle_constraint.mask & valid_frames.unsqueeze(-1) & ~angle_valid
                if bool(invalid_angles.any()):
                    raise ValueError("active angle constraints contain degenerate bones or bend axes")
                collinear = (initial_angles.abs() < 1e-5) | (
                    (math.pi - initial_angles.abs()).abs() < 1e-5
                )
                if angle_constraint.target_radians is not None:
                    initially_violated = (
                        initial_angles - angle_constraint.target_radians
                    ).abs() > 1e-5
                else:
                    assert angle_constraint.min_radians is not None
                    assert angle_constraint.max_radians is not None
                    initially_violated = (initial_angles < angle_constraint.min_radians) | (
                        initial_angles > angle_constraint.max_radians
                    )
                active_collinear = (
                    collinear
                    & initially_violated
                    & angle_constraint.mask
                    & valid_frames.unsqueeze(-1)
                )
                if bool(active_collinear.any()):
                    warnings.warn(
                        "an active bend constraint starts exactly collinear and has no unique "
                        "bend direction; latent angle guidance may stall without a directional "
                        "or position constraint",
                        UserWarning,
                        stacklevel=2,
                    )

        # Adam moments are unstable in low precision. Keep the optimized
        # perturbation in FP32 and cast it only at the decoder boundary.
        delta = torch.nn.Parameter(torch.zeros(z0.shape, device=z0.device, dtype=torch.float32))
        optimizer = torch.optim.Adam([delta], lr=config.learning_rate)
        history: list[dict[str, float]] = []

        with torch.enable_grad():
            for step_index in range(config.steps):
                optimizer.zero_grad(set_to_none=True)
                masked_delta = delta * token_valid.unsqueeze(-1).to(delta.dtype)
                latents = z0 + masked_delta.to(z0.dtype)
                normalized, motion, joints = _decode_latents_to_joints_prepared(
                    vqvae,
                    latents,
                    mean=prepared_mean,
                    std=prepared_std,
                    target_len=target_len,
                    token_mask=token_valid,
                )

                losses: dict[str, Tensor] = {}
                if position_constraint is not None and config.position_weight > 0.0:
                    losses["position"] = joint_position_loss(
                        joints,
                        position_constraint,
                        frame_mask=valid_frames,
                        beta=config.position_huber_beta,
                    )
                if angle_constraint is not None and config.angle_weight > 0.0:
                    losses["angle"] = bend_angle_loss(
                        joints,
                        angle_constraint,
                        frame_mask=valid_frames,
                        beta=config.angle_huber_beta,
                    )
                if config.latent_weight > 0.0:
                    latent_error = masked_delta.pow(2).sum(dim=-1)
                    losses["latent"] = _masked_mean(latent_error, token_valid)
                if config.dynamics_weight > 0.0:
                    losses["dynamics"] = _dynamics_preservation_loss(
                        joints, initial_joints, valid_frames
                    )
                if config.root_weight > 0.0:
                    losses["root"] = _root_trajectory_preservation_loss(
                        joints, initial_joints, valid_frames
                    )
                if config.bone_weight > 0.0:
                    losses["bone"] = _bone_length_preservation_loss(
                        joints, initial_joints, valid_frames
                    )
                if config.foot_skate_weight > 0.0:
                    losses["foot_skate"] = _foot_skate_loss(
                        joints, valid_frames, config.foot_height
                    )
                if config.jerk_weight > 0.0:
                    losses["jerk"] = _jerk_loss(joints, valid_frames)

                zero = _zero_loss(joints)
                total = (
                    config.position_weight * losses.get("position", zero)
                    + config.angle_weight * losses.get("angle", zero)
                    + config.latent_weight * losses.get("latent", zero)
                    + config.dynamics_weight * losses.get("dynamics", zero)
                    + config.root_weight * losses.get("root", zero)
                    + config.bone_weight * losses.get("bone", zero)
                    + config.foot_skate_weight * losses.get("foot_skate", zero)
                    + config.jerk_weight * losses.get("jerk", zero)
                )
                if not bool(torch.isfinite(total)):
                    raise FloatingPointError("non-finite loss during MoMask latent refinement")
                (gradient,) = torch.autograd.grad(total, delta)
                if not bool(torch.isfinite(gradient).all()):
                    raise FloatingPointError("non-finite gradient during MoMask latent refinement")
                delta.grad = gradient
                if config.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_([delta], config.grad_clip_norm)
                optimizer.step()

                with torch.no_grad():
                    delta.mul_(token_valid.unsqueeze(-1))
                    if config.max_delta_norm is not None:
                        norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
                        scale = (config.max_delta_norm / norm.clamp_min(1e-12)).clamp(max=1.0)
                        delta.mul_(scale)
                    should_record = config.history_interval > 0 and (
                        step_index == 0
                        or (step_index + 1) % config.history_interval == 0
                        or step_index + 1 == config.steps
                    )
                    if should_record:
                        record = {
                            name: float(value.detach().cpu()) for name, value in losses.items()
                        }
                        record["total"] = float(total.detach().cpu())
                        history.append(record)

        with torch.no_grad():
            final_latents = z0 + (
                delta * token_valid.unsqueeze(-1).to(delta.dtype)
            ).to(z0.dtype)
            normalized, motion, joints = _decode_latents_to_joints_prepared(
                vqvae,
                final_latents,
                mean=prepared_mean,
                std=prepared_std,
                target_len=target_len,
                token_mask=token_valid,
            )
            metrics: dict[str, float] = {}
            if position_constraint is not None and config.position_weight > 0.0:
                metrics["position_error_initial_m"] = float(
                    joint_position_error(
                        initial_joints, position_constraint, frame_mask=valid_frames
                    ).cpu()
                )
                metrics["position_error_final_m"] = float(
                    joint_position_error(joints, position_constraint, frame_mask=valid_frames).cpu()
                )
            if angle_constraint is not None and config.angle_weight > 0.0:
                metrics["angle_violation_initial_rad"] = float(
                    bend_angle_violation(
                        initial_joints, angle_constraint, frame_mask=valid_frames
                    ).cpu()
                )
                metrics["angle_violation_final_rad"] = float(
                    bend_angle_violation(joints, angle_constraint, frame_mask=valid_frames).cpu()
                )
            valid_delta = delta[token_valid]
            metrics["latent_delta_rms"] = float(valid_delta.pow(2).mean().sqrt().cpu())
        return LatentRefinementResult(
            latents=final_latents.detach(),
            normalized_motion=normalized.detach(),
            motion=motion.detach(),
            joints=joints.detach(),
            initial_motion=initial_motion.detach(),
            initial_joints=initial_joints.detach(),
            metrics=metrics,
            history=history,
        )
    finally:
        vqvae.train(original_mode)
