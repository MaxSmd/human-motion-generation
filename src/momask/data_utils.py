"""Mask-aware helpers shared by MoMask training and evaluation."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor


class _Normalizer(Protocol):
    def transform(self, x: Tensor) -> Tensor: ...


def normalize_motion(x: Tensor, mask: Tensor, normalizer: _Normalizer) -> Tensor:
    """Normalize valid frames and keep batch padding at zero in model space."""
    if x.shape[:2] != mask.shape:
        raise ValueError(f"motion/mask shapes do not match: {tuple(x.shape)} vs {tuple(mask.shape)}")
    normalized = normalizer.transform(x)
    return normalized.masked_fill(~mask.bool().unsqueeze(-1), 0.0)


def token_mask_from_frame_mask(mask: Tensor, token_len: int, downsample: int) -> Tensor:
    """Convert prefix frame masks to exact convolutional latent lengths.

    MoMask's temporal encoder downsamples by powers of two. Its stride-2,
    kernel-4 convolutions produce ``floor(length / 2)`` valid positions per
    stage, so the final valid token count is ``floor(length / downsample)``.
    """
    if mask.ndim != 2:
        raise ValueError(f"frame mask must be (B, T), got {tuple(mask.shape)}")
    if token_len < 0:
        raise ValueError("token_len must be non-negative")
    if downsample < 1 or downsample & (downsample - 1):
        raise ValueError("downsample must be a positive power of two")
    frame_lengths = mask.long().sum(dim=1)
    token_lengths = torch.div(frame_lengths, downsample, rounding_mode="floor").clamp(max=token_len)
    positions = torch.arange(token_len, device=mask.device).unsqueeze(0)
    return positions < token_lengths.unsqueeze(1)
