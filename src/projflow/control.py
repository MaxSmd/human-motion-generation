"""Spatial control with ProjFlow: keyframe sampling + text-conditioned generation.

Mirrors upstream `ACMDM.generate_control` (OmniControl protocol): for every
sample, `density` of its valid frames are drawn uniformly without replacement
and all channels of the controlled joints are observed there. Upstream reads
density 1, 2 or 5 as a keyframe count and anything else as a percentage of the
clip, so 25 / 100 are the paper's "49 / 196 keyframes" on 196-frame clips.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from projflow.models.acmdm import ACMDM, attention_mask_from_lengths
from projflow.sampler import ProjFlowConfig, projflow_sample
from projflow.text import ClipTextEncoder

KEYFRAME_COUNTS = (1, 2, 5)


def keyframes_for_length(length: int, density: int) -> int:
    if density in KEYFRAME_COUNTS:
        return density
    return max(1, int(length * float(density) / 100.0))


def sample_keyframe_mask(lengths: Tensor, joints: Sequence[int], density: int, max_len: int,
                         channels: int = 3) -> Tensor:
    """(B, channels, max_len, 22) 0/1 mask; per sample, keyframes within its valid length.

    Draws with the global RNG on `lengths.device`, like upstream (one randperm per sample).
    """
    B = lengths.shape[0]
    mask = torch.zeros(B, channels, max_len, 22, device=lengths.device)
    joint_idx = torch.as_tensor(sorted(set(joints)), device=lengths.device, dtype=torch.long)
    for b in range(B):
        pool = int(lengths[b])
        k = keyframes_for_length(pool, density)
        frames = torch.sort(torch.randperm(pool, device=lengths.device)[:k]).values
        mask[b, :, frames.unsqueeze(-1), joint_idx] = 1.0
    return mask


@torch.no_grad()
def generate_control(
    model: ACMDM,
    text_encoder: ClipTextEncoder,
    texts: list[str],
    lengths: Tensor,
    control: Tensor,
    joints: Sequence[int],
    density: int,
    cfg_scale: float = 3.0,
    pf: ProjFlowConfig = ProjFlowConfig(),
) -> tuple[Tensor, Tensor]:
    """Generate motions that pass exactly through GT joint positions at keyframes.

    Args:
        control: (B, 3, L, 22) normalised GT motion; only masked cells are read.
        lengths: (B,) valid frames per sample (≤ L).
    Returns:
        (samples, mask), both (B, 3, L, 22) in the normalised space; frames past
        each length are zero (as upstream).
    """
    B, D, L, J = control.shape
    device = control.device
    cond = text_encoder(texts).to(device)
    noise = torch.randn(B, D, L, J, device=device)             # before the mask draws, as upstream
    mask = sample_keyframe_mask(lengths, joints, density, L, channels=D)
    attn = attention_mask_from_lengths(lengths.to(device), L)

    def velocity(x: Tensor, t: Tensor) -> Tensor:
        return model.guided_velocity(x, t, cond, attn, cfg_scale)

    x = projflow_sample(velocity, noise, mask, control * mask, pf)
    valid = torch.arange(L, device=device)[None] < lengths.to(device)[:, None]
    return x * valid[:, None, :, None], mask
