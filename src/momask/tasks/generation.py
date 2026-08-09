"""MoMask text-to-motion sampling helpers."""

from __future__ import annotations

import torch
from torch import Tensor

from momask.models import MaskedMotionTransformer, MotionRVQVAE, ResidualTransformer


@torch.no_grad()
def generate_h3d263(
    *,
    vqvae: MotionRVQVAE,
    masked_transformer: MaskedMotionTransformer,
    residual_transformer: ResidualTransformer,
    cond: Tensor,
    seq_len: int,
    steps: int = 10,
    guidance_scale: float = 4.0,
    temperature: float = 1.0,
    token_mask: Tensor | None = None,
    target_len: int | None = None,
) -> Tensor:
    """Generate 263-D HumanML3D features from text embeddings.

    `cond` is expected to come from the shared text encoders in `rmg.models`.
    The output can be passed directly to the shared Guo evaluator.
    """
    base = masked_transformer.generate(
        cond=cond,
        seq_len=seq_len,
        steps=steps,
        guidance_scale=guidance_scale,
        temperature=temperature,
        mask=token_mask,
    )
    tokens = residual_transformer.generate_residuals(
        base,
        cond=cond,
        guidance_scale=guidance_scale,
        mask=token_mask,
    )
    return vqvae.decode_from_tokens(tokens, target_len=target_len, token_mask=token_mask)
