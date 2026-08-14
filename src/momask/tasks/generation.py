"""MoMask text-to-motion sampling helpers."""

from __future__ import annotations

import torch
from torch import Tensor

from momask.constraints import (
    BendAngleConstraint,
    JointPositionConstraint,
    LatentRefinementConfig,
    LatentRefinementResult,
    refine_motion_latents,
)
from momask.models import (
    CodebookResidualTransformer,
    MaskedMotionTransformer,
    MotionRVQVAE,
    ResidualTransformer,
)

ResidualGenerator = ResidualTransformer | CodebookResidualTransformer


@torch.no_grad()
def _generate_tokens(
    *,
    masked_transformer: MaskedMotionTransformer,
    residual_transformer: ResidualGenerator,
    cond: Tensor,
    seq_len: int,
    steps: int = 10,
    guidance_scale: float = 4.0,
    temperature: float = 1.0,
    topk_filter_thres: float = 1.0,
    sample: bool = False,
    remask_kept_tokens: bool = True,
    token_mask: Tensor | None = None,
) -> Tensor:
    base = masked_transformer.generate(
        cond=cond,
        seq_len=seq_len,
        steps=steps,
        guidance_scale=guidance_scale,
        temperature=temperature,
        topk_filter_thres=topk_filter_thres,
        sample=sample,
        remask_kept_tokens=remask_kept_tokens,
        mask=token_mask,
    )
    return residual_transformer.generate_residuals(
        base,
        cond=cond,
        guidance_scale=guidance_scale,
        temperature=temperature,
        topk_filter_thres=topk_filter_thres,
        sample=sample,
        mask=token_mask,
    )


@torch.no_grad()
def generate_h3d263(
    *,
    vqvae: MotionRVQVAE,
    masked_transformer: MaskedMotionTransformer,
    residual_transformer: ResidualGenerator,
    cond: Tensor,
    seq_len: int,
    steps: int = 10,
    guidance_scale: float = 4.0,
    temperature: float = 1.0,
    topk_filter_thres: float = 1.0,
    sample: bool = False,
    remask_kept_tokens: bool = True,
    token_mask: Tensor | None = None,
    target_len: int | None = None,
) -> Tensor:
    """Generate model-space 263-D HumanML3D features from text embeddings.

    A normalized checkpoint returns normalized features. Callers that need raw
    HumanML3D features must apply the checkpoint normalizer's inverse transform.
    """
    tokens = _generate_tokens(
        masked_transformer=masked_transformer,
        residual_transformer=residual_transformer,
        cond=cond,
        seq_len=seq_len,
        steps=steps,
        guidance_scale=guidance_scale,
        temperature=temperature,
        topk_filter_thres=topk_filter_thres,
        sample=sample,
        remask_kept_tokens=remask_kept_tokens,
        token_mask=token_mask,
    )
    return vqvae.decode_from_tokens(tokens, target_len=target_len, token_mask=token_mask)


def generate_h3d263_constrained(
    *,
    vqvae: MotionRVQVAE,
    masked_transformer: MaskedMotionTransformer,
    residual_transformer: ResidualGenerator,
    cond: Tensor,
    seq_len: int,
    mean: Tensor,
    std: Tensor,
    position_constraint: JointPositionConstraint | None = None,
    angle_constraint: BendAngleConstraint | None = None,
    refinement: LatentRefinementConfig | None = None,
    steps: int = 10,
    guidance_scale: float = 4.0,
    temperature: float = 1.0,
    topk_filter_thres: float = 1.0,
    sample: bool = False,
    remask_kept_tokens: bool = True,
    token_mask: Tensor | None = None,
    frame_mask: Tensor | None = None,
    target_len: int | None = None,
) -> LatentRefinementResult:
    """Generate tokens, then refine their continuous latent under constraints.

    ``mean`` and ``std`` must come from the same checkpoint as ``vqvae``. The
    returned ``motion`` is inverse-normalized raw HumanML3D data, while
    ``normalized_motion`` remains in model space.
    """

    tokens = _generate_tokens(
        masked_transformer=masked_transformer,
        residual_transformer=residual_transformer,
        cond=cond,
        seq_len=seq_len,
        steps=steps,
        guidance_scale=guidance_scale,
        temperature=temperature,
        topk_filter_thres=topk_filter_thres,
        sample=sample,
        remask_kept_tokens=remask_kept_tokens,
        token_mask=token_mask,
    )
    with torch.no_grad():
        initial_latents = vqvae.quantizer.decode(tokens)
    result = refine_motion_latents(
        vqvae,
        initial_latents,
        mean=mean,
        std=std,
        target_len=target_len if target_len is not None else seq_len * vqvae.downsample,
        token_mask=token_mask,
        frame_mask=frame_mask,
        position_constraint=position_constraint,
        angle_constraint=angle_constraint,
        config=refinement,
    )
    result.tokens = tokens.detach()
    return result
