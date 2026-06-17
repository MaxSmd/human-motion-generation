"""MARDM generation task: text -> sampled motion -> HumanML3D 263-D features.

Orchestrates the trained generation branch + frozen AE + text encoder into the
sampling pipeline, plus the eval bridge to 263-D. Kept separate from the model
(`mardm.models`) and training (`mardm.scripts.train_mardm`) per the repo's
models / training / tasks split; `mardm.scripts.evaluate_mardm` is a thin wrapper.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from shared.geometry import Skeleton

from ..models import AE, MARDM
from ..representation import essential_to_h3d


@torch.no_grad()
def sample_latents(mardm: MARDM, cond: Tensor, latent_lens: Tensor, *,
                   timesteps: int, cond_scale: float) -> Tensor:
    """cond: (B, text_dim); latent_lens: (B,) AE-latent lengths -> latents (B, ae_dim, L)."""
    return mardm.generate(cond, m_lens=latent_lens, timesteps=timesteps, cond_scale=cond_scale)


@torch.no_grad()
def generate_h3d_features(
    mardm: MARDM,
    ae: AE,
    text_encoder,
    texts: list[str],
    lengths: Tensor,
    *,
    guidance: float,
    timesteps: int,
    mean: Tensor,
    std: Tensor,
    skeleton: Skeleton,
    device: torch.device,
    humanml3d_repo: str | Path = "external/HumanML3D",
    use_upstream: bool = False,
) -> tuple[list[Tensor], Tensor]:
    """Full text -> per-sample 263-D HumanML3D features (+ their lengths).

    `lengths` are target *essential* feature lengths (one per caption). Sampling
    runs in AE-latent space (length // downsample), so the decoded essential
    length is `(L // ds) * ds`; the 263-D feature is one frame shorter again
    (velocity diff). Returns (feats, feat_lengths) ready for rmg's Guo evaluator.
    """
    ds = ae.downsample_rate
    latent_lens = (lengths.to(device) // ds).clamp(min=1)
    cond = text_encoder.encode(list(texts), device=device)
    latents = sample_latents(mardm, cond, latent_lens, timesteps=timesteps, cond_scale=guidance)
    essential = ae.decode(latents)  # (B, Lmax*ds, 67), z-normalized

    feats: list[Tensor] = []
    feat_lengths: list[int] = []
    for i in range(len(texts)):
        dec_len = int(latent_lens[i]) * ds
        ess_i = essential[i, :dec_len]                       # (dec_len, 67) normalized
        f = essential_to_h3d(ess_i, skeleton, mean=mean, std=std,
                             humanml3d_repo=humanml3d_repo, use_upstream=use_upstream)
        feats.append(f)                                      # (dec_len - 1, 263)
        feat_lengths.append(f.shape[0])
    return feats, torch.tensor(feat_lengths, dtype=torch.long)
