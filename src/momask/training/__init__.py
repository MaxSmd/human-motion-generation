"""Small training-step helpers for MoMask stages."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from momask.models import MaskedMotionTransformer, MotionRVQVAE, ResidualTransformer


@dataclass
class VQTrainerCfg:
    grad_clip: float = 1.0


@dataclass
class MaskedTrainerCfg:
    cond_drop_prob: float = 0.1
    grad_clip: float = 1.0


@dataclass
class ResidualTrainerCfg:
    cond_drop_prob: float = 0.2
    grad_clip: float = 1.0


def train_vq_step(
    model: MotionRVQVAE,
    optimizer: torch.optim.Optimizer,
    motion_263: Tensor,
    mask: Tensor | None = None,
    cfg: VQTrainerCfg | None = None,
) -> dict[str, float]:
    cfg = cfg or VQTrainerCfg()
    out = model(motion_263, mask=mask)
    optimizer.zero_grad(set_to_none=True)
    out.loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    return {
        "loss": float(out.loss.detach()),
        "recon_loss": float(out.recon_loss.detach()),
        "vq_loss": float(out.vq_loss.detach()),
        "perplexity": float(out.perplexity.detach()),
        "grad_norm": float(grad_norm),
    }


def train_masked_step(
    model: MaskedMotionTransformer,
    optimizer: torch.optim.Optimizer,
    base_tokens: Tensor,
    cond: Tensor,
    mask: Tensor | None = None,
    cfg: MaskedTrainerCfg | None = None,
) -> dict[str, float]:
    cfg = cfg or MaskedTrainerCfg()
    loss = model.training_loss(base_tokens, cond=cond, valid_mask=mask, cond_drop_prob=cfg.cond_drop_prob)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    return {"loss": float(loss.detach()), "grad_norm": float(grad_norm)}


def train_residual_step(
    model: ResidualTransformer,
    optimizer: torch.optim.Optimizer,
    tokens: Tensor,
    target_level: int,
    cond: Tensor,
    mask: Tensor | None = None,
    cfg: ResidualTrainerCfg | None = None,
) -> dict[str, float]:
    cfg = cfg or ResidualTrainerCfg()
    loss = model.training_loss(
        tokens,
        target_level=target_level,
        cond=cond,
        valid_mask=mask,
        cond_drop_prob=cfg.cond_drop_prob,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    return {"loss": float(loss.detach()), "grad_norm": float(grad_norm)}


__all__ = [
    "VQTrainerCfg",
    "MaskedTrainerCfg",
    "ResidualTrainerCfg",
    "train_vq_step",
    "train_masked_step",
    "train_residual_step",
]
