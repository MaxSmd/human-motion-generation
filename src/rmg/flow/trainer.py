"""Riemannian conditional flow-matching loss (paper eq. 4).

Model contract:
    model(x_t, t, *, cond=None, drop_cond_mask=None, mask=None) -> ambient_velocity
where:
    x_t: (B, T, D)
    t:   (B,)
    cond: optional conditioning (text features, etc.) — model-specific shape
    drop_cond_mask: (B,) bool tensor; True ⇒ use unconditional embedding for that sample
    mask: (B, T) bool tensor; True = valid frame
The returned velocity may be in the ambient space; the trainer projects it to
T_{x_t}M before computing the loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..manifolds import Manifold
from .interpolation import build_cfm_batch, sample_t
from .prior import WrappedGaussianPrior


@dataclass
class FlowMatchingTrainerCfg:
    cfg_dropout: float = 0.1  # P(replace cond with null) — paper §4.1
    t_eps: float = 1e-3       # keep t away from {0, 1}


class FlowMatchingTrainer:
    """Stateless loss helper. Holds manifold + prior + config; called per step."""

    def __init__(
        self,
        manifold: Manifold,
        prior: WrappedGaussianPrior,
        cfg: FlowMatchingTrainerCfg | None = None,
    ) -> None:
        self.manifold = manifold
        self.prior = prior
        self.cfg = cfg or FlowMatchingTrainerCfg()

    def compute_loss(
        self,
        model: nn.Module,
        x1: Tensor,
        cond: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Riemannian CFM loss for one batch.

        Args:
            x1: (B, T, D) ground-truth motion on the manifold.
            cond: optional conditioning (e.g. text features).
            mask: (B, T) bool, valid-frame mask.

        Returns:
            loss: scalar tensor.
            info: dict with diagnostics (e.g. per-step squared error).
        """
        B, T, D = x1.shape
        device, dtype = x1.device, x1.dtype

        # 1) sample t ~ U[ε, 1-ε], one per sample
        t = sample_t(B, device=device, dtype=dtype, eps=self.cfg.t_eps)
        # 2) sample x_0 from the manifold prior, same (B, T) layout as x_1
        x0 = self.prior.sample((B, T), device=device, dtype=dtype)

        # 3) build the geodesic interpolation and target velocity
        batch = build_cfm_batch(self.manifold, x0, x1, t)

        # 4) classifier-free dropout — per-sample binary mask
        if cond is not None and self.cfg.cfg_dropout > 0.0:
            drop = torch.rand(B, device=device) < self.cfg.cfg_dropout
        else:
            drop = torch.zeros(B, dtype=torch.bool, device=device)

        # 5) forward + tangent projection
        v_pred = model(batch.x_t, t, cond=cond, drop_cond_mask=drop, mask=mask)
        v_pred = self.manifold.project_tangent(batch.x_t, v_pred)

        # 6) MSE in the tangent metric (= ambient L2, since metric is induced)
        diff = v_pred - batch.target  # (B, T, D)
        sq = (diff * diff).sum(dim=-1)  # (B, T)

        if mask is not None:
            mask_f = mask.to(sq.dtype)
            denom = mask_f.sum().clamp_min(1.0)
            loss = (sq * mask_f).sum() / denom
        else:
            loss = sq.mean()

        info = {
            "loss": loss.detach(),
            "t_mean": t.mean().detach(),
            "drop_frac": drop.float().mean().detach(),
            "x_t_offmanifold": (~self.manifold.validate(batch.x_t, atol=1e-3)).float().mean().detach(),
        }
        return loss, info
