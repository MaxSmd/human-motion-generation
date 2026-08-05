"""MARDM masked-autoregressive generation branch.

Adapted from MARDM (neu-vi/MARDM, ``models/MARDM.py``; see the upstream
LICENSE). Changes for this repo:
  * text conditioning is a precomputed feature tensor passed in (CLIP removed —
    we use `rmg`'s text encoders), so `cond` has shape (B, text_dim);
  * timm's `Mlp` is replaced by a plain GELU MLP;
  * action mode, the length estimator, and zero-shot `edit()` are dropped;
  * the diffusion head is SiT-only (`mardm.models.diffmlps.DiffMLPs`).

A single AdaLN transformer over the latent-token sequence produces a per-token
condition `z`; the per-token diffusion head denoises masked latents from `z`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..masking import (
    cosine_schedule,
    eval_decorator,
    get_mask_subset_prob,
    lengths_to_mask,
    uniform,
)
from .diffmlps import DiffMLPs


@dataclass
class MARDMConfig:
    ae_dim: int = 512            # AE latent channel dim (== AE.output_emb_width)
    text_dim: int = 1024         # text-encoder feature dim (Qwen3-Embedding-0.6B)
    latent_dim: int = 384        # MAR transformer width
    ff_size: int = 1024
    num_layers: int = 2
    num_heads: int = 6
    dropout: float = 0.2
    diffmlps_width: int = 512
    diffmlps_depth: int = 6
    diffmlps_batch_mul: int = 4  # repeat factor for the per-token diffusion loss
    cond_drop_prob: float = 0.1  # classifier-free guidance dropout


def modulate_here(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Mlp(nn.Module):
    """Plain replacement for timm's vision-transformer MLP."""

    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class InputProcess(nn.Module):
    def __init__(self, input_feats: int, latent_dim: int):
        super().__init__()
        self.poseEmbedding = nn.Linear(input_feats, latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(1, 0, 2)        # (B, L, C) -> (L, B, C)
        return self.poseEmbedding(x)  # (L, B, latent_dim)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, embed_dim: int = 512, n_head: int = 8, drop_out_rate: float = 0.2):
        super().__init__()
        assert embed_dim % 8 == 0
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_head = n_head

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            att = att.masked_fill(mask[:, None, None, :] != 0, float("-inf"))
        att = self.attn_drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MARTransBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_size: int = 1024, drop_out: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, drop_out_rate=drop_out)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, mlp_size, drop=0.0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x: Tensor, c: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate_here(self.norm1(x), shift_msa, scale_msa), mask=padding_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate_here(self.norm2(x), shift_mlp, scale_mlp))
        return x


class MARDM(nn.Module):
    def __init__(self, cfg: MARDMConfig | None = None):
        super().__init__()
        cfg = cfg or MARDMConfig()
        self.cfg = cfg
        self.ae_dim = cfg.ae_dim
        self.latent_dim = cfg.latent_dim
        self.cond_drop_prob = cfg.cond_drop_prob
        self.diffmlps_batch_mul = cfg.diffmlps_batch_mul

        self.input_process = InputProcess(cfg.ae_dim, cfg.latent_dim)
        self.position_enc = PositionalEncoding(cfg.latent_dim, cfg.dropout)
        self.MARTransformer = nn.ModuleList([
            MARTransBlock(cfg.latent_dim, cfg.num_heads, mlp_size=cfg.ff_size, drop_out=cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        self.cond_emb = nn.Linear(cfg.text_dim, cfg.latent_dim)
        self.mask_latent = nn.Parameter(torch.zeros(1, 1, cfg.ae_dim))

        self.apply(self._init_weights)
        for block in self.MARTransformer:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        self.DiffMLPs = DiffMLPs(
            target_channels=cfg.ae_dim, z_channels=cfg.latent_dim,
            width=cfg.diffmlps_width, depth=cfg.diffmlps_depth,
        )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if module.weight is not None:
                nn.init.ones_(module.weight)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def mask_cond(self, cond: Tensor, force_mask: bool = False) -> Tensor:
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        if self.training and self.cond_drop_prob > 0.0:
            keep = (torch.rand(bs, device=cond.device) >= self.cond_drop_prob).float().view(bs, 1)
            return cond * keep
        return cond

    def forward(self, latents: Tensor, cond: Tensor, padding_mask: Tensor,
                force_mask: bool = False, mask: Tensor | None = None,
                control_residuals: list[Tensor] | None = None) -> Tensor:
        """`control_residuals`: optional per-block additive residuals (one
        (B, L, latent_dim) tensor per MARTransformer block, in original token
        order). None (default) is the exact base computation — no new
        parameters, checkpoints load unchanged."""
        cond = self.mask_cond(cond, force_mask=force_mask)
        x = self.input_process(latents)          # (L, B, latent_dim)
        cond = self.cond_emb(cond)               # (B, latent_dim)
        x = self.position_enc(x)
        x = x.permute(1, 0, 2)                   # (B, L, latent_dim)

        inverse_indices = None
        if mask is not None:  # optional hard pseudo-reorder (bidirectional attn ⇒ ~no-op)
            sort_indices = torch.argsort(mask.to(torch.float), dim=1)
            x = torch.gather(x, 1, sort_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))
            inverse_indices = torch.argsort(sort_indices, dim=1)
            padding_mask = torch.gather(padding_mask, 1, sort_indices)
            if control_residuals is not None:
                control_residuals = [
                    torch.gather(r, 1, sort_indices.unsqueeze(-1).expand(-1, -1, r.size(-1)))
                    for r in control_residuals
                ]

        for i, block in enumerate(self.MARTransformer):
            x = block(x, cond, padding_mask)
            if control_residuals is not None:
                x = x + control_residuals[i]

        if inverse_indices is not None:
            x = torch.gather(x, 1, inverse_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        return x

    def forward_loss(self, latents: Tensor, cond: Tensor, m_lens: Tensor) -> Tensor:
        """latents: (B, ae_dim, L) from AE.encode; cond: (B, text_dim); m_lens: latent lengths."""
        latents = latents.permute(0, 2, 1)       # (B, L, ae_dim)
        b, l, _ = latents.shape
        device = latents.device

        non_pad_mask = lengths_to_mask(m_lens, l)
        latents = torch.where(non_pad_mask.unsqueeze(-1), latents, torch.zeros_like(latents))
        target = latents.clone().detach()
        x_in = latents.clone()

        rand_time = uniform((b,), device=device)
        rand_mask_probs = cosine_schedule(rand_time)
        num_masked = (l * rand_mask_probs).round().clamp(min=1)
        batch_randperm = torch.rand((b, l), device=device).argsort(dim=-1)
        mask = batch_randperm < num_masked.unsqueeze(-1)
        mask &= non_pad_mask

        # BERT-style sub-masking: 10% → random latent, 88% of the rest → mask token.
        mask_rlatents = get_mask_subset_prob(mask, 0.1)
        x_in = torch.where(mask_rlatents.unsqueeze(-1), torch.randn_like(x_in), x_in)
        mask_mlatents = get_mask_subset_prob(mask & ~mask_rlatents, 0.88)
        x_in = torch.where(mask_mlatents.unsqueeze(-1), self.mask_latent.repeat(b, l, 1), x_in)

        z = self.forward(x_in, cond, ~non_pad_mask, force_mask=False)

        target = target.reshape(b * l, -1).repeat(self.diffmlps_batch_mul, 1)
        z = z.reshape(b * l, -1).repeat(self.diffmlps_batch_mul, 1)
        flat_mask = mask.reshape(b * l).repeat(self.diffmlps_batch_mul)
        return self.DiffMLPs(z=z[flat_mask], target=target[flat_mask])

    def forward_with_CFG(self, latents: Tensor, cond: Tensor, padding_mask: Tensor, cfg: float = 3.0,
                         mask: Tensor | None = None, force_mask: bool = False,
                         hard_pseudo_reorder: bool = False) -> Tensor:
        reorder_mask = mask.clone() if hard_pseudo_reorder else None
        if force_mask:
            return self.forward(latents, cond, padding_mask, force_mask=True, mask=reorder_mask)

        logits = self.forward(latents, cond, padding_mask, mask=reorder_mask)
        if cfg != 1:
            aux_logits = self.forward(latents, cond, padding_mask, force_mask=True, mask=reorder_mask)
            mixed_logits = torch.cat([logits, aux_logits], dim=0)
        else:
            mixed_logits = logits
        b, l, d = mixed_logits.size()
        n = b // 2 if cfg != 1 else b          # cond-space batch (mixed_logits doubles only under CFG)
        if mask is not None:
            sel = torch.cat([mask, mask], dim=0) if cfg != 1 else mask
            mixed_logits = mixed_logits.reshape(b * l, d)[sel.reshape(b * l)]
        else:
            mixed_logits = mixed_logits.reshape(b * l, d)
        output = self.DiffMLPs.sample(mixed_logits, 1, cfg)
        scaled_logits = output.chunk(2, dim=0)[0] if cfg != 1 else output
        if mask is not None:
            flat = latents.reshape(n * l, self.ae_dim)
            flat[mask.reshape(n * l)] = scaled_logits
            scaled_logits = flat.reshape(n, l, self.ae_dim)
        return scaled_logits

    @torch.no_grad()
    @eval_decorator
    def generate(self, cond: Tensor, m_lens: Tensor, timesteps: int, cond_scale: float,
                 temperature: float = 1.0, force_mask: bool = False,
                 hard_pseudo_reorder: bool = False) -> Tensor:
        """cond: (B, text_dim) text features; m_lens: latent-space lengths.
        Returns latents (B, ae_dim, L) for AE.decode."""
        device = cond.device
        l = int(max(m_lens))
        b = len(m_lens)
        padding_mask = ~lengths_to_mask(m_lens, l)

        latents = torch.where(
            padding_mask.unsqueeze(-1),
            torch.zeros(b, l, self.ae_dim, device=device),
            self.mask_latent.repeat(b, l, 1),
        )
        masked_rand_schedule = torch.where(padding_mask, 1e5, torch.rand_like(padding_mask, dtype=torch.float))

        for timestep in torch.linspace(0, 1, timesteps, device=device):
            rand_mask_prob = cosine_schedule(timestep)
            num_masked = torch.round(rand_mask_prob * m_lens).clamp(min=1)
            ranks = masked_rand_schedule.argsort(dim=1).argsort(dim=1)
            is_mask = ranks < num_masked.unsqueeze(-1)

            latents = torch.where(is_mask.unsqueeze(-1), self.mask_latent.repeat(b, l, 1), latents)
            logits = self.forward_with_CFG(latents, cond, padding_mask, cfg=cond_scale, mask=is_mask,
                                           force_mask=force_mask, hard_pseudo_reorder=hard_pseudo_reorder)
            latents = torch.where(is_mask.unsqueeze(-1), logits, latents)
            masked_rand_schedule = masked_rand_schedule.masked_fill(~is_mask, 1e5)

        latents = torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(latents), latents)
        return latents.permute(0, 2, 1)
