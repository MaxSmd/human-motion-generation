"""1D Diffusion Transformer (DiT) backbone for RMG.

Adapted from Peebles & Xie (2023) — same AdaLN-Zero conditioning, but the
input is a flat 1D motion sequence (no patchify) where each "token" is a
frame's flattened T+R representation (default 91-d for the paper's 22-joint
HumanML3D setup).

Paper §4.1 — for HumanML3D, conditioning is a fused (text, time) MLP that
drives AdaLN modulation in every block. Output is the same shape as input
and is interpreted as the ambient velocity v_θ; the trainer/sampler project
it onto T_{x_t}M before use.

Two configs ship: `RMG_BASE_CONFIG` (6L/384h/8H/FFN×8, 150k steps, 91→91) and
`RMG_LARGE_CONFIG` (24L/1024h/8H/FFN×4, 600k steps).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .conditioning import ConditioningFusion


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """AdaLN modulation: x * (1 + scale[:, None, :]) + shift[:, None, :]."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """AdaLN-Zero DiT block: pre-norm self-attention + pre-norm MLP, both
    modulated by `c` (B, hidden_dim) via shift/scale/gate triplets that are
    zero-initialized so each block starts as identity (residual only)."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_mult: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        ffn_dim = hidden_dim * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )
        # 6 modulation parameters per token: (shift, scale, gate) × (attn, mlp)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True),
        )
        # Zero init for the modulation linear ⇒ block starts as identity.
        nn.init.zeros_(self.adaln_modulation[-1].weight)
        nn.init.zeros_(self.adaln_modulation[-1].bias)

    def forward(
        self,
        x: Tensor,                         # (B, T, hidden)
        c: Tensor,                         # (B, hidden)
        key_padding_mask: Tensor | None,   # (B, T) bool, True = pad
    ) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln_modulation(c).chunk(6, dim=-1)
        )
        h = _modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(
            h, h, h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + gate_msa.unsqueeze(1) * attn_out

        h = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.ffn(h)
        return x


class FinalLayer(nn.Module):
    """AdaLN-modulated norm + linear projection to ambient velocity."""

    def __init__(self, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
        )
        # Zero init: the model's initial output is exactly zero.
        nn.init.zeros_(self.adaln_modulation[-1].weight)
        nn.init.zeros_(self.adaln_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaln_modulation(c).chunk(2, dim=-1)
        x = _modulate(self.norm(x), shift, scale)
        return self.linear(x)


@dataclass
class DiTConfig:
    input_dim: int = 91          # 3 + 22 * 4 for paper's T+R
    hidden_dim: int = 384
    depth: int = 6
    num_heads: int = 8
    ffn_mult: int = 8
    text_dim: int = 1024         # Qwen3-Embedding-0.6B
    time_freq_dim: int = 256
    max_seq_len: int = 200       # HumanML3D ≤196 frames @ 20 fps


# Two named configs from the paper (Table 7).
RMG_BASE_CONFIG = DiTConfig(hidden_dim=384, depth=6, num_heads=8, ffn_mult=8)
RMG_LARGE_CONFIG = DiTConfig(hidden_dim=1024, depth=24, num_heads=8, ffn_mult=4)


class RMGDiT(nn.Module):
    """Main DiT model. Implements the trainer's expected contract:

        model(x_t, t, *, cond=None, drop_cond_mask=None, mask=None)
            → ambient_velocity of the same shape as x_t.
    """

    def __init__(self, cfg: DiTConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or RMG_BASE_CONFIG
        self.cfg = cfg

        self.x_embed = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, cfg.max_seq_len, cfg.hidden_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.cond = ConditioningFusion(
            text_dim=cfg.text_dim,
            hidden_dim=cfg.hidden_dim,
            time_freq_dim=cfg.time_freq_dim,
        )

        self.blocks = nn.ModuleList(
            [DiTBlock(cfg.hidden_dim, cfg.num_heads, cfg.ffn_mult) for _ in range(cfg.depth)]
        )
        self.final = FinalLayer(cfg.hidden_dim, cfg.input_dim)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        x_t: Tensor,                      # (B, T, D)
        t: Tensor,                        # (B,)
        *,
        cond: Tensor | None = None,       # (B, text_dim)
        drop_cond_mask: Tensor | None = None,  # (B,) bool
        mask: Tensor | None = None,       # (B, T) bool, True = valid
    ) -> Tensor:
        B, T, D = x_t.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}")
        if D != self.cfg.input_dim:
            raise ValueError(f"input dim {D} != configured input_dim {self.cfg.input_dim}")

        h = self.x_embed(x_t) + self.pos_embed[:, :T]
        c = self.cond(t, cond, drop_cond_mask)  # (B, hidden)

        # MultiheadAttention takes True = pad (key is masked out).
        kpm = (~mask) if mask is not None else None

        for block in self.blocks:
            h = block(h, c, kpm)

        out = self.final(h, c)  # (B, T, D) — ambient velocity
        return out
