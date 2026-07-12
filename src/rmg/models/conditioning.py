"""Time + text conditioning for the DiT backbone.

The paper (§4.1) describes the HumanML3D recipe as:
    "We fuse the text features with the time embedding through an MLP and use
     the fused representation as the conditioning input to the Diffusion
     Transformer."

Here `cond` is a per-sample text feature (e.g. Qwen3-Embedding-0.6B's pooled
output, 1024-d). For CFG we keep a learnable null embedding and route around
it whenever `drop_cond_mask[i] = True` (or when `cond is None` entirely, i.e.
unconditional generation).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def sinusoidal_time_embedding(t: Tensor, dim: int, max_period: float = 1e4) -> Tensor:
    """Standard transformer-style sinusoidal embedding for a scalar t ∈ [0, 1].

    Args:
        t: (B,) float tensor.
        dim: embedding dimension (must be even).
    Returns:
        (B, dim) tensor.
    """
    if dim % 2 != 0:
        raise ValueError("sinusoidal_time_embedding requires even dim")
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=t.dtype, device=t.device)
        / max(half - 1, 1)
    )
    args = t.unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ConditioningFusion(nn.Module):
    """Map (t, text_features, drop_mask) → (B, hidden_dim) conditioning vector
    consumed by AdaLN-Zero modulation in every DiT block.

    A learnable null embedding stands in for the dropped/missing condition.
    """

    def __init__(
        self,
        text_dim: int,
        hidden_dim: int,
        time_freq_dim: int = 256,
        time_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.time_freq_dim = time_freq_dim
        # Multiplier applied to t before the sinusoidal embedding. With the
        # default 1.0 and t ∈ [0, 1], the embedding arguments t·f span ≤ 1 rad
        # on every frequency band — i.e. the basis is nearly linear in t and
        # most bands are unused. Standard DiT/flow stacks embed t·1000 instead,
        # giving the conditioning MLP a genuinely multi-scale basis. Kept at
        # 1.0 for checkpoint compatibility with every existing run; set
        # model.time_scale=1000 for future from-scratch runs. Must match
        # between training and evaluation of the same checkpoint.
        self.time_scale = float(time_scale)

        # Learnable null embedding for unconditional / dropped samples.
        self.null_text = nn.Parameter(torch.zeros(text_dim))

        self.text_mlp = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(time_freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fuse = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        t: Tensor,
        cond: Tensor | None,
        drop_cond_mask: Tensor | None = None,
    ) -> Tensor:
        B = t.shape[0]
        # Resolve text features (real, null, or per-sample mix).
        if cond is None:
            cond = self.null_text.unsqueeze(0).expand(B, self.text_dim)
        elif drop_cond_mask is not None:
            null_b = self.null_text.unsqueeze(0).expand(B, self.text_dim)
            cond = torch.where(drop_cond_mask.unsqueeze(-1), null_b, cond)

        text_h = self.text_mlp(cond)  # (B, hidden)
        time_freq = sinusoidal_time_embedding(t * self.time_scale, self.time_freq_dim)
        time_h = self.time_mlp(time_freq)  # (B, hidden)
        c = self.fuse(torch.cat([time_h, text_h], dim=-1))  # (B, hidden)
        return c
