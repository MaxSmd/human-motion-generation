"""ACMDM-Raw flow prior — the backbone ProjFlow samples from.

A port of upstream `models/ACMDM.py` (Meng et al., "Absolute coordinates make
motion generation easy"), restricted to what ProjFlow's released checkpoint
uses: raw 22 × 3 world-space joints, rectified-flow velocity prediction, text
conditioning, patch size (1, 22) (one token per frame), RoPE over frames,
RMS-norm + adaLN DiT blocks with SwiGLU MLPs. Parameter names mirror upstream
so `ACMDM_Raw_Flow_S_PatchSize22/model/latest.tar` (`ema_acmdm`) loads as-is;
the CLIP text encoder lives in projflow.text instead of inside the model.

Input/output layout is upstream's (B, 3, L, 22), in the 22×3 mean/std-
normalised space (external/ProjFlow/utils/22x3_mean_std/t2m).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

NUM_JOINTS = 22


@dataclass(frozen=True)
class ACMDMConfig:
    """ACMDM-Raw-Flow-S-PatchSize22 (upstream `acmdm_raw_flow_s_ps22`)."""

    input_dim: int = 3
    num_joints: int = NUM_JOINTS
    latent_dim: int = 512
    num_layers: int = 8
    num_heads: int = 8
    ff_size: int = 2048
    clip_dim: int = 512
    max_length: int = 196


class RMSNorm(nn.Module):
    """Llama RMS norm, computed in float32 (upstream `LlamaRMSNorm`)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class FrameRoPE:
    """1-D rotary embedding over token positions (upstream `RopeND`, nd=1).

    Base is upstream's automatic choice (int(8 L / π) // 100 + 1) · 100 → 500
    for L = 196.
    """

    def __init__(self, head_dim: int, max_len: int):
        self.head_dim = head_dim
        self.base = (int(8 * max_len / math.pi) // 100 + 1) * 100
        inv_freq = 1.0 / (self.base ** (torch.linspace(0, head_dim, head_dim // 2).float() / head_dim))
        freqs = torch.outer(torch.arange(max_len).float(), inv_freq)
        freqs = torch.cat([freqs, freqs], dim=1)
        self._cos, self._sin = freqs.cos(), freqs.sin()

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def __call__(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        n = q.shape[-2]
        cos = self._cos[:n].to(q.device)[None, None]
        sin = self._sin[:n].to(q.device)[None, None]
        dtype = q.dtype
        q, k = q.float(), k.float()
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q.to(dtype), k.to(dtype)


class Attention(nn.Module):
    """timm-style attention with RMS qk-norm and RoPE (upstream `ACMDMAttention`)."""

    def __init__(self, dim: int, num_heads: int, rope: FrameRoPE):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)
        self.rope = rope

    def forward(self, x: Tensor, attention_mask: Tensor | None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.rope(self.q_norm(q), self.k_norm(k))
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * hidden)
        self.w3 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_size: int, rope: FrameRoPE):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads, rope)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, int(2 / 3 * ff_size))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x: Tensor, c: Tensor, attention_mask: Tensor | None) -> Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_a * self.attn(_modulate(self.norm1(x), shift_a, scale_a), attention_mask)
        return x + gate_m * self.mlp(_modulate(self.norm2(x), shift_m, scale_m))


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out_dim: int, num_joints: int):
        super().__init__()
        self.norm_final = RMSNorm(dim)
        self.linear = nn.Linear(dim, out_dim * num_joints)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out_dim, self.num_joints = out_dim, num_joints

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.linear(_modulate(self.norm_final(x), shift, scale))     # (B, L, J·D) ordered (j, d)
        B, L, _ = x.shape
        return x.view(B, L, self.num_joints, self.out_dim).permute(0, 3, 1, 2)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.freq_dim = freq_dim

    def forward(self, t: Tensor, max_period: float = 10000.0) -> Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half).to(t.device)
        args = t.float()[:, None] * freqs[None]
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


class ACMDM(nn.Module):
    """Text-conditioned rectified-flow velocity field over raw joint positions."""

    def __init__(self, cfg: ACMDMConfig = ACMDMConfig()):
        super().__init__()
        self.cfg = cfg
        rope = FrameRoPE(cfg.latent_dim // cfg.num_heads, cfg.max_length)
        self.t_embedder = TimestepEmbedder(cfg.latent_dim)
        self.x_embedder = nn.Conv2d(cfg.input_dim, cfg.latent_dim,
                                    kernel_size=(1, cfg.num_joints), stride=(1, cfg.num_joints))
        self.y_embedder = nn.Linear(cfg.clip_dim, cfg.latent_dim)
        self.ACMDMTransformer = nn.ModuleList(
            [Block(cfg.latent_dim, cfg.num_heads, cfg.ff_size, rope) for _ in range(cfg.num_layers)])
        self.final_layer = FinalLayer(cfg.latent_dim, cfg.input_dim, cfg.num_joints)

    def forward(self, x: Tensor, t: Tensor, cond: Tensor, attention_mask: Tensor | None) -> Tensor:
        """x (B, 3, L, 22), t (B,), cond (B, clip_dim), attention_mask (B, 1, 1, L) bool."""
        c = (self.t_embedder(t) + self.y_embedder(cond)).unsqueeze(1)
        h = self.x_embedder(x).flatten(2).transpose(1, 2)             # (B, L, latent)
        for block in self.ACMDMTransformer:
            h = block(h, c, attention_mask)
        return self.final_layer(h, c)

    def guided_velocity(self, x: Tensor, t: Tensor, cond: Tensor, attention_mask: Tensor | None,
                        cfg_scale: float) -> Tensor:
        """Classifier-free guidance with a zero text embedding as the unconditional branch."""
        if cfg_scale == 1.0:
            return self(x, t, cond, attention_mask)
        mask2 = None if attention_mask is None else attention_mask.repeat(2, 1, 1, 1)
        v = self(torch.cat([x, x]), torch.cat([t, t]), torch.cat([cond, torch.zeros_like(cond)]), mask2)
        v_cond, v_uncond = v.chunk(2)
        return v_uncond + cfg_scale * (v_cond - v_uncond)

    @classmethod
    def from_upstream_checkpoint(cls, path: str | Path, cfg: ACMDMConfig = ACMDMConfig(),
                                 key: str = "ema_acmdm") -> "ACMDM":
        """Load upstream's `latest.tar`; its CLIP weights (clip_model.*) are ignored."""
        state = torch.load(path, map_location="cpu", weights_only=False)[key]
        state = {k: v for k, v in state.items() if not k.startswith("clip_model.")}
        model = cls(cfg)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
        return model.eval()


def attention_mask_from_lengths(lengths: Tensor, max_len: int) -> Tensor:
    """(B,) valid lengths -> (B, 1, 1, max_len) bool, True where a frame may be attended."""
    valid = torch.arange(max_len, device=lengths.device)[None] < lengths[:, None]
    return valid[:, None, None, :]
