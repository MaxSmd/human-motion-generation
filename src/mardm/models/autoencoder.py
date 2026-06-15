"""1D ResNet motion AutoEncoder for MARDM (stage 1).

Vendored and lightly adapted from MARDM (neu-vi/MARDM, ``models/AE.py``; see the
upstream LICENSE). Encodes a 67-D "essential" HumanML3D feature sequence into a
temporally-downsampled latent (default ×4, 512-d) and reconstructs it, trained
with an L1 loss. The generation branch (`mardm.models.mardm`) diffuses over
these latents.

Shapes:
    encode:  (B, T, input_width)        -> (B, output_emb_width, T // 4)
    decode:  (B, output_emb_width, L)   -> (B, 4L, input_width)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class AEConfig:
    input_width: int = 67           # essential HumanML3D feature dim
    output_emb_width: int = 512     # latent channel dim (== mardm ae_dim)
    down_t: int = 2                 # number of ×stride_t temporal downsamples
    stride_t: int = 2               # → total downsample = stride_t ** down_t = 4
    width: int = 512                # conv hidden width
    depth: int = 3                  # ResNet depth per stage
    dilation_growth_rate: int = 3
    activation: str = "relu"
    norm: str | None = None


class _Swish(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


def _activation(name: str) -> nn.Module:
    return {"relu": nn.ReLU, "silu": _Swish, "gelu": nn.GELU}[name]()


class ResConv1DBlock(nn.Module):
    def __init__(self, n_in: int, n_state: int, dilation: int = 1,
                 activation: str = "relu", norm: str | None = None, dropout: float = 0.2):
        super().__init__()
        self.norm = norm
        if norm == "LN":
            self.norm1, self.norm2 = nn.LayerNorm(n_in), nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(32, n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(32, n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(n_in, eps=1e-6, affine=True)
        else:
            self.norm1, self.norm2 = nn.Identity(), nn.Identity()
        self.activation1 = _activation(activation)
        self.activation2 = _activation(activation)
        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, dilation, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x_orig = x
        if self.norm == "LN":
            x = self.activation1(self.norm1(x.transpose(-2, -1)).transpose(-2, -1))
        else:
            x = self.activation1(self.norm1(x))
        x = self.conv1(x)
        if self.norm == "LN":
            x = self.activation2(self.norm2(x.transpose(-2, -1)).transpose(-2, -1))
        else:
            x = self.activation2(self.norm2(x))
        x = self.conv2(x)
        x = self.dropout(x)
        return x + x_orig


class Resnet1D(nn.Module):
    def __init__(self, n_in: int, n_depth: int, dilation_growth_rate: int = 1,
                 reverse_dilation: bool = True, activation: str = "relu", norm: str | None = None):
        super().__init__()
        blocks = [
            ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** d, activation=activation, norm=norm)
            for d in range(n_depth)
        ]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.model = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class Encoder(nn.Module):
    def __init__(self, input_emb_width: int, output_emb_width: int, down_t: int, stride_t: int,
                 width: int, depth: int, dilation_growth_rate: int, activation: str, norm: str | None):
        super().__init__()
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks: list[nn.Module] = [nn.Conv1d(input_emb_width, width, 3, 1, 1), nn.ReLU()]
        for _ in range(down_t):
            blocks.append(nn.Sequential(
                nn.Conv1d(width, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            ))
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, input_emb_width: int, output_emb_width: int, down_t: int, stride_t: int,
                 width: int, depth: int, dilation_growth_rate: int, activation: str, norm: str | None):
        super().__init__()
        blocks: list[nn.Module] = [nn.Conv1d(output_emb_width, width, 3, 1, 1), nn.ReLU()]
        for _ in range(down_t):
            blocks.append(nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=float(stride_t), mode="nearest"),
                nn.Conv1d(width, width, 3, 1, 1),
            ))
        blocks += [nn.Conv1d(width, width, 3, 1, 1), nn.ReLU(), nn.Conv1d(width, input_emb_width, 3, 1, 1)]
        self.model = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).permute(0, 2, 1)


class AE(nn.Module):
    def __init__(self, cfg: AEConfig | None = None):
        super().__init__()
        cfg = cfg or AEConfig()
        self.cfg = cfg
        self.output_emb_width = cfg.output_emb_width
        self.downsample_rate = cfg.stride_t ** cfg.down_t
        self.encoder = Encoder(cfg.input_width, cfg.output_emb_width, cfg.down_t, cfg.stride_t,
                               cfg.width, cfg.depth, cfg.dilation_growth_rate, cfg.activation, cfg.norm)
        self.decoder = Decoder(cfg.input_width, cfg.output_emb_width, cfg.down_t, cfg.stride_t,
                               cfg.width, cfg.depth, cfg.dilation_growth_rate, cfg.activation, cfg.norm)
        # Per-channel latent scale so `encode` hands the diffusion head ~unit-
        # variance latents (the SiT prior is N(0,1)); without it the raw latents
        # (std ~0.13) sit under the noise floor and the head can't learn the
        # signal. Computed post-training via `compute_latent_scale`; defaults to
        # 1.0 (no-op) so older checkpoints behave unchanged. `forward` bypasses
        # encode/decode, so reconstruction is unaffected.
        self.register_buffer("latent_scale", torch.ones(cfg.output_emb_width))

    @staticmethod
    def _to_channels_first(x: Tensor) -> Tensor:
        return x.permute(0, 2, 1).float()  # (B, T, C) -> (B, C, T)

    def encode(self, x: Tensor) -> Tensor:
        """(B, T, input_width) -> (B, output_emb_width, T // downsample_rate), scaled."""
        z = self.encoder(self._to_channels_first(x))
        return z * self.latent_scale.view(1, -1, 1)

    def decode(self, z: Tensor) -> Tensor:
        """(B, output_emb_width, L) -> (B, L * downsample_rate, input_width). Inverts `encode`'s scale."""
        return self.decoder(z / self.latent_scale.view(1, -1, 1))

    def forward(self, x: Tensor) -> Tensor:
        """(B, T, input_width) -> (B, T, input_width) reconstruction (scale-free)."""
        return self.decoder(self.encoder(self._to_channels_first(x)))

    @torch.no_grad()
    def compute_latent_scale(self, loader, device, max_batches: int = 50,
                             eps: float = 1e-4) -> Tensor:
        """Return per-channel 1/std of the RAW encoder output over `loader`.

        Uses `self.encoder` directly (not `encode`), so it's independent of the
        current `latent_scale`. The caller stores the result on the model (and,
        because this repo's EMA tracks buffers, on the EMA shadow too).
        """
        csum = torch.zeros(self.output_emb_width, device=device)
        csqs = torch.zeros(self.output_emb_width, device=device)
        count = 0
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            x = batch.x1.to(device)
            z = self.encoder(self._to_channels_first(x))          # (B, ae_dim, L) raw
            zc = z.transpose(0, 1).reshape(self.output_emb_width, -1)
            csum += zc.sum(1)
            csqs += (zc * zc).sum(1)
            count += zc.shape[1]
        mean = csum / max(count, 1)
        std = (csqs / max(count, 1) - mean ** 2).clamp(min=eps).sqrt()
        return (1.0 / std.clamp(min=eps)).to(self.latent_scale)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
