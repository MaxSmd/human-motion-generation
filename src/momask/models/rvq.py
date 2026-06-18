"""Residual VQ-VAE components for MoMask.

The implementation keeps the repo-wide `(B, T, D)` motion layout and is meant
to tokenize standard 263-D HumanML3D features returned by
`HumanML3DDataset(..., output_mode="h3d_263")`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class RVQOutput:
    quantized: Tensor          # (B, T, D)
    indices: Tensor            # (B, Q, T)
    loss: Tensor
    perplexity: Tensor


@dataclass
class RVQVAEOutput:
    recon: Tensor              # (B, T, input_dim)
    tokens: Tensor             # (B, Q, T_latent)
    loss: Tensor
    recon_loss: Tensor
    velocity_loss: Tensor
    vq_loss: Tensor
    perplexity: Tensor


class ResidualVectorQuantizer(nn.Module):
    """Stacked residual vector quantizer.

    Each codebook quantizes the residual left by the previous codebooks. During
    training, optional quantization dropout disables a random suffix of layers,
    matching MoMask's idea that early quantizers should carry the coarse motion.
    """

    def __init__(
        self,
        num_quantizers: int = 6,
        codebook_size: int = 512,
        dim: int = 512,
        commitment_weight: float = 0.25,
        quantize_dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        if num_quantizers < 1:
            raise ValueError("num_quantizers must be >= 1")
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.dim = dim
        self.commitment_weight = commitment_weight
        self.quantize_dropout_prob = quantize_dropout_prob
        self.codebooks = nn.ModuleList([nn.Embedding(codebook_size, dim) for _ in range(num_quantizers)])
        for emb in self.codebooks:
            nn.init.uniform_(emb.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def _nearest_indices(self, residual: Tensor, codebook: nn.Embedding) -> Tensor:
        flat = residual.reshape(-1, self.dim)
        weight = codebook.weight
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ weight.t()
            + weight.pow(2).sum(dim=1).unsqueeze(0)
        )
        return dist.argmin(dim=1).reshape(residual.shape[:-1])

    def encode(self, z: Tensor) -> Tensor:
        residual = z
        indices = []
        for codebook in self.codebooks:
            idx = self._nearest_indices(residual, codebook)
            q = codebook(idx)
            indices.append(idx)
            residual = residual - q
        return torch.stack(indices, dim=1)

    def decode(self, indices: Tensor) -> Tensor:
        if indices.dim() != 3:
            raise ValueError(f"indices must be (B, Q, T), got {tuple(indices.shape)}")
        B, Q, T = indices.shape
        if Q > self.num_quantizers:
            raise ValueError(f"indices contain {Q} quantizers, model has {self.num_quantizers}")
        out = torch.zeros(B, T, self.dim, device=indices.device, dtype=self.codebooks[0].weight.dtype)
        for level in range(Q):
            out = out + self.codebooks[level](indices[:, level])
        return out

    def forward(self, z: Tensor) -> RVQOutput:
        if z.shape[-1] != self.dim:
            raise ValueError(f"last dim {z.shape[-1]} != quantizer dim {self.dim}")

        active = self.num_quantizers
        if self.training and self.quantize_dropout_prob > 0.0 and torch.rand(()) < self.quantize_dropout_prob:
            active = int(torch.randint(1, self.num_quantizers + 1, ()).item())

        residual = z
        quantized_sum = torch.zeros_like(z)
        all_indices = []
        losses = []
        one_hot_counts = []

        for level, codebook in enumerate(self.codebooks):
            idx = self._nearest_indices(residual, codebook)
            q = codebook(idx)
            all_indices.append(idx)
            one_hot_counts.append(F.one_hot(idx, self.codebook_size).float().sum(dim=tuple(range(idx.dim()))))

            if level < active:
                quantized_sum = quantized_sum + q
                losses.append(
                    F.mse_loss(q, residual.detach())
                    + self.commitment_weight * F.mse_loss(residual, q.detach())
                )
                residual = residual - q.detach()
            else:
                residual = residual.detach()

        quantized = z + (quantized_sum - z).detach()
        loss = torch.stack(losses).mean() if losses else z.new_tensor(0.0)
        avg_probs = torch.stack(one_hot_counts).sum(dim=0)
        avg_probs = avg_probs / avg_probs.sum().clamp_min(1.0)
        perplexity = torch.exp(-(avg_probs * avg_probs.clamp_min(1e-12).log()).sum())
        return RVQOutput(quantized=quantized, indices=torch.stack(all_indices, dim=1), loss=loss, perplexity=perplexity)


class _ResBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class MotionRVQVAE(nn.Module):
    """Motion tokenizer used by MoMask's transformer stages."""

    def __init__(
        self,
        input_dim: int = 263,
        hidden_dim: int = 512,
        latent_dim: int = 512,
        num_quantizers: int = 6,
        codebook_size: int = 512,
        downsample: int = 1,
        num_res_blocks: int = 2,
        commitment_weight: float = 0.25,
        quantize_dropout_prob: float = 0.2,
        velocity_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if downsample < 1 or downsample & (downsample - 1):
            raise ValueError("downsample must be a power of two")
        if velocity_loss_weight < 0.0:
            raise ValueError("velocity_loss_weight must be non-negative")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.downsample = downsample
        self.velocity_loss_weight = velocity_loss_weight

        enc = [nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1), nn.GELU()]
        stride = downsample
        while stride > 1:
            enc += [nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1), nn.GELU()]
            stride //= 2
        enc += [_ResBlock(hidden_dim) for _ in range(num_res_blocks)]
        enc += [nn.Conv1d(hidden_dim, latent_dim, kernel_size=3, padding=1)]
        self.encoder = nn.Sequential(*enc)

        dec = [nn.Conv1d(latent_dim, hidden_dim, kernel_size=3, padding=1), nn.GELU()]
        dec += [_ResBlock(hidden_dim) for _ in range(num_res_blocks)]
        stride = downsample
        while stride > 1:
            dec += [nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1), nn.GELU()]
            stride //= 2
        dec += [nn.Conv1d(hidden_dim, input_dim, kernel_size=3, padding=1)]
        self.decoder = nn.Sequential(*dec)

        self.quantizer = ResidualVectorQuantizer(
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            dim=latent_dim,
            commitment_weight=commitment_weight,
            quantize_dropout_prob=quantize_dropout_prob,
        )

    def encode_latents(self, x: Tensor) -> Tensor:
        return self.encoder(x.transpose(1, 2)).transpose(1, 2)

    def decode_latents(self, z: Tensor, target_len: int | None = None) -> Tensor:
        x = self.decoder(z.transpose(1, 2)).transpose(1, 2)
        if target_len is not None:
            if x.shape[1] > target_len:
                x = x[:, :target_len]
            elif x.shape[1] < target_len:
                pad = x.new_zeros(x.shape[0], target_len - x.shape[1], x.shape[2])
                x = torch.cat([x, pad], dim=1)
        return x

    @torch.no_grad()
    def encode_to_tokens(self, x: Tensor) -> Tensor:
        return self.quantizer.encode(self.encode_latents(x))

    def decode_from_tokens(self, tokens: Tensor, target_len: int | None = None) -> Tensor:
        return self.decode_latents(self.quantizer.decode(tokens), target_len=target_len)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> RVQVAEOutput:
        z = self.encode_latents(x)
        vq = self.quantizer(z)
        recon = self.decode_latents(vq.quantized, target_len=x.shape[1])
        err = (recon - x).abs()
        if mask is not None:
            err = err * mask.to(err.dtype).unsqueeze(-1)
            recon_loss = err.sum() / (mask.sum().clamp_min(1).to(err.dtype) * x.shape[-1])
        else:
            recon_loss = err.mean()

        if x.shape[1] > 1:
            vel_err = ((recon[:, 1:] - recon[:, :-1]) - (x[:, 1:] - x[:, :-1])).abs()
            if mask is not None:
                vel_mask = (mask[:, 1:] & mask[:, :-1]).to(vel_err.dtype).unsqueeze(-1)
                velocity_loss = (vel_err * vel_mask).sum() / (
                    vel_mask.sum().clamp_min(1.0) * x.shape[-1]
                )
            else:
                velocity_loss = vel_err.mean()
        else:
            velocity_loss = recon_loss.new_tensor(0.0)

        loss = recon_loss + self.velocity_loss_weight * velocity_loss + vq.loss
        return RVQVAEOutput(
            recon=recon,
            tokens=vq.indices,
            loss=loss,
            recon_loss=recon_loss,
            velocity_loss=velocity_loss,
            vq_loss=vq.loss,
            perplexity=vq.perplexity,
        )
