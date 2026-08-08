"""Residual VQ-VAE components for MoMask.

The implementation keeps the repo-wide `(B, T, D)` motion layout and is meant
to tokenize standard 263-D HumanML3D features returned by
`shared.data.H3D263Dataset`.
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
    explicit_loss: Tensor
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
        use_ema: bool = False,
        ema_decay: float = 0.99,
        sample_codebook_temp: float = 0.0,
    ) -> None:
        super().__init__()
        if num_quantizers < 1:
            raise ValueError("num_quantizers must be >= 1")
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.dim = dim
        self.commitment_weight = commitment_weight
        self.quantize_dropout_prob = quantize_dropout_prob
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.sample_codebook_temp = sample_codebook_temp
        self.codebooks = nn.ModuleList([nn.Embedding(codebook_size, dim) for _ in range(num_quantizers)])
        for emb in self.codebooks:
            nn.init.uniform_(emb.weight, -1.0 / codebook_size, 1.0 / codebook_size)
            if use_ema:
                emb.weight.requires_grad_(False)
        if use_ema:
            self.register_buffer("ema_sum", torch.zeros(num_quantizers, codebook_size, dim))
            self.register_buffer("ema_count", torch.zeros(num_quantizers, codebook_size))
            self.register_buffer("ema_initialized", torch.zeros(num_quantizers, dtype=torch.bool))

    def _nearest_indices(self, residual: Tensor, codebook: nn.Embedding) -> Tensor:
        flat = residual.reshape(-1, self.dim)
        weight = codebook.weight
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ weight.t()
            + weight.pow(2).sum(dim=1).unsqueeze(0)
        )
        if self.training and self.sample_codebook_temp > 0.0:
            noise = torch.zeros_like(dist).uniform_(0, 1)
            gumbel = -torch.log(-torch.log(noise.clamp_min(1e-20)).clamp_min(1e-20))
            idx = ((-dist / self.sample_codebook_temp) + gumbel).argmax(dim=1)
        else:
            idx = dist.argmin(dim=1)
        return idx.reshape(residual.shape[:-1])

    def _tile_codes(self, flat: Tensor) -> Tensor:
        if flat.shape[0] < self.codebook_size:
            n_repeats = (self.codebook_size + flat.shape[0] - 1) // flat.shape[0]
            tiled = flat.repeat(n_repeats, 1)
            tiled = tiled + torch.randn_like(tiled) * (0.01 / (self.dim ** 0.5))
        else:
            tiled = flat
        perm = torch.randperm(tiled.shape[0], device=flat.device)
        return tiled[perm[: self.codebook_size]].detach()

    @torch.no_grad()
    def _maybe_init_ema(self, level: int, residual: Tensor) -> None:
        if not self.use_ema or bool(self.ema_initialized[level]):
            return
        flat = residual.reshape(-1, self.dim)
        codes = self._tile_codes(flat)
        self.codebooks[level].weight.data.copy_(codes)
        self.ema_sum[level].copy_(codes)
        self.ema_count[level].fill_(1.0)
        self.ema_initialized[level] = True

    @torch.no_grad()
    def _update_ema(self, level: int, residual: Tensor, idx: Tensor) -> None:
        if not self.use_ema or not self.training:
            return
        flat = residual.reshape(-1, self.dim)
        flat_idx = idx.reshape(-1)
        one_hot = F.one_hot(flat_idx, self.codebook_size).to(flat.dtype)
        code_sum = one_hot.t() @ flat
        code_count = one_hot.sum(dim=0)
        self.ema_sum[level].mul_(self.ema_decay).add_(code_sum, alpha=1.0 - self.ema_decay)
        self.ema_count[level].mul_(self.ema_decay).add_(code_count, alpha=1.0 - self.ema_decay)
        code_update = self.ema_sum[level] / self.ema_count[level].clamp_min(1e-5).unsqueeze(1)
        random_codes = self._tile_codes(flat)
        usage = (self.ema_count[level] >= 1.0).unsqueeze(1)
        self.codebooks[level].weight.data.copy_(torch.where(usage, code_update, random_codes))

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
            self._maybe_init_ema(level, residual)
            idx = self._nearest_indices(residual, codebook)
            q = codebook(idx)
            all_indices.append(idx)
            one_hot_counts.append(F.one_hot(idx, self.codebook_size).float().sum(dim=tuple(range(idx.dim()))))

            if level < active:
                quantized_sum = quantized_sum + q
                if self.use_ema:
                    losses.append(self.commitment_weight * F.mse_loss(residual, q.detach()))
                    self._update_ema(level, residual, idx)
                else:
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
        explicit_loss_weight: float = 0.0,
        recon_loss: str = "l1",
        use_ema_quantizer: bool = False,
        ema_decay: float = 0.99,
        codebook_sample_temp: float = 0.0,
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
        self.explicit_loss_weight = explicit_loss_weight
        if recon_loss not in ("l1", "smooth_l1"):
            raise ValueError("recon_loss must be 'l1' or 'smooth_l1'")
        self.recon_loss = recon_loss

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
            use_ema=use_ema_quantizer,
            ema_decay=ema_decay,
            sample_codebook_temp=codebook_sample_temp,
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
        if self.recon_loss == "smooth_l1":
            err = F.smooth_l1_loss(recon, x, reduction="none")
        else:
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

        if x.shape[-1] >= 67:
            if self.recon_loss == "smooth_l1":
                explicit_err = F.smooth_l1_loss(recon[..., 4:67], x[..., 4:67], reduction="none")
            else:
                explicit_err = (recon[..., 4:67] - x[..., 4:67]).abs()
            if mask is not None:
                explicit_loss = (explicit_err * mask.to(explicit_err.dtype).unsqueeze(-1)).sum() / (
                    mask.sum().clamp_min(1).to(explicit_err.dtype) * explicit_err.shape[-1]
                )
            else:
                explicit_loss = explicit_err.mean()
        else:
            explicit_loss = recon_loss.new_tensor(0.0)

        loss = (
            recon_loss
            + self.velocity_loss_weight * velocity_loss
            + self.explicit_loss_weight * explicit_loss
            + vq.loss
        )
        return RVQVAEOutput(
            recon=recon,
            tokens=vq.indices,
            loss=loss,
            recon_loss=recon_loss,
            velocity_loss=velocity_loss,
            explicit_loss=explicit_loss,
            vq_loss=vq.loss,
            perplexity=vq.perplexity,
        )
