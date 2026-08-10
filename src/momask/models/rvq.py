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

from momask.data_utils import token_mask_from_frame_mask


@dataclass
class RVQOutput:
    quantized: Tensor          # (B, T, D)
    indices: Tensor            # (B, Q, T)
    loss: Tensor
    perplexity: Tensor
    perplexity_per_level: Tensor
    active_codes_per_level: Tensor


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
    perplexity_per_level: Tensor
    active_codes_per_level: Tensor


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
        self._ema_initialized_runtime = [False] * num_quantizers
        self.register_load_state_dict_post_hook(self._reset_ema_runtime_state)
        self.codebooks = nn.ModuleList([nn.Embedding(codebook_size, dim) for _ in range(num_quantizers)])
        for emb in self.codebooks:
            nn.init.uniform_(emb.weight, -1.0 / codebook_size, 1.0 / codebook_size)
            if use_ema:
                emb.weight.requires_grad_(False)
        if use_ema:
            self.register_buffer("ema_sum", torch.zeros(num_quantizers, codebook_size, dim))
            self.register_buffer("ema_count", torch.zeros(num_quantizers, codebook_size))
            self.register_buffer("ema_initialized", torch.zeros(num_quantizers, dtype=torch.bool))

    @staticmethod
    def _reset_ema_runtime_state(module: nn.Module, _incompatible_keys: object) -> None:
        if isinstance(module, ResidualVectorQuantizer):
            module._ema_initialized_runtime = [False] * module.num_quantizers

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
            neg_log_noise = -torch.log(noise.clamp_min(1e-20))
            gumbel = -torch.log(neg_log_noise.clamp_min(1e-20))
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
        return tiled[: self.codebook_size].detach()

    @staticmethod
    def _valid_values(x: Tensor, mask: Tensor | None) -> Tensor:
        return x.reshape(-1, x.shape[-1]) if mask is None else x[mask]

    @torch.no_grad()
    def _maybe_init_ema(self, level: int, residual: Tensor, mask: Tensor | None = None) -> None:
        if not self.use_ema or self._ema_initialized_runtime[level]:
            return
        if bool(self.ema_initialized[level]):
            self._ema_initialized_runtime[level] = True
            return
        if not self.training:
            raise RuntimeError(
                f"EMA codebook level {level} is uninitialized; train or load a VQ checkpoint before evaluation"
            )
        flat = self._valid_values(residual, mask)
        if flat.shape[0] == 0:
            raise ValueError("RVQ received a batch with no valid latent positions")
        codes = self._tile_codes(flat)
        self.codebooks[level].weight.data.copy_(codes)
        self.ema_sum[level].copy_(codes)
        self.ema_count[level].fill_(1.0)
        self.ema_initialized[level] = True
        self._ema_initialized_runtime[level] = True

    @torch.no_grad()
    def _update_ema(
        self,
        level: int,
        residual: Tensor,
        idx: Tensor,
        mask: Tensor | None = None,
    ) -> None:
        if not self.use_ema or not self.training:
            return
        flat = self._valid_values(residual, mask)
        flat_idx = idx.reshape(-1) if mask is None else idx[mask]
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

    def forward(self, z: Tensor, mask: Tensor | None = None) -> RVQOutput:
        if z.shape[-1] != self.dim:
            raise ValueError(f"last dim {z.shape[-1]} != quantizer dim {self.dim}")
        if mask is not None:
            if mask.shape != z.shape[:2]:
                raise ValueError(f"latent mask must be {tuple(z.shape[:2])}, got {tuple(mask.shape)}")
            mask = mask.bool()
            if not bool(mask.any()):
                raise ValueError("RVQ received a batch with no valid latent positions")

        active = self.num_quantizers
        if self.training and self.quantize_dropout_prob > 0.0 and torch.rand(()) < self.quantize_dropout_prob:
            active = int(torch.randint(1, self.num_quantizers + 1, ()).item())

        residual = z
        quantized_sum = torch.zeros_like(z)
        all_indices = []
        losses = []
        one_hot_counts = []

        for level, codebook in enumerate(self.codebooks):
            if level >= active:
                all_indices.append(torch.zeros(z.shape[:2], dtype=torch.long, device=z.device))
                continue
            self._maybe_init_ema(level, residual, mask)
            idx = self._nearest_indices(residual, codebook)
            q = codebook(idx)
            if mask is not None:
                idx = idx.masked_fill(~mask, 0)
            all_indices.append(idx)
            valid_idx = idx.reshape(-1) if mask is None else idx[mask]
            one_hot_counts.append(F.one_hot(valid_idx, self.codebook_size).float().sum(dim=0))

            residual_valid = self._valid_values(residual, mask)
            q_valid = self._valid_values(q, mask)
            if self.use_ema:
                losses.append(self.commitment_weight * F.mse_loss(residual_valid, q_valid.detach()))
                self._update_ema(level, residual, idx, mask)
            else:
                losses.append(
                    F.mse_loss(q_valid, residual_valid.detach())
                    + self.commitment_weight * F.mse_loss(residual_valid, q_valid.detach())
                )

            # Match the official residual quantizer: every active level has its
            # own straight-through path before the quantized levels are summed.
            q_st = residual + (q - residual).detach()
            quantized_sum = quantized_sum + q_st
            residual = residual - q.detach()

        quantized = quantized_sum
        if mask is not None:
            quantized = quantized.masked_fill(~mask.unsqueeze(-1), 0.0)
        loss = torch.stack(losses).mean() if losses else z.new_tensor(0.0)
        level_counts = torch.stack(one_hot_counts)
        level_probs = level_counts / level_counts.sum(dim=1, keepdim=True).clamp_min(1.0)
        level_perplexity = torch.exp(
            -(level_probs * level_probs.clamp_min(1e-12).log()).sum(dim=1)
        )
        active_codes = (level_counts > 0).sum(dim=1)
        perplexity = level_perplexity.mean()
        if active < self.num_quantizers:
            pad = self.num_quantizers - active
            level_perplexity = F.pad(level_perplexity, (0, pad), value=0.0)
            active_codes = F.pad(active_codes, (0, pad), value=0)
        return RVQOutput(
            quantized=quantized,
            indices=torch.stack(all_indices, dim=1),
            loss=loss,
            perplexity=perplexity,
            perplexity_per_level=level_perplexity,
            active_codes_per_level=active_codes,
        )


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


class _PaperResConv1DBlock(nn.Module):
    """Dilated residual block used by the official MoMask RVQ."""

    def __init__(self, dim: int, dilation: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


def _paper_resnet(dim: int, depth: int, dilation_growth: int = 3) -> nn.Sequential:
    dilations = [dilation_growth ** level for level in range(depth)][::-1]
    return nn.Sequential(*[_PaperResConv1DBlock(dim, dilation) for dilation in dilations])


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
        architecture: str = "simple",
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
        if architecture not in ("simple", "paper"):
            raise ValueError("architecture must be 'simple' or 'paper'")
        self.architecture = architecture
        if recon_loss not in ("l1", "smooth_l1"):
            raise ValueError("recon_loss must be 'l1' or 'smooth_l1'")
        self.recon_loss = recon_loss

        if architecture == "paper":
            enc: list[nn.Module] = [nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1), nn.ReLU()]
            stride = downsample
            while stride > 1:
                enc.append(
                    nn.Sequential(
                        nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
                        _paper_resnet(hidden_dim, num_res_blocks),
                    )
                )
                stride //= 2
            enc.append(nn.Conv1d(hidden_dim, latent_dim, kernel_size=3, padding=1))

            dec: list[nn.Module] = [nn.Conv1d(latent_dim, hidden_dim, kernel_size=3, padding=1), nn.ReLU()]
            stride = downsample
            while stride > 1:
                dec.append(
                    nn.Sequential(
                        _paper_resnet(hidden_dim, num_res_blocks),
                        nn.Upsample(scale_factor=2, mode="nearest"),
                        nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    )
                )
                stride //= 2
            dec.extend(
                [
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(hidden_dim, input_dim, kernel_size=3, padding=1),
                ]
            )
        else:
            enc = [nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1), nn.GELU()]
            stride = downsample
            while stride > 1:
                enc += [nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1), nn.GELU()]
                stride //= 2
            enc += [_ResBlock(hidden_dim) for _ in range(num_res_blocks)]
            enc += [nn.Conv1d(hidden_dim, latent_dim, kernel_size=3, padding=1)]

            dec = [nn.Conv1d(latent_dim, hidden_dim, kernel_size=3, padding=1), nn.GELU()]
            dec += [_ResBlock(hidden_dim) for _ in range(num_res_blocks)]
            stride = downsample
            while stride > 1:
                dec += [nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1), nn.GELU()]
                stride //= 2
            dec += [nn.Conv1d(hidden_dim, input_dim, kernel_size=3, padding=1)]
        self.encoder = nn.Sequential(*enc)
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

    def decode_from_tokens(
        self,
        tokens: Tensor,
        target_len: int | None = None,
        token_mask: Tensor | None = None,
    ) -> Tensor:
        latents = self.quantizer.decode(tokens)
        if token_mask is not None:
            if token_mask.shape != latents.shape[:2]:
                raise ValueError(
                    f"token mask must be {tuple(latents.shape[:2])}, got {tuple(token_mask.shape)}"
                )
            latents = latents.masked_fill(~token_mask.bool().unsqueeze(-1), 0.0)
        return self.decode_latents(latents, target_len=target_len)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> RVQVAEOutput:
        z = self.encode_latents(x)
        latent_mask = None
        if mask is not None:
            latent_mask = token_mask_from_frame_mask(mask, z.shape[1], self.downsample)
            z = z.masked_fill(~latent_mask.unsqueeze(-1), 0.0)
        vq = self.quantizer(z, mask=latent_mask)
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
            perplexity_per_level=vq.perplexity_per_level,
            active_codes_per_level=vq.active_codes_per_level,
        )
