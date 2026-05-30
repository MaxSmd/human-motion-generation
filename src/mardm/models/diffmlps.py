"""Per-token diffusion head (SiT velocity) for MARDM.

Vendored/adapted from MARDM (neu-vi/MARDM, ``models/DiffMLPs.py``; see the
upstream LICENSE). Keeps only the SiT variant (linear path, velocity
prediction, ODE sampling) — the DDPM variant and its gaussian-diffusion
dependency are dropped per the scaled-down plan.

`SimpleMLPAdaLN` denoises a single latent token (dim `target_channels`)
conditioned on the MAR transformer's per-token output `z` (dim `z_channels`)
and the diffusion timestep `t`.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..transport import Sampler, create_transport


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """Embeds a scalar diffusion timestep into a vector."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class ResBlock(nn.Module):
    """AdaLN-modulated residual MLP block."""

    def __init__(self, channels: int):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        return x + gate_mlp * self.mlp(h)


class FinalLayer(nn.Module):
    def __init__(self, model_channels: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True))

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class SimpleMLPAdaLN(nn.Module):
    """Diffusion-loss MLP: denoises one latent token conditioned on (t, z)."""

    def __init__(self, in_channels: int, model_channels: int, out_channels: int,
                 z_channels: int, num_res_blocks: int, grad_checkpointing: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)
        self.res_blocks = nn.ModuleList([ResBlock(model_channels) for _ in range(num_res_blocks)])
        self.final_layer = FinalLayer(model_channels, out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x: Tensor, t: Tensor, c: Tensor) -> Tensor:
        x = self.input_proj(x)
        y = self.time_embed(t) + self.cond_embed(c)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)
        return self.final_layer(x, y)

    def forward_with_cfg(self, x: Tensor, t: Tensor, c: Tensor, cfg_scale: float) -> Tensor:
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class DiffMLPs(nn.Module):
    """SiT (velocity, linear path) per-token diffusion head."""

    def __init__(self, target_channels: int, z_channels: int, width: int = 512, depth: int = 6):
        super().__init__()
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels, model_channels=width, out_channels=target_channels,
            z_channels=z_channels, num_res_blocks=depth,
        )
        self.train_transport = create_transport()  # linear path, velocity prediction
        self.gen_transport = Sampler(self.train_transport)

    def forward(self, z: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
        loss_dict = self.train_transport.training_losses(self.net, target, dict(c=z))
        loss = loss_dict["loss"]
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean()

    def sample(self, z: Tensor, temperature: float = 1.0, cfg: float = 1.0) -> Tensor:
        device = z.device
        if cfg != 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels, device=device)
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
            model_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels, device=device)
            model_kwargs = dict(c=z)
            model_fn = self.net.forward
        sample_fn = self.gen_transport.sample_ode()  # default ODE sampling
        return sample_fn(noise, model_fn, **model_kwargs)[-1]
