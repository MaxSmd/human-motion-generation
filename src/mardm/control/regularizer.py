"""Trained condition regularizer for MARDM spatial control (phase 2).

MaskControl's Logits Regularizer translated to continuous masked AR: a
ControlNet-style trainable copy of the (frozen) base MARDM's transformer
block(s) that consumes the spatial control signal and returns per-block
residuals injected into the base via `MARDM.forward(control_residuals=...)`.

Recipe (faithful to ControlNet):
  * the copy's blocks initialize from the base's weights;
  * the control signal enters through a zero-initialized projection added to
    the copy's input;
  * each block's output returns through a zero-initialized linear;
  * so at initialization the regularized model is EXACTLY the base model.

Control-signal encoding: world-frame joint targets at decoded-frame resolution
(the OmniControl/MaskControl convention — see the S-space note in
src/mardm/reports/maskcontrol-differences.md). Latent position i covers frames
4i..4i+3, so per position we stack frames_per_latent * J * (3 target coords,
zeroed where uncontrolled, + 1 mask bit) features and let attention resolve the
global root-cumsum dependency, as MaskControl does.

Only the copy, the control encoder, and the zero projections are trainable;
the base model and AE stay frozen. `state_dict()` therefore contains no base
weights — the regularizer checkpoint is standalone.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from ..models import AE, MARDM
from ..training.masking import cosine_schedule, get_mask_subset_prob, lengths_to_mask, uniform
from .losses import ControlSignal, control_loss, latents_to_joints

NUM_JOINTS = 22


def control_signal_features(control: ControlSignal, num_latents: int,
                            frames_per_latent: int) -> Tensor:
    """(B, T, J, 3)+(B, T, J) -> (B, L, frames_per_latent * J * 4) features.

    Targets are zeroed where uncontrolled; the mask bit rides along. T is
    padded/trimmed to num_latents * frames_per_latent.
    """
    b = control.targets.shape[0]
    t_need = num_latents * frames_per_latent
    targets = torch.zeros(b, t_need, NUM_JOINTS, 3, device=control.targets.device,
                          dtype=control.targets.dtype)
    mask = torch.zeros(b, t_need, NUM_JOINTS, device=control.mask.device, dtype=torch.bool)
    t_have = min(t_need, control.targets.shape[1])
    targets[:, :t_have] = control.targets[:, :t_have]
    mask[:, :t_have] = control.mask[:, :t_have]

    feats = torch.cat([
        torch.where(mask.unsqueeze(-1), targets, torch.zeros_like(targets)),
        mask.unsqueeze(-1).to(targets.dtype),
    ], dim=-1)                                             # (B, T, J, 4)
    return feats.reshape(b, num_latents, frames_per_latent * NUM_JOINTS * 4)


class ControlMARDM(nn.Module):
    """ControlNet-style copy of the base MARDM transformer.

    The base model is NOT registered as a submodule (it stays frozen and is
    checkpointed separately); pass it to `attach_base` after construction /
    loading. `residuals()` returns the per-block injection tensors for
    `MARDM.forward(control_residuals=...)`.
    """

    def __init__(self, base: MARDM, frames_per_latent: int = 4):
        super().__init__()
        cfg = base.cfg
        self.frames_per_latent = frames_per_latent
        self.latent_dim = cfg.latent_dim
        control_feat_dim = frames_per_latent * NUM_JOINTS * 4

        self.copy_blocks = copy.deepcopy(base.MARTransformer)
        for p in self.copy_blocks.parameters():
            p.requires_grad_(True)
        self.control_in = nn.Sequential(
            nn.Linear(control_feat_dim, cfg.latent_dim),
            nn.SiLU(),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )
        self.zero_proj = nn.ModuleList(
            nn.Linear(cfg.latent_dim, cfg.latent_dim) for _ in self.copy_blocks
        )
        # Zero-init the joins: identical-to-base at initialization.
        nn.init.zeros_(self.control_in[-1].weight)
        nn.init.zeros_(self.control_in[-1].bias)
        for proj in self.zero_proj:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

        self.attach_base(base)

    def attach_base(self, base: MARDM) -> None:
        """Store the frozen base without registering it (kept out of state_dict)."""
        object.__setattr__(self, "_base", base)

    @property
    def base(self) -> MARDM:
        return self._base

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def residuals(self, latents: Tensor, cond: Tensor, padding_mask: Tensor,
                  control_feats: Tensor, force_mask: bool = False) -> list[Tensor]:
        """Per-block residuals for `MARDM.forward(control_residuals=...)`.

        latents: (B, L, ae_dim); cond: (B, text_dim);
        control_feats: (B, L, frames_per_latent * 22 * 4) from
        `control_signal_features`. Mirrors the base's input processing with the
        base's FROZEN embedding weights, then runs the trainable copy.
        """
        base = self.base
        cond_e = base.cond_emb(base.mask_cond(cond, force_mask=force_mask))
        x = base.input_process(latents)                    # (L, B, latent_dim)
        x = base.position_enc(x).permute(1, 0, 2)          # (B, L, latent_dim)
        h = x + self.control_in(control_feats)
        out: list[Tensor] = []
        for block, proj in zip(self.copy_blocks, self.zero_proj):
            h = block(h, cond_e, padding_mask)
            out.append(proj(h))
        return out


def control_forward_loss(
    reg: ControlMARDM,
    ae: AE,
    latents: Tensor,
    cond: Tensor,
    m_lens: Tensor,
    control: ControlSignal,
    mean: Tensor,
    std: Tensor,
    ls_ode_steps: int = 4,
) -> tuple[Tensor, Tensor]:
    """Regularizer training step: (L_diff, L_s) under MARDM's masking scheme.

    latents: (B, ae_dim, L) from the frozen AE; cond: (B, text_dim) —
    classifier-free dropout, if any, is the CALLER's job (the frozen base is in
    eval mode, so its own `mask_cond` never drops). Replicates
    `MARDM.forward_loss`'s BERT-style masking, injects the regularizer's
    residuals, then:
      * L_diff — the base's per-token SiT diffusion loss on masked tokens;
      * L_s   — masked tokens sampled FROM PURE NOISE through `ls_ode_steps`
        differentiable euler ODE steps conditioned on z, scattered into the
        TRUE-latent context, decoded through the frozen AE to world-frame
        joints, masked joint error vs `control`.

    L_s must NOT be GT-anchored: the first training run used the one-step
    clean estimate x̂₁ = x_t + (1−t)·v, whose x_t contains t·x₁ — with
    GT-derived targets that loss is minimized by accurate denoising alone, so
    the regularizer never learns to use the control signal (l_s stayed flat
    for 90k steps; reg_only evaluated at the unguided control error). Sampling
    from noise measures the actual generation-time control error, which is the
    quantity MaskControl's DES-based loss measures — ours just does it exactly.
    """
    base = reg.base
    latents = latents.permute(0, 2, 1)               # (B, L, ae_dim)
    b, l, d = latents.shape
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

    mask_rlatents = get_mask_subset_prob(mask, 0.1)
    x_in = torch.where(mask_rlatents.unsqueeze(-1), torch.randn_like(x_in), x_in)
    mask_mlatents = get_mask_subset_prob(mask & ~mask_rlatents, 0.88)
    x_in = torch.where(mask_mlatents.unsqueeze(-1), base.mask_latent.repeat(b, l, 1), x_in)

    control_feats = control_signal_features(control, l, reg.frames_per_latent).to(device)
    padding_mask = ~non_pad_mask
    res = reg.residuals(x_in, cond, padding_mask, control_feats)
    z = base.forward(x_in, cond, padding_mask, control_residuals=res)

    flat_mask = mask.reshape(b * l)
    z_flat = z.reshape(b * l, -1)
    target_flat = target.reshape(b * l, d)

    # L_diff: the base's own per-token diffusion loss (with its batch_mul).
    mul = base.diffmlps_batch_mul
    l_diff = base.DiffMLPs(
        z=z_flat.repeat(mul, 1)[flat_mask.repeat(mul)],
        target=target_flat.repeat(mul, 1)[flat_mask.repeat(mul)],
    )

    # L_s: masked tokens generated from pure noise (differentiable w.r.t. z).
    z_m = z_flat[flat_mask]
    noise = torch.randn(z_m.shape[0], d, device=device)
    sample_fn = base.DiffMLPs.gen_transport.sample_ode(
        sampling_method="euler", num_steps=ls_ode_steps)
    x1_hat = sample_fn(noise, base.DiffMLPs.net, c=z_m)[-1]
    full = target_flat.clone()
    full[flat_mask] = x1_hat
    joints = latents_to_joints(full.reshape(b, l, d), ae, mean, std)
    l_s = control_loss(joints, control.to(device))
    return l_diff, l_s
