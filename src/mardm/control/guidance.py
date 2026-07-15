"""Inference-time spatial control for MARDM via z-optimization (phase 1).

MaskControl's Logits Optimization translated to continuous masked
autoregression: at each masked-AR step the transformer's per-token condition
`z` parameterizes the distribution the diffusion head samples from (the role
logits play in discrete models). We take gradient steps on `z` against the
differentiable control loss (`mardm.control.losses`) before committing the
step's tokens, so the head still maps the perturbed condition through its
learned denoising — outputs stay on the motion manifold. No DES / Gumbel
machinery is needed: the whole path (per-token euler ODE sample -> AE.decode
-> denormalize -> FK) is exactly differentiable.

This module deliberately does NOT touch `MARDM.generate` (which is
`@torch.no_grad`): the masked-AR loop is re-implemented here with gradients
enabled where needed. `inner_iters=0, post_iters=0` reduces to unguided
sampling (euler solver instead of generate()'s adaptive dopri5 — use it as the
matched no-guidance baseline so guidance is the only difference).

An optional post pass (MaskControl Eq. 8 analogue) takes a few gradient steps
directly on the final latent sequence — effective for squeezing out residual
error, but it bypasses the diffusion prior, so keep it short.

Optional re-prediction repair (`repair_rounds > 0`) makes MaskControl's implicit
remask-and-repredict restore explicit for our nested-commit AR loop: tokens the
unguided prior most disagrees with get remasked and re-predicted under light
guidance, pulling the motion back on-manifold without losing the waypoints. See
reports/control-phase1/maskcontrol-differences.md for the exact mapping.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..models import AE, MARDM
from ..training.masking import cosine_schedule, lengths_to_mask
from .losses import ControlSignal, control_loss, latents_to_joints


@dataclass
class GuidanceConfig:
    inner_iters: int = 30        # z gradient steps per masked-AR step (0 = off)
    lr: float = 0.05             # Adam lr on z
    ode_steps_guidance: int = 8  # euler steps for the differentiable sample
    ode_steps_final: int = 25    # euler steps for the committed (no-grad) sample
    post_iters: int = 0          # direct latent optimization after the AR loop
    post_lr: float = 0.01
    # Re-prediction repair (explicit analogue of MaskControl's remask-and-
    # repredict restore; see reports/control-phase1/maskcontrol-differences.md).
    # After the AR loop: remask the `repair_frac` of tokens the UNGUIDED prior
    # most disagrees with and re-predict them under light guidance
    # (`repair_iters` inner steps), `repair_rounds` times.
    repair_rounds: int = 0
    repair_frac: float = 0.5
    repair_iters: int = 10
    verbose: bool = False


def _sample_tokens(mardm: MARDM, z: Tensor, noise: Tensor, cond_scale: float,
                   num_steps: int) -> Tensor:
    """Per-token ODE sample, differentiable w.r.t. z.

    z: (N, latent_dim) or (2N, latent_dim) with the uncond half stacked when
    cond_scale != 1 (matching DiffMLPs.sample's CFG layout); noise: (N, ae_dim).
    Returns (N, ae_dim) — the conditional half under CFG.
    """
    sample_fn = mardm.DiffMLPs.gen_transport.sample_ode(
        sampling_method="euler", num_steps=num_steps)
    if cond_scale != 1.0:
        noise = torch.cat([noise, noise], dim=0)
        out = sample_fn(noise, mardm.DiffMLPs.net.forward_with_cfg, c=z, cfg_scale=cond_scale)[-1]
        return out.chunk(2, dim=0)[0]
    return sample_fn(noise, mardm.DiffMLPs.net, c=z)[-1]


def _compute_z(mardm: MARDM, latents: Tensor, cond: Tensor, padding_mask: Tensor,
               flat_mask: Tensor, cond_scale: float) -> Tensor:
    """Transformer condition for the masked tokens, uncond half stacked under CFG."""
    b, l, _ = latents.shape
    with torch.no_grad():
        z = mardm.forward(latents, cond, padding_mask).reshape(b * l, -1)[flat_mask]
        if cond_scale != 1.0:
            z_uncond = mardm.forward(latents, cond, padding_mask, force_mask=True)
            z = torch.cat([z, z_uncond.reshape(b * l, -1)[flat_mask]], dim=0)
    return z


def _guided_commit(mardm: MARDM, ae: AE, latents: Tensor, is_mask: Tensor, cond: Tensor,
                   padding_mask: Tensor, cond_scale: float, control: ControlSignal,
                   mean: Tensor, std: Tensor, g: GuidanceConfig, inner_iters: int,
                   tag: str) -> Tensor:
    """One masked-prediction step: optimize z for the masked tokens, commit samples."""
    b, l, d = latents.shape
    flat_mask = is_mask.reshape(b * l)
    z = _compute_z(mardm, latents, cond, padding_mask, flat_mask, cond_scale)
    # Fixed noise per step so the inner optimization is deterministic.
    noise = torch.randn(int(flat_mask.sum()), d, device=latents.device)

    if inner_iters > 0 and control.num_constraints > 0:
        z_opt = z.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([z_opt], lr=g.lr)
        context = latents.detach().reshape(b * l, d)
        for it in range(inner_iters):
            x = _sample_tokens(mardm, z_opt, noise, cond_scale, g.ode_steps_guidance)
            full = context.clone()
            full[flat_mask] = x                       # grads flow only via masked tokens
            joints = latents_to_joints(full.reshape(b, l, d), ae, mean, std)
            loss = control_loss(joints, control)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if g.verbose and (it == 0 or it == inner_iters - 1):
                print(f"[guidance] {tag} it {it + 1}/{inner_iters} "
                      f"L_s={float(loss.detach()):.4f}", flush=True)
        z = z_opt.detach()

    with torch.no_grad():
        x = _sample_tokens(mardm, z, noise, cond_scale, g.ode_steps_final)
    flat = latents.reshape(b * l, d)
    flat[flat_mask] = x
    return flat.reshape(b, l, d)


def generate_guided(
    mardm: MARDM,
    ae: AE,
    cond: Tensor,
    m_lens: Tensor,
    control: ControlSignal,
    mean: Tensor,
    std: Tensor,
    *,
    timesteps: int,
    cond_scale: float,
    guidance: GuidanceConfig | None = None,
) -> Tensor:
    """Masked-AR sampling with per-step z-optimization against `control`.

    cond: (B, text_dim); m_lens: (B,) latent lengths; control targets/mask are
    at decoded-frame resolution (see `mardm.control.losses`).
    Returns latents (B, ae_dim, L) for AE.decode, like `MARDM.generate`.
    """
    g = guidance or GuidanceConfig()
    device = cond.device
    control = control.to(device)
    mean, std = mean.to(device), std.to(device)
    l = int(max(m_lens))
    b = len(m_lens)
    d = mardm.ae_dim
    padding_mask = ~lengths_to_mask(m_lens, l)

    latents = torch.where(
        padding_mask.unsqueeze(-1),
        torch.zeros(b, l, d, device=device),
        mardm.mask_latent.detach().repeat(b, l, 1),
    )
    masked_rand_schedule = torch.where(padding_mask, 1e5, torch.rand_like(padding_mask, dtype=torch.float))

    for step, timestep in enumerate(torch.linspace(0, 1, timesteps, device=device)):
        rand_mask_prob = cosine_schedule(timestep)
        num_masked = torch.round(rand_mask_prob * m_lens).clamp(min=1)
        ranks = masked_rand_schedule.argsort(dim=1).argsort(dim=1)
        is_mask = ranks < num_masked.unsqueeze(-1)

        latents = torch.where(is_mask.unsqueeze(-1), mardm.mask_latent.detach().repeat(b, l, 1), latents)
        latents = _guided_commit(mardm, ae, latents, is_mask, cond, padding_mask,
                                 cond_scale, control, mean, std, g, g.inner_iters,
                                 tag=f"step {step + 1}/{timesteps}")
        masked_rand_schedule = masked_rand_schedule.masked_fill(~is_mask, 1e5)

    # Re-prediction repair: remask the tokens the UNGUIDED prior most disagrees
    # with (i.e., the ones guidance pushed furthest off-manifold) and re-predict
    # them under light guidance, so the prior restores realism while the
    # waypoints stay pinned.
    for rnd in range(g.repair_rounds if control.num_constraints > 0 else 0):
        valid = ~padding_mask                              # (b, l)
        flat_valid = valid.reshape(b * l)
        z_prior = _compute_z(mardm, latents, cond, padding_mask, flat_valid, cond_scale)
        with torch.no_grad():
            noise_r = torch.randn(int(flat_valid.sum()), d, device=device)
            x_prior = _sample_tokens(mardm, z_prior, noise_r, cond_scale, g.ode_steps_final)
        disagree = torch.zeros(b, l, device=device)
        disagree.reshape(b * l)[flat_valid] = torch.linalg.vector_norm(
            x_prior - latents.reshape(b * l, d)[flat_valid], dim=-1)
        is_mask = torch.zeros(b, l, dtype=torch.bool, device=device)
        for i in range(b):
            n_i = int(m_lens[i])
            k = max(1, int(round(g.repair_frac * n_i)))
            top = disagree[i].topk(k).indices
            is_mask[i, top] = True
        is_mask &= valid
        if g.verbose:
            print(f"[guidance] repair {rnd + 1}/{g.repair_rounds}: remasking "
                  f"{int(is_mask.sum())} tokens (mean prior disagreement "
                  f"{float(disagree[valid].mean()):.3f})", flush=True)
        latents = torch.where(is_mask.unsqueeze(-1), mardm.mask_latent.detach().repeat(b, l, 1), latents)
        latents = _guided_commit(mardm, ae, latents, is_mask, cond, padding_mask,
                                 cond_scale, control, mean, std, g, g.repair_iters,
                                 tag=f"repair {rnd + 1}/{g.repair_rounds}")

    if g.post_iters > 0 and control.num_constraints > 0:
        lat_opt = latents.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([lat_opt], lr=g.post_lr)
        valid = (~padding_mask).unsqueeze(-1)
        for it in range(g.post_iters):
            joints = latents_to_joints(lat_opt * valid, ae, mean, std)
            loss = control_loss(joints, control)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if g.verbose and (it == 0 or it == g.post_iters - 1):
                print(f"[guidance] post it {it + 1}/{g.post_iters} "
                      f"L_s={float(loss.detach()):.4f}", flush=True)
        latents = lat_opt.detach()

    latents = torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(latents), latents)
    return latents.permute(0, 2, 1)
