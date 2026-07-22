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
guidance, pulling the motion back on-manifold without losing the waypoints.
`repair_every` interleaves such rounds INSIDE the AR loop (over the committed
prefix only) so restoration happens before the frozen context can propagate.
Additional runtime-only trust-region knobs — `tolerance` early stop,
`prox_weight` anchor, `guidance_start_frac`/`guidance_ramp` schedule — are our
own additions, not from MaskControl. See
src/mardm/reports/maskcontrol-differences.md for the exact mapping.
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
    # Runtime trust-region knobs (our own additions, not from MaskControl —
    # label as such if reported; see maskcontrol-differences.md):
    # stop the inner loop once mean control error drops below `tolerance`
    # meters (0 = optimize all inner_iters), and penalize drift from the
    # prior's condition with `prox_weight`·mean‖z − z₀‖².
    tolerance: float = 0.0
    prox_weight: float = 0.0
    # Guidance schedule over AR steps: skip z-optimization while
    # step/(timesteps-1) < `guidance_start_frac` (early steps have an empty
    # context — perturbing them does the most structural damage); with
    # `guidance_ramp` the iteration count ramps linearly from 1 at start_frac
    # to `inner_iters` at the final step instead of switching on at full.
    guidance_start_frac: float = 0.0
    guidance_ramp: bool = False
    post_iters: int = 0          # direct latent optimization after the AR loop
    post_lr: float = 0.01
    # Re-prediction repair (explicit analogue of MaskControl's remask-and-
    # repredict restore; see src/mardm/reports/maskcontrol-differences.md).
    # After the AR loop: remask the `repair_frac` of tokens the UNGUIDED prior
    # most disagrees with and re-predict them under light guidance
    # (`repair_iters` inner steps), `repair_rounds` times. With
    # `repair_every` = N > 0, a repair round also runs every N AR steps over
    # the tokens committed so far — restoring realism while the remaining
    # context is still open, before the freeze can propagate.
    repair_rounds: int = 0
    repair_frac: float = 0.5
    repair_iters: int = 10
    repair_every: int = 0
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
               flat_mask: Tensor, cond_scale: float,
               regularizer=None, control_feats: Tensor | None = None) -> Tensor:
    """Transformer condition for the masked tokens, uncond half stacked under CFG.

    With a trained `regularizer` (phase 2, `mardm.control.regularizer`), its
    residuals are injected into BOTH CFG branches — the spatial signal is
    caption-independent.
    """
    b, l, _ = latents.shape
    with torch.no_grad():
        res = (regularizer.residuals(latents, cond, padding_mask, control_feats)
               if regularizer is not None else None)
        z = mardm.forward(latents, cond, padding_mask,
                          control_residuals=res).reshape(b * l, -1)[flat_mask]
        if cond_scale != 1.0:
            res_u = (regularizer.residuals(latents, cond, padding_mask, control_feats,
                                           force_mask=True)
                     if regularizer is not None else None)
            z_uncond = mardm.forward(latents, cond, padding_mask, force_mask=True,
                                     control_residuals=res_u)
            z = torch.cat([z, z_uncond.reshape(b * l, -1)[flat_mask]], dim=0)
    return z


def _guided_commit(mardm: MARDM, ae: AE, latents: Tensor, is_mask: Tensor, cond: Tensor,
                   padding_mask: Tensor, cond_scale: float, control: ControlSignal,
                   mean: Tensor, std: Tensor, g: GuidanceConfig, inner_iters: int,
                   tag: str, regularizer=None, control_feats: Tensor | None = None) -> Tensor:
    """One masked-prediction step: optimize z for the masked tokens, commit samples."""
    b, l, d = latents.shape
    flat_mask = is_mask.reshape(b * l)
    z = _compute_z(mardm, latents, cond, padding_mask, flat_mask, cond_scale,
                   regularizer=regularizer, control_feats=control_feats)
    # Fixed noise per step so the inner optimization is deterministic.
    noise = torch.randn(int(flat_mask.sum()), d, device=latents.device)

    if inner_iters > 0 and control.num_constraints > 0:
        z0 = z.detach().clone()
        z_opt = z.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([z_opt], lr=g.lr)
        context = latents.detach().reshape(b * l, d)
        for it in range(inner_iters):
            x = _sample_tokens(mardm, z_opt, noise, cond_scale, g.ode_steps_guidance)
            full = context.clone()
            full[flat_mask] = x                       # grads flow only via masked tokens
            joints = latents_to_joints(full.reshape(b, l, d), ae, mean, std)
            ctrl_err = control_loss(joints, control)
            if g.tolerance > 0 and float(ctrl_err.detach()) < g.tolerance:
                if g.verbose:
                    print(f"[guidance] {tag} early stop at it {it + 1}/{inner_iters} "
                          f"L_s={float(ctrl_err.detach()):.4f} < tol {g.tolerance}", flush=True)
                break
            loss = ctrl_err
            if g.prox_weight > 0:
                loss = loss + g.prox_weight * (z_opt - z0).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if g.verbose and (it == 0 or it == inner_iters - 1):
                print(f"[guidance] {tag} it {it + 1}/{inner_iters} "
                      f"L_s={float(ctrl_err.detach()):.4f}", flush=True)
        z = z_opt.detach()

    with torch.no_grad():
        x = _sample_tokens(mardm, z, noise, cond_scale, g.ode_steps_final)
    flat = latents.reshape(b * l, d)
    flat[flat_mask] = x
    return flat.reshape(b, l, d)


def _repair_round(mardm: MARDM, ae: AE, latents: Tensor, eligible: Tensor, cond: Tensor,
                  padding_mask: Tensor, cond_scale: float, control: ControlSignal,
                  mean: Tensor, std: Tensor, g: GuidanceConfig, tag: str,
                  regularizer=None, control_feats: Tensor | None = None) -> Tensor:
    """One remask-and-repredict restore round over the `eligible` tokens.

    Re-predicts every eligible token with the unguided prior, remasks the
    `repair_frac` the prior most disagrees with, and re-predicts those under
    light guidance (`repair_iters`). Post-hoc repair passes eligible = all
    valid tokens; interleaved repair (`repair_every`) passes only the tokens
    committed so far.
    """
    b, l, d = latents.shape
    flat_elig = eligible.reshape(b * l)
    if not bool(flat_elig.any()):
        return latents
    z_prior = _compute_z(mardm, latents, cond, padding_mask, flat_elig, cond_scale,
                         regularizer=regularizer, control_feats=control_feats)
    with torch.no_grad():
        noise_r = torch.randn(int(flat_elig.sum()), d, device=latents.device)
        x_prior = _sample_tokens(mardm, z_prior, noise_r, cond_scale, g.ode_steps_final)
    disagree = torch.full((b, l), -1.0, device=latents.device)
    disagree.reshape(b * l)[flat_elig] = torch.linalg.vector_norm(
        x_prior - latents.reshape(b * l, d)[flat_elig], dim=-1)
    is_mask = torch.zeros(b, l, dtype=torch.bool, device=latents.device)
    for i in range(b):
        n_i = int(eligible[i].sum())
        if n_i == 0:
            continue
        k = max(1, int(round(g.repair_frac * n_i)))
        top = disagree[i].topk(k).indices
        is_mask[i, top] = True
    is_mask &= eligible
    if g.verbose:
        print(f"[guidance] {tag}: remasking {int(is_mask.sum())} tokens "
              f"(mean prior disagreement {float(disagree[eligible].mean()):.3f})", flush=True)
    latents = torch.where(is_mask.unsqueeze(-1), mardm.mask_latent.detach().repeat(b, l, 1), latents)
    return _guided_commit(mardm, ae, latents, is_mask, cond, padding_mask,
                          cond_scale, control, mean, std, g, g.repair_iters,
                          tag=tag, regularizer=regularizer, control_feats=control_feats)


def _iters_for_step(g: GuidanceConfig, step: int, timesteps: int) -> int:
    """Inner-iteration budget for one AR step under the guidance schedule."""
    if g.inner_iters == 0:
        return 0
    frac = step / max(timesteps - 1, 1)
    if frac < g.guidance_start_frac:
        return 0
    if g.guidance_ramp:
        span = max(1.0 - g.guidance_start_frac, 1e-8)
        return max(1, round(g.inner_iters * (frac - g.guidance_start_frac) / span))
    return g.inner_iters


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
    regularizer=None,
) -> Tensor:
    """Masked-AR sampling with per-step z-optimization against `control`.

    cond: (B, text_dim); m_lens: (B,) latent lengths; control targets/mask are
    at decoded-frame resolution (see `mardm.control.losses`).
    `regularizer`: optional trained `ControlMARDM` (phase 2) — its residuals
    are injected into every transformer pass (both CFG branches); combine with
    `inner_iters=0` for regularizer-only sampling, or > 0 for the full
    MaskControl-style regularizer + optimization stack.
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

    control_feats = None
    if regularizer is not None:
        from .regularizer import control_signal_features
        control_feats = control_signal_features(
            control, l, regularizer.frames_per_latent).to(device)

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
                                 cond_scale, control, mean, std, g,
                                 _iters_for_step(g, step, timesteps),
                                 tag=f"step {step + 1}/{timesteps}",
                                 regularizer=regularizer, control_feats=control_feats)
        masked_rand_schedule = masked_rand_schedule.masked_fill(~is_mask, 1e5)

        # Interleaved repair: restore the committed prefix while the rest of
        # the context is still open, so re-prediction pulls toward the prior
        # instead of toward an already-frozen sequence.
        if (g.repair_every > 0 and control.num_constraints > 0
                and (step + 1) % g.repair_every == 0 and step + 1 < timesteps):
            committed = (masked_rand_schedule >= 1e5) & ~padding_mask
            latents = _repair_round(mardm, ae, latents, committed, cond, padding_mask,
                                    cond_scale, control, mean, std, g,
                                    tag=f"inline repair @step {step + 1}/{timesteps}",
                                    regularizer=regularizer, control_feats=control_feats)

    # Re-prediction repair: remask the tokens the UNGUIDED prior most disagrees
    # with (i.e., the ones guidance pushed furthest off-manifold) and re-predict
    # them under light guidance, so the prior restores realism while the
    # waypoints stay pinned.
    for rnd in range(g.repair_rounds if control.num_constraints > 0 else 0):
        latents = _repair_round(mardm, ae, latents, ~padding_mask, cond, padding_mask,
                                cond_scale, control, mean, std, g,
                                tag=f"repair {rnd + 1}/{g.repair_rounds}",
                                regularizer=regularizer, control_feats=control_feats)

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
