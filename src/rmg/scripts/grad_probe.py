"""Locate the source of non-finite gradients in a checkpoint's training step.

Reuses the exact training setup from `rmg.scripts.train` (same data, text
encoder, model, manifold, trainer) and runs a handful of real steps on a loaded
checkpoint, reporting for each:

  * the loss, and whether the *gradient* is finite — computed BOTH in the run's
    AMP precision (e.g. bf16) AND in fp32, so we can tell a precision overflow
    (finite in fp32, inf in bf16 → fixable by computing in fp32) apart from a
    genuine numerical singularity (inf in both).
  * the named parameters whose gradient is non-finite (the layer that blows up).
  * diagnostics that bound the inputs: CFM target magnitude, x_t on-manifold
    fraction, and the model's raw velocity magnitude.

Run via `slurm/rmg/grad_probe.sbatch` with the same MODEL_PRESET / PRESET as the
run under investigation, and CKPT=<path>. Read-only: never writes checkpoints.
"""

from __future__ import annotations

from contextlib import nullcontext

import hydra
import torch
from omegaconf import DictConfig

from rmg.flow import (
    FlowMatchingTrainer,
    FlowMatchingTrainerCfg,
    WrappedGaussianPrior,
)
from rmg.representation import build_representation
from shared.utils import load_checkpoint, set_seed

# Reuse the training builders verbatim so the probe mirrors real training.
from rmg.scripts.train import (
    _build_dataset,
    _build_loader,
    _build_model,
    _build_optimizer,
    _build_text_encoder,
    _infinite,
)


def _grad_report(model: torch.nn.Module) -> tuple[float, list[str]]:
    """Total grad norm (fp32) and the names of parameters with non-finite grad."""
    total_sq = 0.0
    bad: list[str] = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        if not torch.isfinite(g).all():
            bad.append(name)
        total_sq += float(g.pow(2).sum())
    return total_sq ** 0.5, bad


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rep_kwargs = {k: v for k, v in dict(cfg.representation).items() if k != "name"}
    representation = build_representation(cfg.representation.name, **rep_kwargs)

    train_ds = _build_dataset(cfg, split="train", representation=representation)
    train_loader = _build_loader(train_ds, cfg, batch_size=cfg.train.micro_batch_size)
    train_iter = _infinite(train_loader)
    text_encoder = _build_text_encoder(cfg)

    M = representation.build_manifold()
    if hasattr(representation, "prior_mu_from_skeleton") and train_ds._skeleton is not None:
        mu = representation.prior_mu_from_skeleton(train_ds._skeleton)
    else:
        mu = representation.prior_mu()
    prior = WrappedGaussianPrior(M, mu, sigma=cfg.train.prior_sigma)
    model = _build_model(cfg, representation).to(device)
    trainer = FlowMatchingTrainer(
        manifold=M, prior=prior,
        cfg=FlowMatchingTrainerCfg(cfg_dropout=cfg.train.cfg_dropout),
    )
    opt = _build_optimizer(model, cfg)  # only to mirror setup; never stepped

    ckpt = cfg.get("probe_ckpt", None)
    if not ckpt:
        raise ValueError("set +probe_ckpt=<path/to/ckpt.pt>")
    state = load_checkpoint(ckpt, map_location=device)
    model.load_state_dict(state.model)
    print(f"[grad-probe] loaded {ckpt} at step {state.step}", flush=True)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}[cfg.train.precision]
    n_steps = int(cfg.get("probe_steps", 20))
    grad_clip = float(cfg.train.grad_clip)
    print(f"[grad-probe] precision={cfg.train.precision} steps={n_steps} "
          f"grad_clip={grad_clip}", flush=True)

    model.train()
    for i in range(n_steps):
        batch = next(train_iter)
        x1 = batch.x1.to(device, non_blocking=True)
        mask = batch.mask.to(device, non_blocking=True)
        with torch.no_grad():
            cond = text_encoder.encode(batch.texts, device=device)

        # --- run the step under the run's AMP precision ---
        torch.manual_seed(1000 + i)  # fix t / x0 / dropout so amp vs fp32 match
        opt.zero_grad(set_to_none=True)
        amp_ctx = (torch.amp.autocast(device.type, dtype=amp_dtype)
                   if cfg.train.precision != "fp32" else nullcontext())
        with amp_ctx:
            loss_amp, info = trainer.compute_loss(model, x1, cond=cond, mask=mask)
        loss_amp.backward()
        gn_amp = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        norm_amp, bad_amp = _grad_report(model)

        # --- rerun the identical step in fp32 (no autocast) ---
        torch.manual_seed(1000 + i)
        opt.zero_grad(set_to_none=True)
        loss_fp32, _ = trainer.compute_loss(model, x1, cond=cond, mask=mask)
        loss_fp32.backward()
        gn_fp32 = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        norm_fp32, bad_fp32 = _grad_report(model)

        # --- input-side diagnostics (recompute target & x_t for this batch) ---
        with torch.no_grad():
            B, T, _ = x1.shape
            t = torch.rand(B, device=device)
            x0 = prior.sample((B, T), device=device, dtype=x1.dtype)
            from rmg.flow.interpolation import build_cfm_batch
            cb = build_cfm_batch(M, x0, x1, t)
            tgt_max = float(cb.target.abs().max())
            onman = float(M.validate(cb.x_t, atol=1e-2).float().mean())

        print(
            f"[grad-probe] step {i:2d}  "
            f"loss(amp)={float(loss_amp):.3f} loss(fp32)={float(loss_fp32):.3f}  "
            f"gradnorm(amp)={float(gn_amp):.3e} gradnorm(fp32)={float(gn_fp32):.3e}  "
            f"amp_finite={not bad_amp} fp32_finite={not bad_fp32}  "
            f"target_max={tgt_max:.3f} x_t_on_manifold={onman:.3f}",
            flush=True,
        )
        if bad_amp:
            print(f"    amp non-finite grads in: {bad_amp[:6]}"
                  f"{' …' if len(bad_amp) > 6 else ''} ({len(bad_amp)} total)",
                  flush=True)
        if bad_fp32:
            print(f"    fp32 non-finite grads in: {bad_fp32[:6]}"
                  f"{' …' if len(bad_fp32) > 6 else ''} ({len(bad_fp32)} total)",
                  flush=True)

    print("[grad-probe] done", flush=True)


if __name__ == "__main__":
    main()
