"""RMG training entry point — Hydra-driven, single-GPU + grad-accum design.

Usage (local smoke):
    python -m rmg.scripts.train +model=dit_base +train=rmg_base \\
        train.max_steps=20 train.micro_batch_size=4 train.grad_accum=2 \\
        text_encoder.type=random run_name=local-smoke

Usage (cluster, full RMG-base):
    python -m rmg.scripts.train +model=dit_base +train=rmg_base \\
        text_encoder.type=qwen3 +data=cluster_mounted

The training loop:
  * Resamples (t, x_0) per micro-batch via FlowMatchingTrainer.
  * Accumulates gradient over `grad_accum` micro-batches; one optimizer step
    + scheduler.step + ema.update per outer step (effective BS = micro_bs *
    grad_accum).
  * AMP via torch.amp.autocast (bf16 by default; fp16 with GradScaler).
  * Saves a resumable checkpoint every `train.ckpt_every` steps + on exit.
  * Auto-resumes from the latest checkpoint under `<output_dir>/checkpoints/`.
"""

from __future__ import annotations

import math
import signal
import time
from contextlib import nullcontext
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from rmg.data import HumanML3DDataset, collate
from rmg.flow import (
    FlowMatchingTrainer,
    FlowMatchingTrainerCfg,
    RiemannianEulerSampler,
    SamplerCfg,
    WrappedGaussianPrior,
)
from rmg.models import (
    DiTConfig,
    Qwen3EmbeddingEncoder,
    RandomTextEncoder,
    RMGDiT,
    TextEncoder,
)
from rmg.representation import (
    NUM_JOINTS,
    Representation,
    Skeleton,
    build_representation,
)
from shared.utils import (
    EMA,
    Logger,
    LoggerConfig,
    TrainState,
    build_scheduler,
    collect_rng_state,
    find_latest_checkpoint,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    set_seed,
    worker_init_fn,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _build_dataset(cfg: DictConfig, split: str, representation: Representation) -> HumanML3DDataset:
    return HumanML3DDataset(
        root=cfg.data.root,
        split=split,
        max_seq_len=cfg.data.max_seq_len,
        min_seq_len=cfg.data.min_seq_len,
        zip_name=cfg.data.zip_name,
        splits_name=cfg.data.splits_name,
        offsets_name=cfg.data.offsets_name,
        representation=representation,
        subset_fraction=float(cfg.data.get("subset_fraction", 1.0)),
        subset_seed=int(cfg.data.get("subset_seed", 0)),
        subset_n=int(cfg.data.get("subset_n", 0)),
        preload=bool(cfg.data.get("preload", False)),
    )


def _build_loader(ds: HumanML3DDataset, cfg: DictConfig, batch_size: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers and cfg.data.num_workers > 0,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )


def _build_text_encoder(cfg: DictConfig) -> TextEncoder:
    t = cfg.text_encoder.type
    if t == "random":
        return RandomTextEncoder(text_dim=cfg.text_encoder.text_dim)
    if t == "qwen3":
        return Qwen3EmbeddingEncoder(
            model_name=cfg.text_encoder.model_name,
            cache_dir=cfg.text_encoder.cache_dir,
            max_length=cfg.text_encoder.max_length,
        )
    raise ValueError(f"Unknown text_encoder.type {t!r}")


def _build_model(cfg: DictConfig, representation: Representation) -> RMGDiT:
    # input_dim is dictated by the representation (tr=91, tp=69, trp=157, ...)
    dit_cfg = DiTConfig(
        input_dim=representation.ambient_dim,
        hidden_dim=cfg.model.hidden_dim,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        ffn_mult=cfg.model.ffn_mult,
        text_dim=cfg.model.text_dim,
        time_freq_dim=cfg.model.time_freq_dim,
        max_seq_len=cfg.model.max_seq_len,
    )
    return RMGDiT(dit_cfg)


def _build_optimizer(model: torch.nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    o = cfg.train.optimizer
    if o.name != "AdamW":
        raise ValueError(f"unsupported optimizer {o.name!r}")
    return torch.optim.AdamW(
        model.parameters(),
        lr=o.lr,
        betas=tuple(o.betas),
        weight_decay=o.weight_decay,
        eps=o.eps,
    )


# ---------------------------------------------------------------------------
# Iterator helpers
# ---------------------------------------------------------------------------


def _infinite(loader: DataLoader):
    """Yield batches forever, resetting the iterator at the end of each epoch."""
    while True:
        for batch in loader:
            yield batch


# ---------------------------------------------------------------------------
# Graceful pre-walltime stop
# ---------------------------------------------------------------------------

# SLURM (and the job manager's resubmit machinery) deliver SIGTERM/SIGUSR1 a
# little before the 24h walltime kill. We catch them, flush one last checkpoint,
# and exit cleanly WITHOUT writing the `.complete` marker — so the next
# resubmission resumes from latest.pt. The frequent latest.pt saves already cap
# the worst-case loss to a few hundred steps even on a hard kill; this just makes
# the graceful case lossless.
_STOP_REQUESTED = False


def _request_stop(signum, _frame) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"[train] caught signal {signum} — will checkpoint and exit at next step boundary")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Catch the pre-walltime / resubmit signals (see `_request_stop`).
    for _sig in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(_sig, _request_stop)

    set_seed(cfg.seed, deterministic=cfg.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------- representation (paper Figure 3 ablations) --------------------
    rep_kwargs = {k: v for k, v in dict(cfg.representation).items() if k not in ("name",)}
    representation = build_representation(cfg.representation.name, **rep_kwargs)
    print(f"[train] representation={representation.name} ambient_dim={representation.ambient_dim}")

    # -------------------- data --------------------
    train_ds = _build_dataset(cfg, split="train", representation=representation)
    train_loader = _build_loader(train_ds, cfg, batch_size=cfg.train.micro_batch_size)
    train_iter = _infinite(train_loader)

    # -------------------- text encoder --------------------
    text_encoder = _build_text_encoder(cfg)

    # -------------------- model + manifold --------------------
    M = representation.build_manifold()
    # Reference (rest-pose) point — for representations that need a skeleton
    # (T+P, T+R+P), use the dataset's target offsets to compute the canonical
    # T-pose pre-shape; otherwise fall back to the trivial rest pose.
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
    sampler = RiemannianEulerSampler(
        manifold=M, prior=prior,
        cfg=SamplerCfg(num_steps=cfg.train.num_sample_steps,
                       guidance_scale=cfg.train.guidance_scale),
    )

    # -------------------- optimizer / scheduler / ema / scaler --------------------
    opt = _build_optimizer(model, cfg)
    sched = build_scheduler(
        opt,
        total_steps=cfg.train.max_steps,
        warmup_ratio=cfg.train.scheduler.warmup_ratio,
        min_lr_ratio=cfg.train.scheduler.min_lr_ratio,
    )
    ema = EMA(model, decay=cfg.train.ema.decay)

    use_amp = cfg.train.precision in ("bf16", "fp16")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.train.precision]
    scaler = torch.amp.GradScaler(device.type) if cfg.train.precision == "fp16" else None

    # -------------------- resume? --------------------
    step = 0
    latest = find_latest_checkpoint(output_dir)
    if latest is not None:
        print(f"[train] resuming from {latest}")
        state = load_checkpoint(latest, map_location=device)
        model.load_state_dict(state.model)
        ema.load_state_dict(state.ema) if state.ema else None
        opt.load_state_dict(state.optimizer)
        if state.scheduler is not None:
            sched.load_state_dict(state.scheduler)
        if scaler is not None and state.scaler is not None:
            scaler.load_state_dict(state.scaler)
        if state.rng:
            restore_rng_state(state.rng)
        step = state.step

    # -------------------- logger --------------------
    logger = Logger(LoggerConfig(
        run_dir=output_dir,
        run_name=cfg.run_name,
        use_wandb=cfg.logging.use_wandb,
        use_tensorboard=cfg.logging.use_tensorboard,
        wandb_project=cfg.logging.wandb_project,
        wandb_mode=cfg.logging.wandb_mode,
        wandb_entity=cfg.logging.wandb_entity,
        config=OmegaConf.to_container(cfg, resolve=True),
    ))

    print(f"[train] params: {model.num_params():,}  device: {device}  precision: {cfg.train.precision}")
    print(f"[train] start step={step}  max_steps={cfg.train.max_steps}  effective BS={cfg.train.micro_batch_size * cfg.train.grad_accum}")

    # -------------------- training loop --------------------
    t_last = time.time()
    nan_skips = 0  # count of steps skipped due to non-finite loss/grad
    while step < cfg.train.max_steps:
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(cfg.train.grad_accum):
            batch = next(train_iter)
            x1 = batch.x1.to(device, non_blocking=True)
            mask = batch.mask.to(device, non_blocking=True)
            with torch.no_grad():
                cond = text_encoder.encode(batch.texts, device=device)

            ctx = (
                torch.amp.autocast(device.type, dtype=amp_dtype)
                if use_amp else nullcontext()
            )
            with ctx:
                loss, info = trainer.compute_loss(model, x1, cond=cond, mask=mask)
                loss = loss / cfg.train.grad_accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss += loss.detach().item()

        # Optimizer step
        if scaler is not None:
            scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)

        # Non-finite guard. On the sphere's antipodal cut locus the CFM target
        # velocity (θ/sin θ) can blow up to Inf/NaN for an unlucky prior/data
        # pair; clip_grad_norm_ can't help (a NaN grad-norm yields a NaN clip
        # coefficient, so the NaN flows straight into the weights) and a single
        # poisoned step corrupts model + EMA + Adam moments permanently. So we
        # skip the update entirely whenever the loss or grad is non-finite —
        # one wasted step instead of a dead run.
        skip_step = not (math.isfinite(accum_loss) and bool(torch.isfinite(grad_norm)))
        if skip_step:
            nan_skips += 1
            opt.zero_grad(set_to_none=True)
            print(f"[train] WARNING: non-finite step at step {step} "
                  f"(loss={accum_loss}, grad_norm={float(grad_norm)}); skipped "
                  f"optimizer + EMA update (total skips={nan_skips})", flush=True)
        else:
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            ema.update(model)
        sched.step()

        step += 1

        # ------------------------- log -------------------------
        if step % cfg.train.log_every == 0 or step == 1:
            now = time.time()
            steps_per_s = cfg.train.log_every / max(now - t_last, 1e-9)
            t_last = now
            lr = sched.get_last_lr()[0]
            logger.log(
                {
                    "step": step,
                    "loss": accum_loss * cfg.train.grad_accum,  # un-divide
                    "lr": lr,
                    "grad_norm": float(grad_norm),
                    "t_mean": float(info["t_mean"]),
                    "x_t_offmanifold_frac": float(info["x_t_offmanifold"]),
                    "steps_per_s": steps_per_s,
                    "nan_skips": nan_skips,
                },
                step=step,
            )

        # ------------------------- checkpoint -------------------------
        # latest.pt updates much more often than the numbered checkpoints so
        # that a 24h wallclock kill loses at most a few hundred steps. The
        # numbered ckpt-*.pt files are durable history for analysis / rollback.
        # `train.latest_every` decouples the cheap latest.pt cadence from the
        # heavy numbered cadence (a big model's numbered ckpt is GBs, so we keep
        # those sparse but still snapshot latest.pt frequently); defaults to
        # ckpt_every//10 to preserve the original behaviour.
        latest_every = int(cfg.train.get("latest_every", 0)) or max(1, int(cfg.train.ckpt_every) // 10)
        save_latest = step % latest_every == 0 or step == cfg.train.max_steps
        save_numbered = step % cfg.train.ckpt_every == 0 or step == cfg.train.max_steps

        if save_latest or save_numbered:
            state_dict_payload = TrainState(
                step=step,
                model=model.state_dict(),
                ema=ema.state_dict(),
                optimizer=opt.state_dict(),
                scheduler=sched.state_dict(),
                scaler=scaler.state_dict() if scaler is not None else None,
                rng=collect_rng_state(),
                extras={"wandb_run_id": logger.wandb_run_id},
            )
            if save_numbered:
                save_checkpoint(
                    output_dir / "checkpoints" / f"ckpt-{step:09d}.pt",
                    state_dict_payload,
                )
            if save_latest:
                save_checkpoint(
                    output_dir / "checkpoints" / "latest.pt",
                    state_dict_payload,
                )
                # Cheap progress beacon the job manager reads (over SSH) to gate
                # auto-resubmit: it resumes only while this keeps advancing.
                (output_dir / ".progress").write_text(str(step))

        # ------------------------- periodic samples -------------------------
        if step % cfg.train.sample_every == 0:
            sample_texts = [
                "a person walks forward",
                "a person waves their right hand",
                "a person sits down on the floor",
            ]
            with torch.no_grad():
                cond = text_encoder.encode(sample_texts, device=device)
                with ema.swapped(model):
                    samples = sampler.sample(
                        model,
                        shape=(len(sample_texts), cfg.model.max_seq_len // 2),
                        cond=cond,
                    )
            sample_path = output_dir / "samples" / f"step-{step:09d}.pt"
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"texts": sample_texts, "samples": samples.cpu()}, sample_path)

        # ------------------------- graceful stop -------------------------
        # Pre-walltime signal: flush latest.pt (if this step didn't already) and
        # exit without the `.complete` marker so the next job resumes here.
        if _STOP_REQUESTED:
            if not save_latest:
                save_checkpoint(
                    output_dir / "checkpoints" / "latest.pt",
                    TrainState(
                        step=step,
                        model=model.state_dict(),
                        ema=ema.state_dict(),
                        optimizer=opt.state_dict(),
                        scheduler=sched.state_dict(),
                        scaler=scaler.state_dict() if scaler is not None else None,
                        rng=collect_rng_state(),
                        extras={"wandb_run_id": logger.wandb_run_id},
                    ),
                )
                (output_dir / ".progress").write_text(str(step))
            print(f"[train] stopping early at step {step} (signal) — resumable from latest.pt")
            break

    logger.close()
    if step >= cfg.train.max_steps:
        # Durable "training finished" marker. The job manager checks for this to
        # decide done-vs-resubmit, so it must only be written on real completion.
        (output_dir / ".complete").write_text(str(step))
        print(f"[train] done at step {step} (.complete written)")
    else:
        print(f"[train] exited at step {step} / {cfg.train.max_steps} (not complete)")


if __name__ == "__main__":
    main()
