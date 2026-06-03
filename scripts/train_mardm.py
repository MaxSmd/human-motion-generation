"""MARDM generation branch (stage 2) training — Hydra-driven.

Trains the masked-autoregressive transformer + per-token SiT diffusion head on
the *frozen* AE's latents, conditioned on text features. Per step: encode text
-> cond; `AE.encode(motion)` -> latents; `MARDM.forward_loss(latents, cond,
m_lens // downsample)`.

Usage (local smoke, CPU, random text encoder + the tiny AE smoke checkpoint):
    python scripts/train_mardm.py data.root=/tmp/synth stats_path=/tmp/synth/stats.pt \\
        ae_checkpoint=/tmp/mardm_ae_smoke/checkpoints/latest.pt \\
        ae.width=32 ae.output_emb_width=16 ae.depth=2 \\
        text_encoder.type=random text_encoder.text_dim=64 \\
        model.latent_dim=64 model.num_heads=4 model.ff_size=128 \\
        model.diffmlps_width=64 model.diffmlps_depth=2 model.diffmlps_batch_mul=2 \\
        train.max_steps=4 train.micro_batch_size=4 train.grad_accum=1 train.precision=fp32 \\
        data.num_workers=0 logging.use_wandb=false logging.use_tensorboard=false

Usage (cluster):
    python scripts/train_mardm.py ae_checkpoint=runs/mardm-ae-XXXX/checkpoints/latest.pt \\
        +data=cluster_mounted text_encoder.type=qwen3
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from mardm.data import EssentialDataset, collate
from mardm.models import AE, MARDM, AEConfig, MARDMConfig
from rmg.models import Qwen3EmbeddingEncoder, RandomTextEncoder, TextEncoder
from rmg.utils import (
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


def _load_stats(stats_path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    p = Path(stats_path)
    if not p.exists():
        raise FileNotFoundError(
            f"essential mean/std not found at {p}. Run scripts/compute_mardm_stats.py first."
        )
    blob = torch.load(p, weights_only=True)
    return blob["mean"], blob["std"]


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


def _load_frozen_ae(cfg: DictConfig, device: torch.device) -> AE:
    ae = AE(AEConfig(**OmegaConf.to_container(cfg.ae, resolve=True))).to(device)
    state = load_checkpoint(Path(cfg.ae_checkpoint), map_location=device)
    ae.load_state_dict(state.model)
    if cfg.ae_use_ema and state.ema is not None:
        ema = EMA(ae, decay=0.0)
        ema.load_state_dict(state.ema)
        ema.copy_to(ae)
        print(f"[gen] loaded AE EMA weights from step {state.step}")
    else:
        print(f"[gen] loaded AE live weights from step {state.step}")
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


def _build_dataset(cfg: DictConfig, split: str, mean, std, mirror: bool) -> EssentialDataset:
    return EssentialDataset(
        root=cfg.data.root, split=split, mean=mean, std=std, window_size=None,
        mirror_augment=mirror, max_seq_len=cfg.data.max_seq_len, min_seq_len=cfg.data.min_seq_len,
        subset_frac=cfg.get("subset_frac"), limit_clips=cfg.get("limit_clips"),
        zip_name=cfg.data.zip_name, splits_name=cfg.data.splits_name, offsets_name=cfg.data.offsets_name,
    )


def _build_loader(ds: EssentialDataset, cfg: DictConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds, batch_size=cfg.train.micro_batch_size, shuffle=shuffle, collate_fn=collate,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers and cfg.data.num_workers > 0,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
        worker_init_fn=worker_init_fn, drop_last=True,
    )


def _infinite(loader: DataLoader):
    # Guard: an empty loader (micro_batch_size > len(dataset) with drop_last=True)
    # would otherwise turn `while True: for _ in loader: ...` into a silent busy
    # loop — pegging one CPU and producing no batches, no logs, no errors.
    if len(loader) == 0:
        raise RuntimeError(
            f"DataLoader yields 0 batches per epoch (dataset size "
            f"{len(loader.dataset)} < batch_size {loader.batch_size} with "
            "drop_last=True). Reduce train.micro_batch_size or raise "
            "subset_frac/limit_clips."
        )
    while True:
        for batch in loader:
            yield batch


def _step_loss(ae: AE, mardm: MARDM, text_encoder: TextEncoder, batch, device: torch.device) -> torch.Tensor:
    x = batch.x1.to(device, non_blocking=True)
    with torch.no_grad():
        cond = text_encoder.encode(batch.texts, device=device)
        latents = ae.encode(x)                                   # (B, ae_dim, L)
    m_lens = (batch.lengths.to(device) // ae.downsample_rate).clamp(min=1)
    return mardm.forward_loss(latents, cond, m_lens)


@torch.no_grad()
def _validate(ae, mardm, text_encoder, loader, device, max_batches: int) -> float:
    mardm.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        total += float(_step_loss(ae, mardm, text_encoder, batch, device))
        n += 1
    mardm.train()
    return total / max(n, 1)


@hydra.main(config_path="../configs", config_name="mardm/gen", version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean, std = _load_stats(cfg.stats_path)
    text_encoder = _build_text_encoder(cfg)
    ae = _load_frozen_ae(cfg, device)

    mardm = MARDM(MARDMConfig(
        ae_dim=ae.output_emb_width,
        text_dim=cfg.text_encoder.text_dim,
        **OmegaConf.to_container(cfg.model, resolve=True),
    )).to(device)

    train_iter = _infinite(_build_loader(_build_dataset(cfg, "train", mean, std, cfg.data.mirror_augment), cfg, True))
    val_loader = _build_loader(_build_dataset(cfg, "val", mean, std, False), cfg, False)

    o = cfg.train.optimizer
    opt = torch.optim.AdamW(mardm.parameters(), lr=o.lr, betas=tuple(o.betas),
                            weight_decay=o.weight_decay, eps=o.eps)
    sched = build_scheduler(opt, total_steps=cfg.train.max_steps,
                            warmup_ratio=cfg.train.scheduler.warmup_ratio,
                            min_lr_ratio=cfg.train.scheduler.min_lr_ratio)
    ema = EMA(mardm, decay=cfg.train.ema.decay)

    use_amp = cfg.train.precision in ("bf16", "fp16")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.train.precision]
    scaler = torch.amp.GradScaler(device.type) if cfg.train.precision == "fp16" else None

    step = 0
    latest = find_latest_checkpoint(output_dir)
    if latest is not None:
        print(f"[gen] resuming from {latest}")
        state = load_checkpoint(latest, map_location=device)
        mardm.load_state_dict(state.model)
        if state.ema:
            ema.load_state_dict(state.ema)
        opt.load_state_dict(state.optimizer)
        if state.scheduler is not None:
            sched.load_state_dict(state.scheduler)
        if scaler is not None and state.scaler is not None:
            scaler.load_state_dict(state.scaler)
        if state.rng:
            restore_rng_state(state.rng)
        step = state.step

    logger = Logger(LoggerConfig(
        run_dir=output_dir, run_name=cfg.run_name,
        use_wandb=cfg.logging.use_wandb, use_tensorboard=cfg.logging.use_tensorboard,
        wandb_project=cfg.logging.wandb_project, wandb_mode=cfg.logging.wandb_mode,
        wandb_entity=cfg.logging.wandb_entity,
        config=OmegaConf.to_container(cfg, resolve=True),
    ))
    print(f"[gen] MARDM params: {mardm.num_params():,}  device: {device}  precision: {cfg.train.precision}")
    print(f"[gen] start step={step} max_steps={cfg.train.max_steps} "
          f"effective BS={cfg.train.micro_batch_size * cfg.train.grad_accum}")

    t_last = time.time()
    while step < cfg.train.max_steps:
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(cfg.train.grad_accum):
            batch = next(train_iter)
            ctx = torch.amp.autocast(device.type, dtype=amp_dtype) if use_amp else nullcontext()
            with ctx:
                loss = _step_loss(ae, mardm, text_encoder, batch, device) / cfg.train.grad_accum
            (scaler.scale(loss) if scaler is not None else loss).backward()
            accum_loss += float(loss.detach())

        if scaler is not None:
            scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(mardm.parameters(), cfg.train.grad_clip)
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        sched.step()
        ema.update(mardm)
        step += 1

        if step % cfg.train.log_every == 0 or step == 1:
            now = time.time()
            logger.log({
                "gen/loss": accum_loss * cfg.train.grad_accum,
                "lr": sched.get_last_lr()[0],
                "grad_norm": float(grad_norm),
                "steps_per_s": cfg.train.log_every / max(now - t_last, 1e-9),
            }, step=step)
            t_last = now

        if step % cfg.train.val_every == 0 or step == cfg.train.max_steps:
            with ema.swapped(mardm):
                val_loss = _validate(ae, mardm, text_encoder, val_loader, device, cfg.train.val_batches)
            logger.log({"gen/val_loss": val_loss}, step=step)
            print(f"[gen] step {step}: val_loss(ema)={val_loss:.4f}", flush=True)

        latest_every = max(1, int(cfg.train.ckpt_every) // 10)
        if step % latest_every == 0 or step % cfg.train.ckpt_every == 0 or step == cfg.train.max_steps:
            payload = TrainState(
                step=step, model=mardm.state_dict(), ema=ema.state_dict(),
                optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                scaler=scaler.state_dict() if scaler is not None else None,
                rng=collect_rng_state(), extras={"wandb_run_id": logger.wandb_run_id},
            )
            if step % cfg.train.ckpt_every == 0 or step == cfg.train.max_steps:
                save_checkpoint(output_dir / "checkpoints" / f"ckpt-{step:09d}.pt", payload)
            save_checkpoint(output_dir / "checkpoints" / "latest.pt", payload)

    logger.close()
    print(f"[gen] done at step {step}")


if __name__ == "__main__":
    main()
