"""MARDM AutoEncoder (stage 1) training — Hydra-driven, single-GPU + grad-accum.

Trains the 1D-ResNet AE to reconstruct the z-normalized 67-D essential feature
(L1 loss), mirroring upstream MARDM's fixed-window AE training. The generation
branch (stage 2) then diffuses over this AE's frozen latents.

Usage (local smoke, CPU):
    python scripts/compute_mardm_stats.py --data-root /tmp/synth --out /tmp/synth/stats.pt
    python scripts/train_mardm_ae.py data.root=/tmp/synth stats_path=/tmp/synth/stats.pt \\
        train.max_steps=20 train.micro_batch_size=4 train.grad_accum=1 \\
        train.precision=fp32 logging.use_wandb=false logging.use_tensorboard=false

Usage (cluster):
    python scripts/train_mardm_ae.py +data=cluster_mounted
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from mardm.data import EssentialDataset, collate
from mardm.models import AE, AEConfig
from shared.utils import (
    EMA,
    Logger,
    LoggerConfig,
    TrainState,
    build_scheduler,
    collect_rng_state,
    find_latest_checkpoint,
    load_checkpoint,
    resolve_precision,
    restore_rng_state,
    save_checkpoint,
    set_seed,
    worker_init_fn,
)


def _load_stats(stats_path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    p = Path(stats_path)
    if not p.exists():
        raise FileNotFoundError(
            f"essential mean/std not found at {p}. Run "
            "`scripts/compute_mardm_stats.py --data-root <packed> --out <stats_path>` first."
        )
    blob = torch.load(p, weights_only=True)
    return blob["mean"], blob["std"]


def _build_dataset(cfg: DictConfig, split: str, mean, std, mirror: bool) -> EssentialDataset:
    return EssentialDataset(
        root=cfg.data.root,
        split=split,
        mean=mean,
        std=std,
        window_size=cfg.train.window_size,
        mirror_augment=mirror,
        max_seq_len=cfg.data.max_seq_len,
        min_seq_len=cfg.data.min_seq_len,
        subset_frac=cfg.get("subset_frac"),
        limit_clips=cfg.get("limit_clips"),
        zip_name=cfg.data.zip_name,
        splits_name=cfg.data.splits_name,
        offsets_name=cfg.data.offsets_name,
        preload=cfg.data.get("preload", False),
    )


def _build_loader(ds: EssentialDataset, cfg: DictConfig, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers and cfg.data.num_workers > 0,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
        worker_init_fn=worker_init_fn,
        drop_last=True,
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


def _recon_loss(
    ae: AE, x: torch.Tensor, mask: torch.Tensor, aux_loss_joints: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Smooth-L1 reconstruction + auxiliary local-position term.

    Mirrors upstream MARDM (`train_AE.py`):
        loss = criterion(recon, x) + aux_loss_joints * criterion(recon[..., 4:67], x[..., 4:67])
    The explicit term re-weights the 63 local-position dims relative to the 4
    root dims (rot_vel, lin_vel_xz, height). No FK / IK — direct slice on the
    normalized feature. Returns (total, feature_term, joint_term)."""
    recon = ae(x)
    mask_f = mask.to(recon.dtype)
    norm = mask_f.sum().clamp_min(1.0)

    per_frame_full = F.smooth_l1_loss(recon, x, reduction="none").mean(dim=-1)
    feature_l1 = (per_frame_full * mask_f).sum() / norm

    per_frame_pos = F.smooth_l1_loss(recon[..., 4:67], x[..., 4:67], reduction="none").mean(dim=-1)
    joint_l1 = (per_frame_pos * mask_f).sum() / norm

    total = feature_l1 + aux_loss_joints * joint_l1
    return total, feature_l1, joint_l1


@torch.no_grad()
def _validate(
    ae: AE, loader: DataLoader, device: torch.device, max_batches: int,
    aux_loss_joints: float = 1.0,
) -> tuple[float, float, float]:
    ae.eval()
    tot_total = tot_feat = tot_joint = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = batch.x1.to(device)
        mask = batch.mask.to(device)
        total, feat, joint = _recon_loss(ae, x, mask, aux_loss_joints=aux_loss_joints)
        tot_total += float(total)
        tot_feat += float(feat)
        tot_joint += float(joint)
        n += 1
    ae.train()
    denom = max(n, 1)
    return tot_total / denom, tot_feat / denom, tot_joint / denom


@hydra.main(config_path="../configs", config_name="ae", version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve precision against the local GPU and harmonize the dataloader
    # config — both are no-ops when the user already passed sensible values.
    OmegaConf.set_struct(cfg, False)
    cfg.train.precision = resolve_precision(cfg.train.precision)
    if cfg.data.preload and cfg.data.num_workers > 0:
        print(f"[data] preload=true → forcing num_workers=0 (was "
              f"{cfg.data.num_workers}); workers add IPC overhead without "
              "speedup when features live in RAM.")
        cfg.data.num_workers = 0
        cfg.data.persistent_workers = False
    OmegaConf.set_struct(cfg, True)

    mean, std = _load_stats(cfg.stats_path)

    train_ds = _build_dataset(cfg, "train", mean, std, mirror=cfg.data.mirror_augment)
    val_ds = _build_dataset(cfg, "val", mean, std, mirror=False)
    train_iter = _infinite(_build_loader(train_ds, cfg, cfg.train.micro_batch_size, shuffle=True))
    val_loader = _build_loader(val_ds, cfg, cfg.train.micro_batch_size, shuffle=False)

    ae = AE(AEConfig(**OmegaConf.to_container(cfg.ae, resolve=True))).to(device)

    o = cfg.train.optimizer
    opt = torch.optim.AdamW(ae.parameters(), lr=o.lr, betas=tuple(o.betas),
                            weight_decay=o.weight_decay, eps=o.eps)
    sched = build_scheduler(opt, total_steps=cfg.train.max_steps,
                            warmup_ratio=cfg.train.scheduler.warmup_ratio,
                            min_lr_ratio=cfg.train.scheduler.min_lr_ratio)
    ema = EMA(ae, decay=cfg.train.ema.decay)

    use_amp = cfg.train.precision in ("bf16", "fp16")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.train.precision]
    scaler = torch.amp.GradScaler(device.type) if cfg.train.precision == "fp16" else None

    step = 0
    latest = find_latest_checkpoint(output_dir)
    if latest is not None:
        print(f"[ae] resuming from {latest}")
        state = load_checkpoint(latest, map_location=device)
        ae.load_state_dict(state.model)
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
    print(f"[ae] params: {ae.num_params():,}  device: {device}  precision: {cfg.train.precision}")
    print(f"[ae] start step={step} max_steps={cfg.train.max_steps} "
          f"effective BS={cfg.train.micro_batch_size * cfg.train.grad_accum}")

    aux_w = float(cfg.train.get("aux_loss_joints", 1.0))
    t_last = time.time()
    while step < cfg.train.max_steps:
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        accum_feat = 0.0
        accum_joint = 0.0
        for _ in range(cfg.train.grad_accum):
            batch = next(train_iter)
            x = batch.x1.to(device, non_blocking=True)
            mask = batch.mask.to(device, non_blocking=True)
            ctx = torch.amp.autocast(device.type, dtype=amp_dtype) if use_amp else nullcontext()
            with ctx:
                total, feat_l1, joint_l1 = _recon_loss(ae, x, mask, aux_loss_joints=aux_w)
                loss = total / cfg.train.grad_accum
            (scaler.scale(loss) if scaler is not None else loss).backward()
            accum_loss += float(loss.detach())
            accum_feat += float(feat_l1.detach()) / cfg.train.grad_accum
            accum_joint += float(joint_l1.detach()) / cfg.train.grad_accum

        if scaler is not None:
            scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(ae.parameters(), cfg.train.grad_clip)
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        sched.step()
        ema.update(ae)
        step += 1

        if step % cfg.train.log_every == 0 or step == 1:
            now = time.time()
            logger.log({
                "ae/l1": accum_loss * cfg.train.grad_accum,
                "ae/feature_l1": accum_feat,
                "ae/joint_l1": accum_joint,
                "lr": sched.get_last_lr()[0],
                "grad_norm": float(grad_norm),
                "steps_per_s": cfg.train.log_every / max(now - t_last, 1e-9),
            }, step=step)
            t_last = now

        if step % cfg.train.val_every == 0 or step == cfg.train.max_steps:
            with ema.swapped(ae):
                val_total, val_feat, val_joint = _validate(
                    ae, val_loader, device, cfg.train.val_batches, aux_loss_joints=aux_w,
                )
            logger.log({
                "ae/val_l1": val_total,
                "ae/val_feature_l1": val_feat,
                "ae/val_joint_l1": val_joint,
            }, step=step)
            print(
                f"[ae] step {step}: val_l1(ema)={val_total:.4f} "
                f"(feature={val_feat:.4f} joint={val_joint:.4f})",
                flush=True,
            )

        latest_every = max(1, int(cfg.train.ckpt_every) // 10)
        if step % latest_every == 0 or step % cfg.train.ckpt_every == 0 or step == cfg.train.max_steps:
            payload = TrainState(
                step=step, model=ae.state_dict(), ema=ema.state_dict(),
                optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                scaler=scaler.state_dict() if scaler is not None else None,
                rng=collect_rng_state(), extras={"wandb_run_id": logger.wandb_run_id},
            )
            if step % cfg.train.ckpt_every == 0 or step == cfg.train.max_steps:
                save_checkpoint(output_dir / "checkpoints" / f"ckpt-{step:09d}.pt", payload)
            save_checkpoint(output_dir / "checkpoints" / "latest.pt", payload)

    logger.close()
    print(f"[ae] done at step {step}")


if __name__ == "__main__":
    main()
