"""Train the MARDM condition regularizer (spatial control, phase 2).

ControlNet-style copy of the FROZEN base MARDM's transformer that injects
spatial-control residuals (`mardm.control.regularizer`). Per step: encode text
-> cond (with manual CFG dropout — the frozen base won't drop); AE.encode ->
latents; sample a control signal from the batch's own GT joints (one random
joint from the OmniControl set, random keyframe density); loss =
alpha * L_diff + (1 - alpha) * L_s.

Only the regularizer trains — the base MARDM, its diffusion head, and the AE
stay frozen; the regularizer checkpoint is standalone (no base weights).

Usage (cluster):
    python -m mardm.scripts.train_mardm_control \\
        ae_checkpoint=runs/mardm-ae-canon/checkpoints/latest.pt \\
        gen_checkpoint=runs/mardm-gen-m-canon/checkpoints/latest.pt \\
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

from mardm.control import ControlSignal
from mardm.control.regularizer import ControlMARDM, control_forward_loss
from mardm.data import EssentialDataset, collate
from mardm.models import AE, MARDM, AEConfig, MARDMConfig
from mardm.representation import denormalize
from shared.geometry import recover_joints_from_ric
from shared.text import Qwen3EmbeddingEncoder, RandomTextEncoder, TextEncoder
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
        raise FileNotFoundError(f"essential mean/std not found at {p}")
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
    ae.load_state_dict(state.model, strict=False)
    if cfg.ae_use_ema and state.ema is not None:
        ema = EMA(ae, decay=0.0)
        ema.load_state_dict(state.ema)
        ema.copy_to(ae)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


def _load_frozen_base(cfg: DictConfig, ae: AE, device: torch.device) -> MARDM:
    base = MARDM(MARDMConfig(
        ae_dim=ae.output_emb_width, text_dim=cfg.text_encoder.text_dim,
        **OmegaConf.to_container(cfg.model, resolve=True),
    )).to(device)
    state = load_checkpoint(Path(cfg.gen_checkpoint), map_location=device)
    base.load_state_dict(state.model)
    if cfg.gen_use_ema and state.ema is not None:
        ema = EMA(base, decay=0.0)
        ema.load_state_dict(state.ema)
        ema.copy_to(base)
        print(f"[control] loaded base MARDM EMA weights from step {state.step}", flush=True)
    else:
        print(f"[control] loaded base MARDM live weights from step {state.step}", flush=True)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    return base


def _build_dataset(cfg: DictConfig, split: str, mean, std) -> EssentialDataset:
    return EssentialDataset(
        root=cfg.data.root, split=split, mean=mean, std=std, window_size=None,
        max_seq_len=cfg.data.max_seq_len, min_seq_len=cfg.data.min_seq_len,
        subset_frac=cfg.get("subset_frac"), limit_clips=cfg.get("limit_clips"),
        zip_name=cfg.data.zip_name, splits_name=cfg.data.splits_name, offsets_name=cfg.data.offsets_name,
        preload=cfg.data.get("preload", False),
        canonical_dir=cfg.data.get("canonical_dir"),
    )


def _build_loader(ds: EssentialDataset, cfg: DictConfig, shuffle: bool,
                  drop_last: bool = True) -> DataLoader:
    return DataLoader(
        ds, batch_size=cfg.train.micro_batch_size, shuffle=shuffle, collate_fn=collate,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers and cfg.data.num_workers > 0,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
        worker_init_fn=worker_init_fn, drop_last=drop_last,
    )


def _infinite(loader: DataLoader):
    if len(loader) == 0:
        raise RuntimeError(
            f"DataLoader yields 0 batches per epoch (dataset size {len(loader.dataset)} "
            f"< batch_size {loader.batch_size} with drop_last=True)."
        )
    while True:
        for batch in loader:
            yield batch


def _sample_control(x_norm: torch.Tensor, lengths: torch.Tensor, mean, std,
                    joint_set: list[int]) -> ControlSignal:
    """Random control signal from the batch's own GT joints.

    One random joint from `joint_set` per sample, keyframe count uniform in
    [1, valid length] (sparse-to-dense, per MaskControl's density ablation).
    """
    with torch.no_grad():
        joints = recover_joints_from_ric(denormalize(x_norm, mean, std))  # (B, T, 22, 3)
    b, t = joints.shape[:2]
    mask = torch.zeros(b, t, joints.shape[2], dtype=torch.bool, device=joints.device)
    for i in range(b):
        n_i = int(lengths[i])
        j = joint_set[int(torch.randint(len(joint_set), (1,)))]
        k = int(torch.randint(1, max(2, n_i + 1), (1,)))
        frames = torch.randperm(n_i, device=joints.device)[:k]
        mask[i, frames, j] = True
    return ControlSignal(joints, mask)


def _step_loss(ae: AE, reg: ControlMARDM, text_encoder: TextEncoder, batch,
               mean, std, cfg, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = batch.x1.to(device, non_blocking=True)
    lengths = batch.lengths.to(device)
    with torch.no_grad():
        cond = text_encoder.encode(batch.texts, device=device)
        # Manual CFG dropout — frozen base is in eval mode and won't drop.
        drop = float(cfg.control.cond_drop_prob)
        if drop > 0:
            keep = (torch.rand(cond.shape[0], device=device) >= drop).float().unsqueeze(1)
            cond = cond * keep
        latents = ae.encode(x)                                   # (B, ae_dim, L)
    control = _sample_control(x, lengths, mean.to(device), std.to(device),
                              [int(j) for j in cfg.control.joints])
    m_lens = (lengths // ae.downsample_rate).clamp(min=1)
    return control_forward_loss(reg, ae, latents, cond, m_lens, control,
                                mean.to(device), std.to(device))


@torch.no_grad()
def _validate(ae, reg, text_encoder, loader, mean, std, cfg, device, max_batches: int):
    reg.eval()
    tot_d, tot_s, n = 0.0, 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        l_diff, l_s = _step_loss(ae, reg, text_encoder, batch, mean, std, cfg, device)
        tot_d += float(l_diff)
        tot_s += float(l_s)
        n += 1
    reg.train()
    if n == 0:
        return float("nan"), float("nan")
    return tot_d / n, tot_s / n


@hydra.main(config_path="../configs", config_name="control", version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    OmegaConf.set_struct(cfg, False)
    cfg.train.precision = resolve_precision(cfg.train.precision)
    if cfg.data.preload and cfg.data.num_workers > 0:
        cfg.data.num_workers = 0
        cfg.data.persistent_workers = False
    OmegaConf.set_struct(cfg, True)

    mean, std = _load_stats(cfg.stats_path)
    text_encoder = _build_text_encoder(cfg)
    ae = _load_frozen_ae(cfg, device)
    base = _load_frozen_base(cfg, ae, device)
    reg = ControlMARDM(base, frames_per_latent=ae.downsample_rate).to(device)

    train_iter = _infinite(_build_loader(_build_dataset(cfg, "train", mean, std), cfg, True))
    val_loader = _build_loader(_build_dataset(cfg, "val", mean, std), cfg, False, drop_last=False)

    o = cfg.train.optimizer
    opt = torch.optim.AdamW([p for p in reg.parameters() if p.requires_grad],
                            lr=o.lr, betas=tuple(o.betas), weight_decay=o.weight_decay, eps=o.eps)
    sched = build_scheduler(opt, total_steps=cfg.train.max_steps,
                            warmup_ratio=cfg.train.scheduler.warmup_ratio,
                            min_lr_ratio=cfg.train.scheduler.min_lr_ratio)
    ema = EMA(reg, decay=cfg.train.ema.decay)

    use_amp = cfg.train.precision in ("bf16", "fp16")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.train.precision]
    scaler = torch.amp.GradScaler(device.type) if cfg.train.precision == "fp16" else None

    step = 0
    latest = find_latest_checkpoint(output_dir)
    if latest is not None:
        print(f"[control] resuming from {latest}", flush=True)
        state = load_checkpoint(latest, map_location=device)
        reg.load_state_dict(state.model)
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
    alpha = float(cfg.control.alpha)
    print(f"[control] regularizer params: {reg.num_params():,} (base frozen: "
          f"{base.num_params():,})  alpha={alpha}  precision={cfg.train.precision}", flush=True)
    print(f"[control] start step={step} max_steps={cfg.train.max_steps} "
          f"effective BS={cfg.train.micro_batch_size * cfg.train.grad_accum}", flush=True)

    t_last = time.time()
    while step < cfg.train.max_steps:
        opt.zero_grad(set_to_none=True)
        acc_d, acc_s = 0.0, 0.0
        for _ in range(cfg.train.grad_accum):
            batch = next(train_iter)
            ctx = torch.amp.autocast(device.type, dtype=amp_dtype) if use_amp else nullcontext()
            with ctx:
                l_diff, l_s = _step_loss(ae, reg, text_encoder, batch, mean, std, cfg, device)
                loss = (alpha * l_diff + (1.0 - alpha) * l_s) / cfg.train.grad_accum
            (scaler.scale(loss) if scaler is not None else loss).backward()
            acc_d += float(l_diff.detach()) / cfg.train.grad_accum
            acc_s += float(l_s.detach()) / cfg.train.grad_accum

        if scaler is not None:
            scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(reg.parameters(), cfg.train.grad_clip)
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        sched.step()
        ema.update(reg)
        step += 1

        if step % cfg.train.log_every == 0 or step == 1:
            now = time.time()
            logger.log({
                "control/l_diff": acc_d, "control/l_s": acc_s,
                "control/loss": alpha * acc_d + (1 - alpha) * acc_s,
                "lr": sched.get_last_lr()[0], "grad_norm": float(grad_norm),
                "steps_per_s": cfg.train.log_every / max(now - t_last, 1e-9),
            }, step=step)
            print(f"[control] step {step}: l_diff={acc_d:.4f} l_s={acc_s:.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e}", flush=True)
            t_last = now

        if step % cfg.train.val_every == 0 or step == cfg.train.max_steps:
            with ema.swapped(reg):
                vd, vs = _validate(ae, reg, text_encoder, val_loader, mean, std, cfg,
                                   device, cfg.train.val_batches)
            logger.log({"control/val_l_diff": vd, "control/val_l_s": vs}, step=step)
            print(f"[control] step {step}: val(ema) l_diff={vd:.4f} l_s={vs:.4f}", flush=True)

        latest_every = max(1, int(cfg.train.ckpt_every) // 10)
        if step % latest_every == 0 or step % cfg.train.ckpt_every == 0 or step == cfg.train.max_steps:
            payload = TrainState(
                step=step, model=reg.state_dict(), ema=ema.state_dict(),
                optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                scaler=scaler.state_dict() if scaler is not None else None,
                rng=collect_rng_state(), extras={"wandb_run_id": logger.wandb_run_id},
            )
            if step % cfg.train.ckpt_every == 0 or step == cfg.train.max_steps:
                save_checkpoint(output_dir / "checkpoints" / f"ckpt-{step:09d}.pt", payload)
            save_checkpoint(output_dir / "checkpoints" / "latest.pt", payload)

    logger.close()
    print(f"[control] done at step {step}", flush=True)


if __name__ == "__main__":
    main()
