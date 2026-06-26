"""Tiny MoMask overfit/smoke run on the shared HumanML3D packed dataset.

This is not the full MoMask training recipe. It is a cheap check that the data
path and the three model stages can learn something on a very small subset:

  1. Load `HumanML3DDataset(..., output_mode="h3d_263")`.
  2. Train a small MotionRVQVAE for a handful of steps.
  3. Freeze/tokenize with the VQ-VAE.
  4. Train the masked base-token transformer and residual transformer briefly.
  5. Print reconstruction MAE and token losses.
"""

from __future__ import annotations

import argparse
import os
import random
from itertools import cycle
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from momask.models import (
    MaskedMotionTransformer,
    MotionRVQVAE,
    ResidualTransformer,
    TokenTransformerConfig,
)
from rmg.data import HumanML3DDataset, collate
from rmg.models import RandomTextEncoder
from rmg.representation import H3D_FEATURE_DIM, recover_joints_from_ric


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-root",
        default=os.environ.get("RMG_DATA_ROOT", "external/data/humanml3d_packed"),
        help="Directory containing humanml3d.zip, splits.json, target_offsets.pt.",
    )
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--max-clips", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-seq-len", type=int, default=80)
    p.add_argument("--min-seq-len", type=int, default=20)
    p.add_argument("--vq-steps", type=int, default=30)
    p.add_argument("--token-steps", type=int, default=20)
    p.add_argument("--vq-only", action="store_true", help="Train/evaluate only the VQ-VAE tokenizer, then save.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--output-dir", default="runs/momask-smoke")
    p.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="For VQ-only runs, save lightweight VQ training checkpoints every N steps. 0 disables.",
    )
    p.add_argument(
        "--no-momask-normalize",
        action="store_true",
        help="Disable official MoMask-style HumanML3D feature normalization.",
    )
    p.add_argument(
        "--feat-bias",
        type=float,
        default=5.0,
        help="Official MoMask feature bias for root channels and foot contacts.",
    )

    # Small by default, but now scalable enough to test real capacity changes.
    p.add_argument("--vq-hidden-dim", type=int, default=64)
    p.add_argument("--vq-latent-dim", type=int, default=32)
    p.add_argument("--num-quantizers", type=int, default=3)
    p.add_argument("--codebook-size", type=int, default=64)
    p.add_argument("--vq-res-blocks", type=int, default=1)
    p.add_argument("--downsample", type=int, default=1)
    p.add_argument("--quantize-dropout", type=float, default=0.2)
    p.add_argument("--vq-commitment-weight", type=float, default=0.25)
    p.add_argument(
        "--vq-velocity-weight",
        type=float,
        default=0.0,
        help="Weight for matching frame-to-frame feature velocities in the VQ-VAE loss.",
    )

    p.add_argument("--text-dim", type=int, default=64)
    p.add_argument("--transformer-hidden-dim", type=int, default=64)
    p.add_argument("--transformer-depth", type=int, default=2)
    p.add_argument("--transformer-heads", type=int, default=4)
    p.add_argument("--transformer-ffn-dim", type=int, default=128)
    p.add_argument("--transformer-dropout", type=float, default=0.0)
    p.add_argument("--base-cond-drop", type=float, default=0.1)
    p.add_argument("--residual-cond-drop", type=float, default=0.2)
    p.add_argument(
        "--base-full-mask-prob",
        type=float,
        default=0.3,
        help="Probability that base-token training masks every valid token, matching generation startup.",
    )
    p.add_argument("--generation-steps", type=int, default=4)
    p.add_argument("--shared-residual-head", action="store_true")
    p.add_argument("--live-token-crops", action="store_true",
                   help="Train token models from the live dataset instead of cached fixed VQ tokens.")
    return p.parse_args()


class H3DNormalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != (H3D_FEATURE_DIM,) or std.shape != (H3D_FEATURE_DIM,):
            raise ValueError(f"mean/std must be ({H3D_FEATURE_DIM},), got {tuple(mean.shape)} and {tuple(std.shape)}")
        self.mean = mean.float()
        self.std = std.float().clamp_min(1e-6)

    @classmethod
    def identity(cls) -> "H3DNormalizer":
        return cls(torch.zeros(H3D_FEATURE_DIM), torch.ones(H3D_FEATURE_DIM))

    @classmethod
    def from_dataset(cls, ds: Subset, feat_bias: float) -> "H3DNormalizer":
        if feat_bias <= 0:
            raise ValueError("--feat-bias must be positive")
        count = 0
        total = torch.zeros(H3D_FEATURE_DIM)
        total_sq = torch.zeros(H3D_FEATURE_DIM)
        for sample in ds:
            x = sample.x1.float()
            count += x.shape[0]
            total += x.sum(dim=0)
            total_sq += (x * x).sum(dim=0)
        if count <= 1:
            raise RuntimeError("need at least two frames to compute MoMask normalization stats")
        mean = total / count
        var = (total_sq / count - mean * mean).clamp_min(1e-6)
        std = var.sqrt()
        std[0:4] /= feat_bias
        std[259:263] /= feat_bias
        return cls(mean, std)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return x * std + mean

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean.cpu(), "std": self.std.cpu()}


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    err = (pred - target).abs() * mask.to(pred.dtype).unsqueeze(-1)
    return err.sum() / (mask.sum().clamp_min(1).to(pred.dtype) * pred.shape[-1])


def token_mask_from_frame_mask(mask: torch.Tensor, token_len: int) -> torch.Tensor:
    if mask.shape[1] == token_len:
        return mask
    pooled = F.adaptive_max_pool1d(mask.float().unsqueeze(1), token_len).squeeze(1)
    return pooled > 0.5


@torch.no_grad()
def root_trajectory_metrics(motion: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    joints = recover_joints_from_ric(motion.float())
    roots = joints[:, :, 0, :][:, :, [0, 2]]
    drifts, spans = [], []
    for root, valid in zip(roots, mask):
        idx = valid.nonzero(as_tuple=False).flatten()
        if idx.numel() < 2:
            continue
        seq = root[idx]
        drifts.append(float((seq[-1] - seq[0]).norm()))
        spans.append(float((seq.max(dim=0).values - seq.min(dim=0).values).norm()))
    return {
        "end_drift": sum(drifts) / max(len(drifts), 1),
        "xz_span": sum(spans) / max(len(spans), 1),
    }


@torch.no_grad()
def evaluate_vq(
    vqvae: MotionRVQVAE,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    normalizer: H3DNormalizer,
) -> dict[str, float]:
    vqvae.eval()
    maes, raw_maes, losses, velocity_losses, ppls = [], [], [], [], []
    real_drifts, recon_drifts, real_spans, recon_spans = [], [], [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        raw_x = batch.x1.to(device)
        x = normalizer.transform(raw_x)
        mask = batch.mask.to(device)
        out = vqvae(x, mask=mask)
        maes.append(float(masked_mae(out.recon, x, mask)))
        raw_recon = normalizer.inverse(out.recon)
        raw_maes.append(float(masked_mae(raw_recon, raw_x, mask)))
        losses.append(float(out.loss))
        velocity_losses.append(float(out.velocity_loss))
        ppls.append(float(out.perplexity))
        real_root = root_trajectory_metrics(raw_x.cpu(), mask.cpu())
        recon_root = root_trajectory_metrics(raw_recon.detach().cpu(), mask.cpu())
        real_drifts.append(real_root["end_drift"])
        recon_drifts.append(recon_root["end_drift"])
        real_spans.append(real_root["xz_span"])
        recon_spans.append(recon_root["xz_span"])
    vqvae.train()
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "recon_mae": sum(maes) / max(len(maes), 1),
        "raw_recon_mae": sum(raw_maes) / max(len(raw_maes), 1),
        "velocity_loss": sum(velocity_losses) / max(len(velocity_losses), 1),
        "perplexity": sum(ppls) / max(len(ppls), 1),
        "real_end_drift": sum(real_drifts) / max(len(real_drifts), 1),
        "recon_end_drift": sum(recon_drifts) / max(len(recon_drifts), 1),
        "real_xz_span": sum(real_spans) / max(len(real_spans), 1),
        "recon_xz_span": sum(recon_spans) / max(len(recon_spans), 1),
    }


@torch.no_grad()
def evaluate_tokens(
    masked_model: MaskedMotionTransformer,
    residual_model: ResidualTransformer,
    cached_batches: list[dict[str, torch.Tensor | list[str]]],
    device: torch.device,
    max_batches: int,
    generation_steps: int,
) -> dict[str, float]:
    was_training = (masked_model.training, residual_model.training)
    masked_model.eval()
    residual_model.eval()
    base_losses, residual_losses = [], []
    full_mask_losses, full_mask_accs, generated_base_accs = [], [], []
    full_generation_acc_by_level: dict[int, list[float]] = {}
    teacher_residual_acc_by_level: dict[int, list[float]] = {}
    residual_by_level: dict[int, list[float]] = {}
    for i, batch in enumerate(cached_batches):
        if i >= max_batches:
            break
        tok = batch["tokens"].to(device)  # type: ignore[index, union-attr]
        token_mask = batch["token_mask"].to(device)  # type: ignore[index, union-attr]
        cond = batch["cond"].to(device)  # type: ignore[index, union-attr]
        base = masked_model.training_loss(tok[:, 0], cond=cond, valid_mask=token_mask, cond_drop_prob=0.0)
        full_masked = torch.full_like(tok[:, 0], masked_model.mask_token_id)
        full_logits = masked_model(full_masked, cond=cond, mask=token_mask)
        full_pred = full_logits.argmax(dim=-1)
        full_mask_losses.append(float(F.cross_entropy(full_logits[token_mask], tok[:, 0][token_mask])))
        full_mask_accs.append(float((full_pred[token_mask] == tok[:, 0][token_mask]).float().mean()))
        generated_base = masked_model.generate(
            cond=cond,
            seq_len=tok.shape[-1],
            steps=generation_steps,
            guidance_scale=1.0,
            mask=token_mask,
        )
        generated_base_accs.append(float((generated_base[token_mask] == tok[:, 0][token_mask]).float().mean()))
        generated_tokens = residual_model.generate_residuals(generated_base, cond=cond, guidance_scale=1.0, mask=token_mask)
        teacher_residual_tokens = residual_model.generate_residuals(
            tok[:, 0],
            cond=cond,
            guidance_scale=1.0,
            mask=token_mask,
        )
        for level in range(tok.shape[1]):
            full_generation_acc_by_level.setdefault(level, []).append(
                float((generated_tokens[:, level][token_mask] == tok[:, level][token_mask]).float().mean())
            )
            teacher_residual_acc_by_level.setdefault(level, []).append(
                float((teacher_residual_tokens[:, level][token_mask] == tok[:, level][token_mask]).float().mean())
            )
        res_parts = [
            residual_model.training_loss(tok, level, cond=cond, valid_mask=token_mask, cond_drop_prob=0.0)
            for level in range(1, tok.shape[1])
        ]
        for level, loss in zip(range(1, tok.shape[1]), res_parts):
            residual_by_level.setdefault(level, []).append(float(loss))
        residual = torch.stack(res_parts).mean() if res_parts else base.new_tensor(0.0)
        base_losses.append(float(base))
        residual_losses.append(float(residual))
    if was_training[0]:
        masked_model.train()
    if was_training[1]:
        residual_model.train()
    out = {
        "base_ce": sum(base_losses) / max(len(base_losses), 1),
        "residual_ce": sum(residual_losses) / max(len(residual_losses), 1),
        "base_full_mask_ce": sum(full_mask_losses) / max(len(full_mask_losses), 1),
        "base_full_mask_acc": sum(full_mask_accs) / max(len(full_mask_accs), 1),
        "base_generate_acc": sum(generated_base_accs) / max(len(generated_base_accs), 1),
    }
    for level, values in residual_by_level.items():
        out[f"residual_ce_l{level}"] = sum(values) / max(len(values), 1)
    for level, values in full_generation_acc_by_level.items():
        out[f"generated_acc_l{level}"] = sum(values) / max(len(values), 1)
    for level, values in teacher_residual_acc_by_level.items():
        out[f"teacher_residual_acc_l{level}"] = sum(values) / max(len(values), 1)
    return out


@torch.no_grad()
def cache_token_batches(
    vqvae: MotionRVQVAE,
    text_encoder: RandomTextEncoder,
    loader: DataLoader,
    device: torch.device,
    normalizer: H3DNormalizer,
) -> list[dict[str, torch.Tensor | list[str]]]:
    vqvae.eval()
    cached = []
    for batch in loader:
        x = normalizer.transform(batch.x1.to(device))
        frame_mask = batch.mask.to(device)
        tokens = vqvae.encode_to_tokens(x).cpu()
        token_mask = token_mask_from_frame_mask(frame_mask, tokens.shape[-1]).cpu()
        cond = text_encoder.encode(batch.texts, device=device).cpu()
        cached.append({
            "tokens": tokens,
            "token_mask": token_mask,
            "cond": cond,
            "texts": list(batch.texts),
        })
    return cached


def cycle_cached(cached_batches: list[dict[str, torch.Tensor | list[str]]]):
    while True:
        for batch in cached_batches:
            yield batch


def save_vq_train_checkpoint(
    output_dir: Path,
    step: int,
    args: argparse.Namespace,
    normalizer: H3DNormalizer,
    vqvae: MotionRVQVAE,
    vq_opt: torch.optim.Optimizer,
) -> None:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "args": vars(args),
        "normalizer": normalizer.state_dict(),
        "vqvae": vqvae.state_dict(),
        "vq_optimizer": vq_opt.state_dict(),
    }
    step_path = ckpt_dir / f"vq_step_{step:07d}.pt"
    latest_path = ckpt_dir / "vq_latest_train.pt"
    torch.save(ckpt, step_path)
    torch.save(ckpt, latest_path)
    print(f"[save] {step_path}", flush=True)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = HumanML3DDataset(
        root=Path(args.data_root),
        split=args.split,
        max_seq_len=args.max_seq_len,
        min_seq_len=args.min_seq_len,
        mirror_augment=False,
        output_mode="h3d_263",
    )
    n = min(args.max_clips, len(ds))
    if n <= 0:
        raise RuntimeError(f"no clips available in split={args.split!r}")
    small = Subset(ds, list(range(n)))
    normalizer = (
        H3DNormalizer.identity()
        if args.no_momask_normalize
        else H3DNormalizer.from_dataset(small, feat_bias=args.feat_bias)
    )
    loader = DataLoader(small, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    batches = cycle(loader)

    print(f"[momask-smoke] data_root={args.data_root}")
    print(f"[momask-smoke] clips={n} batch_size={args.batch_size} device={device}")
    print(f"[momask-smoke] output_dir={output_dir}")
    print(
        "[momask-smoke] momask_normalize="
        f"{not args.no_momask_normalize} feat_bias={args.feat_bias:.2f}"
    )

    vqvae = MotionRVQVAE(
        input_dim=H3D_FEATURE_DIM,
        hidden_dim=args.vq_hidden_dim,
        latent_dim=args.vq_latent_dim,
        num_quantizers=args.num_quantizers,
        codebook_size=args.codebook_size,
        downsample=args.downsample,
        num_res_blocks=args.vq_res_blocks,
        commitment_weight=args.vq_commitment_weight,
        quantize_dropout_prob=args.quantize_dropout,
        velocity_loss_weight=args.vq_velocity_weight,
    ).to(device)
    vq_opt = torch.optim.AdamW(vqvae.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    first_recon = None
    for step in range(1, args.vq_steps + 1):
        batch = next(batches)
        x = normalizer.transform(batch.x1.to(device))
        mask = batch.mask.to(device)
        out = vqvae(x, mask=mask)
        vq_opt.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(vqvae.parameters(), 1.0)
        vq_opt.step()
        if step == 1:
            first_recon = float(out.recon_loss.detach())
        if step == 1 or step == args.vq_steps or step % args.log_every == 0:
            print(
                f"[vq {step:04d}] loss={out.loss.item():.5f} "
                f"recon_mae={out.recon_loss.item():.5f} "
                f"vel={out.velocity_loss.item():.5f} "
                f"vq={out.vq_loss.item():.5f} ppl={out.perplexity.item():.2f}",
                flush=True,
            )
        if args.vq_only and args.save_every > 0 and step % args.save_every == 0:
            save_vq_train_checkpoint(output_dir, step, args, normalizer, vqvae, vq_opt)

    eval_batch = next(iter(loader))
    eval_raw_x = eval_batch.x1.to(device)
    eval_x = normalizer.transform(eval_raw_x)
    eval_mask = eval_batch.mask.to(device)
    vq_eval = evaluate_vq(vqvae, loader, device, args.eval_batches, normalizer)
    with torch.no_grad():
        eval_out = vqvae(eval_x, mask=eval_mask)
        tokens = eval_out.tokens.detach()
        recon = eval_out.recon.detach()
        raw_recon = normalizer.inverse(recon)
        eval_token_mask = token_mask_from_frame_mask(eval_mask, tokens.shape[-1])
    print(
        f"[vq summary] first_train_recon_mae={first_recon:.5f} "
        f"eval_recon_mae={vq_eval['recon_mae']:.5f} "
        f"eval_raw_recon_mae={vq_eval['raw_recon_mae']:.5f} "
        f"eval_vel={vq_eval['velocity_loss']:.5f} "
        f"eval_loss={vq_eval['loss']:.5f} eval_ppl={vq_eval['perplexity']:.2f}"
    )
    print(
        f"[vq root] real_end_drift={vq_eval['real_end_drift']:.5f} "
        f"recon_end_drift={vq_eval['recon_end_drift']:.5f} "
        f"real_xz_span={vq_eval['real_xz_span']:.5f} "
        f"recon_xz_span={vq_eval['recon_xz_span']:.5f}"
    )

    if args.vq_only:
        ckpt = {
            "args": vars(args),
            "normalizer": normalizer.state_dict(),
            "vqvae": vqvae.state_dict(),
            "vq_optimizer": vq_opt.state_dict(),
            "vq_eval": vq_eval,
            "sample_texts": eval_batch.texts,
            "sample_token_mask": eval_token_mask.detach().cpu(),
            "sample_real": eval_raw_x.detach().cpu(),
            "sample_reconstruction": raw_recon.detach().cpu(),
            "sample_true_tokens": tokens.detach().cpu(),
        }
        out_path = output_dir / "momask_vq_latest.pt"
        torch.save(ckpt, out_path)
        print(f"[save] {out_path}")
        return

    text_encoder = RandomTextEncoder(text_dim=args.text_dim)
    cached_batches = cache_token_batches(vqvae, text_encoder, loader, device, normalizer)
    cached_iter = cycle_cached(cached_batches)
    if args.live_token_crops:
        print("[token data] live random crops from dataset")
    else:
        n_cached = sum(int(batch["tokens"].shape[0]) for batch in cached_batches)  # type: ignore[index, union-attr]
        print(f"[token data] cached fixed VQ tokens batches={len(cached_batches)} samples={n_cached}")

    cfg = TokenTransformerConfig(
        vocab_size=args.codebook_size,
        text_dim=args.text_dim,
        hidden_dim=args.transformer_hidden_dim,
        depth=args.transformer_depth,
        num_heads=args.transformer_heads,
        ffn_dim=args.transformer_ffn_dim,
        max_seq_len=args.max_seq_len,
        dropout=args.transformer_dropout,
    )
    masked_model = MaskedMotionTransformer(cfg).to(device)
    residual_model = ResidualTransformer(
        cfg,
        num_quantizers=args.num_quantizers,
        separate_level_heads=not args.shared_residual_head,
    ).to(device)
    token_opt = torch.optim.AdamW(
        list(masked_model.parameters()) + list(residual_model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    vqvae.eval()
    for step in range(1, args.token_steps + 1):
        if args.live_token_crops:
            batch = next(batches)
            x = normalizer.transform(batch.x1.to(device))
            mask = batch.mask.to(device)
            cond = text_encoder.encode(batch.texts, device=device)
            with torch.no_grad():
                tok = vqvae.encode_to_tokens(x)
            token_mask = token_mask_from_frame_mask(mask, tok.shape[-1])
        else:
            cached = next(cached_iter)
            tok = cached["tokens"].to(device)  # type: ignore[index, union-attr]
            token_mask = cached["token_mask"].to(device)  # type: ignore[index, union-attr]
            cond = cached["cond"].to(device)  # type: ignore[index, union-attr]

        with torch.no_grad():
            tok = tok.long()
            token_mask = token_mask.bool()
            cond = cond.float()
        base_loss = masked_model.training_loss(
            tok[:, 0],
            cond=cond,
            valid_mask=token_mask,
            cond_drop_prob=args.base_cond_drop,
            force_full_mask=bool(torch.rand(()) < args.base_full_mask_prob),
        )
        residual_parts = [
            residual_model.training_loss(
                tok,
                level,
                cond=cond,
                valid_mask=token_mask,
                cond_drop_prob=args.residual_cond_drop,
            )
            for level in range(1, tok.shape[1])
        ]
        res_loss = torch.stack(residual_parts).mean() if residual_parts else base_loss.new_tensor(0.0)
        loss = base_loss + res_loss
        token_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(masked_model.parameters()) + list(residual_model.parameters()), 1.0)
        token_opt.step()
        if step == 1 or step == args.token_steps or step % args.log_every == 0:
            res_levels = " ".join(
                f"res_l{level}={loss.item():.5f}"
                for level, loss in zip(range(1, tok.shape[1]), residual_parts)
            )
            print(
                f"[tok {step:04d}] loss={loss.item():.5f} "
                f"base_ce={base_loss.item():.5f} residual_ce={res_loss.item():.5f} {res_levels}",
                flush=True,
            )

    token_eval = evaluate_tokens(
        masked_model,
        residual_model,
        cached_batches,
        device,
        args.eval_batches,
        args.generation_steps,
    )
    print(
        f"[tok summary] eval_base_ce={token_eval['base_ce']:.5f} "
        f"eval_residual_ce={token_eval['residual_ce']:.5f} "
        f"full_mask_ce={token_eval['base_full_mask_ce']:.5f} "
        f"full_mask_acc={token_eval['base_full_mask_acc']:.3f} "
        f"generate_base_acc={token_eval['base_generate_acc']:.3f}"
    )
    level_summary = " ".join(
        f"{k}={v:.5f}" for k, v in sorted(token_eval.items()) if k.startswith("residual_ce_l")
    )
    if level_summary:
        print(f"[tok summary levels] {level_summary}")
    acc_summary = " ".join(
        f"{k}={v:.3f}"
        for k, v in sorted(token_eval.items())
        if k.startswith("generated_acc_l") or k.startswith("teacher_residual_acc_l")
    )
    if acc_summary:
        print(f"[tok summary acc] {acc_summary}")

    with torch.no_grad():
        cond = text_encoder.encode(eval_batch.texts, device=device)
        recon = normalizer.inverse(vqvae(eval_x, mask=eval_mask).recon)
        base = masked_model.generate(
            cond=cond,
            seq_len=tokens.shape[-1],
            steps=args.generation_steps,
            guidance_scale=1.0,
            mask=eval_token_mask,
        )
        base_only = normalizer.inverse(vqvae.decode_from_tokens(base.unsqueeze(1), target_len=eval_x.shape[1]))
        gen_tokens = residual_model.generate_residuals(base, cond=cond, guidance_scale=1.0, mask=eval_token_mask)
        gen = normalizer.inverse(vqvae.decode_from_tokens(gen_tokens, target_len=eval_x.shape[1]))
        teacher_residual_tokens = residual_model.generate_residuals(
            tokens[:, 0],
            cond=cond,
            guidance_scale=1.0,
            mask=eval_token_mask,
        )
        teacher_residual = normalizer.inverse(
            vqvae.decode_from_tokens(teacher_residual_tokens, target_len=eval_x.shape[1])
        )
    print(f"[generate] tokens={tuple(gen_tokens.shape)} motion={tuple(gen.shape)} finite={bool(torch.isfinite(gen).all())}")

    ckpt = {
        "args": vars(args),
        "normalizer": normalizer.state_dict(),
        "vqvae": vqvae.state_dict(),
        "masked_transformer": masked_model.state_dict(),
        "residual_transformer": residual_model.state_dict(),
        "vq_optimizer": vq_opt.state_dict(),
        "token_optimizer": token_opt.state_dict(),
        "vq_eval": vq_eval,
        "token_eval": token_eval,
        "sample_texts": eval_batch.texts,
        "sample_cond": cond.detach().cpu(),
        "sample_token_mask": eval_token_mask.detach().cpu(),
        "sample_real": eval_raw_x.detach().cpu(),
        "sample_reconstruction": recon.detach().cpu(),
        "sample_base_only": base_only.detach().cpu(),
        "sample_teacher_residual": teacher_residual.detach().cpu(),
        "sample_generated": gen.detach().cpu(),
        "sample_true_tokens": tokens.detach().cpu(),
        "sample_base_tokens": base.detach().cpu(),
        "sample_tokens": gen_tokens.detach().cpu(),
        "sample_teacher_residual_tokens": teacher_residual_tokens.detach().cpu(),
    }
    out_path = output_dir / "momask_smoke_latest.pt"
    torch.save(ckpt, out_path)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
