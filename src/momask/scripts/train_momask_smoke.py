"""Tiny MoMask overfit/smoke run on the shared HumanML3D packed dataset.

This is not the full MoMask training recipe. It is a cheap check that the data
path and the three model stages can learn something on a very small subset:

  1. Load `shared.data.H3D263Dataset`.
  2. Train a small MotionRVQVAE for a handful of steps.
  3. Freeze/tokenize with the VQ-VAE.
  4. Train the masked base-token transformer and residual transformer briefly.
  5. Print reconstruction MAE and token losses.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset

from momask.models import (
    CodebookResidualTransformer,
    MaskedMotionTransformer,
    MotionRVQVAE,
    ResidualTransformer,
    TokenTransformerConfig,
)
from momask.data_utils import normalize_motion, token_mask_from_frame_mask
from shared.data import CanonicalHumanML3DDataset, CanonicalHumanML3DWindowDataset, H3D263Dataset, collate
from shared.text import CLIPTextEncoder, RandomTextEncoder, TextEncoder
from shared.geometry import H3D_FEATURE_DIM, recover_joints_from_ric


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-root",
        default=os.environ.get("RMG_DATA_ROOT", "external/data/humanml3d_packed"),
        help="Directory containing humanml3d.zip, splits.json, target_offsets.pt.",
    )
    p.add_argument(
        "--canonical-h3d-dir",
        default=None,
        help=(
            "Optional canonical HumanML3D new_joint_vecs directory. When set, "
            "MoMask trains on official 263-D features instead of packed-derived features."
        ),
    )
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--max-clips", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-seq-len", type=int, default=80)
    p.add_argument("--min-seq-len", type=int, default=20)
    p.add_argument("--vq-steps", type=int, default=30)
    p.add_argument(
        "--vq-epochs",
        type=int,
        default=0,
        help="When positive, derive VQ training steps from this many complete DataLoader epochs.",
    )
    p.add_argument("--token-steps", type=int, default=20)
    p.add_argument(
        "--token-batch-size",
        type=int,
        default=None,
        help="Batch size for token-transformer training after VQ tokens are cached. Defaults to --batch-size.",
    )
    p.add_argument("--vq-only", action="store_true", help="Train/evaluate only the VQ-VAE tokenizer, then save.")
    p.add_argument(
        "--vq-window-dataset",
        action="store_true",
        help="For canonical VQ-only training, index fixed windows instead of one random crop per clip.",
    )
    p.add_argument("--vq-window-stride", type=int, default=1)
    p.add_argument(
        "--vq-window-preload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preload canonical motions once instead of opening one NumPy file per RVQ window.",
    )
    p.add_argument(
        "--vq-gpu-window-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep canonical fixed-window source frames and their shuffled window index on the GPU. "
            "This avoids host transfers while preserving the configured optimization batch size."
        ),
    )
    p.add_argument(
        "--load-vq-checkpoint",
        default=None,
        help="Load a pretrained VQ-VAE checkpoint and train/evaluate token transformers from it.",
    )
    p.add_argument(
        "--load-token-checkpoint",
        default=None,
        help="Resume token-transformer training from a periodic token checkpoint.",
    )
    p.add_argument(
        "--load-masked-checkpoint",
        default=None,
        help="Load only the masked/base-token transformer from an existing token checkpoint.",
    )
    p.add_argument(
        "--freeze-masked-transformer",
        action="store_true",
        help="Train only the residual transformer after loading/freezing the masked transformer.",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.999)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--lr-milestones", type=int, nargs="*", default=[])
    p.add_argument("--lr-gamma", type=float, default=0.1)
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
    p.add_argument("--h3d-mean", default=None, help="Optional HumanML3D Mean.npy for MoMask normalization.")
    p.add_argument("--h3d-std", default=None, help="Optional HumanML3D Std.npy for MoMask normalization.")

    # Small by default, but now scalable enough to test real capacity changes.
    p.add_argument("--vq-hidden-dim", type=int, default=64)
    p.add_argument("--vq-latent-dim", type=int, default=32)
    p.add_argument("--num-quantizers", type=int, default=3)
    p.add_argument("--codebook-size", type=int, default=64)
    p.add_argument("--vq-res-blocks", type=int, default=1)
    p.add_argument("--downsample", type=int, default=1)
    p.add_argument("--quantize-dropout", type=float, default=0.2)
    p.add_argument("--vq-commitment-weight", type=float, default=0.25)
    p.add_argument("--vq-recon-loss", choices=["l1", "smooth_l1"], default="l1")
    p.add_argument("--vq-use-ema", action="store_true", help="Use EMA-reset codebooks, as in the official MoMask RVQ.")
    p.add_argument("--vq-ema-decay", type=float, default=0.99)
    p.add_argument("--vq-codebook-sample-temp", type=float, default=0.0)
    p.add_argument("--vq-arch", choices=["simple", "paper"], default="simple")
    p.add_argument(
        "--vq-velocity-weight",
        type=float,
        default=0.0,
        help="Weight for matching frame-to-frame feature velocities in the VQ-VAE loss.",
    )
    p.add_argument(
        "--vq-explicit-weight",
        type=float,
        default=0.0,
        help="Weight for the official MoMask auxiliary local-pose reconstruction loss over channels 4:67.",
    )

    p.add_argument("--text-dim", type=int, default=64)
    p.add_argument("--text-encoder", choices=["random", "clip"], default="random")
    p.add_argument("--clip-model", default="ViT-B/32")
    p.add_argument("--clip-cache-dir", default=None)
    p.add_argument("--clip-backend", choices=["auto", "openai", "transformers"], default="auto")
    p.add_argument("--transformer-hidden-dim", type=int, default=64)
    p.add_argument("--transformer-depth", type=int, default=2)
    p.add_argument("--transformer-heads", type=int, default=4)
    p.add_argument("--transformer-ffn-dim", type=int, default=128)
    p.add_argument("--transformer-dropout", type=float, default=0.0)
    p.add_argument("--base-cond-drop", type=float, default=0.1)
    p.add_argument("--residual-cond-drop", type=float, default=0.2)
    p.add_argument("--residual-arch", choices=["simple", "codebook"], default="simple")
    p.add_argument(
        "--residual-share-weight",
        action="store_true",
        help="For --residual-arch codebook, share residual input/output projection weights.",
    )
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
    p.add_argument(
        "--cache-token-device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Where to keep cached VQ tokens for token-transformer training.",
    )
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
    def from_dataset(cls, ds: Dataset, feat_bias: float) -> "H3DNormalizer":
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

    @classmethod
    def from_files(cls, mean_path: str | Path, std_path: str | Path, feat_bias: float) -> "H3DNormalizer":
        mean = torch.from_numpy(np.load(mean_path)).float()
        std = torch.from_numpy(np.load(std_path)).float()
        std = std.clone()
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

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> "H3DNormalizer":
        return cls(state["mean"], state["std"])


def torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def restore_model_args(args: argparse.Namespace, ckpt: dict) -> None:
    saved_args = ckpt.get("args", {})
    if not isinstance(saved_args, dict):
        return
    vq_names = (
        "vq_hidden_dim",
        "vq_latent_dim",
        "num_quantizers",
        "codebook_size",
        "downsample",
        "vq_res_blocks",
        "vq_commitment_weight",
        "quantize_dropout",
        "vq_velocity_weight",
        "vq_explicit_weight",
        "vq_recon_loss",
        "vq_use_ema",
        "vq_ema_decay",
        "vq_codebook_sample_temp",
        "vq_arch",
        "no_momask_normalize",
        "feat_bias",
    )
    transformer_names = (
        "text_encoder",
        "clip_model",
        "clip_backend",
        "text_dim",
        "transformer_hidden_dim",
        "transformer_depth",
        "transformer_heads",
        "transformer_ffn_dim",
        "transformer_dropout",
        "shared_residual_head",
        "residual_arch",
        "residual_share_weight",
        "max_seq_len",
    )
    restore_names = vq_names + (transformer_names if "masked_transformer" in ckpt else ())
    for name in restore_names:
        if name in saved_args:
            setattr(args, name, saved_args[name])
    if not args.canonical_h3d_dir and saved_args.get("canonical_h3d_dir"):
        args.canonical_h3d_dir = saved_args["canonical_h3d_dir"]


def build_text_encoder(args: argparse.Namespace) -> TextEncoder:
    if args.text_encoder == "random":
        return RandomTextEncoder(text_dim=args.text_dim)
    if args.text_encoder == "clip":
        encoder = CLIPTextEncoder(
            model_name=args.clip_model,
            cache_dir=args.clip_cache_dir,
            backend=args.clip_backend,
        )
        args.text_dim = encoder.text_dim
        return encoder
    raise ValueError(f"unknown text encoder: {args.text_encoder}")


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    err = (pred - target).abs() * mask.to(pred.dtype).unsqueeze(-1)
    return err.sum() / (mask.sum().clamp_min(1).to(pred.dtype) * pred.shape[-1])


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
    was_training = vqvae.training
    vqvae.eval()
    maes, raw_maes, losses, velocity_losses, explicit_losses, ppls = [], [], [], [], [], []
    real_drifts, recon_drifts, real_spans, recon_spans = [], [], [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        mask = batch.mask.to(device)
        raw_x = batch.x1.to(device)
        x = normalize_motion(raw_x, mask, normalizer)
        out = vqvae(x, mask=mask)
        maes.append(float(masked_mae(out.recon, x, mask)))
        raw_recon = normalizer.inverse(out.recon)
        raw_maes.append(float(masked_mae(raw_recon, raw_x, mask)))
        losses.append(float(out.loss))
        velocity_losses.append(float(out.velocity_loss))
        explicit_losses.append(float(out.explicit_loss))
        ppls.append(float(out.perplexity))
        real_root = root_trajectory_metrics(raw_x.cpu(), mask.cpu())
        recon_root = root_trajectory_metrics(raw_recon.detach().cpu(), mask.cpu())
        real_drifts.append(real_root["end_drift"])
        recon_drifts.append(recon_root["end_drift"])
        real_spans.append(real_root["xz_span"])
        recon_spans.append(recon_root["xz_span"])
    if was_training:
        vqvae.train()
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "recon_mae": sum(maes) / max(len(maes), 1),
        "raw_recon_mae": sum(raw_maes) / max(len(raw_maes), 1),
        "velocity_loss": sum(velocity_losses) / max(len(velocity_losses), 1),
        "explicit_loss": sum(explicit_losses) / max(len(explicit_losses), 1),
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
        generated_tokens = residual_model.generate_residuals(
            generated_base,
            cond=cond,
            guidance_scale=1.0,
            mask=token_mask,
        )
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
    text_encoder: TextEncoder,
    loader: DataLoader,
    device: torch.device,
    normalizer: H3DNormalizer,
) -> list[dict[str, torch.Tensor | list[str]]]:
    vqvae.eval()
    cached = []
    for batch in loader:
        frame_mask = batch.mask.to(device)
        x = normalize_motion(batch.x1.to(device), frame_mask, normalizer)
        tokens = vqvae.encode_to_tokens(x).cpu()
        token_mask = token_mask_from_frame_mask(frame_mask, tokens.shape[-1], vqvae.downsample).cpu()
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


def stack_token_cache(
    cached_batches: list[dict[str, torch.Tensor | list[str]]],
    device: torch.device,
    max_token_len: int,
) -> dict[str, torch.Tensor]:
    if max_token_len <= 0:
        raise ValueError("max_token_len must be positive")
    tokens = []
    masks = []
    for batch in cached_batches:
        batch_tokens = batch["tokens"]  # type: ignore[assignment]
        batch_mask = batch["token_mask"]  # type: ignore[assignment]
        if not isinstance(batch_tokens, torch.Tensor) or not isinstance(batch_mask, torch.Tensor):
            raise TypeError("cached token batches must contain tensors")
        if batch_tokens.shape[-1] > max_token_len:
            raise ValueError(
                f"cached token length {batch_tokens.shape[-1]} exceeds configured maximum {max_token_len}"
            )
        pad = max_token_len - batch_tokens.shape[-1]
        tokens.append(F.pad(batch_tokens, (0, pad), value=0))
        masks.append(F.pad(batch_mask, (0, pad), value=False))
    return {
        "tokens": torch.cat(tokens, dim=0).to(device),
        "token_mask": torch.cat(masks, dim=0).to(device),
        "cond": torch.cat([batch["cond"] for batch in cached_batches], dim=0).to(device),  # type: ignore[list-item]
    }


def sample_token_cache(cache: dict[str, torch.Tensor], batch_size: int) -> dict[str, torch.Tensor]:
    n = cache["tokens"].shape[0]
    idx = torch.randint(n, (batch_size,), device=cache["tokens"].device)
    return {key: value[idx] for key, value in cache.items()}


def cycle_loader(loader: DataLoader):
    """Iterate forever without caching and replaying the first epoch."""
    while True:
        yield from loader


class FixedWindowTensorBatcher:
    """Shuffle fixed windows while gathering them from one packed device tensor."""

    def __init__(
        self,
        dataset: CanonicalHumanML3DWindowDataset,
        batch_size: int,
        device: torch.device,
        normalizer: H3DNormalizer,
        seed: int,
    ) -> None:
        if batch_size <= 0 or batch_size > len(dataset):
            raise ValueError(f"batch_size must be in [1, {len(dataset)}], got {batch_size}")
        frames_np, offsets_np = dataset.materialize_frames_and_window_offsets()
        self.frames = torch.from_numpy(frames_np).to(device=device, non_blocking=True)
        mean = normalizer.mean.to(device=device, dtype=self.frames.dtype)
        std = normalizer.std.to(device=device, dtype=self.frames.dtype)
        self.frames.sub_(mean).div_(std)
        self.window_offsets = torch.from_numpy(offsets_np).to(device=device, non_blocking=True)
        self.frame_range = torch.arange(dataset.window_size, device=device)
        self.batch_size = batch_size
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.order = torch.empty(0, dtype=torch.long, device=device)
        self.cursor = 0
        self._reshuffle()

    @property
    def cache_nbytes(self) -> int:
        tensors = (self.frames, self.window_offsets, self.frame_range, self.order)
        return sum(t.numel() * t.element_size() for t in tensors)

    def _reshuffle(self) -> None:
        self.order = torch.randperm(
            self.window_offsets.numel(),
            generator=self.generator,
            device=self.window_offsets.device,
        )
        self.cursor = 0

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cursor + self.batch_size > self.order.numel():
            self._reshuffle()
        batch_ids = self.order[self.cursor : self.cursor + self.batch_size]
        self.cursor += self.batch_size
        frame_ids = self.window_offsets[batch_ids, None] + self.frame_range[None, :]
        x = self.frames[frame_ids]
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        return x, mask


def scheduled_lr(
    base_lr: float,
    step: int,
    warmup_steps: int,
    milestones: list[int],
    gamma: float,
) -> float:
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("lr_gamma must be in (0, 1]")
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / (warmup_steps + 1)
    drops = sum(step > milestone for milestone in milestones)
    return base_lr * (gamma ** drops)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


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
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded_token_ckpt = torch_load(args.load_token_checkpoint) if args.load_token_checkpoint else None
    loaded_vq_ckpt = (
        loaded_token_ckpt
        if loaded_token_ckpt is not None
        else torch_load(args.load_vq_checkpoint) if args.load_vq_checkpoint else None
    )
    if loaded_vq_ckpt is not None:
        restore_model_args(args, loaded_vq_ckpt)
    if args.vq_steps <= 0 and args.vq_epochs <= 0 and loaded_vq_ckpt is None:
        raise ValueError(
            "--vq-steps or --vq-epochs must be positive unless "
            "--load-vq-checkpoint or --load-token-checkpoint is set"
        )

    is_window_dataset = bool(args.canonical_h3d_dir and args.vq_only and args.vq_window_dataset)
    if is_window_dataset:
        ds = CanonicalHumanML3DWindowDataset(
            root=Path(args.data_root),
            canonical_dir=Path(args.canonical_h3d_dir),
            split=args.split,
            window_size=args.max_seq_len,
            window_stride=args.vq_window_stride,
            subset_n=args.max_clips,
            preload=args.vq_window_preload,
        )
        feature_source = (
            f"canonical-windows:{args.canonical_h3d_dir} "
            f"window={args.max_seq_len} stride={args.vq_window_stride} "
            f"preload={args.vq_window_preload} skipped_short={ds.num_skipped_short}"
        )
    elif args.canonical_h3d_dir:
        ds = CanonicalHumanML3DDataset(
            root=Path(args.data_root),
            canonical_dir=Path(args.canonical_h3d_dir),
            split=args.split,
            max_seq_len=args.max_seq_len,
            min_seq_len=args.min_seq_len,
        )
        feature_source = f"canonical:{args.canonical_h3d_dir}"
    else:
        ds = H3D263Dataset(
            root=Path(args.data_root),
            split=args.split,
            max_seq_len=args.max_seq_len,
            min_seq_len=args.min_seq_len,
            mirror_augment=False,
        )
        feature_source = "packed-derived h3d_263"
    if is_window_dataset:
        train_ds = ds
        n_items = len(ds)
        n_clips = ds.num_clips  # type: ignore[attr-defined]
    else:
        n_items = min(args.max_clips, len(ds))
        train_ds = Subset(ds, list(range(n_items)))
        n_clips = n_items
    if n_items <= 0:
        raise RuntimeError(f"no clips available in split={args.split!r}")
    if loaded_vq_ckpt is not None and "normalizer" in loaded_vq_ckpt:
        normalizer = H3DNormalizer.from_state_dict(loaded_vq_ckpt["normalizer"])
    else:
        if is_window_dataset and not args.no_momask_normalize and not (args.h3d_mean and args.h3d_std):
            raise ValueError(
                "fixed-window RVQ training requires --h3d-mean and --h3d-std; "
                "scanning every overlapping window would bias and duplicate normalization statistics"
            )
        normalizer = (
            H3DNormalizer.identity()
            if args.no_momask_normalize
            else H3DNormalizer.from_files(args.h3d_mean, args.h3d_std, feat_bias=args.feat_bias)
            if args.h3d_mean and args.h3d_std
            else H3DNormalizer.from_dataset(train_ds, feat_bias=args.feat_bias)
        )
    loader_kwargs: dict[str, object] = {}
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=is_window_dataset and n_items >= args.batch_size,
        **loader_kwargs,
    )
    if args.vq_epochs > 0:
        args.vq_steps = args.vq_epochs * len(loader)
        print(
            f"[momask-smoke] vq_epochs={args.vq_epochs} resolved_vq_steps={args.vq_steps}",
            flush=True,
        )
    batches = cycle_loader(loader)
    gpu_window_batches = None
    if args.vq_gpu_window_cache:
        if not is_window_dataset:
            raise ValueError("--vq-gpu-window-cache requires --vq-window-dataset with canonical VQ-only training")
        if device.type != "cuda":
            raise ValueError("--vq-gpu-window-cache requires a CUDA device")
        gpu_window_batches = FixedWindowTensorBatcher(ds, args.batch_size, device, normalizer, args.seed)
        cache_gib = gpu_window_batches.cache_nbytes / (1024 ** 3)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        print(
            f"[momask-smoke] gpu_window_cache={cache_gib:.2f}GiB "
            f"cuda_free={free_bytes / (1024 ** 3):.2f}GiB/{total_bytes / (1024 ** 3):.2f}GiB",
            flush=True,
        )

    print(f"[momask-smoke] data_root={args.data_root}")
    print(f"[momask-smoke] feature_source={feature_source}")
    print(
        f"[momask-smoke] clips={n_clips} items={n_items} "
        f"batches_per_epoch={len(loader)} batch_size={args.batch_size} "
        f"workers={args.num_workers} device={device}"
    )
    print(f"[momask-smoke] output_dir={output_dir}")
    print(
        "[momask-smoke] momask_normalize="
        f"{not args.no_momask_normalize} feat_bias={args.feat_bias:.2f}"
    )
    if args.h3d_mean and args.h3d_std and not args.no_momask_normalize:
        print(f"[momask-smoke] h3d_mean={args.h3d_mean}")
        print(f"[momask-smoke] h3d_std={args.h3d_std}")
    if args.load_vq_checkpoint:
        print(f"[momask-smoke] load_vq_checkpoint={args.load_vq_checkpoint}")
    if args.load_token_checkpoint:
        print(f"[momask-smoke] load_token_checkpoint={args.load_token_checkpoint}")

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
        explicit_loss_weight=args.vq_explicit_weight,
        recon_loss=args.vq_recon_loss,
        use_ema_quantizer=args.vq_use_ema,
        ema_decay=args.vq_ema_decay,
        codebook_sample_temp=args.vq_codebook_sample_temp,
        architecture=args.vq_arch,
    ).to(device)
    vq_opt = torch.optim.AdamW(
        vqvae.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )
    start_vq_step = 0
    if loaded_vq_ckpt is not None:
        vqvae.load_state_dict(loaded_vq_ckpt["vqvae"])
        if "vq_optimizer" in loaded_vq_ckpt and args.vq_steps > 0:
            vq_opt.load_state_dict(loaded_vq_ckpt["vq_optimizer"])
        start_vq_step = int(loaded_vq_ckpt.get("step", 0)) if args.vq_steps > 0 else 0
        print(
            "[momask-smoke] restored VQ "
            f"quantizers={args.num_quantizers} codebook={args.codebook_size} "
            f"hidden={args.vq_hidden_dim} latent={args.vq_latent_dim} "
            f"downsample={args.downsample} ema={args.vq_use_ema} "
            f"step={start_vq_step} target_steps={args.vq_steps}",
            flush=True,
        )

    first_recon = None
    log_start_step = start_vq_step
    log_start_time = time.perf_counter()
    vqvae.train()
    for step in range(start_vq_step + 1, args.vq_steps + 1):
        current_lr = scheduled_lr(args.lr, step, args.warmup_steps, args.lr_milestones, args.lr_gamma)
        set_optimizer_lr(vq_opt, current_lr)
        if gpu_window_batches is not None:
            x, mask = gpu_window_batches.next()
        else:
            batch = next(batches)
            mask = batch.mask.to(device)
            x = normalize_motion(batch.x1.to(device), mask, normalizer)
        # Fixed windows have no padding. Avoid masked gathers and mask-induced
        # device synchronizations in every RVQ level while keeping ragged
        # batches on the mask-aware path.
        out = vqvae(x, mask=None if is_window_dataset else mask)
        vq_opt.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(vqvae.parameters(), 1.0)
        vq_opt.step()
        if step == start_vq_step + 1 and device.type == "cuda":
            print(
                "[momask-smoke] cuda_memory "
                f"allocated={torch.cuda.memory_allocated(device) / (1024 ** 3):.2f}GiB "
                f"reserved={torch.cuda.memory_reserved(device) / (1024 ** 3):.2f}GiB "
                f"peak={torch.cuda.max_memory_allocated(device) / (1024 ** 3):.2f}GiB",
                flush=True,
            )
        if step == 1:
            first_recon = float(out.recon_loss.detach())
        if step == 1 or step == args.vq_steps or step % args.log_every == 0:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            now = time.perf_counter()
            interval_steps = max(step - log_start_step, 1)
            steps_per_sec = interval_steps / max(now - log_start_time, 1e-9)
            print(
                f"[vq {step:04d}] loss={out.loss.item():.5f} "
                f"recon_mae={out.recon_loss.item():.5f} "
                f"vel={out.velocity_loss.item():.5f} "
                f"explicit={out.explicit_loss.item():.5f} "
                f"vq={out.vq_loss.item():.5f} ppl={out.perplexity.item():.2f} "
                f"lr={current_lr:.3e} steps_per_sec={steps_per_sec:.2f}",
                flush=True,
            )
            log_start_step = step
            log_start_time = now
        if args.vq_only and args.save_every > 0 and step % args.save_every == 0:
            save_vq_train_checkpoint(output_dir, step, args, normalizer, vqvae, vq_opt)

    eval_batch = next(iter(loader))
    eval_raw_x = eval_batch.x1.to(device)
    eval_mask = eval_batch.mask.to(device)
    eval_x = normalize_motion(eval_raw_x, eval_mask, normalizer)
    vq_eval = evaluate_vq(vqvae, loader, device, args.eval_batches, normalizer)
    vqvae.eval()
    with torch.no_grad():
        eval_out = vqvae(eval_x, mask=eval_mask)
        tokens = eval_out.tokens.detach()
        recon = eval_out.recon.detach()
        raw_recon = normalizer.inverse(recon)
        eval_token_mask = token_mask_from_frame_mask(eval_mask, tokens.shape[-1], args.downsample)
    print(
        f"[vq summary] first_train_recon_mae={(first_recon if first_recon is not None else float('nan')):.5f} "
        f"eval_recon_mae={vq_eval['recon_mae']:.5f} "
        f"eval_raw_recon_mae={vq_eval['raw_recon_mae']:.5f} "
        f"eval_vel={vq_eval['velocity_loss']:.5f} "
        f"eval_explicit={vq_eval['explicit_loss']:.5f} "
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
            "step": args.vq_steps,
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

    text_encoder = build_text_encoder(args)
    print(f"[momask-smoke] text_encoder={args.text_encoder} text_dim={args.text_dim}", flush=True)
    print(
        f"[momask-smoke] residual_arch={args.residual_arch} "
        f"residual_share_weight={args.residual_share_weight}",
        flush=True,
    )
    cached_batches = cache_token_batches(vqvae, text_encoder, loader, device, normalizer)
    token_batch_size = args.token_batch_size or args.batch_size
    token_cache = None
    if args.live_token_crops:
        print("[token data] live random crops from dataset")
    else:
        n_cached = sum(int(batch["tokens"].shape[0]) for batch in cached_batches)  # type: ignore[index, union-attr]
        cache_device = device if args.cache_token_device == "cuda" else torch.device("cpu")
        if cache_device.type == "cuda" and device.type != "cuda":
            raise ValueError("--cache-token-device cuda requires --device cuda")
        max_token_len = math.ceil(args.max_seq_len / args.downsample)
        token_cache = stack_token_cache(cached_batches, cache_device, max_token_len)
        print(
            f"[token data] cached fixed VQ tokens batches={len(cached_batches)} "
            f"samples={n_cached} train_batch={token_batch_size} cache_device={cache_device}"
        )

    cfg = TokenTransformerConfig(
        vocab_size=args.codebook_size,
        text_dim=args.text_dim,
        hidden_dim=args.transformer_hidden_dim,
        depth=args.transformer_depth,
        num_heads=args.transformer_heads,
        ffn_dim=args.transformer_ffn_dim,
        max_seq_len=math.ceil(args.max_seq_len / args.downsample),
        dropout=args.transformer_dropout,
    )
    masked_model = MaskedMotionTransformer(cfg).to(device)
    if args.residual_arch == "codebook":
        residual_model = CodebookResidualTransformer(
            cfg,
            num_quantizers=args.num_quantizers,
            code_dim=args.vq_latent_dim,
            share_weight=args.residual_share_weight,
        ).to(device)
    else:
        residual_model = ResidualTransformer(
            cfg,
            num_quantizers=args.num_quantizers,
            separate_level_heads=not args.shared_residual_head,
        ).to(device)
    start_token_step = 0
    optimizer_state = None
    if loaded_token_ckpt is not None:
        masked_model.load_state_dict(loaded_token_ckpt["masked_transformer"])
        residual_model.load_state_dict(loaded_token_ckpt["residual_transformer"])
        if "token_optimizer" in loaded_token_ckpt:
            optimizer_state = loaded_token_ckpt["token_optimizer"]
        start_token_step = int(loaded_token_ckpt.get("step", 0))
        print(
            f"[momask-smoke] restored token transformers from step={start_token_step} "
            f"target_steps={args.token_steps}",
            flush=True,
        )
    if args.load_masked_checkpoint:
        masked_ckpt = torch_load(args.load_masked_checkpoint, map_location=device)
        masked_model.load_state_dict(masked_ckpt["masked_transformer"])
        print(f"[momask-smoke] restored masked transformer only from {args.load_masked_checkpoint}", flush=True)
    if args.freeze_masked_transformer:
        for param in masked_model.parameters():
            param.requires_grad_(False)
        masked_model.eval()
        optimizer_state = None
        print("[momask-smoke] frozen masked transformer; optimizer trains residual transformer only", flush=True)
    token_params = list(residual_model.parameters())
    if not args.freeze_masked_transformer:
        token_params = list(masked_model.parameters()) + token_params
    token_opt = torch.optim.AdamW(
        token_params,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )
    if optimizer_state is not None:
        token_opt.load_state_dict(optimizer_state)

    vqvae.eval()
    for step in range(start_token_step + 1, args.token_steps + 1):
        current_lr = scheduled_lr(args.lr, step, args.warmup_steps, args.lr_milestones, args.lr_gamma)
        set_optimizer_lr(token_opt, current_lr)
        if args.live_token_crops:
            batch = next(batches)
            mask = batch.mask.to(device)
            x = normalize_motion(batch.x1.to(device), mask, normalizer)
            cond = text_encoder.encode(batch.texts, device=device)
            with torch.no_grad():
                tok = vqvae.encode_to_tokens(x)
            token_mask = token_mask_from_frame_mask(mask, tok.shape[-1], args.downsample)
        else:
            if token_cache is None:
                raise RuntimeError("token cache was not built")
            cached = sample_token_cache(token_cache, token_batch_size)
            tok = cached["tokens"].to(device, non_blocking=True)
            token_mask = cached["token_mask"].to(device, non_blocking=True)
            cond = cached["cond"].to(device, non_blocking=True)

        with torch.no_grad():
            tok = tok.long()
            token_mask = token_mask.bool()
            cond = cond.float()
        if args.freeze_masked_transformer:
            with torch.no_grad():
                base_loss = masked_model.training_loss(
                    tok[:, 0],
                    cond=cond,
                    valid_mask=token_mask,
                    cond_drop_prob=0.0,
                    force_full_mask=True,
                )
        else:
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
        loss = res_loss if args.freeze_masked_transformer else base_loss + res_loss
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
                f"base_ce={base_loss.item():.5f} residual_ce={res_loss.item():.5f} "
                f"lr={current_lr:.3e} {res_levels}",
                flush=True,
            )
        if args.save_every > 0 and step % args.save_every == 0:
            ckpt_dir = output_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            step_path = ckpt_dir / f"tokens_step_{step:07d}.pt"
            latest_path = ckpt_dir / "tokens_latest_train.pt"
            ckpt = {
                "step": step,
                "args": vars(args),
                "normalizer": normalizer.state_dict(),
                "vqvae": vqvae.state_dict(),
                "masked_transformer": masked_model.state_dict(),
                "residual_transformer": residual_model.state_dict(),
                "vq_optimizer": vq_opt.state_dict(),
                "token_optimizer": token_opt.state_dict(),
            }
            torch.save(ckpt, step_path)
            torch.save(ckpt, latest_path)
            print(f"[save] {step_path}", flush=True)

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
        base_only = normalizer.inverse(
            vqvae.decode_from_tokens(
                base.unsqueeze(1),
                target_len=eval_x.shape[1],
                token_mask=eval_token_mask,
            )
        )
        gen_tokens = residual_model.generate_residuals(base, cond=cond, guidance_scale=1.0, mask=eval_token_mask)
        gen = normalizer.inverse(
            vqvae.decode_from_tokens(gen_tokens, target_len=eval_x.shape[1], token_mask=eval_token_mask)
        )
        teacher_residual_tokens = residual_model.generate_residuals(
            tokens[:, 0],
            cond=cond,
            guidance_scale=1.0,
            mask=eval_token_mask,
        )
        teacher_residual = normalizer.inverse(
            vqvae.decode_from_tokens(
                teacher_residual_tokens,
                target_len=eval_x.shape[1],
                token_mask=eval_token_mask,
            )
        )
    print(
        f"[generate] tokens={tuple(gen_tokens.shape)} motion={tuple(gen.shape)} "
        f"finite={bool(torch.isfinite(gen).all())}"
    )

    ckpt = {
        "step": args.token_steps,
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
