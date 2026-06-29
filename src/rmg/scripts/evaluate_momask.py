"""Evaluate the local MoMask checkpoint with the shared Guo evaluator.

This targets checkpoints produced by `scripts/train_momask_smoke.py`.

Typical quick run on the cluster:

    python -u scripts/evaluate_momask.py \
        --checkpoint runs/momask-full-tokens-12858/momask_smoke_latest.pt \
        --data-root /mnt/projects/drl4cvb/human-motion-representation/data/data/humanml3d_packed \
        --max-clips 512 \
        --variants full,base

Variants:
  full  = text -> base tokens -> residual tokens -> RVQ decoder
  base  = text -> base tokens only -> RVQ decoder
  recon = real motion -> RVQ encode/decode
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from momask.models import (
    MaskedMotionTransformer,
    MotionRVQVAE,
    ResidualTransformer,
    TokenTransformerConfig,
)
from rmg.data import HumanML3DDataset, collate
from rmg.eval import RandomGuoEvaluator, RealGuoEvaluator, diversity, fid, mm_distance, r_precision
from rmg.models import RandomTextEncoder
from rmg.representation import H3D_FEATURE_DIM


class H3DNormalizer:
    def __init__(self, mean: Tensor, std: Tensor) -> None:
        self.mean = mean.float()
        self.std = std.float().clamp_min(1e-6)
        if self.mean.shape != (H3D_FEATURE_DIM,) or self.std.shape != (H3D_FEATURE_DIM,):
            raise ValueError(f"normalizer must be {H3D_FEATURE_DIM}-D")

    @classmethod
    def from_state_dict(cls, state: dict[str, Tensor]) -> "H3DNormalizer":
        return cls(state["mean"], state["std"])

    def transform(self, x: Tensor) -> Tensor:
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def inverse(self, x: Tensor) -> Tensor:
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return x * std + mean


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to momask_smoke_latest.pt")
    p.add_argument(
        "--data-root",
        default=os.environ.get("RMG_DATA_ROOT", "external/data/humanml3d_packed"),
        help="Directory containing humanml3d.zip, splits.json, target_offsets.pt.",
    )
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--max-clips", type=int, default=512, help="-1 evaluates the whole split.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-seq-len", type=int, default=None, help="Defaults to checkpoint args.max_seq_len.")
    p.add_argument("--min-seq-len", type=int, default=40)
    p.add_argument("--variants", default="full,base", help="Comma list from: full,base,recon")
    p.add_argument("--generation-steps", type=int, default=None, help="Defaults to checkpoint args.generation_steps.")
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--evaluator", choices=["real", "random"], default="real")
    p.add_argument("--text-to-motion-repo", default="external/text-to-motion")
    p.add_argument("--humanml3d-repo", default="external/HumanML3D")
    p.add_argument("--diversity-times", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default=None, help="Defaults to <checkpoint-dir>/eval_momask/results.json")
    return p.parse_args()


def torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def require_path(path: str | Path, what: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


def token_mask_from_frame_mask(mask: Tensor, token_len: int) -> Tensor:
    if mask.shape[1] == token_len:
        return mask
    pooled = F.adaptive_max_pool1d(mask.float().unsqueeze(1), token_len).squeeze(1)
    return pooled > 0.5


def ckpt_args(ckpt: dict) -> dict:
    args = ckpt.get("args", {})
    if not isinstance(args, dict):
        raise ValueError("checkpoint does not contain an args dict")
    return args


def build_models(ckpt: dict, device: torch.device) -> tuple[MotionRVQVAE, MaskedMotionTransformer, ResidualTransformer]:
    a = ckpt_args(ckpt)
    vqvae = MotionRVQVAE(
        input_dim=H3D_FEATURE_DIM,
        hidden_dim=int(a.get("vq_hidden_dim", 64)),
        latent_dim=int(a.get("vq_latent_dim", 32)),
        num_quantizers=int(a.get("num_quantizers", 3)),
        codebook_size=int(a.get("codebook_size", 64)),
        downsample=int(a.get("downsample", 1)),
        num_res_blocks=int(a.get("vq_res_blocks", 1)),
        commitment_weight=float(a.get("vq_commitment_weight", 0.25)),
        quantize_dropout_prob=float(a.get("quantize_dropout", 0.2)),
        velocity_loss_weight=float(a.get("vq_velocity_weight", 0.0)),
    ).to(device)

    cfg = TokenTransformerConfig(
        vocab_size=int(a.get("codebook_size", 64)),
        text_dim=int(a.get("text_dim", 64)),
        hidden_dim=int(a.get("transformer_hidden_dim", 64)),
        depth=int(a.get("transformer_depth", 2)),
        num_heads=int(a.get("transformer_heads", 4)),
        ffn_dim=int(a.get("transformer_ffn_dim", 128)),
        max_seq_len=math.ceil(int(a.get("max_seq_len", 80)) / int(a.get("downsample", 1))),
        dropout=float(a.get("transformer_dropout", 0.0)),
    )
    masked = MaskedMotionTransformer(cfg).to(device)
    residual = ResidualTransformer(
        cfg,
        num_quantizers=int(a.get("num_quantizers", 3)),
        separate_level_heads=not bool(a.get("shared_residual_head", False)),
    ).to(device)

    for key, model in (("vqvae", vqvae), ("masked_transformer", masked), ("residual_transformer", residual)):
        if key not in ckpt:
            raise KeyError(f"checkpoint missing {key!r}")
        model.load_state_dict(ckpt[key])
        model.eval()
    return vqvae, masked, residual


def build_evaluator(args: argparse.Namespace, device: torch.device):
    if args.evaluator == "random":
        return RandomGuoEvaluator()
    require_path(Path(args.text_to_motion_repo) / "networks" / "evaluator_wrapper.py", "text-to-motion evaluator wrapper")
    require_path(
        Path(args.text_to_motion_repo) / "checkpoints" / "t2m" / "text_mot_match" / "model" / "finest.tar",
        "Guo evaluator checkpoint",
    )
    return RealGuoEvaluator(
        text_to_motion_repo=args.text_to_motion_repo,
        humanml3d_repo=args.humanml3d_repo,
        device=device,
    )


def encode_motion(evaluator, motion: Tensor, lengths: Tensor) -> np.ndarray:
    return evaluator.encode_motion(motion.cpu(), lengths.cpu()).cpu().numpy()


@torch.no_grad()
def generate_variant(
    variant: str,
    *,
    vqvae: MotionRVQVAE,
    masked: MaskedMotionTransformer,
    residual: ResidualTransformer,
    normalizer: H3DNormalizer,
    cond: Tensor,
    real_x: Tensor,
    frame_mask: Tensor,
    steps: int,
    guidance_scale: float,
    temperature: float,
) -> Tensor:
    x_norm = normalizer.transform(real_x)

    if variant == "recon":
        return normalizer.inverse(vqvae(x_norm, mask=frame_mask).recon)

    true_tokens = vqvae.encode_to_tokens(x_norm)
    token_mask = token_mask_from_frame_mask(frame_mask, true_tokens.shape[-1])
    base = masked.generate(
        cond=cond,
        seq_len=true_tokens.shape[-1],
        steps=steps,
        guidance_scale=guidance_scale,
        temperature=temperature,
        mask=token_mask,
    )

    if variant == "base":
        tokens = base.unsqueeze(1)
    elif variant == "full":
        tokens = residual.generate_residuals(base, cond=cond, guidance_scale=guidance_scale, mask=token_mask)
    else:
        raise ValueError(f"unknown variant {variant!r}")

    return normalizer.inverse(vqvae.decode_from_tokens(tokens, target_len=real_x.shape[1]))


def compute_metrics(real: np.ndarray, gen: np.ndarray, text: np.ndarray, diversity_times: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "fid": float(fid(real, gen)),
        "r_precision": r_precision(text, gen, top_k=3, rng=rng).tolist(),
        "mm_dist": float(mm_distance(text, gen)),
        "diversity": float(diversity(gen, diversity_times=diversity_times, rng=rng)),
        "diversity_real": float(diversity(real, diversity_times=diversity_times, rng=rng)),
        "num_clips": int(gen.shape[0]),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    ckpt_path = require_path(args.checkpoint, "MoMask checkpoint")
    require_path(Path(args.data_root) / "humanml3d.zip", "packed HumanML3D zip")
    require_path(Path(args.data_root) / "splits.json", "HumanML3D splits")
    require_path(Path(args.data_root) / "target_offsets.pt", "HumanML3D target offsets")

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = sorted(set(variants) - {"full", "base", "recon"})
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")

    ckpt = torch_load(ckpt_path, map_location=device)
    normalizer = H3DNormalizer.from_state_dict(ckpt["normalizer"])
    saved_args = ckpt_args(ckpt)
    steps = int(args.generation_steps or saved_args.get("generation_steps", 10))
    max_seq_len = int(args.max_seq_len or saved_args.get("max_seq_len", 80))
    text_encoder = RandomTextEncoder(text_dim=int(saved_args.get("text_dim", 64)))
    vqvae, masked, residual = build_models(ckpt, device)

    ds = HumanML3DDataset(
        root=args.data_root,
        split=args.split,
        max_seq_len=max_seq_len,
        min_seq_len=args.min_seq_len,
        mirror_augment=False,
        output_mode="h3d_263",
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0, drop_last=False)
    evaluator = build_evaluator(args, device)

    print(
        f"[evaluate_momask] checkpoint={ckpt_path} split={args.split} clips={len(ds)} "
        f"max_clips={args.max_clips} variants={variants} evaluator={args.evaluator} device={device}",
        flush=True,
    )

    real_embs: list[np.ndarray] = []
    text_embs: list[np.ndarray] = []
    gen_embs: dict[str, list[np.ndarray]] = {variant: [] for variant in variants}
    n_seen = 0
    t0 = time.perf_counter()

    for batch in tqdm(loader, desc="evaluate MoMask"):
        take = batch.x1.shape[0]
        if args.max_clips > 0:
            take = min(take, args.max_clips - n_seen)
            if take <= 0:
                break

        real_x = batch.x1[:take].to(device)
        frame_mask = batch.mask[:take].to(device)
        lengths = batch.lengths[:take]
        texts = batch.texts[:take]
        cond = text_encoder.encode(texts, device=device)

        real_embs.append(encode_motion(evaluator, real_x, lengths))
        text_embs.append(evaluator.encode_text_from_strings(texts).cpu().numpy())

        for variant in variants:
            gen = generate_variant(
                variant,
                vqvae=vqvae,
                masked=masked,
                residual=residual,
                normalizer=normalizer,
                cond=cond,
                real_x=real_x,
                frame_mask=frame_mask,
                steps=steps,
                guidance_scale=args.guidance_scale,
                temperature=args.temperature,
            )
            gen_embs[variant].append(encode_motion(evaluator, gen, lengths))

        n_seen += take
        if args.max_clips > 0 and n_seen >= args.max_clips:
            break

    real = np.concatenate(real_embs, axis=0)
    text = np.concatenate(text_embs, axis=0)
    results = {
        "_meta": {
            "checkpoint": str(ckpt_path),
            "split": args.split,
            "max_clips": args.max_clips,
            "num_clips": int(real.shape[0]),
            "max_seq_len": max_seq_len,
            "generation_steps": steps,
            "guidance_scale": args.guidance_scale,
            "temperature": args.temperature,
            "evaluator": args.evaluator,
            "elapsed_sec": time.perf_counter() - t0,
        }
    }

    for variant in variants:
        gen = np.concatenate(gen_embs[variant], axis=0)
        results[variant] = compute_metrics(real, gen, text, args.diversity_times, args.seed)
        print(f"\n[{variant}]\n{json.dumps(results[variant], indent=2)}", flush=True)

    output = Path(args.output) if args.output else ckpt_path.parent / "eval_momask" / "results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[evaluate_momask] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
