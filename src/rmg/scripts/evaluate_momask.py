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
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from momask.models import (
    CodebookResidualTransformer,
    MaskedMotionTransformer,
    MotionRVQVAE,
    ResidualTransformer,
    TokenTransformerConfig,
)
from rmg.data import HumanML3DDataset, collate
from rmg.models import CLIPTextEncoder, RandomTextEncoder, TextEncoder
from rmg.representation import H3D_FEATURE_DIM
from shared.eval import RandomGuoEvaluator, RealGuoEvaluator, diversity, fid, mm_distance, r_precision


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
    p.add_argument("--topk-filter-thres", type=float, default=1.0)
    p.add_argument(
        "--sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sample tokens from the filtered distribution instead of greedy argmax.",
    )
    p.add_argument(
        "--remask-kept-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow previously accepted base tokens to be masked again. "
            "Use --no-remask-kept-tokens for the official MoMask-style sampler."
        ),
    )
    p.add_argument("--text-encoder", choices=["checkpoint", "random", "clip"], default="checkpoint")
    p.add_argument("--clip-model", default=None, help="Defaults to checkpoint args.clip_model or ViT-B/32.")
    p.add_argument("--clip-cache-dir", default=None)
    p.add_argument("--clip-backend", choices=["auto", "openai", "transformers"], default="auto")
    p.add_argument("--evaluator", choices=["real", "random"], default="real")
    p.add_argument("--text-to-motion-repo", default="external/text-to-motion")
    p.add_argument("--humanml3d-repo", default="external/HumanML3D")
    p.add_argument(
        "--real-h3d-dir",
        default=None,
        help=(
            "Optional directory containing canonical HumanML3D new_joint_vecs/*.npy. "
            "When set, FID/real diagnostics use these official real features instead "
            "of packed T+R-derived features."
        ),
    )
    p.add_argument(
        "--model-input-source",
        choices=["auto", "packed", "canonical"],
        default="auto",
        help=(
            "Motion features fed to the VQ/recon path. 'auto' uses canonical when "
            "the checkpoint was trained with --canonical-h3d-dir and --real-h3d-dir is set."
        ),
    )
    p.add_argument(
        "--humanml3d-texts-zip",
        default=None,
        help="Optional direct path to HumanML3D/HumanML3D/texts.zip for VIP caption tokens.",
    )
    p.add_argument(
        "--vip-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use HumanML3D's original word/POS tokens from texts.zip for the Guo "
            "text encoder. This preserves *_VIP tags and should be on for real evaluation."
        ),
    )
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


def build_text_encoder(args: argparse.Namespace, saved_args: dict) -> TextEncoder:
    kind = saved_args.get("text_encoder", "random") if args.text_encoder == "checkpoint" else args.text_encoder
    if kind == "random":
        return RandomTextEncoder(text_dim=int(saved_args.get("text_dim", 64)))
    if kind == "clip":
        return CLIPTextEncoder(
            model_name=args.clip_model or str(saved_args.get("clip_model", "ViT-B/32")),
            cache_dir=args.clip_cache_dir,
            backend=args.clip_backend,
        )
    raise ValueError(f"unknown text encoder: {kind}")


def build_vqvae(ckpt: dict, device: torch.device) -> MotionRVQVAE:
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
    if "vqvae" not in ckpt:
        raise KeyError("checkpoint missing 'vqvae'")
    vqvae.load_state_dict(ckpt["vqvae"])
    vqvae.eval()
    return vqvae


def build_token_models(ckpt: dict, device: torch.device):
    a = ckpt_args(ckpt)
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
    if a.get("residual_arch", "simple") == "codebook":
        residual = CodebookResidualTransformer(
            cfg,
            num_quantizers=int(a.get("num_quantizers", 3)),
            code_dim=int(a.get("vq_latent_dim", 32)),
            share_weight=bool(a.get("residual_share_weight", False)),
        ).to(device)
    else:
        residual = ResidualTransformer(
            cfg,
            num_quantizers=int(a.get("num_quantizers", 3)),
            separate_level_heads=not bool(a.get("shared_residual_head", False)),
        ).to(device)

    for key, model in (("masked_transformer", masked), ("residual_transformer", residual)):
        if key not in ckpt:
            raise KeyError(f"checkpoint missing {key!r}")
        model.load_state_dict(ckpt[key])
        model.eval()
    return masked, residual


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


def load_caption_tokens(texts_zip: str | Path) -> dict[str, dict[str, list[str]]]:
    """clip_id -> {caption_text: [word/POS tokens]} from HumanML3D's texts.zip."""
    zp = Path(texts_zip)
    if not zp.exists():
        raise FileNotFoundError(f"HumanML3D texts.zip not found: {zp}")
    out: dict[str, dict[str, list[str]]] = {}
    with zipfile.ZipFile(zp) as zf:
        for name in zf.namelist():
            if not name.endswith(".txt"):
                continue
            cid = Path(name).stem
            cap_to_tokens: dict[str, list[str]] = {}
            for line in zf.read(name).decode("utf-8").splitlines():
                parts = line.strip().split("#")
                if len(parts) < 2:
                    continue
                cap = parts[0].strip()
                tokens = parts[1].strip().split()
                if cap and tokens:
                    cap_to_tokens[cap] = tokens
            if cap_to_tokens:
                out[cid] = cap_to_tokens
    print(f"[evaluate_momask] loaded VIP caption tokens for {len(out)} clips from {zp}", flush=True)
    return out


def resolve_texts_zip(args: argparse.Namespace) -> Path:
    if args.humanml3d_texts_zip:
        return Path(args.humanml3d_texts_zip)
    repo = Path(args.humanml3d_repo)
    candidates = [
        repo / "HumanML3D" / "texts.zip",
        repo / "texts.zip",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def encode_text_batch(
    evaluator,
    texts: list[str],
    clip_ids: list[str],
    caption_tokens: dict[str, dict[str, list[str]]] | None,
) -> tuple[np.ndarray, int]:
    if caption_tokens is None:
        return evaluator.encode_text_from_strings(texts).cpu().numpy(), 0
    tokens = [caption_tokens.get(cid, {}).get(text) for cid, text in zip(clip_ids, texts)]
    missing_idxs = [i for i, tok in enumerate(tokens) if tok is None]
    present_idxs = [i for i, tok in enumerate(tokens) if tok is not None]
    out = np.empty((len(texts), evaluator.text_dim), dtype=np.float32)
    if present_idxs:
        present_tokens = [tokens[i] for i in present_idxs]
        out[present_idxs] = evaluator.encode_text_from_tokens(present_tokens).cpu().numpy()
    if missing_idxs:
        missing_texts = [texts[i] for i in missing_idxs]
        out[missing_idxs] = evaluator.encode_text_from_strings(missing_texts).cpu().numpy()
    return out, len(missing_idxs)


def encode_motion(evaluator, motion: Tensor, lengths: Tensor) -> np.ndarray:
    return evaluator.encode_motion(motion.cpu(), lengths.cpu()).cpu().numpy()


def load_canonical_motion_batch(
    h3d_dir: str | Path,
    clip_ids: list[str],
    max_len: int,
) -> tuple[Tensor, Tensor, int]:
    root = Path(h3d_dir)
    feats: list[Tensor] = []
    missing = 0
    for cid in clip_ids:
        candidates = [cid]
        if cid.startswith("M") and len(cid) > 1:
            candidates.append(cid[1:])
        path = next((root / f"{name}.npy" for name in candidates if (root / f"{name}.npy").exists()), None)
        if path is None:
            missing += 1
            feats.append(torch.zeros(1, H3D_FEATURE_DIM))
            continue
        arr = np.load(path).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != H3D_FEATURE_DIM:
            raise ValueError(f"canonical HumanML3D feature must be (T, {H3D_FEATURE_DIM}), got {arr.shape}: {path}")
        arr = arr[:max_len]
        if len(arr) < 1:
            missing += 1
            feats.append(torch.zeros(1, H3D_FEATURE_DIM))
            continue
        feats.append(torch.from_numpy(arr))

    tmax = max(f.shape[0] for f in feats)
    out = torch.zeros(len(feats), tmax, H3D_FEATURE_DIM)
    lengths = torch.zeros(len(feats), dtype=torch.long)
    for i, feat in enumerate(feats):
        out[i, : feat.shape[0]] = feat
        lengths[i] = feat.shape[0]
    return out, lengths, missing


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
    topk_filter_thres: float,
    sample: bool,
    remask_kept_tokens: bool,
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
        topk_filter_thres=topk_filter_thres,
        sample=sample,
        remask_kept_tokens=remask_kept_tokens,
        mask=token_mask,
    )

    if variant == "base":
        tokens = base.unsqueeze(1)
    elif variant == "full":
        tokens = residual.generate_residuals(
            base,
            cond=cond,
            guidance_scale=guidance_scale,
            temperature=temperature,
            topk_filter_thres=topk_filter_thres,
            sample=sample,
            mask=token_mask,
        )
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
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    ckpt_path = require_path(args.checkpoint, "MoMask checkpoint")
    require_path(Path(args.data_root) / "humanml3d.zip", "packed HumanML3D zip")
    require_path(Path(args.data_root) / "splits.json", "HumanML3D splits")
    require_path(Path(args.data_root) / "target_offsets.pt", "HumanML3D target offsets")
    real_h3d_dir = Path(args.real_h3d_dir) if args.real_h3d_dir else None
    if real_h3d_dir is not None:
        require_path(real_h3d_dir, "canonical HumanML3D new_joint_vecs dir")

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = sorted(set(variants) - {"full", "base", "recon"})
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    needs_generation = any(v in {"full", "base"} for v in variants)

    ckpt = torch_load(ckpt_path, map_location=device)
    normalizer = H3DNormalizer.from_state_dict(ckpt["normalizer"])
    saved_args = ckpt_args(ckpt)
    model_input_source = args.model_input_source
    if model_input_source == "auto":
        model_input_source = "canonical" if saved_args.get("canonical_h3d_dir") and real_h3d_dir is not None else "packed"
    if model_input_source == "canonical" and real_h3d_dir is None:
        raise ValueError("--model-input-source canonical requires --real-h3d-dir")
    steps = int(args.generation_steps or saved_args.get("generation_steps", 10))
    max_seq_len = int(args.max_seq_len or saved_args.get("max_seq_len", 80))
    vqvae = build_vqvae(ckpt, device)
    text_encoder: TextEncoder | None = None
    masked: MaskedMotionTransformer | None = None
    residual: ResidualTransformer | None = None
    if needs_generation:
        text_encoder = build_text_encoder(args, saved_args)
        if text_encoder.text_dim != int(saved_args.get("text_dim", text_encoder.text_dim)):
            raise ValueError(
                f"text encoder dim {text_encoder.text_dim} does not match checkpoint text_dim "
                f"{saved_args.get('text_dim')}"
            )
        masked, residual = build_token_models(ckpt, device)

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
    caption_tokens = None
    if args.evaluator == "real" and args.vip_tokens:
        texts_zip = resolve_texts_zip(args)
        try:
            caption_tokens = load_caption_tokens(texts_zip)
        except FileNotFoundError as e:
            print(f"[evaluate_momask] WARN: {e}; falling back to spaCy text encoding", flush=True)

    print(
        f"[evaluate_momask] checkpoint={ckpt_path} split={args.split} clips={len(ds)} "
        f"max_clips={args.max_clips} variants={variants} evaluator={args.evaluator} "
        f"text_encoder={saved_args.get('text_encoder', 'random') if needs_generation else '<unused>'} "
        f"text_tokens={'vip' if caption_tokens is not None else 'spacy'} "
        f"real_features={'canonical' if real_h3d_dir is not None else 'packed'} "
        f"model_input={model_input_source} device={device}",
        flush=True,
    )

    real_embs: list[np.ndarray] = []
    packed_real_embs: list[np.ndarray] = []
    text_embs: list[np.ndarray] = []
    gen_embs: dict[str, list[np.ndarray]] = {variant: [] for variant in variants}
    n_seen = 0
    n_text_fallback = 0
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
        clip_ids = batch.clip_ids[:take]
        cond = text_encoder.encode(texts, device=device) if text_encoder is not None else None

        eval_real_x = real_x
        eval_lengths = lengths
        if real_h3d_dir is not None:
            packed_real_embs.append(encode_motion(evaluator, real_x, lengths))
            eval_real_x, eval_lengths, n_missing_real = load_canonical_motion_batch(
                real_h3d_dir,
                clip_ids,
                max_len=real_x.shape[1],
            )
            if n_missing_real:
                raise FileNotFoundError(
                    f"{n_missing_real} canonical HumanML3D feature files missing under {real_h3d_dir}"
                )

        real_embs.append(encode_motion(evaluator, eval_real_x, eval_lengths))
        model_real_x = eval_real_x.to(device) if model_input_source == "canonical" else real_x
        model_lengths = eval_lengths if model_input_source == "canonical" else lengths
        model_frame_mask = torch.arange(model_real_x.shape[1], device=device).unsqueeze(0) < model_lengths.to(device).unsqueeze(1)
        text_np, n_missing = encode_text_batch(evaluator, texts, clip_ids, caption_tokens)
        text_embs.append(text_np)
        n_text_fallback += n_missing

        for variant in variants:
            if variant != "recon":
                if masked is None or residual is None or cond is None:
                    raise RuntimeError(f"variant {variant!r} requires token transformers")
            gen = generate_variant(
                variant,
                vqvae=vqvae,
                masked=masked,  # type: ignore[arg-type]
                residual=residual,  # type: ignore[arg-type]
                normalizer=normalizer,
                cond=cond,  # type: ignore[arg-type]
                real_x=model_real_x,
                frame_mask=model_frame_mask,
                steps=steps,
                guidance_scale=args.guidance_scale,
                temperature=args.temperature,
                topk_filter_thres=args.topk_filter_thres,
                sample=args.sample,
                remask_kept_tokens=args.remask_kept_tokens,
            )
            gen_embs[variant].append(encode_motion(evaluator, gen, eval_lengths))

        n_seen += take
        if args.max_clips > 0 and n_seen >= args.max_clips:
            break

    real = np.concatenate(real_embs, axis=0)
    packed_real = np.concatenate(packed_real_embs, axis=0) if packed_real_embs else None
    text = np.concatenate(text_embs, axis=0)
    diag_rng = np.random.default_rng(args.seed)
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
            "topk_filter_thres": args.topk_filter_thres,
            "sample": args.sample,
            "remask_kept_tokens": args.remask_kept_tokens,
            "evaluator": args.evaluator,
            "real_feature_source": "canonical" if real_h3d_dir is not None else "packed",
            "real_h3d_dir": str(real_h3d_dir) if real_h3d_dir is not None else None,
            "model_input_source": model_input_source,
            "text_token_source": "vip" if caption_tokens is not None else "spacy",
            "vip_token_fallbacks": int(n_text_fallback),
            "elapsed_sec": time.perf_counter() - t0,
        },
        "_diagnostics": {
            "r_precision_real": r_precision(text, real, top_k=3, rng=diag_rng).tolist(),
            "mm_dist_real": float(mm_distance(text, real)),
            "diversity_real": float(diversity(real, diversity_times=args.diversity_times, rng=diag_rng)),
        },
    }
    if packed_real is not None:
        packed_rng = np.random.default_rng(args.seed)
        results["_diagnostics"]["packed_real"] = {
            "fid_vs_canonical": float(fid(real, packed_real)),
            "r_precision": r_precision(text, packed_real, top_k=3, rng=packed_rng).tolist(),
            "mm_dist": float(mm_distance(text, packed_real)),
            "diversity": float(diversity(packed_real, diversity_times=args.diversity_times, rng=packed_rng)),
        }
    if caption_tokens is not None:
        print(f"[evaluate_momask] VIP token fallbacks={n_text_fallback}", flush=True)
    print(f"\n[diagnostics]\n{json.dumps(results['_diagnostics'], indent=2)}", flush=True)

    for variant in variants:
        gen = np.concatenate(gen_embs[variant], axis=0)
        results[variant] = compute_metrics(real, gen, text, args.diversity_times, args.seed)
        if variant == "recon" and packed_real is not None:
            results["recon_vs_packed_real"] = compute_metrics(
                packed_real,
                gen,
                text,
                args.diversity_times,
                args.seed,
            )
        print(f"\n[{variant}]\n{json.dumps(results[variant], indent=2)}", flush=True)
        if variant == "recon" and "recon_vs_packed_real" in results:
            print(f"\n[recon_vs_packed_real]\n{json.dumps(results['recon_vs_packed_real'], indent=2)}", flush=True)

    output = Path(args.output) if args.output else ckpt_path.parent / "eval_momask" / "results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[evaluate_momask] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
