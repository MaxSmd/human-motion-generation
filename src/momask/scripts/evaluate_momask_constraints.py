"""Evaluate simple inference-time trajectory constraints for MoMask.

This is a first constraint baseline, intentionally separate from the standard
FID evaluator:

  1. Generate unconstrained MoMask motion.
  2. Build sparse root-XZ anchors from the paired real test motion.
  3. Interpolate those anchors into a target trajectory.
  4. Project generated root velocity features to follow that target.
  5. Optionally re-encode/decode through the RVQ-VAE to pull the result back
     toward the learned motion-token manifold.

The paired real trajectory is an oracle target used only for measuring whether
the constraint mechanism can satisfy known motion constraints.
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
from shared.data import H3D263Dataset, collate
from shared.text import CLIPTextEncoder, RandomTextEncoder, TextEncoder
from shared.geometry import H3D_FEATURE_DIM, quat_rotate, recover_joints_from_ric
from shared.eval import RandomGuoEvaluator, RealGuoEvaluator, diversity, fid, mm_distance, r_precision


class H3DNormalizer:
    def __init__(self, mean: Tensor, std: Tensor) -> None:
        self.mean = mean.float()
        self.std = std.float().clamp_min(1e-6)

    @classmethod
    def from_state_dict(cls, state: dict[str, Tensor]) -> "H3DNormalizer":
        return cls(state["mean"], state["std"])

    def transform(self, x: Tensor) -> Tensor:
        return (x - self.mean.to(x.device, x.dtype)) / self.std.to(x.device, x.dtype)

    def inverse(self, x: Tensor) -> Tensor:
        return x * self.std.to(x.device, x.dtype) + self.mean.to(x.device, x.dtype)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default=os.environ.get("RMG_DATA_ROOT", "external/data/humanml3d_packed"))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--max-clips", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--min-seq-len", type=int, default=40)
    p.add_argument("--generation-steps", type=int, default=None)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--topk-filter-thres", type=float, default=1.0)
    p.add_argument("--sample", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--remask-kept-tokens", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--anchor-stride", type=int, default=20, help="Frames between root-XZ trajectory anchors.")
    p.add_argument("--text-encoder", choices=["checkpoint", "random", "clip"], default="checkpoint")
    p.add_argument("--clip-model", default=None)
    p.add_argument("--clip-cache-dir", default=None)
    p.add_argument("--clip-backend", choices=["auto", "openai", "transformers"], default="auto")
    p.add_argument("--evaluator", choices=["real", "random"], default="real")
    p.add_argument("--text-to-motion-repo", default="external/text-to-motion")
    p.add_argument("--humanml3d-repo", default="external/HumanML3D")
    p.add_argument(
        "--real-h3d-dir",
        default=None,
        help="Optional canonical HumanML3D new_joint_vecs dir for real features, targets, and canonical checkpoints.",
    )
    p.add_argument(
        "--model-input-source",
        choices=["auto", "packed", "canonical"],
        default="auto",
        help="Motion features used for token length/masks and constraint targets.",
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
        help="Use HumanML3D's original word/POS tokens from texts.zip for R-Precision/MM-Dist.",
    )
    p.add_argument("--diversity-times", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default=None)
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


def build_models(ckpt: dict, device: torch.device):
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
        explicit_loss_weight=float(a.get("vq_explicit_weight", 0.0)),
        use_ema_quantizer=bool(a.get("vq_use_ema", False)),
        ema_decay=float(a.get("vq_ema_decay", 0.99)),
        codebook_sample_temp=float(a.get("vq_codebook_sample_temp", 0.0)),
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
    vqvae.load_state_dict(ckpt["vqvae"])
    masked.load_state_dict(ckpt["masked_transformer"])
    residual.load_state_dict(ckpt["residual_transformer"])
    vqvae.eval()
    masked.eval()
    residual.eval()
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
    print(f"[constraints] loaded VIP caption tokens for {len(out)} clips from {zp}", flush=True)
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


def token_mask_from_frame_mask(mask: Tensor, token_len: int) -> Tensor:
    if mask.shape[1] == token_len:
        return mask
    pooled = F.adaptive_max_pool1d(mask.float().unsqueeze(1), token_len).squeeze(1)
    return pooled > 0.5


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


def root_xz(motion: Tensor) -> Tensor:
    return recover_joints_from_ric(motion.float())[:, :, 0, :][:, :, [0, 2]]


def root_quat_from_features(motion: Tensor) -> Tensor:
    rot_vel = motion[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)
    quat = torch.zeros(*motion.shape[:-1], 4, device=motion.device, dtype=motion.dtype)
    quat[..., 0] = torch.cos(r_rot_ang)
    quat[..., 2] = torch.sin(r_rot_ang)
    return quat


def interpolate_anchor_trajectory(real_root: Tensor, lengths: Tensor, anchor_stride: int) -> tuple[Tensor, Tensor]:
    """Return dense target root-XZ path and sparse anchor mask."""
    B, T, _ = real_root.shape
    target = real_root.clone()
    anchor_mask = torch.zeros(B, T, dtype=torch.bool, device=real_root.device)
    for b in range(B):
        L = int(lengths[b].item())
        if L <= 0:
            continue
        anchors = list(range(0, L, max(1, anchor_stride)))
        if anchors[-1] != L - 1:
            anchors.append(L - 1)
        anchor_mask[b, anchors] = True
        for left, right in zip(anchors[:-1], anchors[1:]):
            span = max(1, right - left)
            alpha = torch.linspace(0, 1, span + 1, device=real_root.device, dtype=real_root.dtype).unsqueeze(-1)
            target[b, left : right + 1] = (1 - alpha) * real_root[b, left] + alpha * real_root[b, right]
    return target, anchor_mask


def project_root_trajectory(motion: Tensor, target_root: Tensor, lengths: Tensor) -> Tensor:
    """Edit HumanML3D root velocity channels so recovered root follows target."""
    out = motion.clone()
    quat = root_quat_from_features(out)
    target3 = torch.zeros(*target_root.shape[:-1], 3, device=out.device, dtype=out.dtype)
    target3[..., 0] = target_root[..., 0]
    target3[..., 2] = target_root[..., 1]
    # HumanML3D stores root XZ velocity rotated by the destination frame's
    # root orientation, matching `_features_from_positions_and_quats`.
    local_delta = quat_rotate(quat[:, 1:], target3[:, 1:] - target3[:, :-1])
    for b in range(out.shape[0]):
        L = int(lengths[b].item())
        if L <= 1:
            continue
        out[b, : L - 1, 1] = local_delta[b, : L - 1, 0]
        out[b, : L - 1, 2] = local_delta[b, : L - 1, 2]
        out[b, :L, 3] = motion[b, :L, 3]
    return out


def trajectory_errors(motion: Tensor, target_root: Tensor, anchor_mask: Tensor, lengths: Tensor) -> dict[str, float]:
    pred = root_xz(motion)
    valid = torch.zeros_like(anchor_mask)
    for b, L in enumerate(lengths.tolist()):
        valid[b, : int(L)] = True
    dense_err = ((pred - target_root).norm(dim=-1)[valid]).mean()
    anchor_valid = anchor_mask & valid
    anchor_err = ((pred - target_root).norm(dim=-1)[anchor_valid]).mean()
    final_errs = []
    for b, L in enumerate(lengths.tolist()):
        L = int(L)
        if L > 0:
            final_errs.append((pred[b, L - 1] - target_root[b, L - 1]).norm())
    final_err = torch.stack(final_errs).mean() if final_errs else dense_err.new_tensor(float("nan"))
    return {
        "traj_dense_l2": float(dense_err.detach().cpu()),
        "traj_anchor_l2": float(anchor_err.detach().cpu()),
        "traj_final_l2": float(final_err.detach().cpu()),
    }


@torch.no_grad()
def generate_full(
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
    tokens = residual.generate_residuals(
        base,
        cond=cond,
        guidance_scale=guidance_scale,
        temperature=temperature,
        topk_filter_thres=topk_filter_thres,
        sample=sample,
        mask=token_mask,
    )
    return normalizer.inverse(vqvae.decode_from_tokens(tokens, target_len=real_x.shape[1]))


@torch.no_grad()
def vq_project(vqvae: MotionRVQVAE, normalizer: H3DNormalizer, motion: Tensor, frame_mask: Tensor) -> Tensor:
    return normalizer.inverse(vqvae(normalizer.transform(motion), mask=frame_mask).recon)


def compute_quality(real: np.ndarray, gen: np.ndarray, text: np.ndarray, diversity_times: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "fid": float(fid(real, gen)),
        "r_precision": r_precision(text, gen, top_k=3, rng=rng).tolist(),
        "mm_dist": float(mm_distance(text, gen)),
        "diversity": float(diversity(gen, diversity_times=diversity_times, rng=rng)),
        "num_clips": int(gen.shape[0]),
    }


def encode_motion(evaluator, motion: Tensor, lengths: Tensor) -> np.ndarray:
    return evaluator.encode_motion(motion.cpu(), lengths.cpu()).cpu().numpy()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    ckpt = torch_load(require_path(args.checkpoint, "MoMask checkpoint"), map_location=device)
    saved_args = ckpt_args(ckpt)
    normalizer = H3DNormalizer.from_state_dict(ckpt["normalizer"])
    real_h3d_dir = Path(args.real_h3d_dir) if args.real_h3d_dir else None
    if real_h3d_dir is not None:
        require_path(real_h3d_dir, "canonical HumanML3D new_joint_vecs dir")
    model_input_source = args.model_input_source
    if model_input_source == "auto":
        model_input_source = "canonical" if saved_args.get("canonical_h3d_dir") and real_h3d_dir is not None else "packed"
    if model_input_source == "canonical" and real_h3d_dir is None:
        raise ValueError("--model-input-source canonical requires --real-h3d-dir")
    steps = int(args.generation_steps or saved_args.get("generation_steps", 10))
    max_seq_len = int(args.max_seq_len or saved_args.get("max_seq_len", 80))
    text_encoder = build_text_encoder(args, saved_args)
    vqvae, masked, residual = build_models(ckpt, device)
    evaluator = build_evaluator(args, device)
    caption_tokens = None
    if args.evaluator == "real" and args.vip_tokens:
        texts_zip = resolve_texts_zip(args)
        try:
            caption_tokens = load_caption_tokens(texts_zip)
        except FileNotFoundError as e:
            print(f"[constraints] WARN: {e}; falling back to spaCy text encoding", flush=True)

    ds = H3D263Dataset(
        root=args.data_root,
        split=args.split,
        max_seq_len=max_seq_len,
        min_seq_len=args.min_seq_len,
        mirror_augment=False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    buckets: dict[str, list[np.ndarray]] = {"real": [], "unconstrained": [], "traj_projected": [], "traj_vq": []}
    text_embs: list[np.ndarray] = []
    traj_sums: dict[str, dict[str, float]] = {
        "unconstrained": {},
        "traj_projected": {},
        "traj_vq": {},
    }
    n_seen = 0
    n_text_fallback = 0
    t0 = time.perf_counter()

    print(
        f"[constraints] ckpt={args.checkpoint} clips={len(ds)} max_clips={args.max_clips} "
        f"steps={steps} guidance={args.guidance_scale} anchor_stride={args.anchor_stride} "
        f"text_tokens={'vip' if caption_tokens is not None else 'spacy'} "
        f"real_features={'canonical' if real_h3d_dir is not None else 'packed'} "
        f"model_input={model_input_source} sample={args.sample} topk={args.topk_filter_thres}",
        flush=True,
    )

    for batch in tqdm(loader, desc="evaluate constrained MoMask"):
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
        cond = text_encoder.encode(texts, device=device)

        eval_real_x = real_x
        eval_lengths = lengths
        if real_h3d_dir is not None:
            eval_real_x, eval_lengths, n_missing_real = load_canonical_motion_batch(
                real_h3d_dir,
                clip_ids,
                max_len=real_x.shape[1],
            )
            if n_missing_real:
                raise FileNotFoundError(
                    f"{n_missing_real} canonical HumanML3D feature files missing under {real_h3d_dir}"
                )

        model_real_x = eval_real_x.to(device) if model_input_source == "canonical" else real_x
        model_lengths = eval_lengths if model_input_source == "canonical" else lengths
        model_frame_mask = torch.arange(model_real_x.shape[1], device=device).unsqueeze(0) < model_lengths.to(device).unsqueeze(1)

        gen = generate_full(
            vqvae=vqvae,
            masked=masked,
            residual=residual,
            normalizer=normalizer,
            cond=cond,
            real_x=model_real_x,
            frame_mask=model_frame_mask,
            steps=steps,
            guidance_scale=args.guidance_scale,
            temperature=args.temperature,
            topk_filter_thres=args.topk_filter_thres,
            sample=args.sample,
            remask_kept_tokens=args.remask_kept_tokens,
        )
        real_root = root_xz(model_real_x)
        target_root, anchor_mask = interpolate_anchor_trajectory(real_root, model_lengths.to(device), args.anchor_stride)
        projected = project_root_trajectory(gen, target_root, model_lengths.to(device))
        traj_vq = vq_project(vqvae, normalizer, projected, model_frame_mask)

        for name, motion in (("unconstrained", gen), ("traj_projected", projected), ("traj_vq", traj_vq)):
            errs = trajectory_errors(motion, target_root, anchor_mask, model_lengths.to(device))
            for key, value in errs.items():
                traj_sums[name][key] = traj_sums[name].get(key, 0.0) + value * take

        buckets["real"].append(encode_motion(evaluator, eval_real_x, eval_lengths))
        buckets["unconstrained"].append(encode_motion(evaluator, gen, eval_lengths))
        buckets["traj_projected"].append(encode_motion(evaluator, projected, eval_lengths))
        buckets["traj_vq"].append(encode_motion(evaluator, traj_vq, eval_lengths))
        text_np, n_missing = encode_text_batch(evaluator, texts, clip_ids, caption_tokens)
        text_embs.append(text_np)
        n_text_fallback += n_missing

        n_seen += take
        if args.max_clips > 0 and n_seen >= args.max_clips:
            break

    real = np.concatenate(buckets["real"], axis=0)
    text = np.concatenate(text_embs, axis=0)
    diag_rng = np.random.default_rng(args.seed)
    results = {
        "_meta": {
            "checkpoint": args.checkpoint,
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
            "anchor_stride": args.anchor_stride,
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
    if caption_tokens is not None:
        print(f"[constraints] VIP token fallbacks={n_text_fallback}", flush=True)
    print(f"\n[diagnostics]\n{json.dumps(results['_diagnostics'], indent=2)}", flush=True)
    for name in ("unconstrained", "traj_projected", "traj_vq"):
        emb = np.concatenate(buckets[name], axis=0)
        quality = compute_quality(real, emb, text, args.diversity_times, args.seed)
        traj = {k: v / max(n_seen, 1) for k, v in traj_sums[name].items()}
        results[name] = {**quality, **traj}
        print(f"\n[{name}]\n{json.dumps(results[name], indent=2)}", flush=True)

    output = Path(args.output) if args.output else Path(args.checkpoint).parent / "constraints" / "trajectory.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[constraints] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
