"""Compare the released MoMask RVQ with this repository's RVQ.

The upstream model is imported directly from a clean checkout of
EricGuo5513/momask-codes. Both models are evaluated on the same shuffled,
drop-last HumanML3D batches through the paired Guo evaluator path used by
official ``evaluation_vqvae``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy import linalg
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from momask.data_utils import normalize_motion
from momask.scripts.evaluate_momask import build_vqvae, load_caption_tokens, torch_load
from momask.scripts.train_momask import H3DNormalizer
from shared.data import CanonicalHumanML3DText2MotionDataset, collate
from shared.eval.guo_evaluator import RealGuoEvaluator
from shared.eval.metrics import (
    calculate_activation_statistics,
    fid as eigen_fid,
    r_precision_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours-checkpoint", required=True)
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--official-checkpoint", required=True)
    parser.add_argument("--official-opt", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--canonical-h3d-dir", required=True)
    parser.add_argument("--texts-zip", required=True)
    parser.add_argument("--humanml3d-split-dir", required=True)
    parser.add_argument("--text-to-motion-repo", default="external/text-to-motion")
    parser.add_argument("--humanml3d-repo", default="external/HumanML3D")
    parser.add_argument("--evaluator-checkpoints-dir", required=True)
    parser.add_argument("--evaluator-normalization-name", default="Comp_v6_KLD005")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-clips", type=int, default=-1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def parse_upstream_opt(path: Path) -> SimpleNamespace:
    values: dict[str, object] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("---") or ": " not in line:
            continue
        key, raw_value = line.split(": ", 1)
        if raw_value in {"True", "False"}:
            value: object = raw_value == "True"
        else:
            try:
                value = int(raw_value)
            except ValueError:
                try:
                    value = float(raw_value)
                except ValueError:
                    value = raw_value
        values[key] = value
    required = {
        "num_quantizers",
        "shared_codebook",
        "quantize_dropout_prob",
        "mu",
        "nb_code",
        "code_dim",
        "down_t",
        "stride_t",
        "width",
        "depth",
        "dilation_growth_rate",
        "vq_act",
        "vq_norm",
    }
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(f"official opt file is missing: {', '.join(missing)}")
    return SimpleNamespace(**values)


def load_upstream_vq(
    repo: Path,
    checkpoint_path: Path,
    opt_path: Path,
    device: torch.device,
):
    model_file = repo / "models" / "vq" / "model.py"
    if not model_file.is_file():
        raise FileNotFoundError(f"official MoMask model not found: {model_file}")
    if "models" in sys.modules:
        raise RuntimeError("a top-level 'models' package was imported before the official MoMask model")
    sys.path.insert(0, str(repo))
    try:
        from models.vq.model import RVQVAE  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)

    opt = parse_upstream_opt(opt_path)
    model = RVQVAE(
        opt,
        263,
        opt.nb_code,
        opt.code_dim,
        opt.code_dim,
        opt.down_t,
        opt.stride_t,
        opt.width,
        opt.depth,
        opt.dilation_growth_rate,
        opt.vq_act,
        None if opt.vq_norm == "None" else opt.vq_norm,
    )
    checkpoint = torch_load(checkpoint_path)
    model_key = "vq_model" if "vq_model" in checkpoint else "net"
    if model_key not in checkpoint:
        raise KeyError("official checkpoint contains neither 'vq_model' nor 'net'")
    model.load_state_dict(checkpoint[model_key], strict=True)
    model.to(device).eval()
    return model, opt, int(checkpoint.get("ep", -1))


def scipy_fid(real: np.ndarray, predicted: np.ndarray, eps: float = 1e-6) -> float:
    """Official HumanML3D FID implementation based on scipy.linalg.sqrtm."""
    mu1, sigma1 = calculate_activation_statistics(real)
    mu2, sigma2 = calculate_activation_statistics(predicted)
    diff = mu1 - mu2
    sqrtm_result = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    covmean = sqrtm_result[0] if isinstance(sqrtm_result, tuple) else sqrtm_result
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        diagonal_imag = np.max(np.abs(np.diagonal(covmean).imag))
        if diagonal_imag > 1e-3:
            raise ValueError(f"FID covariance product has imaginary component {diagonal_imag}")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def official_diversity(features: np.ndarray, count: int, rng: np.random.RandomState) -> float:
    if features.shape[0] <= count:
        count = max(1, features.shape[0] - 1)
    first = rng.choice(features.shape[0], count, replace=False)
    second = rng.choice(features.shape[0], count, replace=False)
    return float(np.linalg.norm(features[first] - features[second], axis=1).mean())


class ProtocolAccumulator:
    def __init__(self) -> None:
        self.real: list[np.ndarray] = []
        self.text: list[np.ndarray] = []
        self.motion: dict[str, list[np.ndarray]] = {"official_vq": [], "ours_vq": []}
        self.r_sum: dict[str, np.ndarray] = {
            "real": np.zeros(3, dtype=np.float64),
            "official_vq": np.zeros(3, dtype=np.float64),
            "ours_vq": np.zeros(3, dtype=np.float64),
        }
        self.mm_sum = {name: 0.0 for name in self.r_sum}
        self.raw_abs_sum = {"official_vq": 0.0, "ours_vq": 0.0}
        self.raw_value_count = 0
        self.num_pairs = 0

    def add(
        self,
        text: np.ndarray,
        real: np.ndarray,
        official: np.ndarray,
        ours: np.ndarray,
        raw: Tensor,
        official_raw: Tensor,
        ours_raw: Tensor,
        mask: Tensor,
    ) -> None:
        batch_size = text.shape[0]
        self.text.append(text)
        self.real.append(real)
        self.motion["official_vq"].append(official)
        self.motion["ours_vq"].append(ours)
        for name, motion in (("real", real), ("official_vq", official), ("ours_vq", ours)):
            self.r_sum[name] += r_precision_batch(text, motion, top_k=3) * batch_size
            self.mm_sum[name] += float(np.linalg.norm(text - motion, axis=1).sum())
        valid = mask.to(raw.dtype).unsqueeze(-1)
        self.raw_abs_sum["official_vq"] += float(((official_raw - raw).abs() * valid).sum())
        self.raw_abs_sum["ours_vq"] += float(((ours_raw - raw).abs() * valid).sum())
        self.raw_value_count += int(mask.sum()) * raw.shape[-1]
        self.num_pairs += batch_size

    def finish(self, seed: int) -> dict[str, object]:
        real = np.concatenate(self.real, axis=0)
        output: dict[str, object] = {
            "num_pairs": self.num_pairs,
            "real": {
                "r_precision": (self.r_sum["real"] / self.num_pairs).tolist(),
                "mm_dist": self.mm_sum["real"] / self.num_pairs,
                "diversity": official_diversity(real, 300, np.random.RandomState(seed)),
            },
        }
        for offset, name in enumerate(("official_vq", "ours_vq"), start=1):
            motion = np.concatenate(self.motion[name], axis=0)
            output[name] = {
                "fid_scipy": scipy_fid(real, motion),
                "fid_eigen": float(eigen_fid(real, motion)),
                "r_precision": (self.r_sum[name] / self.num_pairs).tolist(),
                "mm_dist": self.mm_sum[name] / self.num_pairs,
                "diversity": official_diversity(
                    motion, 300, np.random.RandomState(seed + offset)
                ),
                "raw_feature_mae": self.raw_abs_sum[name] / self.raw_value_count,
            }
        return output


def caption_token_batch(
    lookup: dict[str, dict[str, list[str]]],
    clip_ids: list[str],
    captions: list[str],
) -> list[list[str]]:
    tokens: list[list[str]] = []
    for clip_id, caption in zip(clip_ids, captions):
        base_id = clip_id.split(":segment", 1)[0]
        value = lookup.get(base_id, {}).get(caption)
        if value is None:
            raise KeyError(f"VIP caption tokens missing for {base_id!r}: {caption!r}")
        tokens.append(value)
    return tokens


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    official_model, official_opt, official_epoch = load_upstream_vq(
        Path(args.official_repo),
        Path(args.official_checkpoint),
        Path(args.official_opt),
        device,
    )
    ours_checkpoint = torch_load(args.ours_checkpoint)
    ours_model = build_vqvae(ours_checkpoint, device)
    ours_normalizer = H3DNormalizer.from_state_dict(ours_checkpoint["normalizer"])
    evaluator = RealGuoEvaluator(
        text_to_motion_repo=args.text_to_motion_repo,
        humanml3d_repo=args.humanml3d_repo,
        device=device,
        checkpoints_dir=args.evaluator_checkpoints_dir,
        normalization_name=args.evaluator_normalization_name,
    )
    caption_tokens = load_caption_tokens(args.texts_zip)
    dataset = CanonicalHumanML3DText2MotionDataset(
        root=args.data_root,
        canonical_dir=args.canonical_h3d_dir,
        texts_zip=args.texts_zip,
        split=args.split,
        max_seq_len=196,
        min_seq_len=40,
        unit_length=4,
        split_dir=args.humanml3d_split_dir,
        subset_n=0,
    )
    # Upstream Text2MotionDatasetEval sorts entries by motion length before
    # DataLoader shuffling. Preserve that pre-shuffle order so a fixed seed
    # creates the same style of retrieval batches.
    sorted_indices = sorted(
        range(len(dataset)),
        key=lambda index: dataset.entries[index].end - dataset.entries[index].start,
    )
    eval_dataset = Subset(dataset, sorted_indices)
    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(
        f"[vq-parity] dataset={len(dataset)} batches={len(loader)} "
        f"official_epoch={official_epoch} "
        f"official_params={sum(p.numel() for p in official_model.parameters())} "
        f"ours_params={sum(p.numel() for p in ours_model.parameters())}",
        flush=True,
    )

    repeats: list[dict[str, object]] = []
    for repeat in range(args.repeats):
        accumulator = ProtocolAccumulator()
        for batch_index, batch in enumerate(loader):
            if args.max_clips > 0 and accumulator.num_pairs + args.batch_size > args.max_clips:
                break
            raw = batch.x1.to(device)
            lengths = batch.lengths.to(device)
            mask = batch.mask.to(device)
            tokens = caption_token_batch(caption_tokens, batch.clip_ids, batch.texts)
            with torch.no_grad():
                official_input = evaluator.normalize(raw, lengths)
                official_pred_norm, _, _ = official_model(official_input)
                official_raw = evaluator.denormalize(official_pred_norm, lengths)

                ours_input = normalize_motion(raw, mask, ours_normalizer)
                ours_raw = ours_normalizer.inverse(ours_model(ours_input, mask=None).recon)

                text_emb, real_emb = evaluator.encode_co_embeddings_from_tokens(
                    raw, lengths, tokens
                )
                _, official_emb = evaluator.encode_co_embeddings_from_tokens(
                    official_raw, lengths, tokens
                )
                _, ours_emb = evaluator.encode_co_embeddings_from_tokens(
                    ours_raw, lengths, tokens
                )
            accumulator.add(
                text_emb.cpu().numpy(),
                real_emb.cpu().numpy(),
                official_emb.cpu().numpy(),
                ours_emb.cpu().numpy(),
                raw,
                official_raw,
                ours_raw,
                mask,
            )
            if batch_index == 0 or (batch_index + 1) % 25 == 0:
                print(
                    f"[vq-parity] repeat={repeat + 1}/{args.repeats} "
                    f"batch={batch_index + 1}/{len(loader)} pairs={accumulator.num_pairs}",
                    flush=True,
                )
        result = accumulator.finish(args.seed + repeat * 100)
        repeats.append(result)
        print(f"[vq-parity] repeat={repeat + 1} {json.dumps(result)}", flush=True)

    output = {
        "_meta": {
            "ours_checkpoint": args.ours_checkpoint,
            "official_repo": args.official_repo,
            "official_checkpoint": args.official_checkpoint,
            "official_opt": args.official_opt,
            "official_epoch": official_epoch,
            "official_num_quantizers": official_opt.num_quantizers,
            "split": args.split,
            "batch_size": args.batch_size,
            "drop_last": True,
            "loader_shuffle": True,
            "repeats": args.repeats,
            "seed": args.seed,
            "evaluator_checkpoints_dir": args.evaluator_checkpoints_dir,
            "evaluator_normalization_name": args.evaluator_normalization_name,
        },
        "repeats": repeats,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"[vq-parity] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
