"""Locate divergences between official and local MoMask RVQ evaluation."""

from __future__ import annotations

import argparse
import functools
import inspect
import json
import os
import random
import sys
import types
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy import linalg

from shared.data import CanonicalHumanML3DText2MotionDataset
from shared.eval.metrics import calculate_activation_statistics, r_precision_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--official-checkpoint", required=True)
    parser.add_argument("--official-opt", required=True)
    parser.add_argument("--official-dataset-opt", required=True)
    parser.add_argument("--text-to-motion-repo", required=True)
    parser.add_argument("--evaluator-checkpoints-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--canonical-h3d-dir", required=True)
    parser.add_argument("--texts-zip", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def install_torch_load_compat() -> str:
    original_load = torch.load
    try:
        has_weights_only = "weights_only" in inspect.signature(original_load).parameters
    except (TypeError, ValueError):
        has_weights_only = False
    if not has_weights_only:
        return "legacy-default"

    @functools.wraps(original_load)
    def legacy_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = legacy_load
    return "explicit-weights-only-false"


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
    return SimpleNamespace(**values)


def scipy_fid(real: np.ndarray, predicted: np.ndarray, eps: float = 1e-6) -> float:
    mu1, sigma1 = calculate_activation_statistics(real)
    mu2, sigma2 = calculate_activation_statistics(predicted)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"FID covariance product has imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


class EmbeddingAccumulator:
    def __init__(self) -> None:
        self.text: dict[str, list[np.ndarray]] = {"official": [], "local": []}
        self.motion: dict[str, list[np.ndarray]] = {
            "official_real": [],
            "official_direct": [],
            "official_zero_pad": [],
            "local_real": [],
            "local_direct": [],
            "local_zero_pad": [],
        }
        self.r_sum = {name: np.zeros(3, dtype=np.float64) for name in self.motion}
        self.mm_sum = {name: 0.0 for name in self.motion}
        self.num_pairs = 0

    def add(
        self,
        official_text: np.ndarray,
        local_text: np.ndarray,
        motions: dict[str, np.ndarray],
    ) -> None:
        batch_size = official_text.shape[0]
        self.text["official"].append(official_text)
        self.text["local"].append(local_text)
        for name, motion in motions.items():
            self.motion[name].append(motion)
            text = local_text if name.startswith("local_") else official_text
            self.r_sum[name] += r_precision_batch(text, motion, top_k=3) * batch_size
            self.mm_sum[name] += float(np.linalg.norm(text - motion, axis=1).sum())
        self.num_pairs += batch_size

    def finish(self) -> dict[str, object]:
        text = {name: np.concatenate(parts) for name, parts in self.text.items()}
        motion = {name: np.concatenate(parts) for name, parts in self.motion.items()}
        reference = motion["official_real"]
        metrics: dict[str, object] = {}
        for name, features in motion.items():
            metrics[name] = {
                "fid_vs_official_real": scipy_fid(reference, features),
                "r_precision": (self.r_sum[name] / self.num_pairs).tolist(),
                "mm_dist": self.mm_sum[name] / self.num_pairs,
            }
        return {
            "num_pairs": self.num_pairs,
            "metrics": metrics,
            "embedding_deltas": {
                "text_official_vs_local": array_delta(text["official"], text["local"]),
                "real_official_vs_local": array_delta(
                    motion["official_real"], motion["local_real"]
                ),
                "direct_official_vs_local": array_delta(
                    motion["official_direct"], motion["local_direct"]
                ),
                "official_direct_vs_zero_pad": array_delta(
                    motion["official_direct"], motion["official_zero_pad"]
                ),
                "local_direct_vs_zero_pad": array_delta(
                    motion["local_direct"], motion["local_zero_pad"]
                ),
            },
        }


def array_delta(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    denom = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    cosine = np.sum(left * right, axis=-1) / np.clip(denom, 1e-12, None)
    return {
        "mae": float(delta.mean()),
        "max_abs": float(delta.max()),
        "mean_cosine": float(cosine.mean()),
    }


def module_delta(official: torch.nn.Module, local: torch.nn.Module) -> dict[str, object]:
    official_state = official.state_dict()
    local_state = local.state_dict()
    common = sorted(
        key
        for key in official_state.keys() & local_state.keys()
        if official_state[key].shape == local_state[key].shape
    )
    abs_sum = 0.0
    numel = 0
    max_abs = 0.0
    for key in common:
        delta = (official_state[key].detach().cpu() - local_state[key].detach().cpu()).abs()
        abs_sum += float(delta.sum())
        numel += delta.numel()
        max_abs = max(max_abs, float(delta.max()))
    return {
        "official_keys": len(official_state),
        "local_keys": len(local_state),
        "common_shape_matched_keys": len(common),
        "compared_numel": numel,
        "mae": abs_sum / max(numel, 1),
        "max_abs": max_abs,
        "official_only": sorted(official_state.keys() - local_state.keys()),
        "local_only": sorted(local_state.keys() - official_state.keys()),
    }


def dataset_signature(dataset) -> Counter[tuple[int, tuple[str, ...]]]:
    signature: Counter[tuple[int, tuple[str, ...]]] = Counter()
    for name in dataset.name_list:
        item = dataset.data_dict[name]
        captions = tuple(sorted(text["caption"] for text in item["text"]))
        signature[(int(item["length"]), captions)] += 1
    return signature


def local_dataset_signature(dataset: CanonicalHumanML3DText2MotionDataset) -> Counter:
    return Counter(
        (entry.end - entry.start, tuple(sorted(entry.captions))) for entry in dataset.entries
    )


def compare_datasets(official_dataset, local_dataset) -> dict[str, object]:
    official_signature = dataset_signature(official_dataset)
    local_signature = local_dataset_signature(local_dataset)
    missing_local = official_signature - local_signature
    extra_local = local_signature - official_signature
    return {
        "official_entries": len(official_dataset),
        "local_entries": len(local_dataset),
        "official_source_ids": len(official_dataset.name_list),
        "local_source_ids": local_dataset.num_source_ids,
        "signature_equal": official_signature == local_signature,
        "missing_local_entries": int(sum(missing_local.values())),
        "extra_local_entries": int(sum(extra_local.values())),
        "official_length_min": int(official_dataset.length_arr.min()),
        "official_length_max": int(official_dataset.length_arr.max()),
        "local_length_min": min(entry.end - entry.start for entry in local_dataset.entries),
        "local_length_max": max(entry.end - entry.start for entry in local_dataset.entries),
    }


def build_local_wrapper(checkpoint_root: Path, device: torch.device):
    opt = SimpleNamespace(
        checkpoints_dir=str(checkpoint_root),
        dataset_name="t2m",
        device=device,
        dim_pose=263,
        dim_word=300,
        dim_pos_ohot=15,
        dim_motion_hidden=1024,
        dim_text_hidden=512,
        dim_coemb_hidden=512,
        dim_movement_enc_hidden=512,
        dim_movement_latent=512,
        max_motion_length=196,
        max_text_len=20,
        unit_length=4,
    )
    from networks.evaluator_wrapper import EvaluatorModelWrapper

    return EvaluatorModelWrapper(opt)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    torch_load_mode = install_torch_load_compat()
    if "float" not in np.__dict__:
        np.float = float  # type: ignore[attr-defined]
    sys.modules["clip"] = types.ModuleType("clip")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("the parity diagnostic requires a CUDA compute node")

    official_repo = Path(args.official_repo).resolve()
    sys.path.insert(0, str(official_repo))
    from models.t2m_eval_wrapper import EvaluatorModelWrapper as OfficialEvaluator
    from models.vq.model import RVQVAE
    from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
    from utils.get_opt import get_opt

    old_cwd = Path.cwd()
    os.chdir(official_repo)
    try:
        dataset_opt = get_opt(args.official_dataset_opt, device)
        official_wrapper = OfficialEvaluator(dataset_opt)
        loader, official_dataset = get_dataset_motion_loader(
            args.official_dataset_opt, args.batch_size, "test", device
        )
    finally:
        os.chdir(old_cwd)

    text_to_motion_repo = Path(args.text_to_motion_repo).resolve()
    sys.path.insert(0, str(text_to_motion_repo))
    local_wrapper = build_local_wrapper(Path(args.evaluator_checkpoints_dir), device)

    vq_opt = parse_upstream_opt(Path(args.official_opt))
    official_vq = RVQVAE(
        vq_opt,
        263,
        vq_opt.nb_code,
        vq_opt.code_dim,
        vq_opt.code_dim,
        vq_opt.down_t,
        vq_opt.stride_t,
        vq_opt.width,
        vq_opt.depth,
        vq_opt.dilation_growth_rate,
        vq_opt.vq_act,
        None if vq_opt.vq_norm == "None" else vq_opt.vq_norm,
    )
    checkpoint = torch.load(args.official_checkpoint, map_location="cpu")
    model_key = "vq_model" if "vq_model" in checkpoint else "net"
    official_vq.load_state_dict(checkpoint[model_key], strict=True)
    official_vq.to(device).eval()

    local_dataset = CanonicalHumanML3DText2MotionDataset(
        root=args.data_root,
        canonical_dir=args.canonical_h3d_dir,
        texts_zip=args.texts_zip,
        split="test",
        max_seq_len=196,
        min_seq_len=40,
        unit_length=4,
        split_dir=args.split_dir,
        preload_motions=False,
    )
    dataset_comparison = compare_datasets(official_dataset, local_dataset)

    wrapper_state = {
        "movement_encoder": module_delta(
            official_wrapper.movement_encoder, local_wrapper.movement_encoder
        ),
        "motion_encoder": module_delta(
            official_wrapper.motion_encoder, local_wrapper.motion_encoder
        ),
        "text_encoder": module_delta(
            official_wrapper.text_encoder, local_wrapper.text_encoder
        ),
    }

    accumulator = EmbeddingAccumulator()
    pad_abs_sum = 0.0
    pad_values = 0
    valid_abs_sum = 0.0
    valid_values = 0
    for batch_index, batch in enumerate(loader):
        if args.max_batches > 0 and batch_index >= args.max_batches:
            break
        word_embeddings, pos_one_hots, _, sent_len, motion, lengths, _ = batch
        motion = motion.to(device).float()
        lengths = lengths.to(device).long()
        valid = torch.arange(motion.shape[1], device=device)[None, :] < lengths[:, None]
        with torch.no_grad():
            pred_direct, _, _ = official_vq(motion)
            pred_zero = pred_direct.masked_fill(~valid.unsqueeze(-1), 0.0)

            official_text, official_real = official_wrapper.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len, motion, lengths
            )
            _, official_direct = official_wrapper.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len, pred_direct, lengths
            )
            _, official_zero = official_wrapper.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len, pred_zero, lengths
            )
            local_text, local_real = local_wrapper.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len, motion, lengths
            )
            _, local_direct = local_wrapper.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len, pred_direct, lengths
            )
            _, local_zero = local_wrapper.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len, pred_zero, lengths
            )

        invalid = ~valid.unsqueeze(-1).expand_as(pred_direct)
        valid_full = valid.unsqueeze(-1).expand_as(pred_direct)
        pad_abs_sum += float(pred_direct[invalid].abs().sum())
        pad_values += int(invalid.sum())
        valid_abs_sum += float(pred_direct[valid_full].abs().sum())
        valid_values += int(valid_full.sum())
        accumulator.add(
            official_text.cpu().numpy(),
            local_text.cpu().numpy(),
            {
                "official_real": official_real.cpu().numpy(),
                "official_direct": official_direct.cpu().numpy(),
                "official_zero_pad": official_zero.cpu().numpy(),
                "local_real": local_real.cpu().numpy(),
                "local_direct": local_direct.cpu().numpy(),
                "local_zero_pad": local_zero.cpu().numpy(),
            },
        )
        if batch_index == 0 or (batch_index + 1) % 25 == 0:
            print(
                f"[official-eval-parity] batch={batch_index + 1}/{len(loader)} "
                f"pairs={accumulator.num_pairs}",
                flush=True,
            )

    result = accumulator.finish()
    result.update(
        {
            "_meta": {
                "official_checkpoint": args.official_checkpoint,
                "official_dataset_opt": args.official_dataset_opt,
                "batch_size": args.batch_size,
                "max_batches": args.max_batches,
                "seed": args.seed,
                "torch_load": torch_load_mode,
            },
            "dataset_comparison": dataset_comparison,
            "wrapper_state_deltas": wrapper_state,
            "reconstruction_padding": {
                "predicted_padding_mean_abs": pad_abs_sum / max(pad_values, 1),
                "predicted_valid_mean_abs": valid_abs_sum / max(valid_values, 1),
                "padding_values": pad_values,
            },
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[official-eval-parity] result={json.dumps(result)}", flush=True)
    print(f"[official-eval-parity] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
