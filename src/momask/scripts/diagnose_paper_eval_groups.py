"""Break down real-motion retrieval for the paper-style HumanML3D view."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from momask.scripts.evaluate_momask import encode_motion, encode_text_batch, load_caption_tokens
from shared.data import CanonicalHumanML3DText2MotionDataset, collate
from shared.eval.guo_evaluator import RealGuoEvaluator
from shared.eval.metrics import mm_distance, r_precision, r_precision_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--canonical-h3d-dir", required=True)
    parser.add_argument("--humanml3d-texts-zip", required=True)
    parser.add_argument("--humanml3d-split-dir", required=True)
    parser.add_argument(
        "--humanml3d-index-csv",
        default="",
        help="HumanML3D index.csv; defaults to the parent of the split directory",
    )
    parser.add_argument("--text-to-motion-repo", default="external/text-to-motion")
    parser.add_argument("--humanml3d-repo", default="external/HumanML3D")
    parser.add_argument(
        "--evaluator-checkpoints-dir",
        default="external/text-to-motion/checkpoints/momask_official",
    )
    parser.add_argument("--evaluator-normalization-name", default="Comp_v6_KLD005")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--max-seq-len", type=int, default=196)
    parser.add_argument("--min-seq-len", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-clips", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_index_metadata(
    path: Path,
) -> tuple[dict[str, str], dict[str, int]]:
    """Map canonical HumanML3D ids to AMASS source and expected feature length."""
    source_by_id: dict[str, str] = {}
    expected_length_by_id: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            clip_id = Path(row["new_name"]).stem
            source_parts = Path(row["source_path"].replace("\\", "/")).parts
            try:
                source_index = source_parts.index("pose_data") + 1
                source = source_parts[source_index]
            except (ValueError, IndexError):
                source = source_parts[0] if source_parts else "unknown"
            start = int(row["start_frame"])
            end = int(row["end_frame"])
            source_by_id[clip_id] = source
            if end > start:
                # HumanML3D process_file emits one fewer feature frame because
                # root/local velocities are defined between adjacent frames.
                expected_length_by_id[clip_id] = end - start - 1
    return source_by_id, expected_length_by_id


def base_motion_id(clip_id: str) -> str:
    base_id = clip_id.split(":segment", 1)[0]
    return base_id[1:] if base_id.startswith("M") else base_id


def entry_groups(clip_id: str, source_by_id: dict[str, str]) -> tuple[str, ...]:
    base_id = clip_id.split(":segment", 1)[0]
    source = "mirrored" if base_id.startswith("M") else "original"
    span = "segment" if ":segment" in clip_id else "whole"
    amass_source = source_by_id.get(base_motion_id(clip_id), "unknown")
    return ("all", source, span, f"{source}_{span}", f"source:{amass_source}")


def group_metrics(
    text: np.ndarray,
    motion: np.ndarray,
    lengths: np.ndarray,
    indices: list[int],
    seed: int,
) -> dict[str, object]:
    idx = np.asarray(indices, dtype=np.int64)
    return {
        "num_pairs": int(idx.size),
        "r_precision": r_precision(
            text[idx], motion[idx], top_k=3, rng=np.random.default_rng(seed)
        ).tolist(),
        "mm_dist": float(mm_distance(text[idx], motion[idx])),
        "length_frames": {
            "min": int(lengths[idx].min()),
            "mean": float(lengths[idx].mean()),
            "median": float(np.median(lengths[idx])),
            "max": int(lengths[idx].max()),
        },
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    index_csv = (
        Path(args.humanml3d_index_csv)
        if args.humanml3d_index_csv
        else Path(args.humanml3d_split_dir).parent / "index.csv"
    )
    if not index_csv.is_file():
        raise FileNotFoundError(f"HumanML3D index.csv not found: {index_csv}")
    source_by_id, expected_length_by_id = load_index_metadata(index_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CanonicalHumanML3DText2MotionDataset(
        root=args.data_root,
        canonical_dir=args.canonical_h3d_dir,
        texts_zip=args.humanml3d_texts_zip,
        split=args.split,
        max_seq_len=args.max_seq_len,
        min_seq_len=args.min_seq_len,
        unit_length=4,
        split_dir=args.humanml3d_split_dir,
        subset_n=0,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        drop_last=False,
        generator=torch.Generator().manual_seed(args.seed),
    )
    evaluator = RealGuoEvaluator(
        text_to_motion_repo=args.text_to_motion_repo,
        humanml3d_repo=args.humanml3d_repo,
        device=device,
        checkpoints_dir=args.evaluator_checkpoints_dir,
        normalization_name=args.evaluator_normalization_name,
    )
    caption_tokens = load_caption_tokens(args.humanml3d_texts_zip)

    text_embs: list[np.ndarray] = []
    motion_embs: list[np.ndarray] = []
    all_lengths: list[np.ndarray] = []
    all_clip_ids: list[str] = []
    token_fallbacks = 0
    protocol_pairs = 0
    separate_r_sum = np.zeros(3, dtype=np.float64)
    joint_r_sum = np.zeros(3, dtype=np.float64)
    separate_mm_sum = 0.0
    joint_mm_sum = 0.0
    seen = 0

    for batch in tqdm(loader, desc="paper real retrieval groups"):
        take = batch.x1.shape[0]
        if args.max_clips > 0:
            take = min(take, args.max_clips - seen)
            if take <= 0:
                break
        texts = batch.texts[:take]
        clip_ids = batch.clip_ids[:take]
        lengths = batch.lengths[:take]
        motion_np = encode_motion(evaluator, batch.x1[:take], lengths)
        motion_embs.append(motion_np)
        text_np, missing = encode_text_batch(evaluator, texts, clip_ids, caption_tokens)
        text_embs.append(text_np)
        token_fallbacks += missing

        # Official MoMask computes retrieval inside each DataLoader batch with
        # EvaluatorModelWrapper.get_co_embeddings. Compare that exact route to
        # our independently encoded embeddings using the very same pairs.
        # Ignore the final partial batch, matching the official drop_last=True.
        if take == args.batch_size:
            lookup_ids = [clip_id.split(":segment", 1)[0] for clip_id in clip_ids]
            tokens = [
                caption_tokens.get(clip_id, {}).get(text)
                for clip_id, text in zip(lookup_ids, texts)
            ]
            if any(token is None for token in tokens):
                missing_pairs = sum(token is None for token in tokens)
                raise RuntimeError(
                    "official joint evaluator comparison requires VIP tokens for "
                    f"every pair; missing {missing_pairs}/{take} in one batch"
                )
            joint_text, joint_motion = evaluator.encode_co_embeddings_from_tokens(
                batch.x1[:take],
                lengths,
                [token for token in tokens if token is not None],
            )
            joint_text_np = joint_text.cpu().numpy()
            joint_motion_np = joint_motion.cpu().numpy()
            separate_r_sum += r_precision_batch(text_np, motion_np, top_k=3) * take
            joint_r_sum += r_precision_batch(
                joint_text_np, joint_motion_np, top_k=3
            ) * take
            separate_mm_sum += float(np.linalg.norm(text_np - motion_np, axis=1).sum())
            joint_mm_sum += float(
                np.linalg.norm(joint_text_np - joint_motion_np, axis=1).sum()
            )
            protocol_pairs += take

        all_lengths.append(lengths.cpu().numpy())
        all_clip_ids.extend(clip_ids)
        seen += take

    text = np.concatenate(text_embs, axis=0)
    motion = np.concatenate(motion_embs, axis=0)
    lengths_np = np.concatenate(all_lengths, axis=0)
    groups: dict[str, list[int]] = {}
    for index, clip_id in enumerate(all_clip_ids):
        for group in entry_groups(clip_id, source_by_id):
            groups.setdefault(group, []).append(index)

    checked_lengths: dict[str, tuple[int, int]] = {}
    for entry in dataset.entries:
        clip_id = base_motion_id(entry.clip_id)
        expected = expected_length_by_id.get(clip_id)
        if expected is None or clip_id in checked_lengths:
            continue
        actual = int(np.load(entry.motion_path, mmap_mode="r").shape[0])
        checked_lengths[clip_id] = (actual, expected)
    length_mismatches = {
        clip_id: values
        for clip_id, values in checked_lengths.items()
        if values[0] != values[1]
    }

    if protocol_pairs == 0:
        raise RuntimeError("no full retrieval batch was available for protocol comparison")
    separate_protocol_r = separate_r_sum / protocol_pairs
    joint_protocol_r = joint_r_sum / protocol_pairs
    separate_protocol_mm = separate_mm_sum / protocol_pairs
    joint_protocol_mm = joint_mm_sum / protocol_pairs

    results = {
        "_meta": {
            "split": args.split,
            "num_pairs": int(text.shape[0]),
            "max_seq_len": args.max_seq_len,
            "min_seq_len": args.min_seq_len,
            "seed": args.seed,
            "evaluator_checkpoints_dir": str(evaluator.checkpoints_dir),
            "evaluator_normalization_name": evaluator.normalization_name,
            "evaluator_normalization_path": str(evaluator.normalization_path),
            "evaluator_protocol_version": evaluator.protocol_version,
            "vip_token_fallbacks": int(token_fallbacks),
            "source_ids": dataset.num_source_ids,
            "source_mirror_ids": dataset.num_source_mirror_ids,
            "retained_clips": dataset.num_clips,
            "retained_mirror_clips": dataset.num_mirror_clips,
            "missing_motion": dataset.num_missing_motion,
            "missing_text": dataset.num_missing_text,
            "humanml3d_index_csv": str(index_csv),
            "index_ids": len(source_by_id),
            "lengths_checked": len(checked_lengths),
            "length_mismatches": len(length_mismatches),
            "length_mismatch_examples": [
                {
                    "clip_id": clip_id,
                    "actual": actual,
                    "expected": expected,
                    "source": source_by_id.get(clip_id, "unknown"),
                }
                for clip_id, (actual, expected) in list(length_mismatches.items())[:20]
            ],
        },
        "protocol_checks": {
            "num_pairs": protocol_pairs,
            "batch_size": args.batch_size,
            "loader_shuffle": True,
            "drop_last": True,
            "separate_encoders": {
                "r_precision": separate_protocol_r.tolist(),
                "mm_dist": separate_protocol_mm,
            },
            "joint_get_co_embeddings": {
                "r_precision": joint_protocol_r.tolist(),
                "mm_dist": joint_protocol_mm,
            },
            "absolute_delta": {
                "r_precision": np.abs(
                    separate_protocol_r - joint_protocol_r
                ).tolist(),
                "mm_dist": abs(separate_protocol_mm - joint_protocol_mm),
            },
        },
        "groups": {
            name: group_metrics(text, motion, lengths_np, indices, args.seed)
            for name, indices in groups.items()
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(
        f"[paper-real-groups] pairs={text.shape[0]} vip_fallbacks={token_fallbacks}",
        flush=True,
    )
    for label, metrics in (
        ("separate", results["protocol_checks"]["separate_encoders"]),
        ("joint", results["protocol_checks"]["joint_get_co_embeddings"]),
    ):
        r1, r2, r3 = metrics["r_precision"]
        print(
            f"[paper-real-protocol] {label} n={protocol_pairs} "
            f"R@1/2/3={r1:.4f}/{r2:.4f}/{r3:.4f} "
            f"MM={metrics['mm_dist']:.4f}",
            flush=True,
        )
    delta = results["protocol_checks"]["absolute_delta"]
    dr1, dr2, dr3 = delta["r_precision"]
    print(
        f"[paper-real-protocol] absolute_delta "
        f"R@1/2/3={dr1:.6f}/{dr2:.6f}/{dr3:.6f} "
        f"MM={delta['mm_dist']:.6f}",
        flush=True,
    )
    print(
        f"[paper-real-integrity] index_ids={len(source_by_id)} "
        f"lengths_checked={len(checked_lengths)} "
        f"length_mismatches={len(length_mismatches)}",
        flush=True,
    )
    for name in (
        "all",
        "original",
        "mirrored",
        "whole",
        "segment",
        "original_whole",
        "original_segment",
        "mirrored_whole",
        "mirrored_segment",
    ):
        metrics = results["groups"].get(name)
        if metrics is None:
            continue
        r1, r2, r3 = metrics["r_precision"]
        print(
            f"[paper-real-groups] {name} n={metrics['num_pairs']} "
            f"R@1/2/3={r1:.4f}/{r2:.4f}/{r3:.4f} "
            f"MM={metrics['mm_dist']:.4f}",
            flush=True,
        )
    source_metrics = sorted(
        (
            (name.removeprefix("source:"), metrics)
            for name, metrics in results["groups"].items()
            if name.startswith("source:") and metrics["num_pairs"] >= args.batch_size
        ),
        key=lambda item: item[1]["num_pairs"],
        reverse=True,
    )
    for source, metrics in source_metrics:
        r1, r2, r3 = metrics["r_precision"]
        print(
            f"[paper-real-source] {source} n={metrics['num_pairs']} "
            f"R@1/2/3={r1:.4f}/{r2:.4f}/{r3:.4f} "
            f"MM={metrics['mm_dist']:.4f}",
            flush=True,
        )
    print(f"[paper-real-groups] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
