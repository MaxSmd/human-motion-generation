"""Break down real-motion retrieval for the paper-style HumanML3D view."""

from __future__ import annotations

import argparse
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
from shared.eval.metrics import mm_distance, r_precision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--canonical-h3d-dir", required=True)
    parser.add_argument("--humanml3d-texts-zip", required=True)
    parser.add_argument("--humanml3d-split-dir", required=True)
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


def entry_groups(clip_id: str) -> tuple[str, ...]:
    base_id = clip_id.split(":segment", 1)[0]
    source = "mirrored" if base_id.startswith("M") else "original"
    span = "segment" if ":segment" in clip_id else "whole"
    return ("all", source, span, f"{source}_{span}")


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
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
        drop_last=False,
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
        motion_embs.append(encode_motion(evaluator, batch.x1[:take], lengths))
        text_np, missing = encode_text_batch(evaluator, texts, clip_ids, caption_tokens)
        text_embs.append(text_np)
        token_fallbacks += missing
        all_lengths.append(lengths.cpu().numpy())
        all_clip_ids.extend(clip_ids)
        seen += take

    text = np.concatenate(text_embs, axis=0)
    motion = np.concatenate(motion_embs, axis=0)
    lengths_np = np.concatenate(all_lengths, axis=0)
    groups: dict[str, list[int]] = {}
    for index, clip_id in enumerate(all_clip_ids):
        for group in entry_groups(clip_id):
            groups.setdefault(group, []).append(index)

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
    print(f"[paper-real-groups] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
