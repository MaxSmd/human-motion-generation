"""Audit the HumanML3D population used for paper-style MoMask training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from shared.data import CanonicalHumanML3DText2MotionDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--canonical-h3d-dir", required=True)
    parser.add_argument("--humanml3d-texts-zip", required=True)
    parser.add_argument("--humanml3d-split-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--max-seq-len", type=int, default=196)
    parser.add_argument("--min-seq-len", type=int, default=40)
    parser.add_argument("--unit-length", type=int, default=4)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_packed_split(data_root: Path, split: str) -> list[str]:
    with open(data_root / "splits.json") as handle:
        splits = json.load(handle)
    return list(splits[split])


def summarize_lengths(lengths: list[int]) -> dict[str, float | int]:
    values = np.asarray(lengths, dtype=np.int64)
    return {
        "min": int(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p90": float(np.percentile(values, 90)),
        "max": int(values.max()),
    }


def audit_split(args: argparse.Namespace, split: str) -> dict[str, object]:
    data_root = Path(args.data_root)
    split_dir = Path(args.humanml3d_split_dir)
    official_ids = [
        line.strip()
        for line in (split_dir / f"{split}.txt").read_text().splitlines()
        if line.strip()
    ]
    packed_ids = read_packed_split(data_root, split)
    dataset = CanonicalHumanML3DText2MotionDataset(
        root=data_root,
        canonical_dir=args.canonical_h3d_dir,
        texts_zip=args.humanml3d_texts_zip,
        split=split,
        max_seq_len=args.max_seq_len,
        min_seq_len=args.min_seq_len,
        unit_length=args.unit_length,
        split_dir=split_dir,
        subset_n=0,
        preload_motions=False,
    )
    entry_lengths = [entry.end - entry.start for entry in dataset.entries]
    official_set = set(official_ids)
    packed_set = set(packed_ids)
    return {
        "official_split_file": str(split_dir / f"{split}.txt"),
        "official_ids": len(official_ids),
        "official_mirror_ids": sum(clip_id.startswith("M") for clip_id in official_ids),
        "packed_ids": len(packed_ids),
        "packed_mirror_ids": sum(clip_id.startswith("M") for clip_id in packed_ids),
        "official_only_ids": len(official_set - packed_set),
        "packed_only_ids": len(packed_set - official_set),
        "retained_clips": dataset.num_clips,
        "retained_mirror_clips": dataset.num_mirror_clips,
        "training_entries": len(dataset),
        "training_mirror_entries": sum(entry.clip_id.startswith("M") for entry in dataset.entries),
        "missing_motion": dataset.num_missing_motion,
        "missing_text": dataset.num_missing_text,
        "filtered_parent_length": dataset.num_filtered_motion_length,
        "entry_length_frames": summarize_lengths(entry_lengths),
    }


def main() -> None:
    args = parse_args()
    invalid = sorted(set(args.splits) - {"train", "val", "test"})
    if invalid:
        raise ValueError(f"invalid splits: {invalid}")
    report = {
        "_meta": {
            "data_root": str(Path(args.data_root).resolve()),
            "canonical_h3d_dir": str(Path(args.canonical_h3d_dir).resolve()),
            "humanml3d_texts_zip": str(Path(args.humanml3d_texts_zip).resolve()),
            "humanml3d_split_dir": str(Path(args.humanml3d_split_dir).resolve()),
            "max_seq_len": args.max_seq_len,
            "min_seq_len": args.min_seq_len,
            "unit_length": args.unit_length,
        },
        "splits": {split: audit_split(args, split) for split in args.splits},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print(f"[momask-data-audit] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
