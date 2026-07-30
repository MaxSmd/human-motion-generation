"""Diagnose Guo evaluator R-Precision on canonical HumanML3D features.

This bypasses MoMask/RMG/MARDM completely:
  1. read split clip ids from the packed dataset's splits.json,
  2. load official HumanML3D new_joint_vecs/<clip_id>.npy features,
  3. load HumanML3D texts.zip word/POS tokens,
  4. report real-motion R@1/R@2/R@3, MM-Dist, and Diversity.

If this real-only diagnostic is far below the paper ceiling, the evaluator/data
protocol is still wrong. If it is healthy, low generated R@ belongs to the model.
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from shared.eval import RealGuoEvaluator, diversity, mm_distance, r_precision


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="Packed dataset dir containing splits.json.")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--h3d-dir", required=True, help="Directory containing HumanML3D new_joint_vecs/*.npy.")
    p.add_argument("--texts-zip", required=True, help="Path to HumanML3D/HumanML3D/texts.zip.")
    p.add_argument("--text-to-motion-repo", default="external/text-to-motion")
    p.add_argument("--humanml3d-repo", default="external/HumanML3D")
    p.add_argument("--max-clips", type=int, default=512, help="-1 evaluates the full split.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-seq-len", type=int, default=196)
    p.add_argument("--diversity-times", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default=None)
    return p.parse_args()


def load_splits(data_root: str | Path, split: str) -> list[str]:
    path = Path(data_root) / "splits.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data[split])


def load_caption_tokens(texts_zip: str | Path) -> dict[str, list[tuple[str, list[str]]]]:
    out: dict[str, list[tuple[str, list[str]]]] = {}
    with zipfile.ZipFile(texts_zip) as zf:
        for name in zf.namelist():
            if not name.endswith(".txt"):
                continue
            cid = Path(name).stem
            rows: list[tuple[str, list[str]]] = []
            for line in zf.read(name).decode("utf-8").splitlines():
                parts = line.strip().split("#")
                if len(parts) < 2:
                    continue
                cap = parts[0].strip()
                toks = parts[1].strip().split()
                if cap and toks:
                    rows.append((cap, toks))
            if rows:
                out[cid] = rows
    return out


def index_motion_files(h3d_dir: str | Path) -> dict[str, Path]:
    root = Path(h3d_dir)
    if not root.exists():
        raise FileNotFoundError(f"HumanML3D new_joint_vecs dir not found: {root}")
    return {p.stem: p for p in root.rglob("*.npy")}


def id_candidates(cid: str) -> list[str]:
    candidates = [cid]
    if cid.startswith("M") and len(cid) > 1:
        candidates.append(cid[1:])
    return list(dict.fromkeys(candidates))


def pad_stack(feats: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    tmax = max(f.shape[0] for f in feats)
    out = torch.zeros(len(feats), tmax, 263)
    lengths = torch.zeros(len(feats), dtype=torch.long)
    for i, feat in enumerate(feats):
        out[i, : feat.shape[0]] = feat
        lengths[i] = feat.shape[0]
    return out, lengths


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    clip_ids = load_splits(args.data_root, args.split)
    caption_tokens = load_caption_tokens(args.texts_zip)
    motion_files = index_motion_files(args.h3d_dir)
    evaluator = RealGuoEvaluator(
        text_to_motion_repo=args.text_to_motion_repo,
        humanml3d_repo=args.humanml3d_repo,
        device=device,
    )

    motion_embs: list[np.ndarray] = []
    text_embs: list[np.ndarray] = []
    used_ids: list[str] = []
    missing_motion = 0
    missing_text = 0
    missing_examples: list[dict[str, str]] = []
    n_seen = 0

    batch_feats: list[torch.Tensor] = []
    batch_tokens: list[list[str]] = []

    def flush() -> None:
        if not batch_feats:
            return
        motion, lengths = pad_stack(batch_feats)
        motion_embs.append(evaluator.encode_motion(motion, lengths).cpu().numpy())
        text_embs.append(evaluator.encode_text_from_tokens(batch_tokens).cpu().numpy())
        batch_feats.clear()
        batch_tokens.clear()

    for cid in tqdm(clip_ids, desc="canonical Guo R@"):
        if args.max_clips > 0 and n_seen >= args.max_clips:
            break
        candidates = id_candidates(cid)
        motion_match = next((c for c in candidates if c in motion_files), None)
        text_match = next((c for c in candidates if c in caption_tokens), None)
        motion_path = motion_files[motion_match] if motion_match is not None else None
        text_rows = caption_tokens[text_match] if text_match is not None else None
        if motion_path is None:
            missing_motion += 1
            if len(missing_examples) < 10:
                missing_examples.append({"clip_id": cid, "reason": "missing_motion"})
            continue
        if not text_rows:
            missing_text += 1
            if len(missing_examples) < 10:
                missing_examples.append({"clip_id": cid, "reason": "missing_text"})
            continue
        arr = np.load(motion_path).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 263:
            raise ValueError(f"{motion_path} must have shape (T, 263), got {arr.shape}")
        arr = arr[: args.max_seq_len]
        if len(arr) < 1:
            continue
        _, toks = random.choice(text_rows)
        batch_feats.append(torch.from_numpy(arr))
        batch_tokens.append(toks)
        used_ids.append(motion_match or cid)
        n_seen += 1
        if len(batch_feats) >= args.batch_size:
            flush()
    flush()

    if not motion_embs:
        debug = {
            "split": args.split,
            "num_split_ids": len(clip_ids),
            "num_motion_files": len(motion_files),
            "num_text_ids": len(caption_tokens),
            "first_split_ids": clip_ids[:10],
            "first_motion_ids": sorted(motion_files)[:10],
            "first_text_ids": sorted(caption_tokens)[:10],
            "missing_motion": missing_motion,
            "missing_text": missing_text,
            "missing_examples": missing_examples,
        }
        print(json.dumps({"_debug_no_pairs": debug}, indent=2), flush=True)
        raise RuntimeError("no canonical motion/text pairs were found; see _debug_no_pairs above")

    motion_np = np.concatenate(motion_embs, axis=0)
    text_np = np.concatenate(text_embs, axis=0)
    results = {
        "_meta": {
            "split": args.split,
            "data_root": str(args.data_root),
            "h3d_dir": str(args.h3d_dir),
            "texts_zip": str(args.texts_zip),
            "max_clips": args.max_clips,
            "num_clips": int(motion_np.shape[0]),
            "max_seq_len": args.max_seq_len,
            "missing_motion": missing_motion,
            "missing_text": missing_text,
            "num_motion_files": len(motion_files),
            "num_text_ids": len(caption_tokens),
            "used_ids_preview": used_ids[:10],
        },
        "canonical_real": {
            "r_precision": r_precision(text_np, motion_np, top_k=3, rng=rng).tolist(),
            "mm_dist": float(mm_distance(text_np, motion_np)),
            "diversity": float(diversity(motion_np, diversity_times=args.diversity_times, rng=rng)),
        },
    }
    print(json.dumps(results, indent=2), flush=True)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[diagnose-guo] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
