"""Test the essential->263-D IK bridge in isolation (no AE, no gen model).

Pipeline per clip:
  direct path:   (translation, quats) -> tplusr_to_h3d_features_with_quats -> 263-D
  bridge path:   (translation, quats) -> essential 67-D -> essential_to_h3d -> 263-D

If the bridge faithfully recovers 263-D, FID(direct, bridge) through the Guo
evaluator should be very small (~0). If it's high, the IK is destroying
information that the Guo evaluator picks up on — and no gen model can fix it.

This is the "is the bridge broken at all" test, separate from "is the model
good." If this comes back bad, paper-S retraining will not help.

Usage (inside the container, on the cluster):
    python scripts/diagnose_ik_roundtrip.py \
        --data-root external/data/humanml3d_packed \
        --max-clips 500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from mardm.representation import ESSENTIAL_DIM, encode_essential, essential_to_h3d
from rmg.data.humanml3d import HumanML3DDataset
from rmg.eval import RealGuoEvaluator, fid
from rmg.representation import (
    Skeleton,
    make_continuous,
    normalize_quaternions,
    tplusr_to_h3d_features_with_quats,
)


def _pad_stack(feats):
    L_max = max(f.shape[0] for f in feats)
    out = torch.zeros(len(feats), L_max, feats[0].shape[1])
    lens = torch.zeros(len(feats), dtype=torch.long)
    for i, f in enumerate(feats):
        out[i, : f.shape[0]] = f
        lens[i] = f.shape[0]
    return out, lens


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--max-clips", type=int, default=500)
    p.add_argument("--split", default="test")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[diag] device: {device}")

    ds = HumanML3DDataset(root=args.data_root, split=args.split, mirror_augment=False, min_seq_len=10)
    n = min(args.max_clips, len(ds))
    print(f"[diag] iterating {n} {args.split} clips ...")

    skeleton = ds._skeleton
    assert skeleton is not None, "Skeleton offsets missing from dataset"

    print("[diag] loading Guo evaluator ...")
    evaluator = RealGuoEvaluator(device=device)

    direct_feats, bridge_feats = [], []
    for i in tqdm(range(n)):
        clip_id = ds.clip_ids[i]
        zf = ds._open_zip()
        import io
        blob = torch.load(io.BytesIO(zf.read(f"{clip_id}.pt")), weights_only=False)
        translation = blob["translation"]
        quats = make_continuous(normalize_quaternions(blob["quats"]), time_dim=0)

        direct = tplusr_to_h3d_features_with_quats(translation, quats, skeleton)
        ess = encode_essential(translation, quats, skeleton)
        bridge = essential_to_h3d(ess, skeleton)

        # Bridge is one frame shorter; trim direct to match.
        L = min(direct.shape[0], bridge.shape[0])
        direct_feats.append(direct[:L].float())
        bridge_feats.append(bridge[:L].float())

    direct_padded, direct_lens = _pad_stack(direct_feats)
    bridge_padded, bridge_lens = _pad_stack(bridge_feats)

    print("[diag] encoding both through the Guo evaluator ...")
    with torch.no_grad():
        direct_emb = evaluator.encode_motion(direct_padded, direct_lens).cpu().numpy()
        bridge_emb = evaluator.encode_motion(bridge_padded, bridge_lens).cpu().numpy()

    bridge_fid = fid(direct_emb, bridge_emb)
    print(f"\n[diag] FID(direct, bridge) = {bridge_fid:.4f}")
    print(
        "[diag] verdict: "
        + (
            "bridge preserves info (FID < 0.5)"
            if bridge_fid < 0.5
            else "BRIDGE IS LOSSY — paper-S retrain won't fix this"
            if bridge_fid > 5.0
            else "moderate loss in the bridge — investigate"
        )
    )


if __name__ == "__main__":
    main()
