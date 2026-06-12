"""Compute train-split mean/std of the 67-D essential feature for z-normalization.

MARDM standardizes the essential dims (no manifold normalization), so both the
AutoEncoder and the generation branch need a fixed per-dim mean/std computed
once over the training split. Run this before `scripts/train_mardm_ae.py`:

    python scripts/compute_mardm_stats.py \\
        --data-root external/data/humanml3d_packed \\
        --out external/data/mardm_essential_stats.pt

The output is a dict {"mean": (67,), "std": (67,)} loaded by the train configs
via `stats_path`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mardm.data import EssentialDataset
from mardm.representation import compute_essential_stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True, help="packed dataset dir (contains humanml3d.zip)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, required=True, help="output .pt path for {mean, std}")
    ap.add_argument("--max-clips", type=int, default=None, help="cap clips (debug); default = all")
    args = ap.parse_args()

    # Raw (unnormalized) essential features, no windowing, no mirror.
    ds = EssentialDataset(
        root=args.data_root, split=args.split,
        mean=None, std=None, window_size=None, mirror_augment=False,
    )
    print(f"[stats] computing essential mean/std over {len(ds)} clips (split={args.split}) ...", flush=True)
    mean, std = compute_essential_stats(ds, max_clips=args.max_clips)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mean": mean, "std": std}, args.out)
    print(f"[stats] saved (67,) mean/std → {args.out}")
    print(f"[stats] mean[:4]={[round(v, 4) for v in mean[:4].tolist()]} "
          f"std[:4]={[round(v, 4) for v in std[:4].tolist()]}")


if __name__ == "__main__":
    main()
