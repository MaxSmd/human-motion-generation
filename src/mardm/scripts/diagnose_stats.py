"""Sanity-check the essential mean/std file against a freshly recomputed copy.

If the stats on disk drift far from what the current encode pipeline produces,
the gen model was trained with one normalization and the eval uses another,
which scrambles latents and tanks the metrics silently.

Usage (inside the container, on the cluster):
    python -m mardm.scripts.diagnose_stats \
        --data-root external/data/humanml3d_packed \
        --stats ~/rmg-runs/mardm_essential_stats.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mardm.data import EssentialDataset
from mardm.representation import compute_essential_stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--stats", required=True, help="Existing stats file to compare against")
    p.add_argument("--max-clips", type=int, default=2000, help="Subset of train clips to recompute on")
    args = p.parse_args()

    stats_path = Path(args.stats).expanduser()
    blob = torch.load(stats_path, weights_only=True)
    if isinstance(blob, dict):
        old_mean, old_std = blob["mean"], blob["std"]
    else:
        old_mean, old_std = blob

    print(f"[diag] recomputing stats over {args.max_clips} train clips ...")
    raw_ds = EssentialDataset(args.data_root, "train", min_seq_len=10)
    new_mean, new_std = compute_essential_stats(raw_ds, max_clips=args.max_clips)

    dm = (old_mean - new_mean).abs().max().item()
    ds = (old_std  - new_std).abs().max().item()

    print(f"[diag] mean abs delta: {dm:.6f}")
    print(f"[diag] std  abs delta: {ds:.6f}")
    print(f"[diag] old mean[:6]: {old_mean[:6].tolist()}")
    print(f"[diag] new mean[:6]: {new_mean[:6].tolist()}")
    print(f"[diag] old std[:6]:  {old_std[:6].tolist()}")
    print(f"[diag] new std[:6]:  {new_std[:6].tolist()}")

    verdict = "MATCH (no drift)" if max(dm, ds) < 0.01 else "DRIFT — stats inconsistent"
    print(f"\n[diag] verdict: {verdict}")


if __name__ == "__main__":
    main()
