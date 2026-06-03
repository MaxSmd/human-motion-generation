"""Find clips whose 67-D essential features contain non-finite values.

Iterates the train split, encodes each clip via the shared rmg representation
pipeline (the same path EssentialDataset uses), and reports any clip that
produces NaN/Inf — broken down by feature index so we know which dim is dying.

Usage (inside the container):
    python scripts/diagnose_essential_nans.py \
        --data-root external/data/humanml3d_packed --split train
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

from mardm.data import EssentialDataset


def _idx_label(i: int) -> str:
    if i == 0:
        return "root yaw rate"
    if i == 1:
        return "root vel X"
    if i == 2:
        return "root vel Z"
    if i == 3:
        return "root height Y"
    j = (i - 4) // 3
    axis = "XYZ"[(i - 4) % 3]
    return f"joint{j}.{axis}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--min-seq-len", type=int, default=10)
    p.add_argument("--max-seq-len", type=int, default=10_000)
    p.add_argument("--show-bad", type=int, default=20)
    args = p.parse_args()

    ds = EssentialDataset(
        root=Path(args.data_root), split=args.split, window_size=None,
        mirror_augment=False, min_seq_len=args.min_seq_len, max_seq_len=args.max_seq_len,
    )
    print(f"[diag] iterating {len(ds)} clips in split={args.split} ...")

    bad: list[tuple[str, int, set[int]]] = []
    per_dim = Counter()
    for i in range(len(ds)):
        s = ds[i]
        x = s.x1
        m = ~torch.isfinite(x)
        if not m.any():
            continue
        bad_dims = torch.nonzero(m.any(dim=0), as_tuple=False).flatten().tolist()
        bad.append((s.clip_id, int(m.sum()), set(bad_dims)))
        for d in bad_dims:
            per_dim[d] += 1

    print(f"[diag] {len(bad)} / {len(ds)} clips have non-finite essential features\n")

    if not bad:
        print("[diag] all clean — NaN source is downstream of the dataset, not the encode.")
        return

    print("[diag] non-finite count by feature index:")
    for d, n in sorted(per_dim.items()):
        print(f"  dim {d:>3} ({_idx_label(d):<14}): {n} clips affected")

    print(f"\n[diag] first {min(args.show_bad, len(bad))} bad clips:")
    for cid, total, dims in bad[: args.show_bad]:
        print(f"  {cid}: {total} non-finite values across dims {sorted(dims)}")


if __name__ == "__main__":
    main()
