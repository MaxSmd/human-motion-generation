"""Build a tiny synthetic packed HumanML3D-shaped dataset for smoke testing.

Useful for:
  * Running `scripts/train.py` end-to-end without AMASS (e.g. on a laptop).
  * Reproducing the dataset format documented in `src/rmg/data/humanml3d.py`.

Output layout matches the real `prepare_humanml3d.py pack` output, so the same
config / dataset class works without changes.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import torch

from rmg.representation import NUM_JOINTS


def _clip(T: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    translation = torch.randn(T, 3, generator=g) * 0.05
    quats = torch.randn(T, NUM_JOINTS, 4, generator=g)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    return {
        "translation": translation,
        "quats": quats,
        "texts": [
            f"a person performs action {seed} at a slow pace",
            f"slow {seed}-action",
            f"motion sample number {seed}",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=Path("external/data/humanml3d_packed_synth"))
    ap.add_argument("--num-train", type=int, default=64)
    ap.add_argument("--num-val", type=int, default=8)
    ap.add_argument("--num-test", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=80, help="clip length T (frames)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total = args.num_train + args.num_val + args.num_test

    splits = {"train": [], "val": [], "test": []}
    print(f"[synth] writing {total} clips to {args.output_dir}/humanml3d.zip ...")
    with zipfile.ZipFile(args.output_dir / "humanml3d.zip", "w", compression=zipfile.ZIP_STORED) as zf:
        idx = 0
        for split, n in zip(("train", "val", "test"), (args.num_train, args.num_val, args.num_test)):
            for _ in range(n):
                cid = f"{idx:06d}"
                buf = io.BytesIO()
                torch.save(_clip(T=args.seq_len, seed=idx), buf)
                zf.writestr(f"{cid}.pt", buf.getvalue())
                splits[split].append(cid)
                idx += 1

    (args.output_dir / "splits.json").write_text(json.dumps(splits))
    torch.save(torch.zeros(NUM_JOINTS, 3), args.output_dir / "target_offsets.pt")
    (args.output_dir / "meta.json").write_text(json.dumps({"fps": 20, "num_clips": total, "version": 1}))

    print(f"[synth] done. splits: train={args.num_train} val={args.num_val} test={args.num_test}")
    print(f"[synth] point train.py at this dir with: data.root={args.output_dir}")


if __name__ == "__main__":
    main()
