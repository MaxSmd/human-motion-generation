#!/usr/bin/env python3
"""Normalize joint .npy clips for the showcase's browser player.

lib/npy.js reads C-order float32 (or float64) arrays only. Cluster renders
sometimes come back as float64 and/or non-contiguous; this rewrites them to
C-order float32 (T, 22, 3) so the browser parser is happy and files stay tiny.

Usage:
    python scripts/normalize_npy.py in.npy public/data/out.npy
    python scripts/normalize_npy.py some_dir/*.npy --outdir public/data
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def normalize(src: Path, dst: Path) -> None:
    arr = np.load(src)
    if arr.ndim != 3 or arr.shape[1:] != (22, 3):
        print(f"  ! {src.name}: expected (T,22,3), got {arr.shape} — skipping", file=sys.stderr)
        return
    arr = np.ascontiguousarray(arr, dtype="<f4")
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, arr)
    print(f"  ✓ {src.name} → {dst}  ({arr.shape[0]} frames, {dst.stat().st_size // 1024} KB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--outdir", type=Path, help="write <name>.npy here (else 2nd arg is the dst)")
    args = ap.parse_args()

    if args.outdir:
        for src in args.inputs:
            normalize(src, args.outdir / src.name)
    elif len(args.inputs) == 2:
        normalize(args.inputs[0], args.inputs[1])
    else:
        ap.error("provide `in.npy out.npy`, or `--outdir DIR` with one+ inputs")


if __name__ == "__main__":
    main()
