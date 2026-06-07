#!/usr/bin/env python
"""Quantify motion *jitter* (smoothness) of generated vs ground-truth motions.

Reads the joint `.npy` dumps that `visualize.py` writes next to its GIFs
(`real-<cid>.npy` = GT, `gen-<cid>.npy` = prediction), and compares their
smoothness. Real human motion is smooth (low acceleration / jerk); a jittery
generation has high frame-to-frame acceleration even when the overall pose
trajectory is correct.

Metrics (per-frame, averaged over joints):
  - acceleration magnitude  ‖x_{t+1} - 2x_t + x_{t-1}‖   (the visible "shake")
  - jerk magnitude          ‖Δ³x‖                        (classic smoothness measure)
Lower = smoother. We report the gen/GT ratio — "how many times jerkier than real."

Usage:
    # point at a compare/viz output dir holding real-*.npy / gen-*.npy
    python scripts/plot_jitter.py ~/rmg-runs/viz-compare-XXXX/viz --out runs/jitter.png

    # or pass explicit (GT, gen) pairs
    python scripts/plot_jitter.py real-000857.npy gen-000857.npy --out runs/jitter.png

No GPU / no deps beyond numpy + matplotlib. Run it wherever the .npy files are
(or copy them down and run locally).
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _accel(pos: np.ndarray) -> np.ndarray:
    """Per-frame mean-over-joints acceleration magnitude. pos: (T, J, 3)."""
    acc = pos[2:] - 2.0 * pos[1:-1] + pos[:-2]              # (T-2, J, 3)
    return np.linalg.norm(acc, axis=-1).mean(axis=-1)       # (T-2,)


def _jerk_scalar(pos: np.ndarray) -> float:
    jerk = np.diff(pos, n=3, axis=0)                        # (T-3, J, 3)
    return float(np.linalg.norm(jerk, axis=-1).mean())


def _find_pairs(d: Path) -> list[tuple[str, Path, Path]]:
    pairs = []
    for g in sorted(d.glob("gen-*.npy")):
        cid = g.stem[len("gen-"):]
        r = d / f"real-{cid}.npy"
        if r.exists():
            pairs.append((cid, r, g))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="a viz dir (with real-*.npy / gen-*.npy), or explicit GT,gen pairs")
    ap.add_argument("--out", type=Path, default=Path("jitter.png"))
    args = ap.parse_args()

    # Resolve (cid, gt_path, gen_path) triples.
    pairs: list[tuple[str, Path, Path]] = []
    if len(args.inputs) == 1 and Path(args.inputs[0]).is_dir():
        pairs = _find_pairs(Path(args.inputs[0]))
    else:
        files = [Path(p) for p in args.inputs]
        if len(files) % 2 != 0:
            raise SystemExit("pass a dir, or an even number of files as GT,gen pairs")
        for i in range(0, len(files), 2):
            pairs.append((files[i].stem, files[i], files[i + 1]))
    if not pairs:
        raise SystemExit("no real-*.npy / gen-*.npy pairs found")

    n = len(pairs)
    ncol = min(3, n)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.0 * nrow), squeeze=False)

    print(f"{'clip':<14}{'jerk GT':>10}{'jerk gen':>10}{'ratio':>8}")
    print("-" * 42)
    ratios = []
    for idx, (cid, gtp, genp) in enumerate(pairs):
        gt = np.load(gtp).astype(np.float64)
        gen = np.load(genp).astype(np.float64)
        T = min(len(gt), len(gen))
        gt, gen = gt[:T], gen[:T]

        a_gt, a_gen = _accel(gt), _accel(gen)
        j_gt, j_gen = _jerk_scalar(gt), _jerk_scalar(gen)
        ratio = j_gen / max(j_gt, 1e-9)
        ratios.append(ratio)
        print(f"{cid:<14}{j_gt:>10.4f}{j_gen:>10.4f}{ratio:>7.1f}x")

        ax = axes[idx // ncol][idx % ncol]
        ax.plot(a_gt, color="#16B6A6", lw=1.4, label="GT")
        ax.plot(a_gen, color="#F2683C", lw=1.2, alpha=0.9, label="gen")
        ax.set_title(f"{cid}   gen/GT jerk = {ratio:.1f}×", fontsize=9)
        ax.set_xlabel("frame"); ax.set_ylabel("accel mag")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)

    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fig.suptitle(f"Motion jitter: generated vs GT  (mean gen/GT jerk = {np.mean(ratios):.1f}×)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"\n[plot_jitter] wrote {args.out}   (mean gen/GT jerk ratio {np.mean(ratios):.1f}×)")


if __name__ == "__main__":
    main()
