"""Compare a MARDM eval results.json to paper variants and emit a table.

Saves CSV, Markdown, PNG, and PDF next to the input results.json — slide-ready.

Usage:
    python notebooks/eval_table.py ~/rmg-runs/mardm-eval-latest/eval/results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Paper numbers — HumanML3D test split, Guo et al. evaluator.
# MARDM-S / -XL: MARDM paper (Meng et al. 2025), HumanML3D Table.
PAPER = {
    "MARDM-S (paper, ~30M)": {
        "FID": 0.278, "R@1": 0.534, "R@3": None,
        "MM-Dist": None, "Diversity": None, "MultiModality": None,
    },
    "MARDM-XL (paper, ~290M)": {
        "FID": 0.114, "R@1": 0.555, "R@3": None,
        "MM-Dist": None, "Diversity": None, "MultiModality": None,
    },
}


def row_from_metrics(name: str, metrics: dict) -> dict:
    rp = metrics.get("r_precision") or [None, None, None]
    return {
        "model": name,
        "FID":            round(metrics["fid"], 3),
        "R@1":            round(rp[0], 3) if rp[0] is not None else None,
        "R@3":            round(rp[2], 3) if rp[2] is not None else None,
        "MM-Dist":        round(metrics["mm_dist"], 3),
        "Diversity":      round(metrics["diversity"], 3),
        "MultiModality":  round(metrics.get("multimodality", float("nan")), 3),
    }


def main(results_path: Path) -> None:
    data = json.loads(results_path.read_text())

    rows = []
    # data is keyed by guidance scale (string). One row per scale.
    for omega in sorted(data.keys(), key=float):
        rows.append(row_from_metrics(f"mardm-mini (ours, w={omega})", data[omega]))
    # paper reference rows
    for name, m in PAPER.items():
        rows.append({"model": name, **m})

    df = pd.DataFrame(rows).set_index("model")

    print("\n=== MARDM eval comparison (HumanML3D, Guo evaluator) ===\n")
    try:
        print(df.fillna("—").to_markdown())
    except ImportError:
        print(df.fillna("—").to_string())

    out_dir = results_path.parent
    df.to_csv(out_dir / "comparison.csv")
    try:
        (out_dir / "comparison.md").write_text(df.fillna("—").to_markdown())
    except ImportError:
        pass

    fig, ax = plt.subplots(figsize=(11, 0.6 + 0.4 * len(df)))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.fillna("—").astype(str).values,
        rowLabels=df.index,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.4)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"comparison.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nWrote comparison.{{csv,md,pdf,png}} to {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: eval_table.py <results.json>")
    main(Path(sys.argv[1]).expanduser())
