"""Compare a MARDM eval results.json to the RMG-paper baseline table.

Saves CSV, Markdown, PNG, and PDF next to the input results.json — slide-ready
with columns matching the paper (FID, R@1, Diversity, MultiModality).

Usage:
    python notebooks/eval_table.py ~/rmg-runs/mardm-eval-latest/eval/results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Baseline numbers — HumanML3D test split, Guo et al. evaluator. Taken from
# the RMG paper's Table 4 (HumanML3D format). MARDM here is the paper's XL.
BASELINES = [
    ("GT",                {"FID": 0.002, "R@1": 0.511, "Diversity": 9.503, "MultiModality": 2.799}),
    ("MLD [2023]",        {"FID": 0.473, "R@1": 0.481, "Diversity": 9.724, "MultiModality": 2.413}),
    ("T2M-GPT [2023]",    {"FID": 0.116, "R@1": 0.492, "Diversity": 9.761, "MultiModality": 1.856}),
    ("MotionGPT [2023]",  {"FID": 0.232, "R@1": 0.492, "Diversity": 9.528, "MultiModality": 2.008}),
    ("MoMask [2024]",     {"FID": 0.045, "R@1": 0.521, "Diversity": None,  "MultiModality": 1.241}),
    ("MotionLCM [2024]",  {"FID": 0.304, "R@1": 0.505, "Diversity": 9.607, "MultiModality": 2.259}),
    ("MotionCLR [2024]",  {"FID": 0.269, "R@1": 0.542, "Diversity": 9.607, "MultiModality": 1.985}),
    ("MotionLab [2025]",  {"FID": 0.167, "R@1": None,  "Diversity": 9.593, "MultiModality": 2.912}),
    ("MARDM [2025]",      {"FID": 0.114, "R@1": 0.500, "Diversity": None,  "MultiModality": 2.231}),
    ("RMG (paper)",       {"FID": 0.043, "R@1": 0.525, "Diversity": 9.555, "MultiModality": 2.748}),
]


def row_from_metrics(name: str, metrics: dict) -> dict:
    rp = metrics.get("r_precision") or [None, None, None]
    return {
        "model": name,
        "FID":           round(metrics["fid"], 3),
        "R@1":           round(rp[0], 3) if rp[0] is not None else None,
        "Diversity":     round(metrics["diversity"], 3),
        "MultiModality": round(metrics.get("multimodality", float("nan")), 3),
    }


def main(results_path: Path) -> None:
    data = json.loads(results_path.read_text())

    rows = []
    for name, m in BASELINES:
        rows.append({"model": name, **m})
    for omega in sorted(data.keys(), key=float):
        rows.append(row_from_metrics(f"MARDM-mini (ours, w={omega})", data[omega]))

    df = pd.DataFrame(rows).set_index("model")

    print("\n=== HumanML3D Format (Guo evaluator) — RMG paper Table 4 + our reproduction ===\n")
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

    fig, ax = plt.subplots(figsize=(10, 0.6 + 0.4 * len(df)))
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
    # Bold our rows so they stand out
    n_baselines = len(BASELINES)
    for i in range(n_baselines, len(df)):
        for j in range(len(df.columns)):
            tbl[i + 1, j].set_text_props(weight="bold")
        tbl[i + 1, -1].set_text_props(weight="bold")  # row label
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"comparison.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nWrote comparison.{{csv,md,pdf,png}} to {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: eval_table.py <results.json>")
    main(Path(sys.argv[1]).expanduser())
