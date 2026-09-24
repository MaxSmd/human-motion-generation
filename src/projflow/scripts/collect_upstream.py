"""Collect the Stage A per-cell logs into ProjFlow's Table 1 layout.

slurm/projflow/eval_upstream.sbatch writes one log per (joint, intensity) cell
of the OmniControl protocol. The paper reports each joint as the mean over the
five densities and "Average" as the mean over the six joints, so this script
does exactly that and prints it next to the published numbers.

Usage:
    python -m projflow.scripts.collect_upstream ~/rmg-runs/projflow-upstream-table1-projflow_true
    python -m projflow.scripts.collect_upstream <dir> --out src/projflow/reports/results/upstream-table1

Writes <out>/cells.json (every cell), <out>/table1.json and <out>/table1.md.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

JOINTS = {0: "Pelvis", 10: "Left foot", 11: "Right foot", 15: "Head", 20: "Left wrist", 21: "Right wrist"}
INTENSITIES = (1, 2, 5, 25, 100)
METRICS = ("FID", "R@3", "Diversity", "Foot skate", "Traj err", "Loc err", "Avg err")

# ACMDM-S-PS22+ProjFlow, paper Table 7 (per joint) / Table 1 (average).
PAPER = {
    "Pelvis":      (0.107, 0.784, 10.645, 0.0630, 0.0, 0.0, 0.0),
    "Left foot":   (0.095, 0.771, 10.644, 0.0609, 0.0, 0.0, 0.0),
    "Right foot":  (0.096, 0.770, 10.651, 0.0613, 0.0, 0.0, 0.0),
    "Head":        (0.099, 0.788, 10.754, 0.0595, 0.0, 0.0, 0.0),
    "Left wrist":  (0.089, 0.783, 10.601, 0.0586, 0.0, 0.0, 0.0),
    "Right wrist": (0.096, 0.780, 10.610, 0.0584, 0.0, 0.0, 0.0),
    "Average":     (0.097, 0.779, 10.651, 0.0603, 0.0, 0.0, 0.0),
}

_NUM = r"(-?\d+(?:\.\d+)?)"
_PATTERNS = {
    "FID": rf"FID: {_NUM}",
    "R@3": rf"TOP3\. {_NUM}",
    "Diversity": rf"Diversity: {_NUM}",
    "Foot skate": rf"Foot Skating: {_NUM}",
    "Traj err": rf"Traj Error\. {_NUM}",
    "Loc err": rf"Loc Error\. {_NUM}",
    "Avg err": rf"Avg Error\. {_NUM}",
}


def parse_cell(log: Path) -> dict[str, float] | None:
    """Metrics from the `final result:` block of one upstream eval log."""
    text = log.read_text(errors="replace")
    if "final result:" not in text:
        return None
    final = text.rsplit("final result:", 1)[1]
    cell = {}
    for name, pattern in _PATTERNS.items():
        match = re.search(pattern, final)
        if match is None:
            return None
        cell[name] = float(match.group(1))
    wall = re.search(r"wall_seconds=(\d+)", text)
    if wall:
        cell["wall_seconds"] = int(wall.group(1))
    return cell


def mean_row(cells: list[dict[str, float]]) -> dict[str, float]:
    return {m: sum(c[m] for c in cells) / len(cells) for m in METRICS}


def to_markdown(rows: dict[str, dict[str, float]], complete: dict[str, int]) -> str:
    header = "| Joint | cells | " + " | ".join(METRICS) + " |"
    lines = [header, "|" + "---|" * (len(METRICS) + 2)]
    for joint, row in rows.items():
        ours = " | ".join(f"{row[m]:.4f}" for m in METRICS)
        paper = " | ".join(f"{v:.4f}" for v in PAPER[joint])
        total = len(INTENSITIES) * (len(JOINTS) if joint == "Average" else 1)
        lines.append(f"| **{joint}** (ours) | {complete[joint]}/{total} | {ours} |")
        lines.append(f"| {joint} (paper) | | {paper} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cells: dict[str, dict] = {}
    per_joint: dict[str, list[dict[str, float]]] = {}
    missing = []
    for joint_id, joint in JOINTS.items():
        for intensity in INTENSITIES:
            log = args.log_dir / f"joint{joint_id}_intensity{intensity}.log"
            cell = parse_cell(log) if log.exists() else None
            if cell is None:
                missing.append(log.name)
                continue
            cells[f"{joint}/{intensity}"] = cell
            per_joint.setdefault(joint, []).append(cell)

    rows = {joint: mean_row(cs) for joint, cs in per_joint.items()}
    complete = {joint: len(cs) for joint, cs in per_joint.items()}
    if rows:
        rows["Average"] = mean_row(list(rows.values()))
        complete["Average"] = sum(complete.values())

    table = to_markdown(rows, complete) if rows else "(no finished cells)\n"
    print(table)
    if missing:
        print(f"missing/unfinished: {len(missing)} cells: {', '.join(missing)}")

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "cells.json").write_text(json.dumps(cells, indent=2))
        (args.out / "table1.json").write_text(json.dumps({"rows": rows, "paper": PAPER, "missing": missing}, indent=2))
        (args.out / "table1.md").write_text(table)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
