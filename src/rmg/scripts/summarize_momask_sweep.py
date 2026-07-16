"""Summarize MoMask evaluation sweep JSON files.

The sweep job writes one JSON file per checkpoint / guidance / temperature.
This script collects the main quality metrics into a CSV and a readable table.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("sweep_dir", help="Directory containing per-run evaluation JSON files.")
    p.add_argument("--variant", default="full", help="Variant to summarize, usually 'full'.")
    p.add_argument("--output", default=None, help="CSV output path. Defaults to <sweep_dir>/summary_<variant>.csv.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    rows = []
    for path in sorted(sweep_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data.get("_meta", {})
        metrics = data.get(args.variant)
        if not isinstance(metrics, dict):
            continue
        r = metrics.get("r_precision", [None, None, None])
        row = {
            "file": path.name,
            "checkpoint": meta.get("checkpoint", ""),
            "generation_steps": meta.get("generation_steps", ""),
            "guidance_scale": meta.get("guidance_scale", ""),
            "temperature": meta.get("temperature", ""),
            "fid": metrics.get("fid", ""),
            "r1": r[0] if len(r) > 0 else "",
            "r2": r[1] if len(r) > 1 else "",
            "r3": r[2] if len(r) > 2 else "",
            "mm_dist": metrics.get("mm_dist", ""),
            "diversity": metrics.get("diversity", ""),
            "diversity_real": metrics.get("diversity_real", ""),
            "num_clips": metrics.get("num_clips", ""),
        }
        rows.append(row)

    if not rows:
        raise SystemExit(f"no JSON files with variant {args.variant!r} found in {sweep_dir}")

    rows.sort(key=lambda x: (float(x["fid"]), -float(x["r3"]), float(x["mm_dist"])))
    out_path = Path(args.output) if args.output else sweep_dir / f"summary_{args.variant}.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[summary] wrote {out_path}")
    print()
    print("rank,fid,r1,r2,r3,mm_dist,diversity,guidance,temp,checkpoint")
    for rank, row in enumerate(rows[:20], start=1):
        ckpt = Path(str(row["checkpoint"])).name
        parent = Path(str(row["checkpoint"])).parent.name
        if parent == "checkpoints":
            parent = Path(str(row["checkpoint"])).parent.parent.name
        print(
            f"{rank},"
            f"{float(row['fid']):.4f},"
            f"{float(row['r1']):.4f},"
            f"{float(row['r2']):.4f},"
            f"{float(row['r3']):.4f},"
            f"{float(row['mm_dist']):.4f},"
            f"{float(row['diversity']):.4f},"
            f"{row['guidance_scale']},"
            f"{row['temperature']},"
            f"{parent}/{ckpt}"
        )


if __name__ == "__main__":
    main()
