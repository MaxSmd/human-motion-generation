"""Select an RVQ checkpoint by validation reconstruction FID."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--variant", default="recon_unmasked")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--best-checkpoint", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for result_path in sorted(args.eval_dir.glob("*.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        metrics = result.get(args.variant)
        checkpoint = result.get("_meta", {}).get("checkpoint")
        if not isinstance(metrics, dict) or not isinstance(checkpoint, str):
            continue
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            continue
        fid = metrics.get("fid")
        if not isinstance(fid, (int, float)):
            continue
        rows.append(
            {
                "checkpoint": checkpoint,
                "result": str(result_path),
                "fid": float(fid),
                "r1": float(metrics["r_precision"][0]),
                "r2": float(metrics["r_precision"][1]),
                "r3": float(metrics["r_precision"][2]),
                "mm_dist": float(metrics["mm_dist"]),
                "num_clips": int(metrics["num_clips"]),
            }
        )
    if not rows:
        raise RuntimeError(f"no {args.variant!r} evaluation results found under {args.eval_dir}")

    rows.sort(key=lambda row: float(row["fid"]))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["rank", *rows[0].keys()])
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({"rank": rank, **row})

    best = rows[0]
    source = Path(str(best["checkpoint"]))
    args.best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, args.best_checkpoint)
    selection = {
        "selection_metric": f"validation_{args.variant}_fid",
        "best_checkpoint_source": str(source),
        "best_checkpoint": str(args.best_checkpoint),
        "best_metrics": best,
        "num_candidates": len(rows),
    }
    args.selection_json.parent.mkdir(parents=True, exist_ok=True)
    args.selection_json.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(
        f"[vq-selection] candidates={len(rows)} best={source} "
        f"fid={float(best['fid']):.6f}",
        flush=True,
    )
    print(f"[vq-selection] wrote {args.best_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
