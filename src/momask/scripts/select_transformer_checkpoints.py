"""Discover and rank independently trained MoMask transformer checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

import torch

from momask.scripts.assemble_token_checkpoints import assemble_token_checkpoints


STEP_CHECKPOINT_RE = re.compile(r"tokens_step_(\d+)\.pt$")


@dataclass(frozen=True)
class Candidate:
    label: str
    masked_checkpoint: Path
    residual_checkpoint: Path


def _load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def candidate_identity(candidate: Candidate) -> dict[str, object]:
    return {
        "label": candidate.label,
        "masked": checkpoint_identity(candidate.masked_checkpoint),
        "residual": checkpoint_identity(candidate.residual_checkpoint),
    }


def annotate_result(
    path: Path,
    candidate: Candidate,
    protocol_id: str | None = None,
) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_checkpoint_selection"] = candidate_identity(candidate)
    if protocol_id is not None:
        data["_checkpoint_selection_protocol"] = protocol_id
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def result_matches_candidate(
    path: Path,
    candidate: Candidate,
    protocol_id: str | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("_checkpoint_selection") != candidate_identity(candidate):
            return False
        return (
            protocol_id is None
            or data.get("_checkpoint_selection_protocol") == protocol_id
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False


def discover_stage_checkpoints(
    checkpoint_dir: Path,
    *,
    stride: int = 1,
    include_best_val: bool = True,
    include_latest: bool = True,
    include_final: bool = True,
) -> list[tuple[str, Path]]:
    if stride <= 0:
        raise ValueError("stride must be positive")
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_dir}")

    steps: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("tokens_step_*.pt"):
        match = STEP_CHECKPOINT_RE.fullmatch(path.name)
        if match:
            steps.append((int(match.group(1)), path))
    steps.sort(key=lambda item: item[0])

    selected = [
        (f"step_{step:07d}", path)
        for index, (step, path) in enumerate(steps)
        if index % stride == 0 or index == len(steps) - 1
    ]
    if include_best_val:
        path = checkpoint_dir / "tokens_best_val.pt"
        if path.is_file():
            selected.append(("best_val", path))
    if include_latest:
        path = checkpoint_dir / "tokens_latest_train.pt"
        if path.is_file():
            selected.append(("latest", path))
    if include_final:
        path = checkpoint_dir.parent / "momask_smoke_latest.pt"
        if path.is_file():
            selected.append(("final", path))
    if not selected:
        raise RuntimeError(f"no token checkpoints found under {checkpoint_dir}")
    return selected


def build_candidates(
    *,
    vary: str,
    checkpoint_dir: Path,
    fixed_checkpoint: Path,
    stride: int,
    include_best_val: bool,
    include_latest: bool,
    include_final: bool = True,
) -> list[Candidate]:
    if vary not in {"masked", "residual"}:
        raise ValueError(f"vary must be 'masked' or 'residual', got {vary!r}")
    if not fixed_checkpoint.is_file():
        raise FileNotFoundError(f"fixed checkpoint not found: {fixed_checkpoint}")

    candidates = []
    for label, path in discover_stage_checkpoints(
        checkpoint_dir,
        stride=stride,
        include_best_val=include_best_val,
        include_latest=include_latest,
        include_final=include_final,
    ):
        if vary == "masked":
            candidates.append(Candidate(f"m_{label}", path, fixed_checkpoint))
        else:
            candidates.append(Candidate(f"r_{label}", fixed_checkpoint, path))
    return candidates


def write_manifest(path: Path, candidates: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("label", "masked_checkpoint", "residual_checkpoint"))
        for candidate in candidates:
            writer.writerow(
                (candidate.label, candidate.masked_checkpoint, candidate.residual_checkpoint)
            )


def read_manifest(path: Path) -> list[Candidate]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"label", "masked_checkpoint", "residual_checkpoint"}
        if set(reader.fieldnames or ()) != expected:
            raise ValueError(f"invalid selection manifest columns in {path}")
        candidates = [
            Candidate(
                row["label"],
                Path(row["masked_checkpoint"]),
                Path(row["residual_checkpoint"]),
            )
            for row in reader
        ]
    if not candidates:
        raise RuntimeError(f"selection manifest is empty: {path}")
    labels = [candidate.label for candidate in candidates]
    if len(labels) != len(set(labels)):
        raise ValueError(f"selection manifest contains duplicate labels: {path}")
    return candidates


def rank_candidates(
    candidates: list[Candidate],
    eval_dir: Path,
    variant: str,
    protocol_id: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    missing = []
    for candidate in candidates:
        result_path = eval_dir / f"{candidate.label}.json"
        if not result_path.is_file():
            missing.append(result_path)
            continue
        data = json.loads(result_path.read_text(encoding="utf-8"))
        if data.get("_checkpoint_selection") != candidate_identity(candidate):
            raise ValueError(
                f"{result_path} was not evaluated from the checkpoints in the manifest"
            )
        if (
            protocol_id is not None
            and data.get("_checkpoint_selection_protocol") != protocol_id
        ):
            raise ValueError(f"{result_path} was evaluated with a different protocol")
        metrics = data.get(variant)
        if not isinstance(metrics, dict):
            raise ValueError(f"{result_path} has no {variant!r} metrics")
        r_precision = metrics.get("r_precision")
        if not isinstance(r_precision, list) or len(r_precision) < 3:
            raise ValueError(f"{result_path} has invalid R-precision metrics")
        fid = float(metrics["fid"])
        mm_dist = float(metrics["mm_dist"])
        if not math.isfinite(fid) or not math.isfinite(mm_dist):
            raise ValueError(f"{result_path} contains non-finite metrics")
        meta = data.get("_meta", {})
        rows.append(
            {
                "label": candidate.label,
                "masked_checkpoint": str(candidate.masked_checkpoint),
                "residual_checkpoint": str(candidate.residual_checkpoint),
                "fid": fid,
                "r1": float(r_precision[0]),
                "r2": float(r_precision[1]),
                "r3": float(r_precision[2]),
                "mm_dist": mm_dist,
                "diversity": float(metrics["diversity"]),
                "num_clips": int(metrics["num_clips"]),
                "split": meta.get("split", ""),
                "max_seq_len": meta.get("max_seq_len", ""),
                "generation_steps": meta.get("generation_steps", ""),
                "guidance_scale": meta.get("guidance_scale", ""),
                "temperature": meta.get("temperature", ""),
                "topk_filter_thres": meta.get("topk_filter_thres", ""),
                "sample": meta.get("sample", ""),
                "remask_kept_tokens": meta.get("remask_kept_tokens", ""),
                "seed": meta.get("seed", ""),
                "protocol_id": data.get("_checkpoint_selection_protocol", ""),
                "result": str(result_path),
            }
        )
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(
            f"missing {len(missing)} checkpoint-selection evaluations; first: {preview}"
        )
    rows.sort(key=lambda row: (float(row["fid"]), -float(row["r3"]), float(row["mm_dist"])))
    return rows


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def assemble_best(rows: list[dict[str, object]], output: Path, variant: str) -> None:
    best = rows[0]
    masked_path = Path(str(best["masked_checkpoint"]))
    residual_path = Path(str(best["residual_checkpoint"]))
    merged = assemble_token_checkpoints(
        _load_checkpoint(masked_path),
        _load_checkpoint(residual_path),
        require_validation=False,
    )
    merged["checkpoint_role"] = "best_validation_fid_selection"
    merged["fid_selection"] = {
        "variant": variant,
        "candidate": best["label"],
        "masked_checkpoint": str(masked_path),
        "residual_checkpoint": str(residual_path),
        "protocol_id": best["protocol_id"],
        "metrics": {
            key: best[key]
            for key in ("fid", "r1", "r2", "r3", "mm_dist", "diversity", "num_clips")
        },
        "protocol": {
            key: best[key]
            for key in (
                "split",
                "max_seq_len",
                "generation_steps",
                "guidance_scale",
                "temperature",
                "topk_filter_thres",
                "sample",
                "remask_kept_tokens",
                "seed",
            )
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output)


def write_best_env(path: Path, rows: list[dict[str, object]]) -> None:
    best = rows[0]
    values = {
        "BEST_CANDIDATE": best["label"],
        "BEST_MASKED_CKPT": best["masked_checkpoint"],
        "BEST_RESIDUAL_CKPT": best["residual_checkpoint"],
        "BEST_FID": best["fid"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{key}={shlex.quote(str(value))}\n" for key, value in values.items()),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--vary", choices=("masked", "residual"), required=True)
    manifest.add_argument("--checkpoint-dir", type=Path, required=True)
    manifest.add_argument("--fixed-checkpoint", type=Path, required=True)
    manifest.add_argument("--stride", type=int, default=1)
    manifest.add_argument("--no-best-val", action="store_true")
    manifest.add_argument("--no-latest", action="store_true")
    manifest.add_argument("--no-final", action="store_true")
    manifest.add_argument("--output", type=Path, required=True)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--manifest", type=Path, required=True)
    summarize.add_argument("--eval-dir", type=Path, required=True)
    summarize.add_argument("--variant", default="full")
    summarize.add_argument("--protocol-id")
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument("--best-env", type=Path)
    summarize.add_argument("--best-checkpoint", type=Path)

    for name in ("annotate", "matches"):
        command = subparsers.add_parser(name)
        command.add_argument("--result", type=Path, required=True)
        command.add_argument("--label", required=True)
        command.add_argument("--masked-checkpoint", type=Path, required=True)
        command.add_argument("--residual-checkpoint", type=Path, required=True)
        command.add_argument("--protocol-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command in {"annotate", "matches"}:
        candidate = Candidate(args.label, args.masked_checkpoint, args.residual_checkpoint)
        if args.command == "annotate":
            annotate_result(args.result, candidate, args.protocol_id)
            print(f"[selection] annotated {args.result}")
            return
        raise SystemExit(
            0
            if result_matches_candidate(args.result, candidate, args.protocol_id)
            else 1
        )

    if args.command == "manifest":
        candidates = build_candidates(
            vary=args.vary,
            checkpoint_dir=args.checkpoint_dir,
            fixed_checkpoint=args.fixed_checkpoint,
            stride=args.stride,
            include_best_val=not args.no_best_val,
            include_latest=not args.no_latest,
            include_final=not args.no_final,
        )
        write_manifest(args.output, candidates)
        print(f"[selection] wrote {len(candidates)} candidates to {args.output}")
        return

    candidates = read_manifest(args.manifest)
    rows = rank_candidates(candidates, args.eval_dir, args.variant, args.protocol_id)
    write_summary(args.output, rows)
    if args.best_env:
        write_best_env(args.best_env, rows)
    if args.best_checkpoint:
        assemble_best(rows, args.best_checkpoint, args.variant)
    best = rows[0]
    print(f"[selection] wrote {args.output}")
    print(
        f"[selection] best={best['label']} fid={float(best['fid']):.6f} "
        f"r={float(best['r1']):.4f}/{float(best['r2']):.4f}/{float(best['r3']):.4f}"
    )
    if args.best_checkpoint:
        print(f"[selection] assembled {args.best_checkpoint}")


if __name__ == "__main__":
    main()
