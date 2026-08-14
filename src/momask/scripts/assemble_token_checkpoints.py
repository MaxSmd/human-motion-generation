"""Assemble independently trained MoMask token transformers into one checkpoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch


TOKEN_VALIDATION_VERSION = 1


COMPATIBLE_ARG_NAMES = (
    "vq_hidden_dim",
    "vq_latent_dim",
    "num_quantizers",
    "codebook_size",
    "downsample",
    "vq_res_blocks",
    "vq_commitment_weight",
    "quantize_dropout",
    "vq_velocity_weight",
    "vq_explicit_weight",
    "vq_recon_loss",
    "vq_use_ema",
    "vq_ema_decay",
    "vq_codebook_sample_temp",
    "vq_arch",
    "no_momask_normalize",
    "feat_bias",
    "text_encoder",
    "clip_model",
    "clip_backend",
    "clip_l2_normalize",
    "text_dim",
    "transformer_arch",
    "transformer_hidden_dim",
    "transformer_depth",
    "transformer_heads",
    "transformer_ffn_dim",
    "transformer_dropout",
    "shared_residual_head",
    "residual_arch",
    "residual_share_weight",
    "base_cond_drop",
    "residual_cond_drop",
    "paper_transformer_data",
    "canonical_h3d_dir",
    "humanml3d_texts_zip",
    "max_seq_len",
    "min_seq_len",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--masked-checkpoint", required=True)
    parser.add_argument("--residual-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-unvalidated-components",
        action="store_true",
        help="Allow stage checkpoints that were not selected by validation.",
    )
    return parser.parse_args()


def _load(path: str | Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _assert_state_equal(name: str, first: dict, second: dict) -> None:
    if first.keys() != second.keys():
        raise ValueError(f"{name} state keys differ between checkpoints")
    for key in first:
        left = first[key]
        right = second[key]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            if left != right:
                raise ValueError(f"{name}.{key} differs between checkpoints")
        elif not torch.equal(left, right):
            raise ValueError(f"{name}.{key} differs between checkpoints")


def _validate_component_selection(checkpoint: dict, stage: str) -> None:
    checkpoint_args = checkpoint.get("args", {})
    expected_metric = "base_ce"
    if stage == "residual":
        expected_metric = (
            "residual_sampled_ce"
            if checkpoint_args.get("transformer_arch") == "paper"
            and checkpoint_args.get("residual_arch") == "codebook"
            else "residual_ce"
        )
    if checkpoint.get("checkpoint_role") != "best_validation":
        raise ValueError(f"{stage} checkpoint was not selected as a best-validation checkpoint")
    if checkpoint.get("token_validation_version") != TOKEN_VALIDATION_VERSION:
        raise ValueError(f"{stage} checkpoint uses incompatible validation metadata")
    if checkpoint.get("best_val_metric_name") != expected_metric:
        raise ValueError(
            f"{stage} checkpoint validation metric must be {expected_metric!r}, got "
            f"{checkpoint.get('best_val_metric_name')!r}"
        )
    metric = checkpoint.get("best_val_metric")
    if not isinstance(metric, (int, float)) or not math.isfinite(float(metric)):
        raise ValueError(f"{stage} checkpoint has no finite best validation metric")
    val_eval = checkpoint.get("val_eval")
    if not isinstance(val_eval, dict) or expected_metric not in val_eval:
        raise ValueError(f"{stage} checkpoint does not contain its validation diagnostics")
    if not math.isclose(float(val_eval[expected_metric]), float(metric), rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError(f"{stage} checkpoint metric does not match its validation diagnostics")


def assemble_token_checkpoints(
    masked: dict,
    residual: dict,
    *,
    require_validation: bool = True,
) -> dict:
    for checkpoint, expected_stage in ((masked, "masked"), (residual, "residual")):
        for key in ("args", "normalizer", "vqvae", "masked_transformer", "residual_transformer"):
            if key not in checkpoint:
                raise KeyError(f"{expected_stage} checkpoint is missing {key!r}")
        actual_stage = checkpoint["args"].get("token_stage", "joint")
        if actual_stage != expected_stage:
            raise ValueError(
                f"expected a {expected_stage!r} checkpoint, got token_stage={actual_stage!r}"
            )
        if require_validation:
            _validate_component_selection(checkpoint, expected_stage)

    masked_args = masked["args"]
    residual_args = residual["args"]
    for name in COMPATIBLE_ARG_NAMES:
        if masked_args.get(name) != residual_args.get(name):
            raise ValueError(
                f"incompatible checkpoint argument {name!r}: "
                f"{masked_args.get(name)!r} != {residual_args.get(name)!r}"
            )
    _assert_state_equal("normalizer", masked["normalizer"], residual["normalizer"])
    _assert_state_equal("vqvae", masked["vqvae"], residual["vqvae"])

    merged = dict(masked)
    merged["residual_transformer"] = residual["residual_transformer"]
    merged["step"] = max(int(masked.get("step", 0)), int(residual.get("step", 0)))
    merged["component_steps"] = {
        "masked": int(masked.get("step", 0)),
        "residual": int(residual.get("step", 0)),
    }
    merged["component_validation"] = {
        "masked": {
            "name": masked.get("best_val_metric_name"),
            "value": masked.get("best_val_metric"),
            "step": int(masked.get("step", 0)),
        },
        "residual": {
            "name": residual.get("best_val_metric_name"),
            "value": residual.get("best_val_metric"),
            "step": int(residual.get("step", 0)),
        },
    }
    merged_args = dict(masked_args)
    merged_args["token_stage"] = "assembled"
    merged_args["freeze_masked_transformer"] = False
    merged["args"] = merged_args
    merged["checkpoint_role"] = "assembled"
    merged.pop("token_optimizer", None)
    merged.pop("token_eval", None)
    merged.pop("best_val_metric", None)
    merged.pop("best_val_metric_name", None)
    merged.pop("val_eval", None)
    for key in list(merged):
        if key.startswith("sample_"):
            merged.pop(key)
    return merged


def main() -> None:
    args = parse_args()
    masked_path = Path(args.masked_checkpoint)
    residual_path = Path(args.residual_checkpoint)
    output_path = Path(args.output)
    merged = assemble_token_checkpoints(
        _load(masked_path),
        _load(residual_path),
        require_validation=not args.allow_unvalidated_components,
    )
    merged["component_checkpoints"] = {
        "masked": str(masked_path),
        "residual": str(residual_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output_path)
    print(
        f"[assemble] masked_step={merged['component_steps']['masked']} "
        f"residual_step={merged['component_steps']['residual']}"
    )
    print(f"[assemble] wrote {output_path}")


if __name__ == "__main__":
    main()
