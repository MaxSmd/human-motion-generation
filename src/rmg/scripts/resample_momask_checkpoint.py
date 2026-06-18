"""Regenerate saved MoMask smoke samples from a checkpoint without retraining."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from momask.models import (
    MaskedMotionTransformer,
    MotionRVQVAE,
    ResidualTransformer,
    TokenTransformerConfig,
)
from rmg.representation import H3D_FEATURE_DIM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", help="Path to momask_smoke_latest.pt")
    p.add_argument("--steps", type=int, default=1, help="Base-token generation steps. Use 1 for one-shot.")
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default=None, help="Output checkpoint path. Defaults to *_resampled_s{steps}.pt")
    return p.parse_args()


def _build_models(args: dict, device: torch.device):
    vqvae = MotionRVQVAE(
        input_dim=H3D_FEATURE_DIM,
        hidden_dim=int(args["vq_hidden_dim"]),
        latent_dim=int(args["vq_latent_dim"]),
        num_quantizers=int(args["num_quantizers"]),
        codebook_size=int(args["codebook_size"]),
        downsample=int(args["downsample"]),
        num_res_blocks=int(args["vq_res_blocks"]),
        quantize_dropout_prob=float(args["quantize_dropout"]),
        velocity_loss_weight=float(args.get("vq_velocity_weight", 0.0)),
    ).to(device)
    cfg = TokenTransformerConfig(
        vocab_size=int(args["codebook_size"]),
        text_dim=int(args["text_dim"]),
        hidden_dim=int(args["transformer_hidden_dim"]),
        depth=int(args["transformer_depth"]),
        num_heads=int(args["transformer_heads"]),
        ffn_dim=int(args["transformer_ffn_dim"]),
        max_seq_len=int(args["max_seq_len"]),
        dropout=float(args["transformer_dropout"]),
    )
    masked_model = MaskedMotionTransformer(cfg).to(device)
    residual_model = ResidualTransformer(
        cfg,
        num_quantizers=int(args["num_quantizers"]),
        separate_level_heads=not bool(args.get("shared_residual_head", False)),
    ).to(device)
    return vqvae, masked_model, residual_model


def _level_acc(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    out = {}
    for level in range(target.shape[1]):
        out[f"generated_acc_l{level}"] = float((pred[:, level][mask] == target[:, level][mask]).float().mean())
    return out


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    device = torch.device(args.device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    run_args = ckpt["args"]
    if "sample_cond" not in ckpt:
        raise KeyError(
            "checkpoint does not contain 'sample_cond'. "
            "Re-run train_momask_smoke.py after pulling the latest code."
        )
    if "sample_true_tokens" not in ckpt:
        raise KeyError(
            "checkpoint does not contain 'sample_true_tokens'. "
            "Re-run train_momask_smoke.py after pulling the latest code."
        )

    vqvae, masked_model, residual_model = _build_models(run_args, device)
    vqvae.load_state_dict(ckpt["vqvae"])
    masked_model.load_state_dict(ckpt["masked_transformer"])
    residual_model.load_state_dict(ckpt["residual_transformer"])
    vqvae.eval()
    masked_model.eval()
    residual_model.eval()

    cond = ckpt["sample_cond"].to(device).float()
    true_tokens = ckpt["sample_true_tokens"].to(device).long()
    token_mask = ckpt.get("sample_token_mask")
    if token_mask is None:
        token_mask = torch.ones(true_tokens.shape[0], true_tokens.shape[-1], dtype=torch.bool)
    token_mask = token_mask.to(device).bool()
    target_len = int(ckpt["sample_real"].shape[1])

    with torch.no_grad():
        base = masked_model.generate(
            cond=cond,
            seq_len=true_tokens.shape[-1],
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            mask=token_mask,
        )
        base_only = vqvae.decode_from_tokens(base.unsqueeze(1), target_len=target_len)
        gen_tokens = residual_model.generate_residuals(
            base,
            cond=cond,
            guidance_scale=args.guidance_scale,
            mask=token_mask,
        )
        gen = vqvae.decode_from_tokens(gen_tokens, target_len=target_len)
        teacher_residual_tokens = residual_model.generate_residuals(
            true_tokens[:, 0],
            cond=cond,
            guidance_scale=args.guidance_scale,
            mask=token_mask,
        )
        teacher_residual = vqvae.decode_from_tokens(teacher_residual_tokens, target_len=target_len)

    token_eval = dict(ckpt.get("token_eval", {}))
    token_eval.update(_level_acc(gen_tokens.cpu(), true_tokens.cpu(), token_mask.cpu()))
    token_eval["base_generate_acc"] = token_eval["generated_acc_l0"]
    ckpt["token_eval"] = token_eval
    ckpt["sample_base_only"] = base_only.cpu()
    ckpt["sample_teacher_residual"] = teacher_residual.cpu()
    ckpt["sample_generated"] = gen.cpu()
    ckpt["sample_base_tokens"] = base.cpu()
    ckpt["sample_tokens"] = gen_tokens.cpu()
    ckpt["sample_teacher_residual_tokens"] = teacher_residual_tokens.cpu()
    ckpt["resample"] = {
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "source": str(ckpt_path),
    }

    out_path = Path(args.output) if args.output else ckpt_path.with_name(f"{ckpt_path.stem}_resampled_s{args.steps}.pt")
    torch.save(ckpt, out_path)
    acc = " ".join(f"{k}={v:.3f}" for k, v in sorted(token_eval.items()) if k.startswith("generated_acc_l"))
    print(f"[resample] {acc}")
    print(f"[resample] wrote {out_path}")


if __name__ == "__main__":
    main()
