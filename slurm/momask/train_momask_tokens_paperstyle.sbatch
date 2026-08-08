#!/bin/bash
#SBATCH --job-name=momask-tok-paper
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --output=slurm/logs/momask-tokens-paperstyle-%j.out
#SBATCH --error=slurm/logs/momask-tokens-paperstyle-%j.err
#
# Paper-style MoMask token-transformer training.
# Uses the canonical VQ checkpoint, CLIP text conditioning, and the official-like
# transformer size: hidden=384, ffn=1024, heads=6, layers=8.
# Defaults to the codebook residual stage; set RESIDUAL_ARCH=simple for a
# width-only ablation against our current checkpoint family.
#
# Submit from the repo root:
#     sbatch slurm/momask/train_momask_tokens_paperstyle.sbatch

set -euo pipefail

export VQ_CKPT="${VQ_CKPT:-runs/momask-canonical-vq-150k/momask_vq_latest.pt}"
export CANONICAL_H3D_DIR="${CANONICAL_H3D_DIR:-${HOME}/rmg-runs/h3d-canonical/new_joint_vecs}"
export RUN_NAME="${RUN_NAME:-momask-canonical-tokens-paper384-len196-clip200k}"
export TOKEN_STEPS="${TOKEN_STEPS:-200000}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-196}"
export TOKEN_BATCH_SIZE="${TOKEN_BATCH_SIZE:-128}"
export SAVE_EVERY="${SAVE_EVERY:-5000}"
export LOG_EVERY="${LOG_EVERY:-1000}"
export TEXT_ENCODER="${TEXT_ENCODER:-clip}"
export CLIP_BACKEND="${CLIP_BACKEND:-transformers}"
export RESIDUAL_ARCH="${RESIDUAL_ARCH:-codebook}"
export RESIDUAL_SHARE_WEIGHT="${RESIDUAL_SHARE_WEIGHT:-1}"
export LIVE_TOKEN_CROPS="${LIVE_TOKEN_CROPS:-1}"

export TRANSFORMER_HIDDEN_DIM="${TRANSFORMER_HIDDEN_DIM:-384}"
export TRANSFORMER_DEPTH="${TRANSFORMER_DEPTH:-8}"
export TRANSFORMER_HEADS="${TRANSFORMER_HEADS:-6}"
export TRANSFORMER_FFN_DIM="${TRANSFORMER_FFN_DIM:-1024}"
export TRANSFORMER_DROPOUT="${TRANSFORMER_DROPOUT:-0.2}"

bash slurm/momask/train_momask_tokens_full.sbatch
