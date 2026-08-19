#!/bin/bash
#SBATCH --job-name=momask-tok-paper
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --output=slurm/logs/momask-tokens-paperstyle-%j.out
#SBATCH --error=slurm/logs/momask-tokens-paperstyle-%j.err
#
# Shared paper-style MoMask token-transformer configuration.
# Uses the canonical VQ checkpoint, CLIP text conditioning, and the official-like
# transformer size: hidden=384, ffn=1024, heads=6, layers=8.
# Submit the masked/residual wrappers for paper-faithful independent training.
# Direct submission retains the historical joint stage for compatibility.
#
# Submit from the repo root:
#     sbatch slurm/momask/train_momask_tokens_paperstyle.sbatch

set -euo pipefail

export VQ_CKPT="${VQ_CKPT:-runs/momask-canonical-vq-paper-gumbelfix64-50e/momask_vq_latest.pt}"
export CANONICAL_H3D_DIR="${CANONICAL_H3D_DIR:-${HOME}/rmg-runs/h3d-canonical/new_joint_vecs}"
export HUMANML3D_DIR="${HUMANML3D_DIR:-/mnt/projects/drl4cvb/human-motion-representation/data/HumanML3D/HumanML3D}"
export HUMANML3D_TEXTS_ZIP="${HUMANML3D_TEXTS_ZIP:-${HUMANML3D_DIR}/texts.zip}"
export HUMANML3D_SPLIT_DIR="${HUMANML3D_SPLIT_DIR:-${HUMANML3D_DIR}}"
export RUN_NAME="${RUN_NAME:-momask-canonical-tokens-officialsplits196-clip500e}"
export TOKEN_STAGE="${TOKEN_STAGE:-joint}"
export TOKEN_STEPS="${TOKEN_STEPS:-0}"
export TOKEN_EPOCHS="${TOKEN_EPOCHS:-500}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-196}"
export MIN_SEQ_LEN="${MIN_SEQ_LEN:-40}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export TOKEN_BATCH_SIZE="${TOKEN_BATCH_SIZE:-64}"
export SAVE_EVERY="${SAVE_EVERY:-5000}"
export LOG_EVERY="${LOG_EVERY:-1000}"
export VALIDATE_EVERY="${VALIDATE_EVERY:-5000}"
export VAL_EVAL_BATCHES="${VAL_EVAL_BATCHES:-32}"
export TEXT_ENCODER="${TEXT_ENCODER:-clip}"
export CLIP_BACKEND="${CLIP_BACKEND:-transformers}"
export CLIP_L2_NORMALIZE="${CLIP_L2_NORMALIZE:-0}"
export TRANSFORMER_ARCH="${TRANSFORMER_ARCH:-paper}"
export RESIDUAL_ARCH="${RESIDUAL_ARCH:-codebook}"
export RESIDUAL_SHARE_WEIGHT="${RESIDUAL_SHARE_WEIGHT:-1}"
export LIVE_TOKEN_CROPS="${LIVE_TOKEN_CROPS:-1}"
export PAPER_TRANSFORMER_DATA="${PAPER_TRANSFORMER_DATA:-1}"
export BASE_FULL_MASK_PROB="${BASE_FULL_MASK_PROB:-0.0}"
export TOKEN_LR="${TOKEN_LR:-2e-4}"
export ADAM_BETA2="${ADAM_BETA2:-0.99}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
export TOKEN_GRAD_CLIP="${TOKEN_GRAD_CLIP:-0.0}"
export WARMUP_STEPS="${WARMUP_STEPS:-2000}"
export LR_MILESTONES="${LR_MILESTONES:-50000}"
export LR_GAMMA="${LR_GAMMA:-0.1}"
export SEED="${SEED:-3407}"

export TRANSFORMER_HIDDEN_DIM="${TRANSFORMER_HIDDEN_DIM:-384}"
export TRANSFORMER_DEPTH="${TRANSFORMER_DEPTH:-8}"
export TRANSFORMER_HEADS="${TRANSFORMER_HEADS:-6}"
export TRANSFORMER_FFN_DIM="${TRANSFORMER_FFN_DIM:-1024}"
export TRANSFORMER_DROPOUT="${TRANSFORMER_DROPOUT:-0.2}"

bash slurm/momask/train_momask_tokens_full.sbatch
