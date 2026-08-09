#!/bin/bash
#SBATCH --job-name=momask-vq-paper
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --output=slurm/logs/momask-vq-paperstyle-%j.out
#SBATCH --error=slurm/logs/momask-vq-paperstyle-%j.err
#
# More paper-faithful MoMask RVQ-VAE training.
# Official MoMask trains the RVQ before the token transformers, with short
# motion windows and a downsampled 512-D EMA residual codebook.
#
# Submit from the repo root:
#     sbatch slurm/momask/train_momask_vq_paperstyle.sbatch

set -euo pipefail

HUMANML3D_DIR="${HUMANML3D_DIR:-/mnt/projects/drl4cvb/human-motion-representation/data/HumanML3D/HumanML3D}"

export CANONICAL_H3D_DIR="${CANONICAL_H3D_DIR:-${HOME}/rmg-runs/h3d-canonical/new_joint_vecs}"
export H3D_MEAN="${H3D_MEAN:-${HUMANML3D_DIR}/Mean.npy}"
export H3D_STD="${H3D_STD:-${HUMANML3D_DIR}/Std.npy}"
export RUN_NAME="${RUN_NAME:-momask-canonical-vq-paper-fixed-windows64-150k}"
export VQ_STEPS="${VQ_STEPS:-150000}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-64}"
export MIN_SEQ_LEN="${MIN_SEQ_LEN:-64}"
export VQ_WINDOW_DATASET="${VQ_WINDOW_DATASET:-1}"
export VQ_WINDOW_STRIDE="${VQ_WINDOW_STRIDE:-1}"
export VQ_WINDOW_PRELOAD="${VQ_WINDOW_PRELOAD:-1}"
export BATCH_SIZE="${BATCH_SIZE:-512}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export SAVE_EVERY="${SAVE_EVERY:-5000}"
export LOG_EVERY="${LOG_EVERY:-1000}"
export NO_MOMASK_NORMALIZE="${NO_MOMASK_NORMALIZE:-0}"

export VQ_LR="${VQ_LR:-2e-4}"
export ADAM_BETA2="${ADAM_BETA2:-0.99}"
export VQ_HIDDEN_DIM="${VQ_HIDDEN_DIM:-512}"
export VQ_LATENT_DIM="${VQ_LATENT_DIM:-512}"
export NUM_QUANTIZERS="${NUM_QUANTIZERS:-6}"
export CODEBOOK_SIZE="${CODEBOOK_SIZE:-512}"
export VQ_RES_BLOCKS="${VQ_RES_BLOCKS:-3}"
export DOWNSAMPLE="${DOWNSAMPLE:-4}"
export VQ_COMMITMENT_WEIGHT="${VQ_COMMITMENT_WEIGHT:-0.02}"
export VQ_RECON_LOSS="${VQ_RECON_LOSS:-smooth_l1}"
export VQ_VELOCITY_WEIGHT="${VQ_VELOCITY_WEIGHT:-0.0}"
export VQ_EXPLICIT_WEIGHT="${VQ_EXPLICIT_WEIGHT:-0.5}"
export QUANTIZE_DROPOUT="${QUANTIZE_DROPOUT:-0.2}"
export VQ_USE_EMA="${VQ_USE_EMA:-1}"
export VQ_EMA_DECAY="${VQ_EMA_DECAY:-0.99}"
export VQ_CODEBOOK_SAMPLE_TEMP="${VQ_CODEBOOK_SAMPLE_TEMP:-0.5}"
export VQ_ARCH="${VQ_ARCH:-paper}"
export WARMUP_STEPS="${WARMUP_STEPS:-2000}"
export LR_MILESTONES="${LR_MILESTONES:-150000 250000}"
export LR_GAMMA="${LR_GAMMA:-0.05}"

bash slurm/momask/train_momask_vq_full.sbatch
