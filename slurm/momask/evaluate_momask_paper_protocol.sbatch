#!/bin/bash
#SBATCH --job-name=momask-paper-eval
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --output=slurm/logs/momask-paper-eval-%j.out
#SBATCH --error=slurm/logs/momask-paper-eval-%j.err

set -euo pipefail

export CKPT="${CKPT:-runs/momask-canonical-tokens-releaseparity196-valfidselection-full/momask_best_val_fid.pt}"
export SPLIT="${SPLIT:-test}"
export MAX_CLIPS="${MAX_CLIPS:--1}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-196}"
export MIN_SEQ_LEN="${MIN_SEQ_LEN:-40}"
export VARIANTS="${VARIANTS:-full,base,recon}"
export GENERATION_STEPS="${GENERATION_STEPS:-10}"
export GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
export RESIDUAL_GUIDANCE_SCALE="${RESIDUAL_GUIDANCE_SCALE:-5.0}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOPK_FILTER_THRES="${TOPK_FILTER_THRES:-0.9}"
export SAMPLE="${SAMPLE:-1}"
export REMASK_KEPT_TOKENS="${REMASK_KEPT_TOKENS:-0}"
export PAPER_TRANSFORMER_DATA="${PAPER_TRANSFORMER_DATA:-1}"
export CLIP_BACKEND="${CLIP_BACKEND:-checkpoint}"
export SEED="${SEED:-3407}"
export OUTPUT="${OUTPUT:-$(dirname "${CKPT}")/eval_momask/paper_protocol_test_seed${SEED}.json}"

echo "[momask-paper-eval] M guidance=${GUIDANCE_SCALE} R guidance=${RESIDUAL_GUIDANCE_SCALE}"
echo "[momask-paper-eval] steps=${GENERATION_STEPS} topk=${TOPK_FILTER_THRES} sample=${SAMPLE} remask=${REMASK_KEPT_TOKENS}"

bash slurm/momask/evaluate_momask.sbatch
