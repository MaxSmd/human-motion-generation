#!/bin/bash
#SBATCH --job-name=momask-tok-full
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --output=slurm/logs/momask-tokens-full-%j.out
#SBATCH --error=slurm/logs/momask-tokens-full-%j.err
#
# Full-dataset MoMask token-transformer training from a pretrained VQ-VAE.
#
# Submit from the repo root:
#     sbatch --partition=24g slurm/momask/train_momask_tokens_full.sbatch
#
# Useful overrides:
#     VQ_CKPT=runs/momask-full-vq-raw-12668/momask_vq_latest.pt sbatch --partition=24g slurm/momask/train_momask_tokens_full.sbatch
#     TOKEN_STEPS=50000 TOKEN_BATCH_SIZE=128 RUN_NAME=momask-full-tokens-v2 sbatch --partition=24g slurm/momask/train_momask_tokens_full.sbatch
#     TOKEN_CKPT=runs/momask-full-tokens-12767/checkpoints/tokens_latest_train.pt sbatch --partition=24g slurm/momask/train_momask_tokens_full.sbatch

set -euo pipefail
mkdir -p slurm/logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
fi

IMAGE="${IMAGE:-${HOME}/rmg-momask.sqsh}"
HF_CACHE="${HF_CACHE:-${HOME}/hf-cache}"
DATA_ROOT="${DATA_ROOT:-/mnt/projects/drl4cvb/human-motion-representation/data/data/humanml3d_packed}"
CANONICAL_H3D_DIR="${CANONICAL_H3D_DIR:-}"
VQ_CKPT="${VQ_CKPT:-runs/momask-full-vq-raw-12668/momask_vq_latest.pt}"
TOKEN_CKPT="${TOKEN_CKPT:-}"
MASKED_CKPT="${MASKED_CKPT:-}"
RUN_NAME="${RUN_NAME:-momask-full-tokens-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/${RUN_NAME}}"
TOKEN_STEPS="${TOKEN_STEPS:-100000}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
LOG_EVERY="${LOG_EVERY:-1000}"
EVAL_BATCHES="${EVAL_BATCHES:-32}"
MAX_CLIPS="${MAX_CLIPS:-999999}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TOKEN_BATCH_SIZE="${TOKEN_BATCH_SIZE:-128}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-80}"
TEXT_ENCODER="${TEXT_ENCODER:-random}"
CLIP_MODEL="${CLIP_MODEL:-ViT-B/32}"
CLIP_CACHE_DIR="${CLIP_CACHE_DIR:-${HF_CACHE}}"
CLIP_BACKEND="${CLIP_BACKEND:-auto}"
TOKEN_LR="${TOKEN_LR:-1e-4}"
TRANSFORMER_HIDDEN_DIM="${TRANSFORMER_HIDDEN_DIM:-512}"
TRANSFORMER_DEPTH="${TRANSFORMER_DEPTH:-8}"
TRANSFORMER_HEADS="${TRANSFORMER_HEADS:-8}"
TRANSFORMER_FFN_DIM="${TRANSFORMER_FFN_DIM:-2048}"
TRANSFORMER_DROPOUT="${TRANSFORMER_DROPOUT:-0.0}"
BASE_COND_DROP="${BASE_COND_DROP:-0.1}"
RESIDUAL_COND_DROP="${RESIDUAL_COND_DROP:-0.2}"
RESIDUAL_ARCH="${RESIDUAL_ARCH:-simple}"
RESIDUAL_SHARE_WEIGHT="${RESIDUAL_SHARE_WEIGHT:-0}"
FREEZE_MASKED="${FREEZE_MASKED:-0}"
BASE_FULL_MASK_PROB="${BASE_FULL_MASK_PROB:-0.3}"
GENERATION_STEPS="${GENERATION_STEPS:-10}"
CACHE_TOKEN_DEVICE="${CACHE_TOKEN_DEVICE:-cuda}"

TOKEN_CKPT_ARG=""
if [ -n "${TOKEN_CKPT}" ]; then
    TOKEN_CKPT_ARG="--load-token-checkpoint '${TOKEN_CKPT}'"
fi

MASKED_CKPT_ARG=""
if [ -n "${MASKED_CKPT}" ]; then
    MASKED_CKPT_ARG="--load-masked-checkpoint '${MASKED_CKPT}'"
fi

CANONICAL_ARG=""
if [ -n "${CANONICAL_H3D_DIR}" ]; then
    CANONICAL_ARG="--canonical-h3d-dir '${CANONICAL_H3D_DIR}'"
fi

echo "[momask-tokens] host=$(hostname)"
echo "[momask-tokens] repo=${REPO}"
echo "[momask-tokens] image=${IMAGE}"
echo "[momask-tokens] hf_cache=${HF_CACHE}"
echo "[momask-tokens] data_root=${DATA_ROOT}"
echo "[momask-tokens] canonical_h3d_dir=${CANONICAL_H3D_DIR:-<checkpoint-or-packed-derived>}"
echo "[momask-tokens] vq_ckpt=${VQ_CKPT}"
echo "[momask-tokens] token_ckpt=${TOKEN_CKPT:-<none>}"
echo "[momask-tokens] masked_ckpt=${MASKED_CKPT:-<none>}"
echo "[momask-tokens] output_dir=${OUTPUT_DIR}"
echo "[momask-tokens] token_steps=${TOKEN_STEPS} max_clips=${MAX_CLIPS} batch_size=${BATCH_SIZE} token_batch_size=${TOKEN_BATCH_SIZE} max_seq_len=${MAX_SEQ_LEN}"
echo "[momask-tokens] text_encoder=${TEXT_ENCODER} clip_model=${CLIP_MODEL} clip_backend=${CLIP_BACKEND}"
echo "[momask-tokens] transformer hidden=${TRANSFORMER_HIDDEN_DIM} depth=${TRANSFORMER_DEPTH} heads=${TRANSFORMER_HEADS} ffn=${TRANSFORMER_FFN_DIM} dropout=${TRANSFORMER_DROPOUT}"
echo "[momask-tokens] lr=${TOKEN_LR} residual_arch=${RESIDUAL_ARCH} residual_share_weight=${RESIDUAL_SHARE_WEIGHT} freeze_masked=${FREEZE_MASKED} base_full_mask_prob=${BASE_FULL_MASK_PROB} generation_steps=${GENERATION_STEPS}"

RESIDUAL_SHARE_ARG=""
if [ "${RESIDUAL_SHARE_WEIGHT}" = "1" ]; then
    RESIDUAL_SHARE_ARG="--residual-share-weight"
fi
FREEZE_MASKED_ARG=""
if [ "${FREEZE_MASKED}" = "1" ]; then
    FREEZE_MASKED_ARG="--freeze-masked-transformer"
fi

enroot start \
    --root \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --mount "${REPO}:${REPO}" \
    --env "RMG_DATA_ROOT=${DATA_ROOT}" \
    --env "HF_HOME=${HF_CACHE}" \
    --env "PYTHONPATH=${REPO}/src" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        cd ${REPO}
        pip install -e . --no-deps --quiet
        python -u -m momask.scripts.train_momask_smoke \\
            --data-root '${DATA_ROOT}' \\
            ${CANONICAL_ARG} \\
            --max-clips ${MAX_CLIPS} \\
            --batch-size ${BATCH_SIZE} \\
            --token-batch-size ${TOKEN_BATCH_SIZE} \\
            --max-seq-len ${MAX_SEQ_LEN} \\
            --text-encoder ${TEXT_ENCODER} \\
            --clip-model '${CLIP_MODEL}' \\
            --clip-cache-dir '${CLIP_CACHE_DIR}' \\
            --clip-backend ${CLIP_BACKEND} \\
            --load-vq-checkpoint '${VQ_CKPT}' \\
            ${TOKEN_CKPT_ARG} \\
            ${MASKED_CKPT_ARG} \\
            --vq-steps 0 \\
            --token-steps ${TOKEN_STEPS} \\
            --lr ${TOKEN_LR} \\
            --text-dim 64 \\
            --transformer-hidden-dim ${TRANSFORMER_HIDDEN_DIM} \\
            --transformer-depth ${TRANSFORMER_DEPTH} \\
            --transformer-heads ${TRANSFORMER_HEADS} \\
            --transformer-ffn-dim ${TRANSFORMER_FFN_DIM} \\
            --transformer-dropout ${TRANSFORMER_DROPOUT} \\
            --base-cond-drop ${BASE_COND_DROP} \\
            --residual-cond-drop ${RESIDUAL_COND_DROP} \\
            --residual-arch ${RESIDUAL_ARCH} \\
            ${RESIDUAL_SHARE_ARG} \\
            ${FREEZE_MASKED_ARG} \\
            --base-full-mask-prob ${BASE_FULL_MASK_PROB} \\
            --generation-steps ${GENERATION_STEPS} \\
            --cache-token-device ${CACHE_TOKEN_DEVICE} \\
            --log-every ${LOG_EVERY} \\
            --eval-batches ${EVAL_BATCHES} \\
            --save-every ${SAVE_EVERY} \\
            --output-dir '${OUTPUT_DIR}'
    "
