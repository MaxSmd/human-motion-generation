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
HUMANML3D_TEXTS_ZIP="${HUMANML3D_TEXTS_ZIP:-}"
HUMANML3D_SPLIT_DIR="${HUMANML3D_SPLIT_DIR:-}"
VQ_CKPT="${VQ_CKPT:-runs/momask-full-vq-raw-12668/momask_vq_latest.pt}"
TOKEN_CKPT="${TOKEN_CKPT:-}"
MASKED_CKPT="${MASKED_CKPT:-}"
TOKEN_FAIL_IF_OUTPUT_EXISTS="${TOKEN_FAIL_IF_OUTPUT_EXISTS:-0}"
ASSEMBLE_AFTER_TRAIN="${ASSEMBLE_AFTER_TRAIN:-0}"
ASSEMBLE_MASKED_CKPT="${ASSEMBLE_MASKED_CKPT:-}"
ASSEMBLE_OUTPUT="${ASSEMBLE_OUTPUT:-}"
RUN_NAME="${RUN_NAME:-momask-full-tokens-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/${RUN_NAME}}"
TOKEN_STEPS="${TOKEN_STEPS:-100000}"
TOKEN_EPOCHS="${TOKEN_EPOCHS:-0}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
LOG_EVERY="${LOG_EVERY:-1000}"
EVAL_BATCHES="${EVAL_BATCHES:-32}"
VALIDATE_EVERY="${VALIDATE_EVERY:-0}"
VAL_EVAL_BATCHES="${VAL_EVAL_BATCHES:-32}"
MAX_CLIPS="${MAX_CLIPS:-999999}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TOKEN_BATCH_SIZE="${TOKEN_BATCH_SIZE:-128}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-80}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:-20}"
TEXT_ENCODER="${TEXT_ENCODER:-random}"
CLIP_MODEL="${CLIP_MODEL:-ViT-B/32}"
CLIP_CACHE_DIR="${CLIP_CACHE_DIR:-${HF_CACHE}}"
CLIP_BACKEND="${CLIP_BACKEND:-auto}"
CLIP_L2_NORMALIZE="${CLIP_L2_NORMALIZE:-1}"
TOKEN_LR="${TOKEN_LR:-1e-4}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
TOKEN_GRAD_CLIP="${TOKEN_GRAD_CLIP:-1.0}"
TRANSFORMER_ARCH="${TRANSFORMER_ARCH:-legacy}"
TRANSFORMER_HIDDEN_DIM="${TRANSFORMER_HIDDEN_DIM:-512}"
TRANSFORMER_DEPTH="${TRANSFORMER_DEPTH:-8}"
TRANSFORMER_HEADS="${TRANSFORMER_HEADS:-8}"
TRANSFORMER_FFN_DIM="${TRANSFORMER_FFN_DIM:-2048}"
TRANSFORMER_DROPOUT="${TRANSFORMER_DROPOUT:-0.0}"
BASE_COND_DROP="${BASE_COND_DROP:-0.1}"
RESIDUAL_COND_DROP="${RESIDUAL_COND_DROP:-0.2}"
RESIDUAL_ARCH="${RESIDUAL_ARCH:-simple}"
RESIDUAL_SHARE_WEIGHT="${RESIDUAL_SHARE_WEIGHT:-0}"
RESIDUAL_PREDICT_PAD="${RESIDUAL_PREDICT_PAD:-0}"
OFFICIAL_MASK_SCHEDULE="${OFFICIAL_MASK_SCHEDULE:-0}"
FREEZE_MASKED="${FREEZE_MASKED:-0}"
TOKEN_STAGE="${TOKEN_STAGE:-joint}"
BASE_FULL_MASK_PROB="${BASE_FULL_MASK_PROB:-0.3}"
GENERATION_STEPS="${GENERATION_STEPS:-10}"
CACHE_TOKEN_DEVICE="${CACHE_TOKEN_DEVICE:-cuda}"
LIVE_TOKEN_CROPS="${LIVE_TOKEN_CROPS:-0}"
PAPER_TRANSFORMER_DATA="${PAPER_TRANSFORMER_DATA:-0}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
LR_MILESTONES="${LR_MILESTONES:-}"
LR_GAMMA="${LR_GAMMA:-0.1}"
SEED="${SEED:-0}"

if [ "${TOKEN_FAIL_IF_OUTPUT_EXISTS}" = "1" ] && [ -z "${TOKEN_CKPT}" ] \
    && [ -d "${OUTPUT_DIR}" ]; then
    echo "[momask-tokens] ERROR: fresh-training output directory already exists: ${OUTPUT_DIR}" >&2
    echo "[momask-tokens] choose a new RUN_NAME, or set TOKEN_CKPT to resume explicitly" >&2
    exit 2
fi

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
echo "[momask-tokens] humanml3d_texts_zip=${HUMANML3D_TEXTS_ZIP:-<unused>}"
echo "[momask-tokens] humanml3d_split_dir=${HUMANML3D_SPLIT_DIR:-<packed-splits.json>}"
echo "[momask-tokens] vq_ckpt=${VQ_CKPT}"
echo "[momask-tokens] token_ckpt=${TOKEN_CKPT:-<none>}"
echo "[momask-tokens] masked_ckpt=${MASKED_CKPT:-<none>}"
echo "[momask-tokens] fail_if_output_exists=${TOKEN_FAIL_IF_OUTPUT_EXISTS}"
echo "[momask-tokens] assemble_after_train=${ASSEMBLE_AFTER_TRAIN}"
echo "[momask-tokens] assemble_masked_ckpt=${ASSEMBLE_MASKED_CKPT:-<none>}"
echo "[momask-tokens] assemble_output=${ASSEMBLE_OUTPUT:-<none>}"
echo "[momask-tokens] output_dir=${OUTPUT_DIR}"
echo "[momask-tokens] token_steps=${TOKEN_STEPS} token_epochs=${TOKEN_EPOCHS} max_clips=${MAX_CLIPS} batch_size=${BATCH_SIZE} token_batch_size=${TOKEN_BATCH_SIZE} num_workers=${NUM_WORKERS} max_seq_len=${MAX_SEQ_LEN} min_seq_len=${MIN_SEQ_LEN}"
echo "[momask-tokens] validate_every=${VALIDATE_EVERY} val_eval_batches=${VAL_EVAL_BATCHES}"
echo "[momask-tokens] text_encoder=${TEXT_ENCODER} clip_model=${CLIP_MODEL} clip_backend=${CLIP_BACKEND} clip_l2_normalize=${CLIP_L2_NORMALIZE}"
echo "[momask-tokens] transformer arch=${TRANSFORMER_ARCH} hidden=${TRANSFORMER_HIDDEN_DIM} depth=${TRANSFORMER_DEPTH} heads=${TRANSFORMER_HEADS} ffn=${TRANSFORMER_FFN_DIM} dropout=${TRANSFORMER_DROPOUT}"
echo "[momask-tokens] token_stage=${TOKEN_STAGE} lr=${TOKEN_LR} adam_beta2=${ADAM_BETA2} weight_decay=${WEIGHT_DECAY} grad_clip=${TOKEN_GRAD_CLIP} residual_arch=${RESIDUAL_ARCH} residual_share_weight=${RESIDUAL_SHARE_WEIGHT} residual_predict_pad=${RESIDUAL_PREDICT_PAD} official_mask_schedule=${OFFICIAL_MASK_SCHEDULE} freeze_masked=${FREEZE_MASKED} live_token_crops=${LIVE_TOKEN_CROPS} paper_data=${PAPER_TRANSFORMER_DATA} base_full_mask_prob=${BASE_FULL_MASK_PROB} generation_steps=${GENERATION_STEPS}"
echo "[momask-tokens] warmup_steps=${WARMUP_STEPS} lr_milestones=${LR_MILESTONES:-<none>} lr_gamma=${LR_GAMMA}"
echo "[momask-tokens] seed=${SEED}"

RESIDUAL_SHARE_ARG=""
if [ "${RESIDUAL_SHARE_WEIGHT}" = "1" ]; then
    RESIDUAL_SHARE_ARG="--residual-share-weight"
fi
RESIDUAL_PREDICT_PAD_ARG=""
if [ "${RESIDUAL_PREDICT_PAD}" = "1" ]; then
    RESIDUAL_PREDICT_PAD_ARG="--residual-predict-pad"
fi
OFFICIAL_MASK_SCHEDULE_ARG=""
if [ "${OFFICIAL_MASK_SCHEDULE}" = "1" ]; then
    OFFICIAL_MASK_SCHEDULE_ARG="--official-mask-schedule"
fi
FREEZE_MASKED_ARG=""
if [ "${FREEZE_MASKED}" = "1" ]; then
    FREEZE_MASKED_ARG="--freeze-masked-transformer"
fi
LIVE_TOKEN_CROPS_ARG=""
if [ "${LIVE_TOKEN_CROPS}" = "1" ]; then
    LIVE_TOKEN_CROPS_ARG="--live-token-crops"
fi
PAPER_DATA_ARG=""
if [ "${PAPER_TRANSFORMER_DATA}" = "1" ]; then
    if [ -z "${HUMANML3D_TEXTS_ZIP}" ]; then
        echo "[momask-tokens] ERROR: PAPER_TRANSFORMER_DATA=1 requires HUMANML3D_TEXTS_ZIP" >&2
        exit 2
    fi
    PAPER_DATA_ARG="--paper-transformer-data --humanml3d-texts-zip '${HUMANML3D_TEXTS_ZIP}'"
    if [ -n "${HUMANML3D_SPLIT_DIR}" ]; then
        PAPER_DATA_ARG="${PAPER_DATA_ARG} --humanml3d-split-dir '${HUMANML3D_SPLIT_DIR}'"
    fi
fi
CLIP_NORMALIZE_ARG="--clip-l2-normalize"
if [ "${CLIP_L2_NORMALIZE}" = "0" ]; then
    CLIP_NORMALIZE_ARG="--no-clip-l2-normalize"
fi
LR_MILESTONES_ARG=""
if [ -n "${LR_MILESTONES}" ]; then
    LR_MILESTONES_ARG="--lr-milestones ${LR_MILESTONES}"
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
        python -u -m momask.scripts.train_momask \\
            --data-root '${DATA_ROOT}' \\
            ${CANONICAL_ARG} \\
            --max-clips ${MAX_CLIPS} \\
            --batch-size ${BATCH_SIZE} \\
            --num-workers ${NUM_WORKERS} \\
            --token-batch-size ${TOKEN_BATCH_SIZE} \\
            --max-seq-len ${MAX_SEQ_LEN} \\
            --min-seq-len ${MIN_SEQ_LEN} \\
            ${PAPER_DATA_ARG} \\
            --text-encoder ${TEXT_ENCODER} \\
            --clip-model '${CLIP_MODEL}' \\
            --clip-cache-dir '${CLIP_CACHE_DIR}' \\
            --clip-backend ${CLIP_BACKEND} \\
            ${CLIP_NORMALIZE_ARG} \\
            --load-vq-checkpoint '${VQ_CKPT}' \\
            ${TOKEN_CKPT_ARG} \\
            ${MASKED_CKPT_ARG} \\
            --vq-steps 0 \\
            --token-steps ${TOKEN_STEPS} \\
            --token-epochs ${TOKEN_EPOCHS} \\
            --lr ${TOKEN_LR} \\
            --adam-beta2 ${ADAM_BETA2} \\
            --weight-decay ${WEIGHT_DECAY} \\
            --token-grad-clip ${TOKEN_GRAD_CLIP} \\
            --warmup-steps ${WARMUP_STEPS} \\
            ${LR_MILESTONES_ARG} \\
            --lr-gamma ${LR_GAMMA} \\
            --seed ${SEED} \\
            --text-dim 64 \\
            --transformer-arch ${TRANSFORMER_ARCH} \\
            --transformer-hidden-dim ${TRANSFORMER_HIDDEN_DIM} \\
            --transformer-depth ${TRANSFORMER_DEPTH} \\
            --transformer-heads ${TRANSFORMER_HEADS} \\
            --transformer-ffn-dim ${TRANSFORMER_FFN_DIM} \\
            --transformer-dropout ${TRANSFORMER_DROPOUT} \\
            --base-cond-drop ${BASE_COND_DROP} \\
            --residual-cond-drop ${RESIDUAL_COND_DROP} \\
            --residual-arch ${RESIDUAL_ARCH} \\
            ${RESIDUAL_SHARE_ARG} \\
            ${RESIDUAL_PREDICT_PAD_ARG} \\
            ${OFFICIAL_MASK_SCHEDULE_ARG} \\
            ${FREEZE_MASKED_ARG} \\
            --token-stage ${TOKEN_STAGE} \\
            ${LIVE_TOKEN_CROPS_ARG} \\
            --base-full-mask-prob ${BASE_FULL_MASK_PROB} \\
            --generation-steps ${GENERATION_STEPS} \\
            --cache-token-device ${CACHE_TOKEN_DEVICE} \\
            --log-every ${LOG_EVERY} \\
            --eval-batches ${EVAL_BATCHES} \\
            --validate-every ${VALIDATE_EVERY} \\
            --val-eval-batches ${VAL_EVAL_BATCHES} \\
            --save-every ${SAVE_EVERY} \\
            --output-dir '${OUTPUT_DIR}'

        if [ '${ASSEMBLE_AFTER_TRAIN}' = '1' ]; then
            if [ -z '${ASSEMBLE_MASKED_CKPT}' ] || [ -z '${ASSEMBLE_OUTPUT}' ]; then
                echo '[momask-tokens] ERROR: assembly requires both ASSEMBLE_MASKED_CKPT and ASSEMBLE_OUTPUT' >&2
                exit 2
            fi
            python -u -m momask.scripts.assemble_token_checkpoints \\
                --masked-checkpoint '${ASSEMBLE_MASKED_CKPT}' \\
                --residual-checkpoint '${OUTPUT_DIR}/checkpoints/tokens_best_val.pt' \\
                --output '${ASSEMBLE_OUTPUT}'
        fi
    "
