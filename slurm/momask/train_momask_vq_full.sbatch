#!/bin/bash
#SBATCH --job-name=momask-vq-full
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --output=slurm/logs/momask-vq-full-%j.out
#SBATCH --error=slurm/logs/momask-vq-full-%j.err
#
# Full-dataset MoMask RVQ-VAE training run.
#
# Submit from the repo root:
#     sbatch --partition=24g slurm/momask/train_momask_vq_full.sbatch
#
# Useful overrides:
#     VQ_STEPS=150000 RUN_NAME=momask-full-vq-raw-v2 sbatch --partition=24g slurm/momask/train_momask_vq_full.sbatch
#     IMAGE=/mnt/home/$USER/rmg-momask.sqsh sbatch --partition=24g slurm/momask/train_momask_vq_full.sbatch

set -euo pipefail
mkdir -p slurm/logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
fi

IMAGE="${IMAGE:-${HOME}/rmg-momask.sqsh}"
DATA_ROOT="${DATA_ROOT:-/mnt/projects/drl4cvb/human-motion-representation/data/data/humanml3d_packed}"
CANONICAL_H3D_DIR="${CANONICAL_H3D_DIR:-}"
H3D_MEAN="${H3D_MEAN:-}"
H3D_STD="${H3D_STD:-}"
LOAD_VQ_CKPT="${LOAD_VQ_CKPT:-}"
RUN_NAME="${RUN_NAME:-momask-full-vq-raw-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/${RUN_NAME}}"
VQ_STEPS="${VQ_STEPS:-100000}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
LOG_EVERY="${LOG_EVERY:-1000}"
EVAL_BATCHES="${EVAL_BATCHES:-32}"
MAX_CLIPS="${MAX_CLIPS:-999999}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-80}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:-20}"
NO_MOMASK_NORMALIZE="${NO_MOMASK_NORMALIZE:-1}"
VQ_LR="${VQ_LR:-1e-4}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"
VQ_HIDDEN_DIM="${VQ_HIDDEN_DIM:-384}"
VQ_LATENT_DIM="${VQ_LATENT_DIM:-256}"
NUM_QUANTIZERS="${NUM_QUANTIZERS:-6}"
CODEBOOK_SIZE="${CODEBOOK_SIZE:-512}"
VQ_RES_BLOCKS="${VQ_RES_BLOCKS:-4}"
DOWNSAMPLE="${DOWNSAMPLE:-1}"
VQ_COMMITMENT_WEIGHT="${VQ_COMMITMENT_WEIGHT:-0.1}"
VQ_RECON_LOSS="${VQ_RECON_LOSS:-l1}"
VQ_VELOCITY_WEIGHT="${VQ_VELOCITY_WEIGHT:-0.0}"
VQ_EXPLICIT_WEIGHT="${VQ_EXPLICIT_WEIGHT:-0.0}"
QUANTIZE_DROPOUT="${QUANTIZE_DROPOUT:-0.0}"
VQ_USE_EMA="${VQ_USE_EMA:-0}"
VQ_EMA_DECAY="${VQ_EMA_DECAY:-0.99}"
VQ_CODEBOOK_SAMPLE_TEMP="${VQ_CODEBOOK_SAMPLE_TEMP:-0.0}"

echo "[momask-vq] host=$(hostname)"
echo "[momask-vq] repo=${REPO}"
echo "[momask-vq] image=${IMAGE}"
echo "[momask-vq] data_root=${DATA_ROOT}"
echo "[momask-vq] canonical_h3d_dir=${CANONICAL_H3D_DIR:-<packed-derived>}"
echo "[momask-vq] h3d_mean=${H3D_MEAN:-<dataset-scan-or-identity>} h3d_std=${H3D_STD:-<dataset-scan-or-identity>}"
echo "[momask-vq] load_vq_ckpt=${LOAD_VQ_CKPT:-<none>}"
echo "[momask-vq] output_dir=${OUTPUT_DIR}"
echo "[momask-vq] vq_steps=${VQ_STEPS} max_clips=${MAX_CLIPS} batch_size=${BATCH_SIZE} num_workers=${NUM_WORKERS} max_seq_len=${MAX_SEQ_LEN} min_seq_len=${MIN_SEQ_LEN} no_momask_normalize=${NO_MOMASK_NORMALIZE}"
echo "[momask-vq] lr=${VQ_LR} adam_beta2=${ADAM_BETA2} hidden=${VQ_HIDDEN_DIM} latent=${VQ_LATENT_DIM} quantizers=${NUM_QUANTIZERS} codebook=${CODEBOOK_SIZE} res_blocks=${VQ_RES_BLOCKS} downsample=${DOWNSAMPLE}"
echo "[momask-vq] recon_loss=${VQ_RECON_LOSS} commit=${VQ_COMMITMENT_WEIGHT} vel=${VQ_VELOCITY_WEIGHT} explicit=${VQ_EXPLICIT_WEIGHT} qdrop=${QUANTIZE_DROPOUT} ema=${VQ_USE_EMA} ema_decay=${VQ_EMA_DECAY} codebook_temp=${VQ_CODEBOOK_SAMPLE_TEMP}"

CANONICAL_ARG=""
if [ -n "${CANONICAL_H3D_DIR}" ]; then
    CANONICAL_ARG="--canonical-h3d-dir '${CANONICAL_H3D_DIR}'"
fi
NORMALIZE_ARG=""
if [ "${NO_MOMASK_NORMALIZE}" = "1" ]; then
    NORMALIZE_ARG="--no-momask-normalize"
fi
H3D_STATS_ARG=""
if [ -n "${H3D_MEAN}" ] && [ -n "${H3D_STD}" ]; then
    H3D_STATS_ARG="--h3d-mean '${H3D_MEAN}' --h3d-std '${H3D_STD}'"
fi
VQ_EMA_ARG=""
if [ "${VQ_USE_EMA}" = "1" ]; then
    VQ_EMA_ARG="--vq-use-ema"
fi
LOAD_VQ_ARG=""
if [ -n "${LOAD_VQ_CKPT}" ]; then
    LOAD_VQ_ARG="--load-vq-checkpoint '${LOAD_VQ_CKPT}'"
fi

enroot start \
    --root \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --mount "${REPO}:${REPO}" \
    --env "RMG_DATA_ROOT=${DATA_ROOT}" \
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
            --num-workers ${NUM_WORKERS} \\
            --max-seq-len ${MAX_SEQ_LEN} \\
            --min-seq-len ${MIN_SEQ_LEN} \\
            ${H3D_STATS_ARG} \\
            --vq-only \\
            ${LOAD_VQ_ARG} \\
            --vq-steps ${VQ_STEPS} \\
            --lr ${VQ_LR} \\
            --adam-beta2 ${ADAM_BETA2} \\
            --vq-hidden-dim ${VQ_HIDDEN_DIM} \\
            --vq-latent-dim ${VQ_LATENT_DIM} \\
            --num-quantizers ${NUM_QUANTIZERS} \\
            --codebook-size ${CODEBOOK_SIZE} \\
            --vq-res-blocks ${VQ_RES_BLOCKS} \\
            --downsample ${DOWNSAMPLE} \\
            --vq-commitment-weight ${VQ_COMMITMENT_WEIGHT} \\
            --vq-recon-loss ${VQ_RECON_LOSS} \\
            --vq-velocity-weight ${VQ_VELOCITY_WEIGHT} \\
            --vq-explicit-weight ${VQ_EXPLICIT_WEIGHT} \\
            --quantize-dropout ${QUANTIZE_DROPOUT} \\
            ${VQ_EMA_ARG} \\
            --vq-ema-decay ${VQ_EMA_DECAY} \\
            --vq-codebook-sample-temp ${VQ_CODEBOOK_SAMPLE_TEMP} \\
            ${NORMALIZE_ARG} \\
            --log-every ${LOG_EVERY} \\
            --eval-batches ${EVAL_BATCHES} \\
            --save-every ${SAVE_EVERY} \\
            --output-dir '${OUTPUT_DIR}'
    "
