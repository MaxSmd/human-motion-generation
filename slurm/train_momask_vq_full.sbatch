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
#     sbatch --partition=24g slurm/train_momask_vq_full.sbatch
#
# Useful overrides:
#     VQ_STEPS=150000 RUN_NAME=momask-full-vq-raw-v2 sbatch --partition=24g slurm/train_momask_vq_full.sbatch
#     IMAGE=/mnt/home/$USER/rmg-momask.sqsh sbatch --partition=24g slurm/train_momask_vq_full.sbatch

set -euo pipefail
mkdir -p slurm/logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(dirname "${SCRIPT_DIR}")}"
fi

IMAGE="${IMAGE:-${HOME}/rmg-momask.sqsh}"
DATA_ROOT="${DATA_ROOT:-/mnt/projects/drl4cvb/human-motion-representation/data/data/humanml3d_packed}"
CANONICAL_H3D_DIR="${CANONICAL_H3D_DIR:-}"
RUN_NAME="${RUN_NAME:-momask-full-vq-raw-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/${RUN_NAME}}"
VQ_STEPS="${VQ_STEPS:-100000}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
LOG_EVERY="${LOG_EVERY:-1000}"
EVAL_BATCHES="${EVAL_BATCHES:-32}"
MAX_CLIPS="${MAX_CLIPS:-999999}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-80}"
NO_MOMASK_NORMALIZE="${NO_MOMASK_NORMALIZE:-1}"

echo "[momask-vq] host=$(hostname)"
echo "[momask-vq] repo=${REPO}"
echo "[momask-vq] image=${IMAGE}"
echo "[momask-vq] data_root=${DATA_ROOT}"
echo "[momask-vq] canonical_h3d_dir=${CANONICAL_H3D_DIR:-<packed-derived>}"
echo "[momask-vq] output_dir=${OUTPUT_DIR}"
echo "[momask-vq] vq_steps=${VQ_STEPS} max_clips=${MAX_CLIPS} batch_size=${BATCH_SIZE} max_seq_len=${MAX_SEQ_LEN} no_momask_normalize=${NO_MOMASK_NORMALIZE}"

CANONICAL_ARG=""
if [ -n "${CANONICAL_H3D_DIR}" ]; then
    CANONICAL_ARG="--canonical-h3d-dir '${CANONICAL_H3D_DIR}'"
fi
NORMALIZE_ARG=""
if [ "${NO_MOMASK_NORMALIZE}" = "1" ]; then
    NORMALIZE_ARG="--no-momask-normalize"
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
        python -u -m rmg.scripts.train_momask_smoke \\
            --data-root '${DATA_ROOT}' \\
            ${CANONICAL_ARG} \\
            --max-clips ${MAX_CLIPS} \\
            --batch-size ${BATCH_SIZE} \\
            --max-seq-len ${MAX_SEQ_LEN} \\
            --vq-only \\
            --vq-steps ${VQ_STEPS} \\
            --lr 1e-4 \\
            --vq-hidden-dim 384 \\
            --vq-latent-dim 256 \\
            --num-quantizers 6 \\
            --codebook-size 512 \\
            --vq-res-blocks 4 \\
            --vq-commitment-weight 0.1 \\
            --vq-velocity-weight 0.0 \\
            --quantize-dropout 0.0 \\
            ${NORMALIZE_ARG} \\
            --log-every ${LOG_EVERY} \\
            --eval-batches ${EVAL_BATCHES} \\
            --save-every ${SAVE_EVERY} \\
            --output-dir '${OUTPUT_DIR}'
    "
