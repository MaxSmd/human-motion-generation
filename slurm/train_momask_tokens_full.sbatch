#!/bin/bash
#SBATCH --job-name=momask-tok-full
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=23:55:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm/logs/momask-tokens-full-%j.out
#SBATCH --error=slurm/logs/momask-tokens-full-%j.err
#
# Full-dataset MoMask token-transformer training from a pretrained VQ-VAE.
#
# Submit from the repo root:
#     sbatch --partition=24g slurm/train_momask_tokens_full.sbatch
#
# Useful overrides:
#     VQ_CKPT=runs/momask-full-vq-raw-12668/momask_vq_latest.pt sbatch --partition=24g slurm/train_momask_tokens_full.sbatch
#     TOKEN_STEPS=50000 RUN_NAME=momask-full-tokens-v2 sbatch --partition=24g slurm/train_momask_tokens_full.sbatch

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
VQ_CKPT="${VQ_CKPT:-runs/momask-full-vq-raw-12668/momask_vq_latest.pt}"
RUN_NAME="${RUN_NAME:-momask-full-tokens-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/${RUN_NAME}}"
TOKEN_STEPS="${TOKEN_STEPS:-100000}"
LOG_EVERY="${LOG_EVERY:-1000}"
EVAL_BATCHES="${EVAL_BATCHES:-32}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-80}"

echo "[momask-tokens] host=$(hostname)"
echo "[momask-tokens] repo=${REPO}"
echo "[momask-tokens] image=${IMAGE}"
echo "[momask-tokens] data_root=${DATA_ROOT}"
echo "[momask-tokens] vq_ckpt=${VQ_CKPT}"
echo "[momask-tokens] output_dir=${OUTPUT_DIR}"
echo "[momask-tokens] token_steps=${TOKEN_STEPS} batch_size=${BATCH_SIZE} max_seq_len=${MAX_SEQ_LEN}"

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
        python -u scripts/train_momask_smoke.py \\
            --data-root '${DATA_ROOT}' \\
            --max-clips 999999 \\
            --batch-size ${BATCH_SIZE} \\
            --max-seq-len ${MAX_SEQ_LEN} \\
            --load-vq-checkpoint '${VQ_CKPT}' \\
            --vq-steps 0 \\
            --token-steps ${TOKEN_STEPS} \\
            --lr 1e-4 \\
            --text-dim 64 \\
            --transformer-hidden-dim 512 \\
            --transformer-depth 8 \\
            --transformer-heads 8 \\
            --transformer-ffn-dim 2048 \\
            --transformer-dropout 0.0 \\
            --base-cond-drop 0.0 \\
            --residual-cond-drop 0.0 \\
            --base-full-mask-prob 1.0 \\
            --generation-steps 1 \\
            --log-every ${LOG_EVERY} \\
            --eval-batches ${EVAL_BATCHES} \\
            --output-dir '${OUTPUT_DIR}'
    "
