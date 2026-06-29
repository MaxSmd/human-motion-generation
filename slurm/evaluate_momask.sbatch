#!/bin/bash
#SBATCH --job-name=momask-eval
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --output=slurm/logs/momask-eval-%j.out
#SBATCH --error=slurm/logs/momask-eval-%j.err
#
# Submit from repo root:
#   MAX_CLIPS=512 VARIANTS=full,base sbatch slurm/evaluate_momask.sbatch
#
# Full diagnostic:
#   MAX_CLIPS=-1 VARIANTS=full,base,recon sbatch slurm/evaluate_momask.sbatch

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
CKPT="${CKPT:-runs/momask-full-tokens-12858/momask_smoke_latest.pt}"
MAX_CLIPS="${MAX_CLIPS:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-80}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:-40}"
VARIANTS="${VARIANTS:-full,base}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
EVALUATOR="${EVALUATOR:-real}"
OUTPUT="${OUTPUT:-}"

if [ -z "${OUTPUT}" ]; then
    OUTPUT="$(dirname "${CKPT}")/eval_momask/results_${MAX_CLIPS}_${VARIANTS//,/plus}.json"
fi

echo "[momask-eval] host=$(hostname)"
echo "[momask-eval] repo=${REPO}"
echo "[momask-eval] image=${IMAGE}"
echo "[momask-eval] data_root=${DATA_ROOT}"
echo "[momask-eval] ckpt=${CKPT}"
echo "[momask-eval] max_clips=${MAX_CLIPS} batch_size=${BATCH_SIZE} variants=${VARIANTS}"
echo "[momask-eval] output=${OUTPUT}"

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
        python -m spacy download en_core_web_sm --quiet
        python -u -m rmg.scripts.evaluate_momask \
            --checkpoint '${CKPT}' \
            --data-root '${DATA_ROOT}' \
            --split test \
            --max-clips ${MAX_CLIPS} \
            --batch-size ${BATCH_SIZE} \
            --max-seq-len ${MAX_SEQ_LEN} \
            --min-seq-len ${MIN_SEQ_LEN} \
            --variants '${VARIANTS}' \
            --guidance-scale ${GUIDANCE_SCALE} \
            --temperature ${TEMPERATURE} \
            --evaluator '${EVALUATOR}' \
            --output '${OUTPUT}'
    "
