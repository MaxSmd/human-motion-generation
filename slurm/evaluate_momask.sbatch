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
HF_CACHE="${HF_CACHE:-${HOME}/hf-cache}"
DATA_ROOT="${DATA_ROOT:-/mnt/projects/drl4cvb/human-motion-representation/data/data/humanml3d_packed}"
HUMANML3D_REPO="${HUMANML3D_REPO:-external/HumanML3D}"
HUMANML3D_TEXTS_ZIP="${HUMANML3D_TEXTS_ZIP:-/mnt/projects/drl4cvb/human-motion-representation/data/HumanML3D/HumanML3D/texts.zip}"
REAL_H3D_DIR="${REAL_H3D_DIR:-${HOME}/rmg-runs/h3d-canonical/new_joint_vecs}"
CKPT="${CKPT:-runs/momask-full-tokens-12858/momask_smoke_latest.pt}"
MAX_CLIPS="${MAX_CLIPS:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-80}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:-40}"
VARIANTS="${VARIANTS:-full,base}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
GENERATION_STEPS="${GENERATION_STEPS:-}"
TEXT_ENCODER="${TEXT_ENCODER:-checkpoint}"
CLIP_MODEL="${CLIP_MODEL:-}"
CLIP_CACHE_DIR="${CLIP_CACHE_DIR:-${HF_CACHE}}"
CLIP_BACKEND="${CLIP_BACKEND:-auto}"
EVALUATOR="${EVALUATOR:-real}"
OUTPUT="${OUTPUT:-}"

if [ -z "${OUTPUT}" ]; then
    OUTPUT="$(dirname "${CKPT}")/eval_momask/results_${MAX_CLIPS}_${VARIANTS//,/plus}_canonicalreal.json"
fi

echo "[momask-eval] host=$(hostname)"
echo "[momask-eval] repo=${REPO}"
echo "[momask-eval] image=${IMAGE}"
echo "[momask-eval] hf_cache=${HF_CACHE}"
echo "[momask-eval] data_root=${DATA_ROOT}"
echo "[momask-eval] humanml3d_repo=${HUMANML3D_REPO}"
echo "[momask-eval] humanml3d_texts_zip=${HUMANML3D_TEXTS_ZIP:-<auto>}"
echo "[momask-eval] real_h3d_dir=${REAL_H3D_DIR:-<packed>}"
echo "[momask-eval] ckpt=${CKPT}"
echo "[momask-eval] max_clips=${MAX_CLIPS} batch_size=${BATCH_SIZE} variants=${VARIANTS}"
echo "[momask-eval] generation_steps=${GENERATION_STEPS:-checkpoint-default}"
echo "[momask-eval] text_encoder=${TEXT_ENCODER} clip_model=${CLIP_MODEL:-checkpoint-default} clip_backend=${CLIP_BACKEND}"
echo "[momask-eval] output=${OUTPUT}"

GENERATION_ARGS=""
if [ -n "${GENERATION_STEPS}" ]; then
    case "${GENERATION_STEPS}" in
        ''|*[!0-9]*)
            echo "[momask-eval] ERROR: GENERATION_STEPS must be a positive integer, got '${GENERATION_STEPS}'" >&2
            exit 1
            ;;
    esac
    GENERATION_ARGS="--generation-steps ${GENERATION_STEPS}"
fi

TEXTS_ZIP_ARG=""
if [ -n "${HUMANML3D_TEXTS_ZIP}" ]; then
    TEXTS_ZIP_ARG="--humanml3d-texts-zip ${HUMANML3D_TEXTS_ZIP}"
fi

REAL_H3D_ARG=""
if [ -n "${REAL_H3D_DIR}" ]; then
    REAL_H3D_ARG="--real-h3d-dir ${REAL_H3D_DIR}"
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
        python -m spacy download en_core_web_sm --quiet
        python -u -m rmg.scripts.evaluate_momask \
            --checkpoint '${CKPT}' \
            --data-root '${DATA_ROOT}' \
            --humanml3d-repo '${HUMANML3D_REPO}' \
            ${TEXTS_ZIP_ARG} \
            ${REAL_H3D_ARG} \
            --split test \
            --max-clips ${MAX_CLIPS} \
            --batch-size ${BATCH_SIZE} \
            --max-seq-len ${MAX_SEQ_LEN} \
            --min-seq-len ${MIN_SEQ_LEN} \
            --variants '${VARIANTS}' \
            ${GENERATION_ARGS} \
            --text-encoder '${TEXT_ENCODER}' \
            --clip-model '${CLIP_MODEL}' \
            --clip-cache-dir '${CLIP_CACHE_DIR}' \
            --clip-backend '${CLIP_BACKEND}' \
            --guidance-scale ${GUIDANCE_SCALE} \
            --temperature ${TEMPERATURE} \
            --evaluator '${EVALUATOR}' \
            --output '${OUTPUT}'
    "
