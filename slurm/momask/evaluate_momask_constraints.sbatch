#!/bin/bash
#SBATCH --job-name=momask-constraints
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --output=slurm/logs/momask-constraints-%j.out
#SBATCH --error=slurm/logs/momask-constraints-%j.err
#
# Evaluate inference-time root-trajectory constraints for a MoMask checkpoint.
#
# Submit from repo root, for example:
#   CKPT=runs/momask-full-tokens-clip150k/momask_smoke_latest.pt \
#   MAX_CLIPS=256 GUIDANCE_SCALE=4 ANCHOR_STRIDE=20 \
#   sbatch slurm/momask/evaluate_momask_constraints.sbatch

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
HUMANML3D_REPO="${HUMANML3D_REPO:-external/HumanML3D}"
HUMANML3D_TEXTS_ZIP="${HUMANML3D_TEXTS_ZIP:-/mnt/projects/drl4cvb/human-motion-representation/data/HumanML3D/HumanML3D/texts.zip}"
REAL_H3D_DIR="${REAL_H3D_DIR:-${HOME}/rmg-runs/h3d-canonical/new_joint_vecs}"
CKPT="${CKPT:-runs/momask-full-tokens-clip150k/momask_smoke_latest.pt}"
MAX_CLIPS="${MAX_CLIPS:-256}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-80}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:-40}"
GENERATION_STEPS="${GENERATION_STEPS:-10}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOPK_FILTER_THRES="${TOPK_FILTER_THRES:-1.0}"
SAMPLE="${SAMPLE:-0}"
REMASK_KEPT_TOKENS="${REMASK_KEPT_TOKENS:-1}"
ANCHOR_STRIDE="${ANCHOR_STRIDE:-20}"
TEXT_ENCODER="${TEXT_ENCODER:-checkpoint}"
CLIP_MODEL="${CLIP_MODEL:-}"
CLIP_CACHE_DIR="${CLIP_CACHE_DIR:-${HF_CACHE}}"
CLIP_BACKEND="${CLIP_BACKEND:-auto}"
EVALUATOR="${EVALUATOR:-real}"
OUTPUT="${OUTPUT:-}"

if [ -z "${OUTPUT}" ]; then
    OUTPUT="$(dirname "${CKPT}")/constraints/trajectory_${MAX_CLIPS}_stride${ANCHOR_STRIDE}_g${GUIDANCE_SCALE}_t${TEMPERATURE}_sample${SAMPLE}_canonicalreal.json"
fi

echo "[momask-constraints] host=$(hostname)"
echo "[momask-constraints] repo=${REPO}"
echo "[momask-constraints] image=${IMAGE}"
echo "[momask-constraints] hf_cache=${HF_CACHE}"
echo "[momask-constraints] data_root=${DATA_ROOT}"
echo "[momask-constraints] humanml3d_repo=${HUMANML3D_REPO}"
echo "[momask-constraints] humanml3d_texts_zip=${HUMANML3D_TEXTS_ZIP:-<auto>}"
echo "[momask-constraints] real_h3d_dir=${REAL_H3D_DIR:-<packed>}"
echo "[momask-constraints] ckpt=${CKPT}"
echo "[momask-constraints] max_clips=${MAX_CLIPS} batch_size=${BATCH_SIZE}"
echo "[momask-constraints] steps=${GENERATION_STEPS} guidance=${GUIDANCE_SCALE} anchor_stride=${ANCHOR_STRIDE}"
echo "[momask-constraints] sample=${SAMPLE} topk_filter_thres=${TOPK_FILTER_THRES} remask_kept_tokens=${REMASK_KEPT_TOKENS}"
echo "[momask-constraints] output=${OUTPUT}"

TEXTS_ZIP_ARG=""
if [ -n "${HUMANML3D_TEXTS_ZIP}" ]; then
    TEXTS_ZIP_ARG="--humanml3d-texts-zip ${HUMANML3D_TEXTS_ZIP}"
fi

REAL_H3D_ARG=""
if [ -n "${REAL_H3D_DIR}" ]; then
    REAL_H3D_ARG="--real-h3d-dir ${REAL_H3D_DIR}"
fi

SAMPLING_ARGS=""
if [ "${SAMPLE}" = "1" ]; then
    SAMPLING_ARGS="${SAMPLING_ARGS} --sample"
else
    SAMPLING_ARGS="${SAMPLING_ARGS} --no-sample"
fi
if [ "${REMASK_KEPT_TOKENS}" = "1" ]; then
    SAMPLING_ARGS="${SAMPLING_ARGS} --remask-kept-tokens"
else
    SAMPLING_ARGS="${SAMPLING_ARGS} --no-remask-kept-tokens"
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
        python -u -m momask.scripts.evaluate_momask_constraints \
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
            --generation-steps ${GENERATION_STEPS} \
            --guidance-scale ${GUIDANCE_SCALE} \
            --temperature ${TEMPERATURE} \
            --topk-filter-thres ${TOPK_FILTER_THRES} \
            ${SAMPLING_ARGS} \
            --anchor-stride ${ANCHOR_STRIDE} \
            --text-encoder '${TEXT_ENCODER}' \
            --clip-model '${CLIP_MODEL}' \
            --clip-cache-dir '${CLIP_CACHE_DIR}' \
            --clip-backend '${CLIP_BACKEND}' \
            --evaluator '${EVALUATOR}' \
            --output '${OUTPUT}'
    "
