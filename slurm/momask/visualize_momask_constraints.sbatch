#!/bin/bash
#SBATCH --job-name=momask-constraint-viz
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --output=slurm/logs/momask-constraint-viz-%j.out
#SBATCH --error=slurm/logs/momask-constraint-viz-%j.err

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
REAL_H3D_DIR="${REAL_H3D_DIR:-${HOME}/rmg-runs/h3d-canonical/new_joint_vecs}"
CKPT="${CKPT:-runs/momask-full-tokens-clip150k/momask_smoke_latest.pt}"
SAMPLE="${SAMPLE:-0}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:-40}"
GENERATION_STEPS="${GENERATION_STEPS:-10}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOPK_FILTER_THRES="${TOPK_FILTER_THRES:-1.0}"
SAMPLE_TOKENS="${SAMPLE_TOKENS:-0}"
REMASK_KEPT_TOKENS="${REMASK_KEPT_TOKENS:-1}"
ANCHOR_STRIDE="${ANCHOR_STRIDE:-20}"
CONSTRAINT_VARIANT="${CONSTRAINT_VARIANT:-both}"
AUTO_SAMPLE_MOVING="${AUTO_SAMPLE_MOVING:-1}"
MIN_ROOT_SPAN="${MIN_ROOT_SPAN:-0.75}"
MAX_FRAMES="${MAX_FRAMES:-80}"
FPS="${FPS:-20}"
SEED="${SEED:-0}"
TEXT_ENCODER="${TEXT_ENCODER:-checkpoint}"
CLIP_MODEL="${CLIP_MODEL:-}"
CLIP_CACHE_DIR="${CLIP_CACHE_DIR:-${HF_CACHE}}"
CLIP_BACKEND="${CLIP_BACKEND:-auto}"
RUN_DIR="$(dirname "${CKPT}")"
if [ "$(basename "${RUN_DIR}")" = "checkpoints" ]; then
    RUN_DIR="$(dirname "${RUN_DIR}")"
fi
OUT_DIR="${OUT_DIR:-${RUN_DIR}/constraints/viz}"

echo "[momask-constraint-viz] host=$(hostname)"
echo "[momask-constraint-viz] repo=${REPO}"
echo "[momask-constraint-viz] ckpt=${CKPT}"
echo "[momask-constraint-viz] real_h3d_dir=${REAL_H3D_DIR:-<packed>}"
echo "[momask-constraint-viz] sample=${SAMPLE} max_seq_len=${MAX_SEQ_LEN:-checkpoint-default} anchor_stride=${ANCHOR_STRIDE} constraint_variant=${CONSTRAINT_VARIANT} auto_sample_moving=${AUTO_SAMPLE_MOVING} min_root_span=${MIN_ROOT_SPAN} out_dir=${OUT_DIR}"
echo "[momask-constraint-viz] guidance=${GUIDANCE_SCALE} temp=${TEMPERATURE} sample_tokens=${SAMPLE_TOKENS} topk=${TOPK_FILTER_THRES} seed=${SEED}"

REAL_H3D_ARG=""
if [ -n "${REAL_H3D_DIR}" ]; then
    REAL_H3D_ARG="--real-h3d-dir ${REAL_H3D_DIR}"
fi
MAX_SEQ_LEN_ARG=""
if [ -n "${MAX_SEQ_LEN}" ]; then
    MAX_SEQ_LEN_ARG="--max-seq-len ${MAX_SEQ_LEN}"
fi

SAMPLING_ARGS=""
if [ "${SAMPLE_TOKENS}" = "1" ]; then
    SAMPLING_ARGS="${SAMPLING_ARGS} --sample-tokens"
else
    SAMPLING_ARGS="${SAMPLING_ARGS} --no-sample-tokens"
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
        python -u -m momask.scripts.visualize_momask_constraints \
            --checkpoint '${CKPT}' \
            --data-root '${DATA_ROOT}' \
            ${REAL_H3D_ARG} \
            --sample ${SAMPLE} \
            --split test \
            ${MAX_SEQ_LEN_ARG} \
            --min-seq-len ${MIN_SEQ_LEN} \
            --generation-steps ${GENERATION_STEPS} \
            --guidance-scale ${GUIDANCE_SCALE} \
            --temperature ${TEMPERATURE} \
            --topk-filter-thres ${TOPK_FILTER_THRES} \
            ${SAMPLING_ARGS} \
            --anchor-stride ${ANCHOR_STRIDE} \
            --constraint-variant ${CONSTRAINT_VARIANT} \
            --auto-sample-moving ${AUTO_SAMPLE_MOVING} \
            --min-root-span ${MIN_ROOT_SPAN} \
            --max-frames ${MAX_FRAMES} \
            --fps ${FPS} \
            --seed ${SEED} \
            --text-encoder '${TEXT_ENCODER}' \
            --clip-model '${CLIP_MODEL}' \
            --clip-cache-dir '${CLIP_CACHE_DIR}' \
            --clip-backend '${CLIP_BACKEND}' \
            --out-dir '${OUT_DIR}'
    "
