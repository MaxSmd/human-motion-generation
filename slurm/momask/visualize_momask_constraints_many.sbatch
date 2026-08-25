#!/bin/bash
#SBATCH --job-name=momask-constraint-viz-many
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --output=slurm/logs/momask-constraint-viz-many-%j.out
#SBATCH --error=slurm/logs/momask-constraint-viz-many-%j.err

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
MOMASK_RUNS_ROOT="${MOMASK_RUNS_ROOT:-/mnt/projects/drl4cvb/human-motion-representation/runs}"
CKPT="${CKPT:-${MOMASK_RUNS_ROOT}/momask-canonical-tokens-vq150k-maskfix196-valfidselection-full/momask_best_val_fid.pt}"
SAMPLES="${SAMPLES:-0 500 1000 1500 2000}"
SEEDS="${SEEDS:-0 1 2}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-196}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:-40}"
GENERATION_STEPS="${GENERATION_STEPS:-10}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
RESIDUAL_GUIDANCE_SCALE="${RESIDUAL_GUIDANCE_SCALE:-5.0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOPK_FILTER_THRES="${TOPK_FILTER_THRES:-0.9}"
SAMPLE_TOKENS="${SAMPLE_TOKENS:-1}"
REMASK_KEPT_TOKENS="${REMASK_KEPT_TOKENS:-0}"
ANCHOR_STRIDE="${ANCHOR_STRIDE:-20}"
CONSTRAINT_VARIANT="${CONSTRAINT_VARIANT:-both}"
AUTO_SAMPLE_MOVING="${AUTO_SAMPLE_MOVING:-1}"
MIN_ROOT_SPAN="${MIN_ROOT_SPAN:-2.0}"
MAX_FRAMES="${MAX_FRAMES:-196}"
FPS="${FPS:-20}"
TEXT_ENCODER="${TEXT_ENCODER:-checkpoint}"
CLIP_MODEL="${CLIP_MODEL:-}"
CLIP_CACHE_DIR="${CLIP_CACHE_DIR:-${HF_CACHE}}"
CLIP_BACKEND="${CLIP_BACKEND:-checkpoint}"
SKIP_EDITABLE_INSTALL="${SKIP_EDITABLE_INSTALL:-1}"

RUN_DIR="$(dirname "${CKPT}")"
if [ "$(basename "${RUN_DIR}")" = "checkpoints" ]; then
    RUN_DIR="$(dirname "${RUN_DIR}")"
fi
OUT_DIR="${OUT_DIR:-${RUN_DIR}/constraints/viz_many}"

echo "[momask-constraint-viz-many] host=$(hostname)"
echo "[momask-constraint-viz-many] repo=${REPO}"
echo "[momask-constraint-viz-many] ckpt=${CKPT}"
echo "[momask-constraint-viz-many] samples=${SAMPLES}"
echo "[momask-constraint-viz-many] seeds=${SEEDS}"
echo "[momask-constraint-viz-many] max_seq_len=${MAX_SEQ_LEN:-checkpoint-default}"
echo "[momask-constraint-viz-many] guidance=${GUIDANCE_SCALE} residual_guidance=${RESIDUAL_GUIDANCE_SCALE} temp=${TEMPERATURE} sample_tokens=${SAMPLE_TOKENS} topk=${TOPK_FILTER_THRES}"
echo "[momask-constraint-viz-many] out_dir=${OUT_DIR}"

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
        mkdir -p '${OUT_DIR}'
        if [ '${SKIP_EDITABLE_INSTALL}' != '1' ]; then
            pip install -e . --no-deps --quiet
        fi
        for sample in ${SAMPLES}; do
            for seed in ${SEEDS}; do
                echo \"[momask-constraint-viz-many] rendering sample=\${sample} seed=\${seed}\"
                python -u -m momask.scripts.visualize_momask_constraints \
                    --checkpoint '${CKPT}' \
                    --data-root '${DATA_ROOT}' \
                    ${REAL_H3D_ARG} \
                    --sample \"\${sample}\" \
                    --split test \
                    ${MAX_SEQ_LEN_ARG} \
                    --min-seq-len ${MIN_SEQ_LEN} \
                    --generation-steps ${GENERATION_STEPS} \
                    --guidance-scale ${GUIDANCE_SCALE} \
                    --residual-guidance-scale ${RESIDUAL_GUIDANCE_SCALE} \
                    --temperature ${TEMPERATURE} \
                    --topk-filter-thres ${TOPK_FILTER_THRES} \
                    ${SAMPLING_ARGS} \
                    --anchor-stride ${ANCHOR_STRIDE} \
                    --constraint-variant ${CONSTRAINT_VARIANT} \
                    --auto-sample-moving ${AUTO_SAMPLE_MOVING} \
                    --min-root-span ${MIN_ROOT_SPAN} \
                    --max-frames ${MAX_FRAMES} \
                    --fps ${FPS} \
                    --seed \"\${seed}\" \
                    --text-encoder '${TEXT_ENCODER}' \
                    --clip-model '${CLIP_MODEL}' \
                    --clip-cache-dir '${CLIP_CACHE_DIR}' \
                    --clip-backend '${CLIP_BACKEND}' \
                    --out-dir '${OUT_DIR}'
            done
        done
    "
