#!/bin/bash
#SBATCH --job-name=momask-joint-angle-viz
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --output=slurm/logs/momask-joint-angle-viz-%j.out
#SBATCH --error=slurm/logs/momask-joint-angle-viz-%j.err

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
CKPT="${CKPT:-runs/momask-canonical-tokens-clip200k/checkpoints/tokens_step_0145000.pt}"
SAMPLES="${SAMPLES:-0 500 1000 1500 2000}"
SEEDS="${SEEDS:-0}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-80}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:-40}"
GENERATION_STEPS="${GENERATION_STEPS:-10}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOPK_FILTER_THRES="${TOPK_FILTER_THRES:-1.0}"
SAMPLE_TOKENS="${SAMPLE_TOKENS:-0}"
REMASK_KEPT_TOKENS="${REMASK_KEPT_TOKENS:-1}"
ANCHOR_STRIDE="${ANCHOR_STRIDE:-20}"
JOINT_IDS="${JOINT_IDS:-20,21}"
JOINT_TARGET_SPACE="${JOINT_TARGET_SPACE:-root-relative}"
ANGLE_JOINTS="${ANGLE_JOINTS:-4,5,18,19}"
ANGLE_TOLERANCE_DEG="${ANGLE_TOLERANCE_DEG:-5.0}"
REFINEMENT_STEPS="${REFINEMENT_STEPS:-50}"
REFINEMENT_LR="${REFINEMENT_LR:-0.01}"
POSITION_WEIGHT="${POSITION_WEIGHT:-1.0}"
ANGLE_WEIGHT="${ANGLE_WEIGHT:-1.0}"
LATENT_WEIGHT="${LATENT_WEIGHT:-0.01}"
DYNAMICS_WEIGHT="${DYNAMICS_WEIGHT:-0.1}"
ROOT_WEIGHT="${ROOT_WEIGHT:-0.1}"
BONE_WEIGHT="${BONE_WEIGHT:-0.1}"
MAX_DELTA_NORM="${MAX_DELTA_NORM:-1.0}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
MAX_FRAMES="${MAX_FRAMES:-80}"
FPS="${FPS:-20}"
TEXT_ENCODER="${TEXT_ENCODER:-checkpoint}"
CLIP_MODEL="${CLIP_MODEL:-}"
CLIP_CACHE_DIR="${CLIP_CACHE_DIR:-${HF_CACHE}}"
CLIP_BACKEND="${CLIP_BACKEND:-auto}"

RUN_DIR="$(dirname "${CKPT}")"
if [ "$(basename "${RUN_DIR}")" = "checkpoints" ]; then
    RUN_DIR="$(dirname "${RUN_DIR}")"
fi
OUT_DIR="${OUT_DIR:-${RUN_DIR}/constraints/viz_joint_angle_step145k}"

echo "[momask-joint-angle-viz] host=$(hostname)"
echo "[momask-joint-angle-viz] repo=${REPO}"
echo "[momask-joint-angle-viz] ckpt=${CKPT}"
echo "[momask-joint-angle-viz] samples=${SAMPLES} seeds=${SEEDS}"
echo "[momask-joint-angle-viz] joints=${JOINT_IDS} target_space=${JOINT_TARGET_SPACE}"
echo "[momask-joint-angle-viz] angle_centers=${ANGLE_JOINTS} tolerance=${ANGLE_TOLERANCE_DEG}deg"
echo "[momask-joint-angle-viz] refinement_steps=${REFINEMENT_STEPS} refinement_lr=${REFINEMENT_LR}"
echo "[momask-joint-angle-viz] out_dir=${OUT_DIR}"

SAMPLING_ARGS="--no-sample-tokens"
if [ "${SAMPLE_TOKENS}" = "1" ]; then
    SAMPLING_ARGS="--sample-tokens"
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
        cd '${REPO}'
        mkdir -p '${OUT_DIR}'
        pip install -e . --no-deps --quiet
        for sample in ${SAMPLES}; do
            for seed in ${SEEDS}; do
                echo \"[momask-joint-angle-viz] rendering sample=\${sample} seed=\${seed}\"
                python -u -m momask.scripts.visualize_momask_constraints \
                    --checkpoint '${CKPT}' \
                    --data-root '${DATA_ROOT}' \
                    --real-h3d-dir '${REAL_H3D_DIR}' \
                    --model-input-source canonical \
                    --sample \"\${sample}\" \
                    --split test \
                    --max-seq-len ${MAX_SEQ_LEN} \
                    --min-seq-len ${MIN_SEQ_LEN} \
                    --generation-steps ${GENERATION_STEPS} \
                    --guidance-scale ${GUIDANCE_SCALE} \
                    --temperature ${TEMPERATURE} \
                    --topk-filter-thres ${TOPK_FILTER_THRES} \
                    ${SAMPLING_ARGS} \
                    --anchor-stride ${ANCHOR_STRIDE} \
                    --constraint-variant joint-angle \
                    --joint-ids '${JOINT_IDS}' \
                    --joint-target-space '${JOINT_TARGET_SPACE}' \
                    --angle-joints '${ANGLE_JOINTS}' \
                    --angle-tolerance-deg ${ANGLE_TOLERANCE_DEG} \
                    --refinement-steps ${REFINEMENT_STEPS} \
                    --refinement-lr ${REFINEMENT_LR} \
                    --position-weight ${POSITION_WEIGHT} \
                    --angle-weight ${ANGLE_WEIGHT} \
                    --latent-weight ${LATENT_WEIGHT} \
                    --dynamics-weight ${DYNAMICS_WEIGHT} \
                    --root-weight ${ROOT_WEIGHT} \
                    --bone-weight ${BONE_WEIGHT} \
                    --max-delta-norm ${MAX_DELTA_NORM} \
                    --grad-clip-norm ${GRAD_CLIP_NORM} \
                    --auto-sample-moving 0 \
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
