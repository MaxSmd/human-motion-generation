#!/bin/bash
#SBATCH --job-name=momask-scene-viz
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --output=slurm/logs/momask-scene-viz-%j.out
#SBATCH --error=slurm/logs/momask-scene-viz-%j.err

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
SAMPLES="${SAMPLES:-0 1 2}"
SEEDS="${SEEDS:-0}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-80}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:-40}"
GENERATION_STEPS="${GENERATION_STEPS:-10}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOPK_FILTER_THRES="${TOPK_FILTER_THRES:-1.0}"
SAMPLE_TOKENS="${SAMPLE_TOKENS:-0}"
REMASK_KEPT_TOKENS="${REMASK_KEPT_TOKENS:-1}"
REFINEMENT_STEPS="${REFINEMENT_STEPS:-200}"
REFINEMENT_LR="${REFINEMENT_LR:-0.02}"
SCENE_WEIGHT="${SCENE_WEIGHT:-50.0}"
SCENE_ROOT_WEIGHT="${SCENE_ROOT_WEIGHT:-0.0}"
LATENT_WEIGHT="${LATENT_WEIGHT:-0.001}"
DYNAMICS_WEIGHT="${DYNAMICS_WEIGHT:-0.05}"
BONE_WEIGHT="${BONE_WEIGHT:-0.1}"
MAX_DELTA_NORM="${MAX_DELTA_NORM:-2.0}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-5.0}"
SCENE_ROOM_WIDTH="${SCENE_ROOM_WIDTH:-6.0}"
SCENE_ROOM_DEPTH="${SCENE_ROOM_DEPTH:-8.0}"
SCENE_ROOM_HEIGHT="${SCENE_ROOM_HEIGHT:-3.0}"
SCENE_SPAWN_X="${SCENE_SPAWN_X:-0.0}"
SCENE_SPAWN_Z="${SCENE_SPAWN_Z:--2.0}"
SCENE_SPAWN_YAW="${SCENE_SPAWN_YAW:-0.0}"
SCENE_PADDING="${SCENE_PADDING:-0.02}"
SCENE_BODY_RADIUS="${SCENE_BODY_RADIUS:-0.05}"
SCENE_BONE_SAMPLES="${SCENE_BONE_SAMPLES:-2}"
SCENE_OBSTACLE_KIND="${SCENE_OBSTACLE_KIND:-box}"
SCENE_OBSTACLE_X="${SCENE_OBSTACLE_X:-0.0}"
SCENE_OBSTACLE_Y="${SCENE_OBSTACLE_Y:-0.5}"
SCENE_OBSTACLE_Z="${SCENE_OBSTACLE_Z:--1.0}"
SCENE_OBSTACLE_WIDTH="${SCENE_OBSTACLE_WIDTH:-1.0}"
SCENE_OBSTACLE_HEIGHT="${SCENE_OBSTACLE_HEIGHT:-1.0}"
SCENE_OBSTACLE_DEPTH="${SCENE_OBSTACLE_DEPTH:-0.4}"
SCENE_OBSTACLE_RADIUS="${SCENE_OBSTACLE_RADIUS:-0.4}"
SCENE_OBSTACLE_YAW="${SCENE_OBSTACLE_YAW:-0.0}"
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
OUT_DIR="${OUT_DIR:-${RUN_DIR}/constraints/viz_scene_ref${REFINEMENT_STEPS}_w${SCENE_WEIGHT}}"

echo "[momask-scene-viz] host=$(hostname)"
echo "[momask-scene-viz] repo=${REPO}"
echo "[momask-scene-viz] ckpt=${CKPT}"
echo "[momask-scene-viz] samples=${SAMPLES} seeds=${SEEDS} out_dir=${OUT_DIR}"
echo "[momask-scene-viz] refinement_steps=${REFINEMENT_STEPS} refinement_lr=${REFINEMENT_LR} scene_weight=${SCENE_WEIGHT}"
echo "[momask-scene-viz] room=${SCENE_ROOM_WIDTH}x${SCENE_ROOM_DEPTH}x${SCENE_ROOM_HEIGHT} obstacle=${SCENE_OBSTACLE_KIND}@(${SCENE_OBSTACLE_X},${SCENE_OBSTACLE_Y},${SCENE_OBSTACLE_Z})"

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
                echo \"[momask-scene-viz] rendering sample=\${sample} seed=\${seed}\"
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
                    --constraint-variant scene \
                    --refinement-steps ${REFINEMENT_STEPS} \
                    --refinement-lr ${REFINEMENT_LR} \
                    --scene-weight ${SCENE_WEIGHT} \
                    --scene-root-weight ${SCENE_ROOT_WEIGHT} \
                    --latent-weight ${LATENT_WEIGHT} \
                    --dynamics-weight ${DYNAMICS_WEIGHT} \
                    --bone-weight ${BONE_WEIGHT} \
                    --max-delta-norm ${MAX_DELTA_NORM} \
                    --grad-clip-norm ${GRAD_CLIP_NORM} \
                    --scene-room-width ${SCENE_ROOM_WIDTH} \
                    --scene-room-depth ${SCENE_ROOM_DEPTH} \
                    --scene-room-height ${SCENE_ROOM_HEIGHT} \
                    --scene-spawn-x ${SCENE_SPAWN_X} \
                    --scene-spawn-z ${SCENE_SPAWN_Z} \
                    --scene-spawn-yaw ${SCENE_SPAWN_YAW} \
                    --scene-padding ${SCENE_PADDING} \
                    --scene-body-radius ${SCENE_BODY_RADIUS} \
                    --scene-bone-samples ${SCENE_BONE_SAMPLES} \
                    --scene-obstacle-kind '${SCENE_OBSTACLE_KIND}' \
                    --scene-obstacle-x ${SCENE_OBSTACLE_X} \
                    --scene-obstacle-y ${SCENE_OBSTACLE_Y} \
                    --scene-obstacle-z ${SCENE_OBSTACLE_Z} \
                    --scene-obstacle-width ${SCENE_OBSTACLE_WIDTH} \
                    --scene-obstacle-height ${SCENE_OBSTACLE_HEIGHT} \
                    --scene-obstacle-depth ${SCENE_OBSTACLE_DEPTH} \
                    --scene-obstacle-radius ${SCENE_OBSTACLE_RADIUS} \
                    --scene-obstacle-yaw ${SCENE_OBSTACLE_YAW} \
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
