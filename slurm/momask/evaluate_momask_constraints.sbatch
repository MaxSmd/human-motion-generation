#!/bin/bash
#SBATCH --job-name=momask-constraints
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --output=slurm/logs/momask-constraints-%j.out
#SBATCH --error=slurm/logs/momask-constraints-%j.err
#
# Evaluate root-trajectory, pose, body-fixed, and room-geometry constraints.
#
# Submit from repo root, for example:
#   MAX_CLIPS=256 LATENT_VARIANTS=joint,angle \
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
MODEL_INPUT_SOURCE="${MODEL_INPUT_SOURCE:-canonical}"
MOMASK_RUNS_ROOT="${MOMASK_RUNS_ROOT:-/mnt/projects/drl4cvb/human-motion-representation/runs}"
CKPT="${CKPT:-${MOMASK_RUNS_ROOT}/momask-canonical-tokens-vq150k-maskfix196-valfidselection-full/momask_best_val_fid.pt}"
MAX_CLIPS="${MAX_CLIPS:-256}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-196}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:-40}"
GENERATION_STEPS="${GENERATION_STEPS:-10}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
RESIDUAL_GUIDANCE_SCALE="${RESIDUAL_GUIDANCE_SCALE:-5.0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOPK_FILTER_THRES="${TOPK_FILTER_THRES:-0.9}"
SAMPLE="${SAMPLE:-1}"
REMASK_KEPT_TOKENS="${REMASK_KEPT_TOKENS:-0}"
ANCHOR_STRIDE="${ANCHOR_STRIDE:-20}"
LATENT_VARIANTS="${LATENT_VARIANTS:-joint,angle}"
JOINT_IDS="${JOINT_IDS:-20,21}"
JOINT_TARGET_SPACE="${JOINT_TARGET_SPACE:-root-relative}"
ANGLE_JOINTS="${ANGLE_JOINTS:-4,5,18,19}"
ANGLE_TOLERANCE_DEG="${ANGLE_TOLERANCE_DEG:-5.0}"
BODY_FIXED_JOINT_IDS="${BODY_FIXED_JOINT_IDS:-17,19,21}"
BODY_REFERENCE_FRAME="${BODY_REFERENCE_FRAME:-0}"
REFINEMENT_STEPS="${REFINEMENT_STEPS:-50}"
REFINEMENT_LR="${REFINEMENT_LR:-0.01}"
POSITION_WEIGHT="${POSITION_WEIGHT:-1.0}"
TORSO_RELATIVE_WEIGHT="${TORSO_RELATIVE_WEIGHT:-1.0}"
PARENT_RELATIVE_WEIGHT="${PARENT_RELATIVE_WEIGHT:-${TORSO_RELATIVE_WEIGHT}}"
ANGLE_WEIGHT="${ANGLE_WEIGHT:-1.0}"
LATENT_WEIGHT="${LATENT_WEIGHT:-0.01}"
DYNAMICS_WEIGHT="${DYNAMICS_WEIGHT:-0.1}"
ROOT_WEIGHT="${ROOT_WEIGHT:-0.1}"
BONE_WEIGHT="${BONE_WEIGHT:-0.1}"
MAX_DELTA_NORM="${MAX_DELTA_NORM:-1.0}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
SCENE_WEIGHT="${SCENE_WEIGHT:-10.0}"
SCENE_PEAK_WEIGHT="${SCENE_PEAK_WEIGHT:-50.0}"
SCENE_PEAK_TEMPERATURE="${SCENE_PEAK_TEMPERATURE:-0.01}"
SCENE_PENALTY_GROWTH="${SCENE_PENALTY_GROWTH:-2.0}"
SCENE_PENALTY_INTERVAL="${SCENE_PENALTY_INTERVAL:-50}"
SCENE_MAX_PENALTY_SCALE="${SCENE_MAX_PENALTY_SCALE:-16.0}"
SCENE_VIOLATION_TOLERANCE="${SCENE_VIOLATION_TOLERANCE:-0.005}"
SCENE_REGULARIZATION_FLOOR="${SCENE_REGULARIZATION_FLOOR:-0.1}"
SCENE_ROOT_WEIGHT="${SCENE_ROOT_WEIGHT:-0.0}"
SCENE_ROOM_WIDTH="${SCENE_ROOM_WIDTH:-6.0}"
SCENE_ROOM_DEPTH="${SCENE_ROOM_DEPTH:-8.0}"
SCENE_ROOM_HEIGHT="${SCENE_ROOM_HEIGHT:-3.0}"
SCENE_SPAWN_X="${SCENE_SPAWN_X:-0.0}"
SCENE_SPAWN_Z="${SCENE_SPAWN_Z:--2.0}"
SCENE_SPAWN_YAW="${SCENE_SPAWN_YAW:-0.0}"
SCENE_PADDING="${SCENE_PADDING:-0.02}"
SCENE_BODY_RADIUS="${SCENE_BODY_RADIUS:-0.05}"
SCENE_BONE_SAMPLES="${SCENE_BONE_SAMPLES:-2}"
SCENE_SWEPT_SAMPLES="${SCENE_SWEPT_SAMPLES:-3}"
SCENE_OBSTACLE_KIND="${SCENE_OBSTACLE_KIND:-box}"
SCENE_OBSTACLE_X="${SCENE_OBSTACLE_X:-0.0}"
SCENE_OBSTACLE_Y="${SCENE_OBSTACLE_Y:-0.5}"
SCENE_OBSTACLE_Z="${SCENE_OBSTACLE_Z:--1.0}"
SCENE_OBSTACLE_WIDTH="${SCENE_OBSTACLE_WIDTH:-1.0}"
SCENE_OBSTACLE_HEIGHT="${SCENE_OBSTACLE_HEIGHT:-1.0}"
SCENE_OBSTACLE_DEPTH="${SCENE_OBSTACLE_DEPTH:-0.4}"
SCENE_OBSTACLE_RADIUS="${SCENE_OBSTACLE_RADIUS:-0.4}"
SCENE_OBSTACLE_YAW="${SCENE_OBSTACLE_YAW:-0.0}"
TEXT_ENCODER="${TEXT_ENCODER:-checkpoint}"
CLIP_MODEL="${CLIP_MODEL:-}"
CLIP_CACHE_DIR="${CLIP_CACHE_DIR:-${HF_CACHE}}"
CLIP_BACKEND="${CLIP_BACKEND:-checkpoint}"
EVALUATOR="${EVALUATOR:-real}"
OUTPUT="${OUTPUT:-}"
SKIP_EDITABLE_INSTALL="${SKIP_EDITABLE_INSTALL:-1}"

if [ -z "${OUTPUT}" ]; then
    CKPT_DIR="$(dirname "${CKPT}")"
    if [ "$(basename "${CKPT_DIR}")" = "checkpoints" ]; then
        RUN_DIR="$(dirname "${CKPT_DIR}")"
    else
        RUN_DIR="${CKPT_DIR}"
    fi
    VARIANT_TAG="${LATENT_VARIANTS//,/plus}"
    OUTPUT="${RUN_DIR}/constraints/${VARIANT_TAG}_${MAX_CLIPS}_stride${ANCHOR_STRIDE}_g${GUIDANCE_SCALE}_t${TEMPERATURE}_ref${REFINEMENT_STEPS}_sample${SAMPLE}.json"
fi

echo "[momask-constraints] host=$(hostname)"
echo "[momask-constraints] repo=${REPO}"
echo "[momask-constraints] image=${IMAGE}"
echo "[momask-constraints] hf_cache=${HF_CACHE}"
echo "[momask-constraints] data_root=${DATA_ROOT}"
echo "[momask-constraints] humanml3d_repo=${HUMANML3D_REPO}"
echo "[momask-constraints] humanml3d_texts_zip=${HUMANML3D_TEXTS_ZIP:-<auto>}"
echo "[momask-constraints] real_h3d_dir=${REAL_H3D_DIR:-<packed>}"
echo "[momask-constraints] model_input_source=${MODEL_INPUT_SOURCE}"
echo "[momask-constraints] ckpt=${CKPT}"
echo "[momask-constraints] max_clips=${MAX_CLIPS} batch_size=${BATCH_SIZE} max_seq_len=${MAX_SEQ_LEN:-checkpoint-default}"
echo "[momask-constraints] steps=${GENERATION_STEPS} guidance=${GUIDANCE_SCALE} residual_guidance=${RESIDUAL_GUIDANCE_SCALE} anchor_stride=${ANCHOR_STRIDE}"
echo "[momask-constraints] sample=${SAMPLE} topk_filter_thres=${TOPK_FILTER_THRES} remask_kept_tokens=${REMASK_KEPT_TOKENS}"
echo "[momask-constraints] latent_variants=${LATENT_VARIANTS} joint_ids=${JOINT_IDS} joint_space=${JOINT_TARGET_SPACE}"
echo "[momask-constraints] angle_joints=${ANGLE_JOINTS} angle_tolerance_deg=${ANGLE_TOLERANCE_DEG}"
echo "[momask-constraints] body_fixed_joint_ids=${BODY_FIXED_JOINT_IDS} body_reference_frame=${BODY_REFERENCE_FRAME}"
echo "[momask-constraints] refinement_steps=${REFINEMENT_STEPS} refinement_lr=${REFINEMENT_LR} position_weight=${POSITION_WEIGHT} parent_relative_weight=${PARENT_RELATIVE_WEIGHT} angle_weight=${ANGLE_WEIGHT}"
echo "[momask-constraints] scene room=${SCENE_ROOM_WIDTH}x${SCENE_ROOM_DEPTH}x${SCENE_ROOM_HEIGHT} spawn=${SCENE_SPAWN_X},${SCENE_SPAWN_Z},${SCENE_SPAWN_YAW} obstacle=${SCENE_OBSTACLE_KIND}@${SCENE_OBSTACLE_X},${SCENE_OBSTACLE_Y},${SCENE_OBSTACLE_Z}"
echo "[momask-constraints] scene weights collision=${SCENE_WEIGHT} peak=${SCENE_PEAK_WEIGHT} swept_samples=${SCENE_SWEPT_SAMPLES}"
echo "[momask-constraints] scene adaptive temp=${SCENE_PEAK_TEMPERATURE} growth=${SCENE_PENALTY_GROWTH} interval=${SCENE_PENALTY_INTERVAL} max_scale=${SCENE_MAX_PENALTY_SCALE} tolerance=${SCENE_VIOLATION_TOLERANCE} reg_floor=${SCENE_REGULARIZATION_FLOOR}"
echo "[momask-constraints] output=${OUTPUT}"
echo "[momask-constraints] skip_editable_install=${SKIP_EDITABLE_INSTALL}"

TEXTS_ZIP_ARG=""
if [ -n "${HUMANML3D_TEXTS_ZIP}" ]; then
    TEXTS_ZIP_ARG="--humanml3d-texts-zip ${HUMANML3D_TEXTS_ZIP}"
fi

REAL_H3D_ARG=""
if [ -n "${REAL_H3D_DIR}" ]; then
    REAL_H3D_ARG="--real-h3d-dir ${REAL_H3D_DIR}"
fi
MAX_SEQ_LEN_ARG=""
if [ -n "${MAX_SEQ_LEN}" ]; then
    MAX_SEQ_LEN_ARG="--max-seq-len ${MAX_SEQ_LEN}"
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
        if [ '${SKIP_EDITABLE_INSTALL}' != '1' ]; then
            pip install -e . --no-deps --quiet
        fi
        python -m spacy download en_core_web_sm --quiet
        python -u -m momask.scripts.evaluate_momask_constraints \
            --checkpoint '${CKPT}' \
            --data-root '${DATA_ROOT}' \
            --humanml3d-repo '${HUMANML3D_REPO}' \
            ${TEXTS_ZIP_ARG} \
            ${REAL_H3D_ARG} \
            --model-input-source '${MODEL_INPUT_SOURCE}' \
            --split test \
            --max-clips ${MAX_CLIPS} \
            --batch-size ${BATCH_SIZE} \
            ${MAX_SEQ_LEN_ARG} \
            --min-seq-len ${MIN_SEQ_LEN} \
            --generation-steps ${GENERATION_STEPS} \
            --guidance-scale ${GUIDANCE_SCALE} \
            --residual-guidance-scale ${RESIDUAL_GUIDANCE_SCALE} \
            --temperature ${TEMPERATURE} \
            --topk-filter-thres ${TOPK_FILTER_THRES} \
            ${SAMPLING_ARGS} \
            --anchor-stride ${ANCHOR_STRIDE} \
            --latent-variants '${LATENT_VARIANTS}' \
            --joint-ids '${JOINT_IDS}' \
            --joint-target-space '${JOINT_TARGET_SPACE}' \
            --angle-joints '${ANGLE_JOINTS}' \
            --angle-tolerance-deg ${ANGLE_TOLERANCE_DEG} \
            --body-fixed-joint-ids '${BODY_FIXED_JOINT_IDS}' \
            --body-reference-frame ${BODY_REFERENCE_FRAME} \
            --refinement-steps ${REFINEMENT_STEPS} \
            --refinement-lr ${REFINEMENT_LR} \
            --position-weight ${POSITION_WEIGHT} \
            --parent-relative-weight ${PARENT_RELATIVE_WEIGHT} \
            --angle-weight ${ANGLE_WEIGHT} \
            --latent-weight ${LATENT_WEIGHT} \
            --dynamics-weight ${DYNAMICS_WEIGHT} \
            --root-weight ${ROOT_WEIGHT} \
            --bone-weight ${BONE_WEIGHT} \
            --max-delta-norm ${MAX_DELTA_NORM} \
            --grad-clip-norm ${GRAD_CLIP_NORM} \
            --scene-weight ${SCENE_WEIGHT} \
            --scene-peak-weight ${SCENE_PEAK_WEIGHT} \
            --scene-peak-temperature ${SCENE_PEAK_TEMPERATURE} \
            --scene-penalty-growth ${SCENE_PENALTY_GROWTH} \
            --scene-penalty-interval ${SCENE_PENALTY_INTERVAL} \
            --scene-max-penalty-scale ${SCENE_MAX_PENALTY_SCALE} \
            --scene-violation-tolerance ${SCENE_VIOLATION_TOLERANCE} \
            --scene-regularization-floor ${SCENE_REGULARIZATION_FLOOR} \
            --scene-root-weight ${SCENE_ROOT_WEIGHT} \
            --scene-room-width ${SCENE_ROOM_WIDTH} \
            --scene-room-depth ${SCENE_ROOM_DEPTH} \
            --scene-room-height ${SCENE_ROOM_HEIGHT} \
            --scene-spawn-x ${SCENE_SPAWN_X} \
            --scene-spawn-z ${SCENE_SPAWN_Z} \
            --scene-spawn-yaw ${SCENE_SPAWN_YAW} \
            --scene-padding ${SCENE_PADDING} \
            --scene-body-radius ${SCENE_BODY_RADIUS} \
            --scene-bone-samples ${SCENE_BONE_SAMPLES} \
            --scene-swept-samples ${SCENE_SWEPT_SAMPLES} \
            --scene-obstacle-kind ${SCENE_OBSTACLE_KIND} \
            --scene-obstacle-x ${SCENE_OBSTACLE_X} \
            --scene-obstacle-y ${SCENE_OBSTACLE_Y} \
            --scene-obstacle-z ${SCENE_OBSTACLE_Z} \
            --scene-obstacle-width ${SCENE_OBSTACLE_WIDTH} \
            --scene-obstacle-height ${SCENE_OBSTACLE_HEIGHT} \
            --scene-obstacle-depth ${SCENE_OBSTACLE_DEPTH} \
            --scene-obstacle-radius ${SCENE_OBSTACLE_RADIUS} \
            --scene-obstacle-yaw ${SCENE_OBSTACLE_YAW} \
            --text-encoder '${TEXT_ENCODER}' \
            --clip-model '${CLIP_MODEL}' \
            --clip-cache-dir '${CLIP_CACHE_DIR}' \
            --clip-backend '${CLIP_BACKEND}' \
            --evaluator '${EVALUATOR}' \
            --output '${OUTPUT}'
    "
