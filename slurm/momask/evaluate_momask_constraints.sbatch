#!/bin/bash
#SBATCH --job-name=momask-constraints
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --output=slurm/logs/momask-constraints-%j.out
#SBATCH --error=slurm/logs/momask-constraints-%j.err
#
# Evaluate root-trajectory, joint-position, and bend-angle constraints.
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
CKPT="${CKPT:-runs/momask-canonical-tokens-clip200k/checkpoints/tokens_step_0145000.pt}"
MAX_CLIPS="${MAX_CLIPS:-256}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:-40}"
GENERATION_STEPS="${GENERATION_STEPS:-10}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOPK_FILTER_THRES="${TOPK_FILTER_THRES:-1.0}"
SAMPLE="${SAMPLE:-0}"
REMASK_KEPT_TOKENS="${REMASK_KEPT_TOKENS:-1}"
ANCHOR_STRIDE="${ANCHOR_STRIDE:-20}"
LATENT_VARIANTS="${LATENT_VARIANTS:-joint,angle}"
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
TEXT_ENCODER="${TEXT_ENCODER:-checkpoint}"
CLIP_MODEL="${CLIP_MODEL:-}"
CLIP_CACHE_DIR="${CLIP_CACHE_DIR:-${HF_CACHE}}"
CLIP_BACKEND="${CLIP_BACKEND:-auto}"
EVALUATOR="${EVALUATOR:-real}"
OUTPUT="${OUTPUT:-}"

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
echo "[momask-constraints] steps=${GENERATION_STEPS} guidance=${GUIDANCE_SCALE} anchor_stride=${ANCHOR_STRIDE}"
echo "[momask-constraints] sample=${SAMPLE} topk_filter_thres=${TOPK_FILTER_THRES} remask_kept_tokens=${REMASK_KEPT_TOKENS}"
echo "[momask-constraints] latent_variants=${LATENT_VARIANTS} joint_ids=${JOINT_IDS} joint_space=${JOINT_TARGET_SPACE}"
echo "[momask-constraints] angle_joints=${ANGLE_JOINTS} angle_tolerance_deg=${ANGLE_TOLERANCE_DEG}"
echo "[momask-constraints] refinement_steps=${REFINEMENT_STEPS} refinement_lr=${REFINEMENT_LR} position_weight=${POSITION_WEIGHT} angle_weight=${ANGLE_WEIGHT}"
echo "[momask-constraints] output=${OUTPUT}"

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
        pip install -e . --no-deps --quiet
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
            --temperature ${TEMPERATURE} \
            --topk-filter-thres ${TOPK_FILTER_THRES} \
            ${SAMPLING_ARGS} \
            --anchor-stride ${ANCHOR_STRIDE} \
            --latent-variants '${LATENT_VARIANTS}' \
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
            --text-encoder '${TEXT_ENCODER}' \
            --clip-model '${CLIP_MODEL}' \
            --clip-cache-dir '${CLIP_CACHE_DIR}' \
            --clip-backend '${CLIP_BACKEND}' \
            --evaluator '${EVALUATOR}' \
            --output '${OUTPUT}'
    "
