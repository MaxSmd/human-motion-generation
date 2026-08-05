#!/bin/bash
#SBATCH --job-name=mardm-ctrl-eval
#SBATCH --partition=12g
#SBATCH --nodelist=frankfurt,hamburg,kassel,leipzig
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --output=slurm/logs/mardm-ctrl-eval-%j.out
#SBATCH --error=slurm/logs/mardm-ctrl-eval-%j.err
#
# Layer-2 control evaluation: FID / R-precision / Diversity of GUIDED samples
# vs the real test distribution, plus Traj/Loc/Avg control error, with a
# same-seed unguided baseline. Guo evaluator harness as in evaluate.
#
# Usage:
#     AE_CKPT=.../mardm-ae-canon/checkpoints/latest.pt \
#     GEN_CKPT=.../mardm-gen-m-canon/checkpoints/latest.pt \
#     MARDM_STATS=.../mardm_essential_stats_canonical.pt \
#     CANONICAL_DIR=.../h3d-canonical/new_joint_vecs \
#     MAX_CLIPS=512 sbatch slurm/mardm/evaluate_control.sbatch

set -euo pipefail
mkdir -p slurm/logs

AE_CKPT="${AE_CKPT:?set AE_CKPT to the stage-1 AE checkpoint}"
GEN_CKPT="${GEN_CKPT:-${1:?set GEN_CKPT env var or pass the gen checkpoint as the first positional arg}}"
CONFIG_NAME="${CONFIG_NAME:-gen}"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(dirname "$(dirname "${SCRIPT_DIR}")")}"
fi
IMAGE="${IMAGE:-${HOME}/rmg.sqsh}"
RUNS_ROOT="${RUNS_ROOT:-${HOME}/rmg-runs}"
HF_CACHE="${HF_CACHE:-${HOME}/hf-cache}"
DATA_ROOT="${RMG_DATA_ROOT:-${REPO}/external/data/humanml3d_packed}"
MARDM_STATS="${MARDM_STATS:-${RUNS_ROOT}/mardm_essential_stats.pt}"
CANONICAL_DIR="${CANONICAL_DIR:-}"
GUIDANCE="${GUIDANCE:-3.0}"
MAX_CLIPS="${MAX_CLIPS:--1}"
INNER_ITERS="${INNER_ITERS:-30}"
JOINTS="${JOINTS:-[0]}"
NUM_KEYFRAMES="${NUM_KEYFRAMES:-5}"
RUN_NAME="${RUN_NAME:-mardm-ctrl-eval-${SLURM_JOB_ID:-local}}"
OVERRIDES="${OVERRIDES:-}"

CANON_FLAG=""
REAL_H3D_FLAG=""
if [ -n "${CANONICAL_DIR}" ]; then
    CANON_FLAG="data.canonical_dir='${CANONICAL_DIR}'"
    REAL_H3D_FLAG="+ctrl.real_h3d_dir='${CANONICAL_DIR}'"
fi

enroot start \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --env "HF_HOME=${HF_CACHE}" \
    --env "RMG_RUNS_DIR=${RUNS_ROOT}" \
    --env "RMG_DATA_ROOT=${DATA_ROOT}" \
    --env "MARDM_STATS=${MARDM_STATS}" \
    --env "PYTHONPATH=${REPO}/src" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        cd ${REPO}
        pip install -e . --no-deps --quiet
        python -m spacy download en_core_web_sm --quiet || true
        python -u -m mardm.scripts.evaluate_control \\
            --config-name=${CONFIG_NAME} \\
            ae_checkpoint='${AE_CKPT}' \\
            text_encoder.type=qwen3 \\
            ${CANON_FLAG} \\
            ${REAL_H3D_FLAG} \\
            +ctrl.checkpoint='${GEN_CKPT}' \\
            +ctrl.evaluator=real \\
            +ctrl.guidance='${GUIDANCE}' \\
            +ctrl.max_clips='${MAX_CLIPS}' \\
            +ctrl.inner_iters='${INNER_ITERS}' \\
            +ctrl.joints='${JOINTS}' \\
            +ctrl.num_keyframes='${NUM_KEYFRAMES}' \\
            run_name='${RUN_NAME}' \\
            ${OVERRIDES}
    "
