#!/bin/bash
#SBATCH --job-name=momask-rgrp
#SBATCH --partition=12g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1,ccc:75
#SBATCH --nodes=1
#SBATCH --output=slurm/logs/momask-real-groups-%j.out
#SBATCH --error=slurm/logs/momask-real-groups-%j.err

set -euo pipefail
mkdir -p slurm/logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
fi

IMAGE="${IMAGE:-${HOME}/rmg-momask.sqsh}"
DATA_ROOT="${DATA_ROOT:-/mnt/projects/drl4cvb/human-motion-representation/data/data/humanml3d_packed}"
HUMANML3D_DIR="${HUMANML3D_DIR:-/mnt/projects/drl4cvb/human-motion-representation/data/HumanML3D/HumanML3D}"
HUMANML3D_TEXTS_ZIP="${HUMANML3D_TEXTS_ZIP:-${HUMANML3D_DIR}/texts.zip}"
REAL_H3D_DIR="${REAL_H3D_DIR:-${HOME}/rmg-runs/h3d-canonical/new_joint_vecs}"
EVALUATOR_CHECKPOINTS_DIR="${EVALUATOR_CHECKPOINTS_DIR:-external/text-to-motion/checkpoints/momask_official}"
EVALUATOR_NORMALIZATION_NAME="${EVALUATOR_NORMALIZATION_NAME:-Comp_v6_KLD005}"
SPLIT="${SPLIT:-test}"
MAX_CLIPS="${MAX_CLIPS:--1}"
SEED="${SEED:-3407}"
OUTPUT="${OUTPUT:-runs/momask_diagnostics/paper_eval_real_groups_${SPLIT}_${EVALUATOR_NORMALIZATION_NAME}.json}"

echo "[momask-real-groups] host=$(hostname)"
echo "[momask-real-groups] split=${SPLIT} max_clips=${MAX_CLIPS} output=${OUTPUT}"
echo "[momask-real-groups] evaluator_checkpoints=${EVALUATOR_CHECKPOINTS_DIR} normalization=${EVALUATOR_NORMALIZATION_NAME}"

enroot start \
    --root \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --mount "${REPO}:${REPO}" \
    --env "PYTHONPATH=${REPO}/src" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        cd '${REPO}'
        pip install -e . --no-deps --quiet
        python -u -m momask.scripts.diagnose_paper_eval_groups \\
            --data-root '${DATA_ROOT}' \\
            --canonical-h3d-dir '${REAL_H3D_DIR}' \\
            --humanml3d-texts-zip '${HUMANML3D_TEXTS_ZIP}' \\
            --humanml3d-split-dir '${HUMANML3D_DIR}' \\
            --split '${SPLIT}' \\
            --max-seq-len 196 \\
            --min-seq-len 40 \\
            --batch-size 32 \\
            --max-clips '${MAX_CLIPS}' \\
            --seed '${SEED}' \\
            --evaluator-checkpoints-dir '${EVALUATOR_CHECKPOINTS_DIR}' \\
            --evaluator-normalization-name '${EVALUATOR_NORMALIZATION_NAME}' \\
            --output '${OUTPUT}'
    "
