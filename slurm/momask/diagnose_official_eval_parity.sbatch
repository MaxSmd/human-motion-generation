#!/bin/bash
#SBATCH --job-name=momask-eval-parity
#SBATCH --partition=12g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1,ccc:75
#SBATCH --nodes=1
#SBATCH --output=slurm/logs/momask-eval-parity-%j.out
#SBATCH --error=slurm/logs/momask-eval-parity-%j.err

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
CANONICAL_H3D_DIR="${CANONICAL_H3D_DIR:-${HOME}/rmg-runs/h3d-canonical/new_joint_vecs}"
TEXTS_ZIP="${TEXTS_ZIP:-${HUMANML3D_DIR}/texts.zip}"
OFFICIAL_REPO="${OFFICIAL_MOMASK_REPO:-${REPO}/external/momask-codes-official}"
OFFICIAL_VQ_NAME="${OFFICIAL_VQ_NAME:-rvq_nq6_dc512_nc512_noshare_qdp0.2}"
OFFICIAL_CKPT="${OFFICIAL_CKPT:-${OFFICIAL_REPO}/checkpoints/t2m/${OFFICIAL_VQ_NAME}/model/net_best_fid.tar}"
OFFICIAL_OPT="${OFFICIAL_OPT:-${OFFICIAL_REPO}/checkpoints/t2m/${OFFICIAL_VQ_NAME}/opt.txt}"
OFFICIAL_DATASET_OPT="${OFFICIAL_DATASET_OPT:-${OFFICIAL_REPO}/checkpoints/t2m/Comp_v6_KLD005/opt.txt}"
EVALUATOR_ROOT="${EVALUATOR_ROOT:-${REPO}/external/text-to-motion/checkpoints/momask_official}"
TEXT_TO_MOTION_REPO="${TEXT_TO_MOTION_REPO:-${REPO}/external/text-to-motion}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_BATCHES="${MAX_BATCHES:--1}"
SEED="${SEED:-3407}"
OUTPUT="${OUTPUT:-runs/momask_diagnostics/official_eval_same_batch_parity.json}"
DIAGNOSTIC_SCRIPT="${DIAGNOSTIC_SCRIPT:-${REPO}/src/momask/scripts/diagnose_official_eval_parity.py}"

require_file() {
    if [ ! -s "$1" ]; then
        echo "[momask-eval-parity] ERROR: $2 not found: $1" >&2
        exit 2
    fi
}

require_file "${IMAGE}" "enroot image"
require_file "${OFFICIAL_REPO}/eval_t2m_vq.py" "official evaluator"
require_file "${OFFICIAL_CKPT}" "official VQ checkpoint"
require_file "${OFFICIAL_OPT}" "official VQ options"
require_file "${OFFICIAL_DATASET_OPT}" "official dataset options"
require_file "${EVALUATOR_ROOT}/t2m/text_mot_match/model/finest.tar" "local evaluator checkpoint"
require_file "${EVALUATOR_ROOT}/t2m/Comp_v6_KLD005/meta/mean.npy" "evaluator mean"
require_file "${EVALUATOR_ROOT}/t2m/Comp_v6_KLD005/meta/std.npy" "evaluator std"
require_file "${TEXT_TO_MOTION_REPO}/networks/evaluator_wrapper.py" "local evaluator wrapper"
require_file "${TEXTS_ZIP}" "HumanML3D captions"
require_file "${HUMANML3D_DIR}/test.txt" "HumanML3D test split"
require_file "${CANONICAL_H3D_DIR}/004822.npy" "canonical test motion"
require_file "${OFFICIAL_REPO}/dataset/HumanML3D/test.txt" "official test split link"
require_file "${OFFICIAL_REPO}/dataset/HumanML3D/texts/004822.txt" "official caption overlay"
require_file "${OFFICIAL_REPO}/dataset/HumanML3D/new_joint_vecs/004822.npy" "official motion overlay"
require_file "${DIAGNOSTIC_SCRIPT}" "parity diagnostic"

echo "[momask-eval-parity] host=$(hostname) repo=${REPO}"
echo "[momask-eval-parity] official_repo=${OFFICIAL_REPO}"
echo "[momask-eval-parity] checkpoint=${OFFICIAL_CKPT}"
echo "[momask-eval-parity] max_batches=${MAX_BATCHES} output=${OUTPUT}"

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
        python -u '${DIAGNOSTIC_SCRIPT}' \\
            --official-repo '${OFFICIAL_REPO}' \\
            --official-checkpoint '${OFFICIAL_CKPT}' \\
            --official-opt '${OFFICIAL_OPT}' \\
            --official-dataset-opt '${OFFICIAL_DATASET_OPT}' \\
            --text-to-motion-repo '${TEXT_TO_MOTION_REPO}' \\
            --evaluator-checkpoints-dir '${EVALUATOR_ROOT}' \\
            --data-root '${DATA_ROOT}' \\
            --canonical-h3d-dir '${CANONICAL_H3D_DIR}' \\
            --texts-zip '${TEXTS_ZIP}' \\
            --split-dir '${HUMANML3D_DIR}' \\
            --batch-size '${BATCH_SIZE}' \\
            --max-batches '${MAX_BATCHES}' \\
            --seed '${SEED}' \\
            --output '${OUTPUT}'
    "

echo "[momask-eval-parity] DONE output=${OUTPUT}"
