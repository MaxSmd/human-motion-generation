#!/bin/bash
#SBATCH --job-name=momask-vq-parity
#SBATCH --partition=12g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1,ccc:75
#SBATCH --nodes=1
#SBATCH --output=slurm/logs/momask-vq-parity-%j.out
#SBATCH --error=slurm/logs/momask-vq-parity-%j.err

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
OURS_CKPT="${OURS_CKPT:-runs/momask-canonical-vq-releaseparity64-50e-v2/val_fid_selection/momask_best_val_fid.pt}"
EVALUATOR_ROOT="${EVALUATOR_ROOT:-external/text-to-motion/checkpoints/momask_official}"
REPEATS="${REPEATS:-1}"
MAX_CLIPS="${MAX_CLIPS:--1}"
SEED="${SEED:-3407}"
OUTPUT="${OUTPUT:-runs/momask_diagnostics/official_vq_parity.json}"

echo "[momask-vq-parity] host=$(hostname) repo=${REPO}"
echo "[momask-vq-parity] official_repo=${OFFICIAL_REPO}"
echo "[momask-vq-parity] official_ckpt=${OFFICIAL_CKPT}"
echo "[momask-vq-parity] ours_ckpt=${OURS_CKPT}"
echo "[momask-vq-parity] repeats=${REPEATS} max_clips=${MAX_CLIPS} seed=${SEED}"
echo "[momask-vq-parity] output=${OUTPUT}"

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
        python -u -m momask.scripts.compare_official_vq \\
            --ours-checkpoint '${OURS_CKPT}' \\
            --official-repo '${OFFICIAL_REPO}' \\
            --official-checkpoint '${OFFICIAL_CKPT}' \\
            --official-opt '${OFFICIAL_OPT}' \\
            --data-root '${DATA_ROOT}' \\
            --canonical-h3d-dir '${CANONICAL_H3D_DIR}' \\
            --texts-zip '${TEXTS_ZIP}' \\
            --humanml3d-split-dir '${HUMANML3D_DIR}' \\
            --evaluator-checkpoints-dir '${EVALUATOR_ROOT}' \\
            --repeats ${REPEATS} \\
            --max-clips ${MAX_CLIPS} \\
            --seed ${SEED} \\
            --output '${OUTPUT}'
    "
