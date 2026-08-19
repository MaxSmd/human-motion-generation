#!/bin/bash
#SBATCH --job-name=momask-data-audit
#SBATCH --partition=data
#SBATCH --qos=students_normal
#SBATCH --nodes=1
#SBATCH --output=slurm/logs/momask-data-audit-%j.out
#SBATCH --error=slurm/logs/momask-data-audit-%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
fi

IMAGE="${IMAGE:-${HOME}/rmg-momask.sqsh}"
DATA_ROOT="${DATA_ROOT:-/mnt/projects/drl4cvb/human-motion-representation/data/data/humanml3d_packed}"
CANONICAL_H3D_DIR="${CANONICAL_H3D_DIR:-${HOME}/rmg-runs/h3d-canonical/new_joint_vecs}"
HUMANML3D_DIR="${HUMANML3D_DIR:-/mnt/projects/drl4cvb/human-motion-representation/data/HumanML3D/HumanML3D}"
HUMANML3D_TEXTS_ZIP="${HUMANML3D_TEXTS_ZIP:-${HUMANML3D_DIR}/texts.zip}"
OUTPUT="${OUTPUT:-runs/momask_diagnostics/paper_transformer_data_audit.json}"

echo "[momask-data-audit] host=$(hostname)"
echo "[momask-data-audit] repo=${REPO}"
echo "[momask-data-audit] data_root=${DATA_ROOT}"
echo "[momask-data-audit] canonical_h3d_dir=${CANONICAL_H3D_DIR}"
echo "[momask-data-audit] humanml3d_dir=${HUMANML3D_DIR}"
echo "[momask-data-audit] output=${OUTPUT}"

enroot start \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --mount "${REPO}:${REPO}" \
    --env "PYTHONPATH=${REPO}/src" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        cd '${REPO}'
        python -u -m momask.scripts.audit_transformer_data \\
            --data-root '${DATA_ROOT}' \\
            --canonical-h3d-dir '${CANONICAL_H3D_DIR}' \\
            --humanml3d-texts-zip '${HUMANML3D_TEXTS_ZIP}' \\
            --humanml3d-split-dir '${HUMANML3D_DIR}' \\
            --max-seq-len 196 \\
            --min-seq-len 40 \\
            --unit-length 4 \\
            --output '${OUTPUT}'
    "
