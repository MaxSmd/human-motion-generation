#!/bin/bash
#SBATCH --job-name=guo-rdiag
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --output=slurm/logs/guo-rdiag-%j.out
#SBATCH --error=slurm/logs/guo-rdiag-%j.err

set -euo pipefail
mkdir -p slurm/logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(dirname "${SCRIPT_DIR}")}"
fi

IMAGE="${IMAGE:-${HOME}/rmg-momask.sqsh}"
DATA_ROOT="${DATA_ROOT:-/mnt/projects/drl4cvb/human-motion-representation/data/data/humanml3d_packed}"
H3D_DIR="${H3D_DIR:-/mnt/projects/drl4cvb/human-motion-representation/data/HumanML3D/HumanML3D/new_joint_vecs}"
TEXTS_ZIP="${TEXTS_ZIP:-/mnt/projects/drl4cvb/human-motion-representation/data/HumanML3D/HumanML3D/texts.zip}"
HUMANML3D_REPO="${HUMANML3D_REPO:-external/HumanML3D}"
MAX_CLIPS="${MAX_CLIPS:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-196}"
OUTPUT="${OUTPUT:-runs/momask_diagnostics/guo_canonical_real_${MAX_CLIPS}.json}"

echo "[guo-rdiag] host=$(hostname)"
echo "[guo-rdiag] repo=${REPO}"
echo "[guo-rdiag] image=${IMAGE}"
echo "[guo-rdiag] data_root=${DATA_ROOT}"
echo "[guo-rdiag] h3d_dir=${H3D_DIR}"
echo "[guo-rdiag] texts_zip=${TEXTS_ZIP}"
echo "[guo-rdiag] max_clips=${MAX_CLIPS} batch_size=${BATCH_SIZE} max_seq_len=${MAX_SEQ_LEN}"
echo "[guo-rdiag] output=${OUTPUT}"

enroot start \
    --root \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --mount "${REPO}:${REPO}" \
    --env "RMG_DATA_ROOT=${DATA_ROOT}" \
    --env "PYTHONPATH=${REPO}/src" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        cd ${REPO}
        pip install -e . --no-deps --quiet
        python -u -m rmg.scripts.diagnose_guo_rprecision \
            --data-root '${DATA_ROOT}' \
            --h3d-dir '${H3D_DIR}' \
            --texts-zip '${TEXTS_ZIP}' \
            --humanml3d-repo '${HUMANML3D_REPO}' \
            --max-clips ${MAX_CLIPS} \
            --batch-size ${BATCH_SIZE} \
            --max-seq-len ${MAX_SEQ_LEN} \
            --output '${OUTPUT}'
    "
