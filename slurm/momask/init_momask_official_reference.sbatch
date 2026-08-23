#!/bin/bash
#SBATCH --job-name=momask-ref-assets
#SBATCH --partition=data
#SBATCH --nodes=1
#SBATCH --output=slurm/logs/momask-ref-assets-%j.out
#SBATCH --error=slurm/logs/momask-ref-assets-%j.err

set -euo pipefail
mkdir -p slurm/logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
fi

OFFICIAL_REPO="${OFFICIAL_MOMASK_REPO:-${REPO}/external/momask-codes-official}"
DOWNLOAD_DIR="${MOMASK_REFERENCE_DOWNLOAD_DIR:-${HOME}/rmg-momask-official-assets}"
ARCHIVE="${DOWNLOAD_DIR}/humanml3d_models.zip"
ARCHIVE_ID="${MOMASK_MODELS_GDRIVE_ID:-1vXS7SHJBgWPt59wupQ5UUzhFObrnGkQ0}"
VQ_NAME="${OFFICIAL_VQ_NAME:-rvq_nq6_dc512_nc512_noshare_qdp0.2}"
VQ_CHECKPOINT="${OFFICIAL_REPO}/checkpoints/t2m/${VQ_NAME}/model/net_best_fid.tar"

echo "[momask-ref-assets] host=$(hostname) repo=${REPO}"
echo "[momask-ref-assets] official_repo=${OFFICIAL_REPO}"
echo "[momask-ref-assets] vq_checkpoint=${VQ_CHECKPOINT}"

if [ ! -f "${OFFICIAL_REPO}/models/vq/model.py" ]; then
    if [ -e "${OFFICIAL_REPO}" ]; then
        echo "[momask-ref-assets] ERROR: incomplete path already exists: ${OFFICIAL_REPO}" >&2
        exit 2
    fi
    echo "[momask-ref-assets] cloning official MoMask repository"
    git clone --depth 1 https://github.com/EricGuo5513/momask-codes.git "${OFFICIAL_REPO}"
fi

if [ -s "${VQ_CHECKPOINT}" ]; then
    echo "[momask-ref-assets] official RVQ already present"
    echo "[momask-ref-assets] DONE"
    exit 0
fi

if ! command -v unzip >/dev/null 2>&1; then
    echo "[momask-ref-assets] ERROR: unzip is unavailable on this data node" >&2
    exit 2
fi
mkdir -p "${DOWNLOAD_DIR}" "${OFFICIAL_REPO}/checkpoints/t2m"
if ! command -v gdown >/dev/null 2>&1; then
    python -m pip install --user --quiet gdown
    export PATH="${HOME}/.local/bin:${PATH}"
fi
if [ -f "${ARCHIVE}" ] && ! unzip -tq "${ARCHIVE}" >/dev/null 2>&1; then
    echo "[momask-ref-assets] removing invalid cached archive"
    rm -f "${ARCHIVE}"
fi
if [ ! -f "${ARCHIVE}" ]; then
    echo "[momask-ref-assets] downloading official HumanML3D model archive"
    python -m gdown "${ARCHIVE_ID}" -O "${ARCHIVE}"
fi
unzip -tq "${ARCHIVE}" >/dev/null
unzip -oq "${ARCHIVE}" -d "${OFFICIAL_REPO}/checkpoints/t2m"

if [ ! -s "${VQ_CHECKPOINT}" ]; then
    echo "[momask-ref-assets] ERROR: expected RVQ checkpoint was not extracted" >&2
    find "${OFFICIAL_REPO}/checkpoints/t2m" -maxdepth 4 -type f -name '*.tar' -print
    exit 2
fi
test -s "${OFFICIAL_REPO}/checkpoints/t2m/${VQ_NAME}/opt.txt"
echo "[momask-ref-assets] checkpoint_size=$(stat -c %s "${VQ_CHECKPOINT}")"
echo "[momask-ref-assets] DONE"
