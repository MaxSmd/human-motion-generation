#!/bin/bash
#SBATCH --job-name=momask-copybest
#SBATCH --partition=data
#SBATCH --qos=students_normal
#SBATCH --nodes=1
#SBATCH --output=slurm/logs/momask-copybest-%j.out
#SBATCH --error=slurm/logs/momask-copybest-%j.err
#
# Copy the assembled validation-FID-selected MoMask checkpoint to shared storage.

set -euo pipefail
mkdir -p slurm/logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
fi

RUN_NAME="${RUN_NAME:-momask-canonical-tokens-paperfaithful196-valfidselection-full}"
SOURCE="${SOURCE:-${REPO}/runs/${RUN_NAME}/momask_best_val_fid.pt}"
DEST_DIR="${DEST_DIR:-/mnt/projects/drl4cvb/human-motion-representation/runs/${RUN_NAME}}"
DESTINATION="${DEST_DIR}/momask_best_val_fid.pt"

echo "[momask-copybest] host=$(hostname)"
echo "[momask-copybest] source=${SOURCE}"
echo "[momask-copybest] destination=${DESTINATION}"

if [ ! -f "${SOURCE}" ]; then
    echo "[momask-copybest] ERROR: selected checkpoint not found: ${SOURCE}" >&2
    exit 1
fi

mkdir -p "${DEST_DIR}"
if command -v rsync >/dev/null 2>&1; then
    rsync -a --partial --human-readable --info=progress2 \
        "${SOURCE}" "${DEST_DIR}/"
else
    cp -a "${SOURCE}" "${DEST_DIR}/"
fi

if ! cmp -s "${SOURCE}" "${DESTINATION}"; then
    echo "[momask-copybest] ERROR: destination differs from source" >&2
    exit 1
fi

ls -lh "${SOURCE}" "${DESTINATION}"
echo "[momask-copybest] verified byte-for-byte"
echo "[momask-copybest] DONE"
