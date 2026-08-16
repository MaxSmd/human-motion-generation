#!/bin/bash
#SBATCH --job-name=momask-copy196
#SBATCH --partition=data
#SBATCH --qos=students_normal
#SBATCH --nodes=1
#SBATCH --output=slurm/logs/momask-copy196-%j.out
#SBATCH --error=slurm/logs/momask-copy196-%j.err
#
# Copy only the paper-faithful 196-frame transformer checkpoints to shared
# project storage. This is intentionally a data-node job, never a head-node copy.

set -euo pipefail
mkdir -p slurm/logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
fi

SOURCE_ROOT="${SOURCE_ROOT:-${REPO}/runs}"
DEST_ROOT="${DEST_ROOT:-/mnt/projects/drl4cvb/human-motion-representation/runs}"
RUN_NAMES="${RUN_NAMES:-momask-canonical-mtrans-paperfaithful196-clip500e momask-canonical-rtrans-paperfaithful196-clip500e}"

echo "[momask-copy196] host=$(hostname)"
echo "[momask-copy196] source_root=${SOURCE_ROOT}"
echo "[momask-copy196] dest_root=${DEST_ROOT}"

mkdir -p "${DEST_ROOT}"

copy_tree() {
    local source="$1"
    local destination="$2"
    mkdir -p "${destination}"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --partial --human-readable --info=progress2 \
            "${source}/" "${destination}/"
    else
        cp -a "${source}/." "${destination}/"
    fi
}

for run_name in ${RUN_NAMES}; do
    source_run="${SOURCE_ROOT}/${run_name}"
    destination_run="${DEST_ROOT}/${run_name}"
    if [ ! -d "${source_run}/checkpoints" ]; then
        echo "[momask-copy196] ERROR: checkpoint directory missing: ${source_run}/checkpoints" >&2
        exit 1
    fi

    echo "[momask-copy196] copying ${run_name}/checkpoints"
    copy_tree "${source_run}/checkpoints" "${destination_run}/checkpoints"

    if [ -f "${source_run}/momask_smoke_latest.pt" ]; then
        echo "[momask-copy196] copying ${run_name}/momask_smoke_latest.pt"
        if command -v rsync >/dev/null 2>&1; then
            rsync -a --partial --human-readable --info=progress2 \
                "${source_run}/momask_smoke_latest.pt" "${destination_run}/"
        else
            cp -a "${source_run}/momask_smoke_latest.pt" "${destination_run}/"
        fi
    fi

    echo "[momask-copy196] copied files for ${run_name}:"
    find "${destination_run}" -type f -name '*.pt' -printf '%P\n' | sort
    du -sh "${destination_run}"
done

echo "[momask-copy196] DONE"
