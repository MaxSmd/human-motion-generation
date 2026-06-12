#!/bin/bash
#SBATCH --job-name=rmg-publish
#SBATCH --partition=12g
#SBATCH --nodelist=frankfurt,hamburg,kassel,leipzig
#SBATCH --qos=students_normal
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/publish-%j.out
#SBATCH --error=slurm/logs/publish-%j.err
#
# Copy a finished run from <project>/runs/<kind>/<run> to the shared project dir.
# Runs on a compute node because the head has a 200 MB file-size ulimit
# that breaks rsync on the checkpoint files.
#
# Usage:
#     RUN=rmg-base-v1 sbatch slurm/publish_run.sbatch
#     # optional override of the destination root:
#     RUN=rmg-base-v1 DEST=/mnt/projects/drl4cvb/human-motion-representation/rmg/runs \
#         sbatch slurm/publish_run.sbatch

set -euo pipefail
mkdir -p slurm/logs

RUN="${RUN:?set RUN=<run-name>, e.g. RUN=rmg-base-v1}"
SRC="${SRC:-$(ls -d "${HOME}/riemann-motion-generation/runs/"*/"${RUN}" 2>/dev/null | head -1 || true)}"
DEST="${DEST:-/mnt/projects/drl4cvb/human-motion-representation/rmg/runs}"

test -d "${SRC}" || { echo "no such run: ${SRC}"; exit 1; }
mkdir -p "${DEST}"

echo "[publish] ${SRC}  ->  ${DEST}/${RUN}"
rsync -avhP --stats "${SRC}/" "${DEST}/${RUN}/"
echo "[publish] DONE"
ls -lh "${DEST}/${RUN}/"
