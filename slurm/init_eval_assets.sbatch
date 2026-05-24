#!/bin/bash
#SBATCH --job-name=rmg-init-eval
#SBATCH --partition=data
#SBATCH --qos=students_normal
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/init-eval-%j.out
#SBATCH --error=slurm/logs/init-eval-%j.err
#
# One-shot setup for evaluation:
#   1. Init the `external/text-to-motion` submodule (head node OOMs on clone,
#      so we do it on a data node with low-memory git options).
#   2. Place the Guo pretrained checkpoint at the path the evaluator expects.
#
# Pre-req: the Guo checkpoint tar must already exist somewhere on the cluster.
# Pass its absolute path via GUO_TAR (default: scans likely locations).
#
# Usage:
#     sbatch slurm/init_eval_assets.sbatch
#     # or pin the source explicitly:
#     GUO_TAR=/mnt/home/julianmunzer/.../finest.tar sbatch slurm/init_eval_assets.sbatch

set -euo pipefail
mkdir -p slurm/logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(dirname "${SCRIPT_DIR}")}"
fi

cd "${REPO}"
echo "[init-eval] repo=${REPO}"

# --- 1. submodule init (low-mem flags) ---
SUBMOD="external/text-to-motion"
if [ ! -f "${SUBMOD}/networks/modules.py" ]; then
    echo "[init-eval] cloning ${SUBMOD} (shallow, single-threaded)..."
    git -c pack.threads=1 -c pack.windowMemory=64m -c pack.deltaCacheSize=64m \
        submodule update --init --depth 1 -- "${SUBMOD}"
else
    echo "[init-eval] ${SUBMOD} already initialized — skipping clone"
fi

# --- 2. Guo checkpoint placement ---
DEST_DIR="${SUBMOD}/checkpoints/t2m/text_mot_match/model"
DEST="${DEST_DIR}/finest.tar"

if [ -f "${DEST}" ]; then
    echo "[init-eval] ${DEST} already in place — done"
    ls -lh "${DEST}"
    exit 0
fi

# Find the source tar. Either passed in, or scan a few likely spots.
if [ -z "${GUO_TAR:-}" ]; then
    echo "[init-eval] scanning for finest.tar..."
    CANDIDATES=$(find \
        "${REPO}/external/TODO" \
        "${REPO}/external" \
        "${HOME}" \
        -maxdepth 6 -name 'finest.tar' 2>/dev/null | head -10 || true)
    echo "[init-eval] candidates found:"
    echo "${CANDIDATES}"
    GUO_TAR=$(echo "${CANDIDATES}" | head -1)
fi

if [ -z "${GUO_TAR}" ] || [ ! -f "${GUO_TAR}" ]; then
    echo "[init-eval] ERROR: could not find finest.tar."
    echo "[init-eval] pass it as: GUO_TAR=/abs/path/to/finest.tar sbatch slurm/init_eval_assets.sbatch"
    exit 1
fi

echo "[init-eval] copying ${GUO_TAR}  ->  ${DEST}"
mkdir -p "${DEST_DIR}"
cp "${GUO_TAR}" "${DEST}"
ls -lh "${DEST}"

# --- sanity ---
echo
echo "[init-eval] checking submodule contents:"
test -f "${SUBMOD}/networks/modules.py"          && echo "  OK: networks/modules.py"
test -f "${SUBMOD}/networks/evaluator_wrapper.py" && echo "  OK: networks/evaluator_wrapper.py"
test -d "${SUBMOD}/glove"                         && echo "  OK: glove/"
test -f "${DEST}"                                  && echo "  OK: ${DEST}"

echo "[init-eval] DONE — you can now resubmit slurm/evaluate.sbatch"
