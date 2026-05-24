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
#   1. Ensure `external/text-to-motion` submodule is properly cloned (networks/,
#      utils/, glove/ all present). Handles the case where finest.tar was
#      placed in checkpoints/ before the submodule was initialized — git
#      submodule update silently skips non-empty dirs, so we stash the
#      checkpoint, wipe the dir, clone, and restore.
#   2. Verify the Guo pretrained checkpoint is at the path the evaluator expects.
#
# Runs on a data node because the head node OOMs on git submodule clones.

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

SUBMOD="external/text-to-motion"
DEST_DIR="${SUBMOD}/checkpoints/t2m/text_mot_match/model"
DEST="${DEST_DIR}/finest.tar"
STASH="/tmp/rmg-finest-${SLURM_JOB_ID:-$$}.tar"

needs_clone=true
if [ -f "${SUBMOD}/networks/modules.py" ] \
   && [ -f "${SUBMOD}/networks/evaluator_wrapper.py" ] \
   && [ -d "${SUBMOD}/glove" ]; then
    needs_clone=false
fi

if [ "${needs_clone}" = "true" ]; then
    echo "[init-eval] submodule is incomplete — re-cloning"

    # Stash any existing checkpoint so we don't lose the 235 MB download.
    if [ -f "${DEST}" ]; then
        echo "[init-eval] stashing existing finest.tar to ${STASH}"
        cp "${DEST}" "${STASH}"
    fi

    # Wipe the submodule dir (git refuses to clone into a non-empty dir,
    # silently — that's why the previous attempt looked like it worked).
    if [ -d "${SUBMOD}" ]; then
        echo "[init-eval] wiping ${SUBMOD}"
        rm -rf "${SUBMOD}"
    fi
    # Also clear the .git/modules cache so the clone is fully fresh.
    rm -rf ".git/modules/external/text-to-motion"

    echo "[init-eval] cloning ${SUBMOD} (shallow, single-threaded)..."
    git -c pack.threads=1 -c pack.windowMemory=64m -c pack.deltaCacheSize=64m \
        submodule update --init --depth 1 --force -- "${SUBMOD}"

    # Restore the checkpoint. The upstream repo ships `checkpoints` as a
    # symlink (or similar non-dir) — mkdir -p fails on those. Remove
    # anything in the way, then create.
    if [ -f "${STASH}" ]; then
        echo "[init-eval] restoring finest.tar  ->  ${DEST}"
        # Walk down each ancestor of DEST_DIR and turn any non-dir into a dir.
        ancestor="${SUBMOD}"
        for seg in checkpoints t2m text_mot_match model; do
            ancestor="${ancestor}/${seg}"
            if [ -e "${ancestor}" ] && [ ! -d "${ancestor}" ]; then
                echo "[init-eval]   removing non-dir at ${ancestor}"
                rm -f "${ancestor}"
            fi
            [ -d "${ancestor}" ] || mkdir "${ancestor}"
        done
        mv "${STASH}" "${DEST}"
    fi
else
    echo "[init-eval] submodule already complete"
fi

# --- find finest.tar if still missing ---
if [ ! -f "${DEST}" ]; then
    echo "[init-eval] finest.tar missing — scanning for it..."
    CANDIDATES=$(find \
        "${REPO}/external" \
        "${HOME}" \
        -maxdepth 6 -name 'finest.tar' 2>/dev/null | head -10 || true)
    echo "[init-eval] candidates: ${CANDIDATES}"
    SRC=$(echo "${CANDIDATES}" | head -1)
    if [ -z "${SRC}" ] || [ ! -f "${SRC}" ]; then
        echo "[init-eval] ERROR: finest.tar not found anywhere."
        echo "[init-eval] pass GUO_TAR=/abs/path/to/finest.tar"
        exit 1
    fi
    ancestor="${SUBMOD}"
    for seg in checkpoints t2m text_mot_match model; do
        ancestor="${ancestor}/${seg}"
        if [ -L "${ancestor}" ] || ([ -e "${ancestor}" ] && [ ! -d "${ancestor}" ]); then
            echo "[init-eval]   removing non-dir / symlink at ${ancestor}"
            rm -f "${ancestor}"
        fi
        [ -d "${ancestor}" ] || mkdir "${ancestor}"
    done
    cp "${SRC}" "${DEST}"
fi

# --- sanity ---
echo
echo "[init-eval] verifying:"
test -f "${SUBMOD}/networks/modules.py"           && echo "  OK: networks/modules.py"           || { echo "  MISSING: networks/modules.py"; exit 1; }
test -f "${SUBMOD}/networks/evaluator_wrapper.py" && echo "  OK: networks/evaluator_wrapper.py" || { echo "  MISSING: networks/evaluator_wrapper.py"; exit 1; }
test -d "${SUBMOD}/glove"                         && echo "  OK: glove/"                         || { echo "  MISSING: glove/"; exit 1; }
test -f "${DEST}"                                  && echo "  OK: ${DEST}"                        || { echo "  MISSING: ${DEST}"; exit 1; }

echo "[init-eval] DONE — you can now resubmit slurm/evaluate.sbatch"
