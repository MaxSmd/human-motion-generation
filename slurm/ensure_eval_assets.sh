#!/bin/bash
# Ensure the Guo (text-to-motion) evaluator assets are present:
#   1. the `external/text-to-motion` submodule (networks/, utils/, glove/), and
#   2. the pretrained matching-model checkpoint `finest.tar` at the path the
#      evaluator loads it from.
#
# Idempotent: a fast presence check short-circuits when everything is already in
# place (the normal case — the assets persist on shared $HOME once set up, so
# every eval after the first one skips straight through). Only the first call
# does the heavy submodule clone + checkpoint placement.
#
# Used two ways:
#   - run by slurm/rmg_eval.sbatch on startup (auto-setup, host side)
#   - by hand if the first-time clone needs a network/memory-capable node:
#       REPO=$PWD sbatch --partition=data --time=00:30:00 \
#           --wrap 'bash slurm/ensure_eval_assets.sh'
#
# (The git submodule clone needs network + memory the head/GPU nodes may lack;
# the data partition is the safe place for that one-time step.)

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
cd "${REPO}"

SUBMOD="external/text-to-motion"
DEST_DIR="${SUBMOD}/checkpoints/t2m/text_mot_match/model"
DEST="${DEST_DIR}/finest.tar"
STASH="/tmp/rmg-finest-${SLURM_JOB_ID:-$$}.tar"

# --- fast path: already set up -> nothing to do ---
if [ -f "${SUBMOD}/networks/modules.py" ] \
   && [ -f "${SUBMOD}/networks/evaluator_wrapper.py" ] \
   && [ -d "${SUBMOD}/glove" ] \
   && [ -f "${DEST}" ]; then
    echo "[eval-assets] present — skipping setup"
    exit 0
fi

echo "[eval-assets] missing assets — running one-time setup (repo=${REPO})"

needs_clone=true
if [ -f "${SUBMOD}/networks/modules.py" ] \
   && [ -f "${SUBMOD}/networks/evaluator_wrapper.py" ] \
   && [ -d "${SUBMOD}/glove" ]; then
    needs_clone=false
fi

if [ "${needs_clone}" = "true" ]; then
    echo "[eval-assets] submodule is incomplete — re-cloning"

    # Stash any existing checkpoint so we don't lose the 235 MB download.
    if [ -f "${DEST}" ]; then
        echo "[eval-assets] stashing existing finest.tar to ${STASH}"
        cp "${DEST}" "${STASH}"
    fi

    # Wipe the submodule dir (git refuses to clone into a non-empty dir,
    # silently — that's why a previous attempt could look like it worked).
    if [ -d "${SUBMOD}" ]; then
        echo "[eval-assets] wiping ${SUBMOD}"
        rm -rf "${SUBMOD}"
    fi
    # Also clear the .git/modules cache so the clone is fully fresh.
    rm -rf ".git/modules/external/text-to-motion"

    echo "[eval-assets] cloning ${SUBMOD} (shallow, single-threaded)..."
    git -c pack.threads=1 -c pack.windowMemory=64m -c pack.deltaCacheSize=64m \
        submodule update --init --depth 1 --force -- "${SUBMOD}"

    # Restore the checkpoint. The upstream repo ships `checkpoints` as a
    # symlink (or similar non-dir) — mkdir -p fails on those. Remove
    # anything in the way, then create.
    if [ -f "${STASH}" ]; then
        echo "[eval-assets] restoring finest.tar  ->  ${DEST}"
        # Walk down each ancestor of DEST_DIR and turn any non-dir into a dir.
        ancestor="${SUBMOD}"
        for seg in checkpoints t2m text_mot_match model; do
            ancestor="${ancestor}/${seg}"
            if [ -e "${ancestor}" ] && [ ! -d "${ancestor}" ]; then
                echo "[eval-assets]   removing non-dir at ${ancestor}"
                rm -f "${ancestor}"
            fi
            [ -d "${ancestor}" ] || mkdir "${ancestor}"
        done
        mv "${STASH}" "${DEST}"
    fi
fi

# --- find finest.tar if still missing ---
if [ ! -f "${DEST}" ]; then
    echo "[eval-assets] finest.tar missing — scanning for it..."
    CANDIDATES=$(find \
        "${REPO}/external" \
        "${HOME}" \
        -maxdepth 6 -name 'finest.tar' 2>/dev/null | head -10 || true)
    echo "[eval-assets] candidates: ${CANDIDATES}"
    SRC=$(echo "${CANDIDATES}" | head -1)
    if [ -z "${SRC}" ] || [ ! -f "${SRC}" ]; then
        echo "[eval-assets] ERROR: finest.tar not found anywhere."
        echo "[eval-assets] pass GUO_TAR=/abs/path/to/finest.tar"
        exit 1
    fi
    ancestor="${SUBMOD}"
    for seg in checkpoints t2m text_mot_match model; do
        ancestor="${ancestor}/${seg}"
        if [ -L "${ancestor}" ] || ([ -e "${ancestor}" ] && [ ! -d "${ancestor}" ]); then
            echo "[eval-assets]   removing non-dir / symlink at ${ancestor}"
            rm -f "${ancestor}"
        fi
        [ -d "${ancestor}" ] || mkdir "${ancestor}"
    done
    cp "${SRC}" "${DEST}"
fi

# --- sanity ---
echo "[eval-assets] verifying:"
test -f "${SUBMOD}/networks/modules.py"           && echo "  OK: networks/modules.py"           || { echo "  MISSING: networks/modules.py"; exit 1; }
test -f "${SUBMOD}/networks/evaluator_wrapper.py" && echo "  OK: networks/evaluator_wrapper.py" || { echo "  MISSING: networks/evaluator_wrapper.py"; exit 1; }
test -d "${SUBMOD}/glove"                         && echo "  OK: glove/"                         || { echo "  MISSING: glove/"; exit 1; }
test -f "${DEST}"                                  && echo "  OK: ${DEST}"                        || { echo "  MISSING: ${DEST}"; exit 1; }

echo "[eval-assets] DONE"
