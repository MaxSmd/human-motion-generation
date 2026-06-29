#!/bin/bash
# Ensure the Guo (text-to-motion) evaluator assets are present:
#   1. the `external/text-to-motion` submodule (networks/, utils/, glove/),
#   2. the pretrained matching-model checkpoint `finest.tar`, and
#   3. the Comp_v6_KLD01 evaluator normalization files.
#
# Idempotent: a fast presence check short-circuits when everything is already in
# place. The first setup may need the data partition because the head node should
# not run the submodule clone or model download.
#
# Used two ways:
#   - run by slurm/rmg/eval.sbatch on startup (auto-setup, host side)
#   - by hand if first-time setup needs a network/memory-capable node:
#       REPO=$PWD DOWNLOAD_GUO=1 sbatch slurm/init_eval_assets.sbatch
#
# Optional:
#   GUO_TAR=/abs/path/to/finest.tar sbatch slurm/init_eval_assets.sbatch
#   DOWNLOAD_GUO=1 sbatch slurm/init_eval_assets.sbatch

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
cd "${REPO}"

SUBMOD="external/text-to-motion"
DEST_DIR="${SUBMOD}/checkpoints/t2m/text_mot_match/model"
DEST="${DEST_DIR}/finest.tar"
META_DIR="${SUBMOD}/checkpoints/t2m/Comp_v6_KLD01/meta"
META_MEAN="${META_DIR}/mean.npy"
META_STD="${META_DIR}/std.npy"
STASH="/tmp/rmg-finest-${SLURM_JOB_ID:-$$}.tar"
T2M_GDRIVE_ID="${T2M_GDRIVE_ID:-1IgrFCnxeg4olBtURUHimzS03ZI0df_6W}"
DOWNLOAD_DIR="${HOME}/rmg-eval-assets"
ZIP="${DOWNLOAD_DIR}/t2m.zip"

ensure_dest_dir() {
    ancestor="${SUBMOD}"
    for seg in checkpoints t2m text_mot_match model; do
        ancestor="${ancestor}/${seg}"
        if [ -L "${ancestor}" ] || ([ -e "${ancestor}" ] && [ ! -d "${ancestor}" ]); then
            echo "[eval-assets]   removing non-dir / symlink at ${ancestor}"
            rm -f "${ancestor}"
        fi
        [ -d "${ancestor}" ] || mkdir "${ancestor}"
    done
}

download_t2m_zip() {
    echo "[eval-assets] ensuring official HumanML3D pretrained models archive..."
    mkdir -p "${DOWNLOAD_DIR}"
    if ! command -v gdown >/dev/null 2>&1; then
        python -m pip install --user --quiet gdown
        export PATH="${HOME}/.local/bin:${PATH}"
    fi
    if [ -f "${ZIP}" ] && ! unzip -tq "${ZIP}" >/dev/null 2>&1; then
        echo "[eval-assets] existing ${ZIP} is not a valid zip; re-downloading"
        rm -f "${ZIP}"
    fi
    [ -f "${ZIP}" ] || python -m gdown "${T2M_GDRIVE_ID}" -O "${ZIP}"
    unzip -tq "${ZIP}" >/dev/null
}

extract_zip_member() {
    pattern="$1"
    dest="$2"
    tmp="${dest}.tmp"
    mkdir -p "$(dirname "${dest}")"
    rm -f "${tmp}"
    unzip -p "${ZIP}" "${pattern}" > "${tmp}"
    if [ ! -s "${tmp}" ]; then
        echo "[eval-assets] ERROR: failed to extract ${pattern} from ${ZIP}"
        rm -f "${tmp}"
        exit 1
    fi
    mv "${tmp}" "${dest}"
}

# Fast path: already set up.
if [ -f "${SUBMOD}/networks/modules.py" ] \
   && [ -f "${SUBMOD}/networks/evaluator_wrapper.py" ] \
   && [ -d "${SUBMOD}/glove" ] \
   && [ -f "${DEST}" ] \
   && [ -f "${META_MEAN}" ] \
   && [ -f "${META_STD}" ]; then
    echo "[eval-assets] present - skipping setup"
    exit 0
fi

echo "[eval-assets] missing assets - running setup (repo=${REPO})"

needs_clone=true
if [ -f "${SUBMOD}/networks/modules.py" ] \
   && [ -f "${SUBMOD}/networks/evaluator_wrapper.py" ] \
   && [ -d "${SUBMOD}/glove" ]; then
    needs_clone=false
fi

if [ "${needs_clone}" = "true" ]; then
    echo "[eval-assets] submodule is incomplete - re-cloning"

    if [ -f "${DEST}" ]; then
        echo "[eval-assets] stashing existing finest.tar to ${STASH}"
        cp "${DEST}" "${STASH}"
    fi

    if [ -d "${SUBMOD}" ]; then
        echo "[eval-assets] wiping ${SUBMOD}"
        rm -rf "${SUBMOD}"
    fi
    rm -rf ".git/modules/external/text-to-motion"

    echo "[eval-assets] cloning ${SUBMOD} (shallow, single-threaded)..."
    git -c pack.threads=1 -c pack.windowMemory=64m -c pack.deltaCacheSize=64m \
        submodule update --init --depth 1 --force -- "${SUBMOD}"

    if [ -f "${STASH}" ]; then
        echo "[eval-assets] restoring finest.tar -> ${DEST}"
        ensure_dest_dir
        mv "${STASH}" "${DEST}"
    fi
fi

if [ ! -f "${DEST}" ]; then
    echo "[eval-assets] finest.tar missing - scanning for it..."
    CANDIDATES=$(find \
        "${REPO}/external" \
        "${HOME}" \
        -maxdepth 6 -name 'finest.tar' 2>/dev/null | head -10 || true)
    echo "[eval-assets] candidates: ${CANDIDATES}"
    SRC=$(echo "${CANDIDATES}" | head -1)
    if [ -n "${GUO_TAR:-}" ]; then
        SRC="${GUO_TAR}"
    fi
    if [ -n "${SRC}" ] && [ -f "${SRC}" ]; then
        ensure_dest_dir
        cp "${SRC}" "${DEST}"
    elif [ "${DOWNLOAD_GUO:-0}" = "1" ]; then
        download_t2m_zip
        ensure_dest_dir
        extract_zip_member '*/text_mot_match/model/finest.tar' "${DEST}"
    else
        echo "[eval-assets] ERROR: finest.tar not found anywhere."
        echo "[eval-assets] pass GUO_TAR=/abs/path/to/finest.tar or set DOWNLOAD_GUO=1"
        exit 1
    fi
fi

if [ ! -f "${META_MEAN}" ] || [ ! -f "${META_STD}" ]; then
    echo "[eval-assets] evaluator normalization missing"
    if [ "${DOWNLOAD_GUO:-0}" = "1" ]; then
        download_t2m_zip
        extract_zip_member '*/Comp_v6_KLD01/meta/mean.npy' "${META_MEAN}"
        extract_zip_member '*/Comp_v6_KLD01/meta/std.npy' "${META_STD}"
    else
        echo "[eval-assets] ERROR: evaluator mean/std not found."
        echo "[eval-assets] set DOWNLOAD_GUO=1 to extract Comp_v6_KLD01/meta/{mean,std}.npy"
        exit 1
    fi
fi

echo "[eval-assets] verifying:"
test -f "${SUBMOD}/networks/modules.py"           && echo "  OK: networks/modules.py"           || { echo "  MISSING: networks/modules.py"; exit 1; }
test -f "${SUBMOD}/networks/evaluator_wrapper.py" && echo "  OK: networks/evaluator_wrapper.py" || { echo "  MISSING: evaluator_wrapper.py"; exit 1; }
test -d "${SUBMOD}/glove"                         && echo "  OK: glove/"                         || { echo "  MISSING: glove/"; exit 1; }
test -f "${DEST}"                                  && echo "  OK: ${DEST}"                        || { echo "  MISSING: ${DEST}"; exit 1; }
test -f "${META_MEAN}"                             && echo "  OK: ${META_MEAN}"                   || { echo "  MISSING: ${META_MEAN}"; exit 1; }
test -f "${META_STD}"                              && echo "  OK: ${META_STD}"                    || { echo "  MISSING: ${META_STD}"; exit 1; }

echo "[eval-assets] DONE"
