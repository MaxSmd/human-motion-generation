#!/bin/bash
# Install the evaluator archive used by the official MoMask release into an
# isolated checkpoint root. Run this only from a compute/data-node batch job.

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
cd "${REPO}"

SUBMOD="external/text-to-motion"
OFFICIAL_ROOT="${MOMASK_EVALUATOR_ROOT:-${SUBMOD}/checkpoints/momask_official}"
T2M_ROOT="${OFFICIAL_ROOT}/t2m"
CHECKPOINT="${T2M_ROOT}/text_mot_match/model/finest.tar"
META_MEAN="${T2M_ROOT}/Comp_v6_KLD005/meta/mean.npy"
META_STD="${T2M_ROOT}/Comp_v6_KLD005/meta/std.npy"
OPT_FILE="${T2M_ROOT}/Comp_v6_KLD005/opt.txt"
DOWNLOAD_DIR="${HOME}/rmg-eval-assets"
ARCHIVE="${DOWNLOAD_DIR}/humanml3d_evaluator.zip"
ARCHIVE_ID="${MOMASK_EVALUATOR_GDRIVE_ID:-19C_eiEr0kMGlYVJy_yFL6_Dhk3RvmwhM}"

if [ -f "${CHECKPOINT}" ] && [ -f "${META_MEAN}" ] && [ -f "${META_STD}" ]; then
    echo "[momask-eval-assets] present root=${OFFICIAL_ROOT}"
    exit 0
fi

if [ ! -f "${SUBMOD}/networks/evaluator_wrapper.py" ] || [ ! -d "${SUBMOD}/glove" ]; then
    echo "[momask-eval-assets] ERROR: external/text-to-motion is incomplete" >&2
    echo "[momask-eval-assets] initialize the evaluator submodule first" >&2
    exit 1
fi
if [ "${DOWNLOAD_MOMASK_EVALUATOR:-0}" != "1" ]; then
    echo "[momask-eval-assets] ERROR: official MoMask evaluator assets are missing" >&2
    echo "[momask-eval-assets] submit slurm/momask/init_momask_evaluator_assets.sbatch" >&2
    exit 1
fi
if ! command -v unzip >/dev/null 2>&1; then
    echo "[momask-eval-assets] ERROR: unzip is not available on this node" >&2
    exit 1
fi

mkdir -p "${DOWNLOAD_DIR}"
if ! command -v gdown >/dev/null 2>&1; then
    python -m pip install --user --quiet gdown
    export PATH="${HOME}/.local/bin:${PATH}"
fi
if [ -f "${ARCHIVE}" ] && ! unzip -tq "${ARCHIVE}" >/dev/null 2>&1; then
    echo "[momask-eval-assets] invalid cached archive; downloading it again"
    rm -f "${ARCHIVE}"
fi
if [ ! -f "${ARCHIVE}" ]; then
    echo "[momask-eval-assets] downloading official HumanML3D evaluator archive"
    python -m gdown "${ARCHIVE_ID}" -O "${ARCHIVE}"
fi
unzip -tq "${ARCHIVE}" >/dev/null

archive_member() {
    suffix="$1"
    unzip -Z1 "${ARCHIVE}" | awk -v suffix="${suffix}" '
        length($0) >= length(suffix) &&
        substr($0, length($0) - length(suffix) + 1) == suffix && !found {
            print
            found = 1
        }
    '
}

extract_member() {
    suffix="$1"
    destination="$2"
    member="$(archive_member "${suffix}")"
    if [ -z "${member}" ]; then
        echo "[momask-eval-assets] ERROR: ${suffix} is absent from ${ARCHIVE}" >&2
        exit 1
    fi
    mkdir -p "$(dirname "${destination}")"
    temporary="${destination}.tmp-${SLURM_JOB_ID:-$$}"
    rm -f "${temporary}"
    unzip -p "${ARCHIVE}" "${member}" > "${temporary}"
    if [ ! -s "${temporary}" ]; then
        echo "[momask-eval-assets] ERROR: failed to extract ${member}" >&2
        rm -f "${temporary}"
        exit 1
    fi
    mv "${temporary}" "${destination}"
    echo "[momask-eval-assets] extracted ${member} -> ${destination}"
}

extract_member "text_mot_match/model/finest.tar" "${CHECKPOINT}"
extract_member "Comp_v6_KLD005/meta/mean.npy" "${META_MEAN}"
extract_member "Comp_v6_KLD005/meta/std.npy" "${META_STD}"
if [ ! -f "${OPT_FILE}" ]; then
    extract_member "Comp_v6_KLD005/opt.txt" "${OPT_FILE}"
fi

test -s "${CHECKPOINT}"
test -s "${META_MEAN}"
test -s "${META_STD}"

compare_asset() {
    label="$1"
    official="$2"
    legacy="$3"
    if [ ! -f "${legacy}" ]; then
        echo "[momask-eval-assets] compare ${label}: legacy asset absent"
    elif cmp -s "${official}" "${legacy}"; then
        echo "[momask-eval-assets] compare ${label}: identical"
    else
        echo "[momask-eval-assets] compare ${label}: DIFFERENT"
    fi
}

compare_asset \
    "finest.tar" \
    "${CHECKPOINT}" \
    "${SUBMOD}/checkpoints/t2m/text_mot_match/model/finest.tar"
compare_asset \
    "mean.npy KLD005-vs-KLD01" \
    "${META_MEAN}" \
    "${SUBMOD}/checkpoints/t2m/Comp_v6_KLD01/meta/mean.npy"
compare_asset \
    "std.npy KLD005-vs-KLD01" \
    "${META_STD}" \
    "${SUBMOD}/checkpoints/t2m/Comp_v6_KLD01/meta/std.npy"
echo "[momask-eval-assets] DONE root=${OFFICIAL_ROOT}"
