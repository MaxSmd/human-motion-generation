#!/bin/bash
#SBATCH --job-name=momask-vq-official
#SBATCH --partition=12g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1,ccc:75
#SBATCH --nodes=1
#SBATCH --output=slurm/logs/momask-vq-official-%j.out
#SBATCH --error=slurm/logs/momask-vq-official-%j.err

set -euo pipefail
mkdir -p slurm/logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
fi

IMAGE="${IMAGE:-${HOME}/rmg-momask.sqsh}"
OFFICIAL_REPO="${OFFICIAL_MOMASK_REPO:-${REPO}/external/momask-codes-official}"
HUMANML3D_DIR="${HUMANML3D_DIR:-/mnt/projects/drl4cvb/human-motion-representation/data/HumanML3D/HumanML3D}"
CANONICAL_H3D_DIR="${CANONICAL_H3D_DIR:-${HOME}/rmg-runs/h3d-canonical/new_joint_vecs}"
TEXTS_ZIP="${TEXTS_ZIP:-${HUMANML3D_DIR}/texts.zip}"
TEXTS_CACHE_ROOT="${TEXTS_CACHE_ROOT:-${HOME}/rmg-runs/h3d-official-eval}"
GLOVE_DIR="${GLOVE_DIR:-${REPO}/external/text-to-motion/glove}"
EVALUATOR_ROOT="${EVALUATOR_ROOT:-${REPO}/external/text-to-motion/checkpoints/momask_official}"
VQ_NAME="${OFFICIAL_VQ_NAME:-rvq_nq6_dc512_nc512_noshare_qdp0.2}"
WHICH_EPOCH="${WHICH_EPOCH:-net_best_fid.tar}"
EXT="${EXT:-local_humanml3d_official_protocol}"
OUTPUT="${OUTPUT:-runs/momask_diagnostics/official_eval_t2m_vq.log}"
if [[ "${OUTPUT}" = /* ]]; then
    OUTPUT_PATH="${OUTPUT}"
else
    OUTPUT_PATH="${REPO}/${OUTPUT}"
fi

require_file() {
    if [ ! -s "$1" ]; then
        echo "[momask-vq-official] ERROR: $2 not found: $1" >&2
        exit 2
    fi
}

require_file "${OFFICIAL_REPO}/eval_t2m_vq.py" "official evaluator entrypoint"
require_file "${OFFICIAL_REPO}/checkpoints/t2m/${VQ_NAME}/model/${WHICH_EPOCH}" "official RVQ checkpoint"
require_file "${OFFICIAL_REPO}/checkpoints/t2m/${VQ_NAME}/opt.txt" "official RVQ options"
require_file "${EVALUATOR_ROOT}/t2m/Comp_v6_KLD005/opt.txt" "downloaded Guo evaluator options"
require_file "${EVALUATOR_ROOT}/t2m/text_mot_match/model/finest.tar" "downloaded Guo evaluator"
require_file "${HUMANML3D_DIR}/test.txt" "HumanML3D test split"
require_file "${GLOVE_DIR}/our_vab_data.npy" "GloVe vectors"
if [ ! -d "${CANONICAL_H3D_DIR}" ]; then
    echo "[momask-vq-official] ERROR: canonical motions not found: ${CANONICAL_H3D_DIR}" >&2
    exit 2
fi
require_file "${CANONICAL_H3D_DIR}/004822.npy" "canonical test motion"

mkdir -p "${OFFICIAL_REPO}/dataset/HumanML3D" "$(dirname "${OUTPUT_PATH}")"

link_directory() {
    local target="$1"
    local link="$2"
    local label="$3"
    if [ -L "${link}" ]; then
        if [ "$(readlink -f "${link}")" != "$(readlink -f "${target}")" ]; then
            echo "[momask-vq-official] ERROR: ${label} symlink points elsewhere: ${link}" >&2
            exit 2
        fi
    elif [ -e "${link}" ]; then
        echo "[momask-vq-official] ERROR: ${label} path exists and is not a symlink: ${link}" >&2
        exit 2
    else
        ln -s "${target}" "${link}"
    fi
}

if [ ! -e "${OFFICIAL_REPO}/checkpoints/t2m/Comp_v6_KLD005" ]; then
    link_directory \
        "${EVALUATOR_ROOT}/t2m/Comp_v6_KLD005" \
        "${OFFICIAL_REPO}/checkpoints/t2m/Comp_v6_KLD005" \
        "Guo evaluator options"
fi
if [ ! -e "${OFFICIAL_REPO}/checkpoints/t2m/text_mot_match" ]; then
    link_directory \
        "${EVALUATOR_ROOT}/t2m/text_mot_match" \
        "${OFFICIAL_REPO}/checkpoints/t2m/text_mot_match" \
        "Guo evaluator checkpoint"
fi
require_file "${OFFICIAL_REPO}/checkpoints/t2m/Comp_v6_KLD005/opt.txt" "linked Guo evaluator options"
require_file "${OFFICIAL_REPO}/checkpoints/t2m/text_mot_match/model/finest.tar" "linked Guo evaluator"

if [ -d "${HUMANML3D_DIR}/texts" ]; then
    TEXTS_DIR="${HUMANML3D_DIR}/texts"
else
    require_file "${TEXTS_ZIP}" "HumanML3D caption archive"
    if ! command -v unzip >/dev/null 2>&1; then
        echo "[momask-vq-official] ERROR: unzip is unavailable" >&2
        exit 2
    fi
    mkdir -p "${TEXTS_CACHE_ROOT}"
    if [ ! -s "${TEXTS_CACHE_ROOT}/texts/000000.txt" ] && [ ! -s "${TEXTS_CACHE_ROOT}/000000.txt" ]; then
        echo "[momask-vq-official] extracting caption files on compute node"
        unzip -oq "${TEXTS_ZIP}" -d "${TEXTS_CACHE_ROOT}"
    fi
    if [ -s "${TEXTS_CACHE_ROOT}/texts/000000.txt" ]; then
        TEXTS_DIR="${TEXTS_CACHE_ROOT}/texts"
    elif [ -s "${TEXTS_CACHE_ROOT}/000000.txt" ]; then
        TEXTS_DIR="${TEXTS_CACHE_ROOT}"
    else
        echo "[momask-vq-official] ERROR: extracted caption files were not found" >&2
        exit 2
    fi
fi

link_directory "${CANONICAL_H3D_DIR}" "${OFFICIAL_REPO}/dataset/HumanML3D/new_joint_vecs" "canonical motions"
link_directory "${TEXTS_DIR}" "${OFFICIAL_REPO}/dataset/HumanML3D/texts" "HumanML3D captions"
link_directory "${GLOVE_DIR}" "${OFFICIAL_REPO}/glove" "GloVe"
for FILE in train.txt val.txt test.txt train_val.txt; do
    if [ -s "${HUMANML3D_DIR}/${FILE}" ] && [ ! -e "${OFFICIAL_REPO}/dataset/HumanML3D/${FILE}" ]; then
        ln -s "${HUMANML3D_DIR}/${FILE}" "${OFFICIAL_REPO}/dataset/HumanML3D/${FILE}"
    fi
done

echo "[momask-vq-official] host=$(hostname) repo=${REPO}"
echo "[momask-vq-official] official_repo=${OFFICIAL_REPO}"
echo "[momask-vq-official] motions=${CANONICAL_H3D_DIR} texts=${TEXTS_DIR}"
echo "[momask-vq-official] checkpoint=${VQ_NAME}/model/${WHICH_EPOCH}"
echo "[momask-vq-official] output=${OUTPUT}"

enroot start \
    --root \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --mount "${REPO}:${REPO}" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        cd '${OFFICIAL_REPO}'
        if ! python -c 'import clip' >/dev/null 2>&1; then
            echo '[momask-vq-official] installing official OpenAI CLIP dependency'
            pip install --quiet --no-cache-dir \
                'git+https://github.com/openai/CLIP.git'
        fi
        python -u eval_t2m_vq.py \\
            --gpu_id 0 \\
            --name '${VQ_NAME}' \\
            --dataset_name t2m \\
            --which_epoch '${WHICH_EPOCH}' \\
            --ext '${EXT}'
        cp -- 'checkpoints/t2m/${VQ_NAME}/eval/${EXT}.log' '${OUTPUT_PATH}'
    "

echo "[momask-vq-official] DONE output=${OUTPUT}"
