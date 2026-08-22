#!/bin/bash
#SBATCH --job-name=momask-vqsel
#SBATCH --partition=12g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1,ccc:75
#SBATCH --nodes=1
#SBATCH --output=slurm/logs/momask-vqsel-%j.out
#SBATCH --error=slurm/logs/momask-vqsel-%j.err

set -euo pipefail
mkdir -p slurm/logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
fi

IMAGE="${IMAGE:-${HOME}/rmg-momask.sqsh}"
DATA_ROOT="${DATA_ROOT:-/mnt/projects/drl4cvb/human-motion-representation/data/data/humanml3d_packed}"
HUMANML3D_REPO="${HUMANML3D_REPO:-external/HumanML3D}"
HUMANML3D_DIR="${HUMANML3D_DIR:-/mnt/projects/drl4cvb/human-motion-representation/data/HumanML3D/HumanML3D}"
HUMANML3D_TEXTS_ZIP="${HUMANML3D_TEXTS_ZIP:-${HUMANML3D_DIR}/texts.zip}"
REAL_H3D_DIR="${REAL_H3D_DIR:-${HOME}/rmg-runs/h3d-canonical/new_joint_vecs}"
EVALUATOR_CHECKPOINTS_DIR="${EVALUATOR_CHECKPOINTS_DIR:-external/text-to-motion/checkpoints/momask_official}"
EVALUATOR_NORMALIZATION_NAME="${EVALUATOR_NORMALIZATION_NAME:-Comp_v6_KLD005}"

RUN_NAME="${RUN_NAME:-momask-canonical-vq-releaseparity64-50e}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-runs/${RUN_NAME}/checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/${RUN_NAME}/val_fid_selection}"
MAX_CLIPS="${MAX_CLIPS:--1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-196}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:-40}"
SEED="${SEED:-3407}"
OVERWRITE="${OVERWRITE:-1}"

echo "[momask-vqsel] host=$(hostname)"
echo "[momask-vqsel] checkpoint_dir=${CHECKPOINT_DIR} output_dir=${OUTPUT_DIR}"
echo "[momask-vqsel] split=val max_clips=${MAX_CLIPS} max_seq_len=${MAX_SEQ_LEN}"

enroot start \
    --root \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --mount "${REPO}:${REPO}" \
    --env "RMG_DATA_ROOT=${DATA_ROOT}" \
    --env "PYTHONPATH=${REPO}/src" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        cd '${REPO}'
        mkdir -p '${OUTPUT_DIR}/evals'
        pip install -e . --no-deps --quiet
        shopt -s nullglob
        checkpoints=( '${CHECKPOINT_DIR}'/vq_step_*.pt )
        if [ \${#checkpoints[@]} -eq 0 ]; then
            echo '[momask-vqsel] ERROR: no epoch checkpoints found under ${CHECKPOINT_DIR}' >&2
            exit 2
        fi
        for checkpoint in "\${checkpoints[@]}"; do
            label=\$(basename "\${checkpoint}" .pt)
            result='${OUTPUT_DIR}/evals/'"\${label}"'.json'
            if [ '${OVERWRITE}' != 1 ] && [ -s "\${result}" ] && [ "\${result}" -nt "\${checkpoint}" ]; then
                echo "[momask-vqsel] reuse \${result}"
                continue
            fi
            if [ -s "\${result}" ]; then
                echo "[momask-vqsel] refresh stale result \${result}"
            fi
            echo "[momask-vqsel] evaluate \${checkpoint}"
            python -u -m momask.scripts.evaluate_momask \
                --checkpoint "\${checkpoint}" \
                --data-root '${DATA_ROOT}' \
                --humanml3d-repo '${HUMANML3D_REPO}' \
                --humanml3d-texts-zip '${HUMANML3D_TEXTS_ZIP}' \
                --humanml3d-split-dir '${HUMANML3D_DIR}' \
                --paper-transformer-data \
                --real-h3d-dir '${REAL_H3D_DIR}' \
                --split val \
                --max-clips ${MAX_CLIPS} \
                --batch-size ${BATCH_SIZE} \
                --max-seq-len ${MAX_SEQ_LEN} \
                --min-seq-len ${MIN_SEQ_LEN} \
                --variants recon_unmasked \
                --seed ${SEED} \
                --evaluator real \
                --evaluator-checkpoints-dir '${EVALUATOR_CHECKPOINTS_DIR}' \
                --evaluator-normalization-name '${EVALUATOR_NORMALIZATION_NAME}' \
                --output "\${result}"
        done
        python -u -m momask.scripts.select_vq_checkpoints \
            --eval-dir '${OUTPUT_DIR}/evals' \
            --variant recon_unmasked \
            --summary '${OUTPUT_DIR}/summary.csv' \
            --best-checkpoint '${OUTPUT_DIR}/momask_best_val_fid.pt' \
            --selection-json '${OUTPUT_DIR}/selection.json'
        echo '[momask-vqsel] DONE'
    "
