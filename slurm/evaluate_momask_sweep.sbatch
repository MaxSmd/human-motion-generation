#!/bin/bash
#SBATCH --job-name=momask-sweep
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --output=slurm/logs/momask-sweep-%j.out
#SBATCH --error=slurm/logs/momask-sweep-%j.err

set -euo pipefail
mkdir -p slurm/logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(dirname "${SCRIPT_DIR}")}"
fi

IMAGE="${IMAGE:-${HOME}/rmg-momask.sqsh}"
HF_CACHE="${HF_CACHE:-${HOME}/hf-cache}"
DATA_ROOT="${DATA_ROOT:-/mnt/projects/drl4cvb/human-motion-representation/data/data/humanml3d_packed}"
HUMANML3D_REPO="${HUMANML3D_REPO:-external/HumanML3D}"
HUMANML3D_TEXTS_ZIP="${HUMANML3D_TEXTS_ZIP:-}"
MAX_CLIPS="${MAX_CLIPS:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-80}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:-40}"
VARIANTS="${VARIANTS:-full}"
GENERATION_STEPS="${GENERATION_STEPS:-10}"
GUIDANCE_SCALES="${GUIDANCE_SCALES:-1 2 4 6}"
TEMPERATURES="${TEMPERATURES:-0.7 1.0}"
TEXT_ENCODER="${TEXT_ENCODER:-checkpoint}"
CLIP_MODEL="${CLIP_MODEL:-}"
CLIP_CACHE_DIR="${CLIP_CACHE_DIR:-${HF_CACHE}}"
CLIP_BACKEND="${CLIP_BACKEND:-auto}"
EVALUATOR="${EVALUATOR:-real}"
SWEEP_NAME="${SWEEP_NAME:-clip_checkpoint_sampling_512}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/momask_sweeps/${SWEEP_NAME}}"
CKPTS="${CKPTS:-runs/momask-full-tokens-clip150k/momask_smoke_latest.pt runs/momask-full-tokens-clip300k/checkpoints/tokens_step_0200000.pt runs/momask-full-tokens-clip300k/checkpoints/tokens_step_0230000.pt runs/momask-full-tokens-clip300k/checkpoints/tokens_step_0250000.pt runs/momask-full-tokens-clip300k/checkpoints/tokens_step_0270000.pt}"

echo "[momask-sweep] host=$(hostname)"
echo "[momask-sweep] repo=${REPO}"
echo "[momask-sweep] image=${IMAGE}"
echo "[momask-sweep] humanml3d_repo=${HUMANML3D_REPO}"
echo "[momask-sweep] humanml3d_texts_zip=${HUMANML3D_TEXTS_ZIP:-<auto>}"
echo "[momask-sweep] output_dir=${OUTPUT_DIR}"
echo "[momask-sweep] max_clips=${MAX_CLIPS} variants=${VARIANTS} steps=${GENERATION_STEPS}"
echo "[momask-sweep] guidance=${GUIDANCE_SCALES}"
echo "[momask-sweep] temperatures=${TEMPERATURES}"
echo "[momask-sweep] ckpts=${CKPTS}"

enroot start \
    --root \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --mount "${REPO}:${REPO}" \
    --env "RMG_DATA_ROOT=${DATA_ROOT}" \
    --env "HF_HOME=${HF_CACHE}" \
    --env "PYTHONPATH=${REPO}/src" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        cd ${REPO}
        mkdir -p '${OUTPUT_DIR}'
        pip install -e . --no-deps --quiet
        python -m spacy download en_core_web_sm --quiet
        TEXTS_ZIP_ARG=''
        if [ -n '${HUMANML3D_TEXTS_ZIP}' ]; then
            TEXTS_ZIP_ARG=\"--humanml3d-texts-zip '${HUMANML3D_TEXTS_ZIP}'\"
        fi

        for ckpt in ${CKPTS}; do
            if [ ! -f \"\${ckpt}\" ]; then
                echo \"[momask-sweep] WARN missing checkpoint: \${ckpt}\" >&2
                continue
            fi
            ckpt_base=\"\$(basename \"\${ckpt}\" .pt)\"
            ckpt_dir=\"\$(basename \"\$(dirname \"\${ckpt}\")\")\"
            if [ \"\${ckpt_dir}\" = checkpoints ]; then
                ckpt_dir=\"\$(basename \"\$(dirname \"\$(dirname \"\${ckpt}\")\")\")\"
            fi
            for guidance in ${GUIDANCE_SCALES}; do
                for temp in ${TEMPERATURES}; do
                    tag=\"\${ckpt_dir}_\${ckpt_base}_g\${guidance}_t\${temp}\"
                    tag=\"\${tag//./p}\"
                    out=\"${OUTPUT_DIR}/\${tag}.json\"
                    echo \"[momask-sweep] evaluating ckpt=\${ckpt} guidance=\${guidance} temp=\${temp} out=\${out}\"
                    python -u -m rmg.scripts.evaluate_momask \
                        --checkpoint \"\${ckpt}\" \
                        --data-root '${DATA_ROOT}' \
                        --humanml3d-repo '${HUMANML3D_REPO}' \
                        \${TEXTS_ZIP_ARG} \
                        --split test \
                        --max-clips ${MAX_CLIPS} \
                        --batch-size ${BATCH_SIZE} \
                        --max-seq-len ${MAX_SEQ_LEN} \
                        --min-seq-len ${MIN_SEQ_LEN} \
                        --variants '${VARIANTS}' \
                        --generation-steps ${GENERATION_STEPS} \
                        --text-encoder '${TEXT_ENCODER}' \
                        --clip-model '${CLIP_MODEL}' \
                        --clip-cache-dir '${CLIP_CACHE_DIR}' \
                        --clip-backend '${CLIP_BACKEND}' \
                        --guidance-scale \"\${guidance}\" \
                        --temperature \"\${temp}\" \
                        --evaluator '${EVALUATOR}' \
                        --output \"\${out}\"
                done
            done
        done

        python -u -m rmg.scripts.summarize_momask_sweep '${OUTPUT_DIR}' --variant full
    "
