#!/bin/bash
#SBATCH --job-name=momask-fidsel
#SBATCH --partition=12g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1,ccc:75
#SBATCH --nodes=1
#SBATCH --output=slurm/logs/momask-fidsel-%j.out
#SBATCH --error=slurm/logs/momask-fidsel-%j.err
#
# Select M- and R-Transformer checkpoints with the paper's independent
# validation-FID protocols. All work runs inside this compute allocation.

set -euo pipefail
mkdir -p slurm/logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
fi

IMAGE="${IMAGE:-${HOME}/rmg-momask.sqsh}"
HF_CACHE="${HF_CACHE:-${HOME}/hf-cache}"
DATA_ROOT="${DATA_ROOT:-/mnt/projects/drl4cvb/human-motion-representation/data/data/humanml3d_packed}"
HUMANML3D_REPO="${HUMANML3D_REPO:-external/HumanML3D}"
HUMANML3D_DIR="${HUMANML3D_DIR:-/mnt/projects/drl4cvb/human-motion-representation/data/HumanML3D/HumanML3D}"
HUMANML3D_TEXTS_ZIP="${HUMANML3D_TEXTS_ZIP:-${HUMANML3D_DIR}/texts.zip}"
REAL_H3D_DIR="${REAL_H3D_DIR:-${HOME}/rmg-runs/h3d-canonical/new_joint_vecs}"

MASKED_CHECKPOINT_DIR="${MASKED_CHECKPOINT_DIR:-runs/momask-canonical-mtrans-releaseparity196-clip500e/checkpoints}"
RESIDUAL_CHECKPOINT_DIR="${RESIDUAL_CHECKPOINT_DIR:-runs/momask-canonical-rtrans-releaseparity196-clip500e/checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/momask-canonical-tokens-releaseparity196-valfidselection-full}"
CHECKPOINT_STRIDE="${CHECKPOINT_STRIDE:-1}"
OVERWRITE="${OVERWRITE:-0}"

SPLIT="${SPLIT:-val}"
MAX_CLIPS="${MAX_CLIPS:--1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-196}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:-40}"
MASKED_GENERATION_STEPS="${MASKED_GENERATION_STEPS:-18}"
MASKED_GUIDANCE_SCALE="${MASKED_GUIDANCE_SCALE:-4.0}"
RESIDUAL_GENERATION_STEPS="${RESIDUAL_GENERATION_STEPS:-1}"
RESIDUAL_GUIDANCE_SCALE="${RESIDUAL_GUIDANCE_SCALE:-2.0}"
FINAL_GENERATION_STEPS="${FINAL_GENERATION_STEPS:-10}"
FINAL_GUIDANCE_SCALE="${FINAL_GUIDANCE_SCALE:-4.0}"
FINAL_RESIDUAL_GUIDANCE_SCALE="${FINAL_RESIDUAL_GUIDANCE_SCALE:-5.0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOPK_FILTER_THRES="${TOPK_FILTER_THRES:-0.9}"
SAMPLE="${SAMPLE:-1}"
REMASK_KEPT_TOKENS="${REMASK_KEPT_TOKENS:-0}"
SEED="${SEED:-3407}"

echo "[momask-fidsel] host=$(hostname)"
echo "[momask-fidsel] repo=${REPO}"
echo "[momask-fidsel] masked_dir=${MASKED_CHECKPOINT_DIR}"
echo "[momask-fidsel] residual_dir=${RESIDUAL_CHECKPOINT_DIR}"
echo "[momask-fidsel] output_dir=${OUTPUT_DIR} stride=${CHECKPOINT_STRIDE} overwrite=${OVERWRITE}"
echo "[momask-fidsel] split=${SPLIT} max_clips=${MAX_CLIPS} max_seq_len=${MAX_SEQ_LEN}"
echo "[momask-fidsel] humanml3d_dir=${HUMANML3D_DIR} data_view=official_text2motion"
echo "[momask-fidsel] masked_steps=${MASKED_GENERATION_STEPS} masked_guidance=${MASKED_GUIDANCE_SCALE}"
echo "[momask-fidsel] residual_guidance=${RESIDUAL_GUIDANCE_SCALE} final_steps=${FINAL_GENERATION_STEPS} final_residual_guidance=${FINAL_RESIDUAL_GUIDANCE_SCALE}"
echo "[momask-fidsel] temp=${TEMPERATURE} topk=${TOPK_FILTER_THRES} sample=${SAMPLE} remask=${REMASK_KEPT_TOKENS} seed=${SEED}"

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
        cd '${REPO}'
        mkdir -p '${OUTPUT_DIR}/masked_evals' '${OUTPUT_DIR}/residual_evals'
        pip install -e . --no-deps --quiet
        python -m spacy download en_core_web_sm --quiet

        evaluate_manifest() {
            local manifest=\"\$1\"
            local eval_dir=\"\$2\"
            local variant=\"\$3\"
            local generation_steps=\"\$4\"
            local guidance_scale=\"\$5\"
            local protocol_id=\"\$6\"
            local scratch='${OUTPUT_DIR}/.candidate.pt'
            local label masked_ckpt residual_ckpt output
            while IFS=\$'\\t' read -r label masked_ckpt residual_ckpt; do
                if [ \"\${label}\" = label ]; then
                    continue
                fi
                output=\"\${eval_dir}/\${label}.json\"
                if [ '${OVERWRITE}' != 1 ] && [ -s \"\${output}\" ]; then
                    if python -m momask.scripts.select_transformer_checkpoints matches \\
                        --result \"\${output}\" \\
                        --label \"\${label}\" \\
                        --masked-checkpoint \"\${masked_ckpt}\" \\
                        --residual-checkpoint \"\${residual_ckpt}\" \\
                        --protocol-id \"\${protocol_id}\"; then
                        echo \"[momask-fidsel] reuse \${output}\"
                        continue
                    fi
                    echo \"[momask-fidsel] stale result will be replaced: \${output}\"
                fi
                echo \"[momask-fidsel] evaluate \${label}\"
                python -u -m momask.scripts.assemble_token_checkpoints \\
                    --masked-checkpoint \"\${masked_ckpt}\" \\
                    --residual-checkpoint \"\${residual_ckpt}\" \\
                    --output \"\${scratch}\" \\
                    --allow-unvalidated-components
                python -u -m momask.scripts.evaluate_momask \\
                    --checkpoint \"\${scratch}\" \\
                    --data-root '${DATA_ROOT}' \\
                    --humanml3d-repo '${HUMANML3D_REPO}' \\
                    --humanml3d-texts-zip '${HUMANML3D_TEXTS_ZIP}' \\
                    --humanml3d-split-dir '${HUMANML3D_DIR}' \\
                    --paper-transformer-data \\
                    --real-h3d-dir '${REAL_H3D_DIR}' \\
                    --split '${SPLIT}' \\
                    --max-clips ${MAX_CLIPS} \\
                    --batch-size ${BATCH_SIZE} \\
                    --max-seq-len ${MAX_SEQ_LEN} \\
                    --min-seq-len ${MIN_SEQ_LEN} \\
                    --variants \"\${variant}\" \\
                    --generation-steps \"\${generation_steps}\" \\
                    --text-encoder checkpoint \\
                    --clip-cache-dir '${HF_CACHE}' \\
                    --clip-backend checkpoint \\
                    --guidance-scale \"\${guidance_scale}\" \\
                    --temperature ${TEMPERATURE} \\
                    --topk-filter-thres ${TOPK_FILTER_THRES} \\
                    $([ "${SAMPLE}" = 1 ] && printf '%s' '--sample' || printf '%s' '--no-sample') \\
                    $([ "${REMASK_KEPT_TOKENS}" = 1 ] && printf '%s' '--remask-kept-tokens' || printf '%s' '--no-remask-kept-tokens') \\
                    --seed ${SEED} \\
                    --evaluator real \\
                    --output \"\${output}\"
                python -m momask.scripts.select_transformer_checkpoints annotate \\
                    --result \"\${output}\" \\
                    --label \"\${label}\" \\
                    --masked-checkpoint \"\${masked_ckpt}\" \\
                    --residual-checkpoint \"\${residual_ckpt}\" \\
                    --protocol-id \"\${protocol_id}\"
                rm -f \"\${scratch}\"
            done < \"\${manifest}\"
        }

        common_protocol='split=${SPLIT};max_clips=${MAX_CLIPS};batch_size=${BATCH_SIZE}'
        common_protocol=\"\${common_protocol};max_seq_len=${MAX_SEQ_LEN};min_seq_len=${MIN_SEQ_LEN}\"
        common_protocol=\"\${common_protocol};temperature=${TEMPERATURE};topk=${TOPK_FILTER_THRES}\"
        common_protocol=\"\${common_protocol};sample=${SAMPLE};remask=${REMASK_KEPT_TOKENS};seed=${SEED}\"
        common_protocol=\"\${common_protocol};data_root=${DATA_ROOT}\"
        common_protocol=\"\${common_protocol};texts=${HUMANML3D_TEXTS_ZIP};real=${REAL_H3D_DIR}\"
        common_protocol=\"\${common_protocol};splits=${HUMANML3D_DIR};view=official_text2motion\"
        masked_protocol=\"phase=masked;variant=base;steps=${MASKED_GENERATION_STEPS}\"
        masked_protocol=\"\${masked_protocol};guidance=${MASKED_GUIDANCE_SCALE};\${common_protocol}\"
        residual_protocol='phase=residual;variant=teacher_residual'
        residual_protocol=\"\${residual_protocol};steps=${RESIDUAL_GENERATION_STEPS}\"
        residual_protocol=\"\${residual_protocol};guidance=${RESIDUAL_GUIDANCE_SCALE}\"
        residual_protocol=\"\${residual_protocol};\${common_protocol}\"
        final_protocol=\"phase=final;variant=full;steps=${FINAL_GENERATION_STEPS}\"
        final_protocol=\"\${final_protocol};guidance=${FINAL_GUIDANCE_SCALE}\"
        final_protocol=\"\${final_protocol};residual_guidance=${FINAL_RESIDUAL_GUIDANCE_SCALE};\${common_protocol}\"

        residual_reference='${RESIDUAL_CHECKPOINT_DIR}/tokens_best_val.pt'
        python -u -m momask.scripts.select_transformer_checkpoints manifest \\
            --vary masked \\
            --checkpoint-dir '${MASKED_CHECKPOINT_DIR}' \\
            --fixed-checkpoint \"\${residual_reference}\" \\
            --stride ${CHECKPOINT_STRIDE} \\
            --output '${OUTPUT_DIR}/masked_manifest.tsv'
        evaluate_manifest \\
            '${OUTPUT_DIR}/masked_manifest.tsv' \\
            '${OUTPUT_DIR}/masked_evals' \\
            base \\
            ${MASKED_GENERATION_STEPS} \\
            ${MASKED_GUIDANCE_SCALE} \\
            \"\${masked_protocol}\"
        python -u -m momask.scripts.select_transformer_checkpoints summarize \\
            --manifest '${OUTPUT_DIR}/masked_manifest.tsv' \\
            --eval-dir '${OUTPUT_DIR}/masked_evals' \\
            --variant base \\
            --protocol-id \"\${masked_protocol}\" \\
            --output '${OUTPUT_DIR}/summary_masked.csv' \\
            --best-env '${OUTPUT_DIR}/best_masked.env'

        source '${OUTPUT_DIR}/best_masked.env'
        echo \"[momask-fidsel] selected masked=\${BEST_MASKED_CKPT} fid=\${BEST_FID}\"
        python -u -m momask.scripts.select_transformer_checkpoints manifest \\
            --vary residual \\
            --checkpoint-dir '${RESIDUAL_CHECKPOINT_DIR}' \\
            --fixed-checkpoint \"\${BEST_MASKED_CKPT}\" \\
            --stride ${CHECKPOINT_STRIDE} \\
            --output '${OUTPUT_DIR}/residual_manifest.tsv'
        evaluate_manifest \\
            '${OUTPUT_DIR}/residual_manifest.tsv' \\
            '${OUTPUT_DIR}/residual_evals' \\
            teacher_residual \\
            ${RESIDUAL_GENERATION_STEPS} \\
            ${RESIDUAL_GUIDANCE_SCALE} \\
            \"\${residual_protocol}\"
        python -u -m momask.scripts.select_transformer_checkpoints summarize \\
            --manifest '${OUTPUT_DIR}/residual_manifest.tsv' \\
            --eval-dir '${OUTPUT_DIR}/residual_evals' \\
            --variant teacher_residual \\
            --protocol-id \"\${residual_protocol}\" \\
            --output '${OUTPUT_DIR}/summary_residual.csv' \\
            --best-env '${OUTPUT_DIR}/best_final.env' \\
            --best-checkpoint '${OUTPUT_DIR}/momask_best_val_fid.pt'

        source '${OUTPUT_DIR}/best_final.env'
        final_output='${OUTPUT_DIR}/selected_full_validation.json'
        if [ '${OVERWRITE}' != 1 ] && [ -s \"\${final_output}\" ] && \\
            python -m momask.scripts.select_transformer_checkpoints matches \\
                --result \"\${final_output}\" \\
                --label selected_pair \\
                --masked-checkpoint \"\${BEST_MASKED_CKPT}\" \\
                --residual-checkpoint \"\${BEST_RESIDUAL_CKPT}\" \\
                --protocol-id \"\${final_protocol}\"; then
            echo \"[momask-fidsel] reuse \${final_output}\"
        else
            python -u -m momask.scripts.evaluate_momask \\
                --checkpoint '${OUTPUT_DIR}/momask_best_val_fid.pt' \\
                --data-root '${DATA_ROOT}' \\
                --humanml3d-repo '${HUMANML3D_REPO}' \\
                --humanml3d-texts-zip '${HUMANML3D_TEXTS_ZIP}' \\
                --humanml3d-split-dir '${HUMANML3D_DIR}' \\
                --paper-transformer-data \\
                --real-h3d-dir '${REAL_H3D_DIR}' \\
                --split '${SPLIT}' \\
                --max-clips ${MAX_CLIPS} \\
                --batch-size ${BATCH_SIZE} \\
                --max-seq-len ${MAX_SEQ_LEN} \\
                --min-seq-len ${MIN_SEQ_LEN} \\
                --variants full \\
                --generation-steps ${FINAL_GENERATION_STEPS} \\
                --text-encoder checkpoint \\
                --clip-cache-dir '${HF_CACHE}' \\
                --clip-backend checkpoint \\
                --guidance-scale ${FINAL_GUIDANCE_SCALE} \\
                --residual-guidance-scale ${FINAL_RESIDUAL_GUIDANCE_SCALE} \\
                --temperature ${TEMPERATURE} \\
                --topk-filter-thres ${TOPK_FILTER_THRES} \\
                $([ "${SAMPLE}" = 1 ] && printf '%s' '--sample' || printf '%s' '--no-sample') \\
                $([ "${REMASK_KEPT_TOKENS}" = 1 ] && printf '%s' '--remask-kept-tokens' || printf '%s' '--no-remask-kept-tokens') \\
                --seed ${SEED} \\
                --evaluator real \\
                --output \"\${final_output}\"
            python -m momask.scripts.select_transformer_checkpoints annotate \\
                --result \"\${final_output}\" \\
                --label selected_pair \\
                --masked-checkpoint \"\${BEST_MASKED_CKPT}\" \\
                --residual-checkpoint \"\${BEST_RESIDUAL_CKPT}\" \\
                --protocol-id \"\${final_protocol}\"
        fi
        rm -f '${OUTPUT_DIR}/.candidate.pt'
        echo '[momask-fidsel] DONE'
        cat '${OUTPUT_DIR}/best_final.env'
    "
