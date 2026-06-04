#!/bin/bash
#SBATCH --job-name=rmg-eval
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --output=slurm/logs/rmg-eval-%j.out
#SBATCH --error=slurm/logs/rmg-eval-%j.err
#
# Evaluate an RMG checkpoint. Defaults to the full Table 4 / Figure 3a
# protocol (test split, 8 guidance scales) but every knob is overridable
# via env var for fast iteration cycles.
#
# Common usages:
#
#   # Full protocol (~45 min on 24g)
#   CKPT=/path sbatch slurm/evaluate.sbatch
#
#   # Fast test-split eval (~5 min): 256 clips, single guidance, reduced MM
#   CKPT=/path MAX_CLIPS=256 GUIDANCE_SCALES='[6.5]' MM_NUM_TEXTS=10 MM_REPEATS=3 \
#       sbatch slurm/evaluate.sbatch
#
#   # Train-subset eval (overfit sanity check on what the model actually saw):
#   CKPT=/path EVAL_SPLIT=train SUBSET_FRACTION=0.01 SUBSET_SEED=0 \
#       MAX_CLIPS=128 GUIDANCE_SCALES='[6.5]' MM_NUM_TEXTS=10 MM_REPEATS=3 \
#       sbatch slurm/evaluate.sbatch

set -euo pipefail
mkdir -p slurm/logs

CKPT="${CKPT:-${1:?set CKPT env var or pass as first positional arg}}"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(dirname "${SCRIPT_DIR}")}"
fi
IMAGE="${IMAGE:-${HOME}/rmg.sqsh}"
HF_CACHE="${HF_CACHE:-${HOME}/hf-cache}"
PRESET="${PRESET:-rmg_base}"
MODEL_PRESET="${MODEL_PRESET:-dit_base}"

# Eval-protocol knobs (all overridable for fast iteration)
EVAL_SPLIT="${EVAL_SPLIT:-test}"
MAX_CLIPS="${MAX_CLIPS:--1}"
GUIDANCE_SCALES="${GUIDANCE_SCALES:-[2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5]}"
MM_NUM_TEXTS="${MM_NUM_TEXTS:-30}"
MM_REPEATS="${MM_REPEATS:-10}"
SUBSET_FRACTION="${SUBSET_FRACTION:-1.0}"
SUBSET_SEED="${SUBSET_SEED:-0}"
NUM_SAMPLE_STEPS="${NUM_SAMPLE_STEPS:-50}"

enroot start \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --env "HF_HOME=${HF_CACHE}" \
    --env "PYTHONPATH=${REPO}/src" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        cd ${REPO}
        pip install -e . --no-deps --quiet
        python -m spacy download en_core_web_sm --quiet || true
        python -u scripts/evaluate.py \\
            model=${MODEL_PRESET} \\
            train=${PRESET} \\
            text_encoder.type=qwen3 \\
            data.subset_fraction=${SUBSET_FRACTION} \\
            data.subset_seed=${SUBSET_SEED} \\
            +eval.checkpoint=${CKPT} \\
            +eval.evaluator=real \\
            +eval.split=${EVAL_SPLIT} \\
            +eval.max_clips=${MAX_CLIPS} \\
            +eval.mm_num_texts=${MM_NUM_TEXTS} \\
            +eval.mm_repeats=${MM_REPEATS} \\
            +eval.num_sample_steps=${NUM_SAMPLE_STEPS} \\
            +eval.guidance_scales='${GUIDANCE_SCALES}' \\
            run_name=eval-\$(basename ${CKPT} .pt)-\$(date +%H%M%S)
    "
