#!/bin/bash
#SBATCH --job-name=rmg-eval
#SBATCH --partition=24g
#SBATCH --qos=students_normal
# The 12g partition mixes GPU generations and `ccc` in each node's GRES is its
# compute capability x10: ulm/jena are TITANX (5.2) and dortmund is TITANXp (6.1),
# which this CUDA container has no kernels for — a job landing there dies with
# "no kernel image is available for execution on the device", and the traceback
# points into the text encoder so it reads like a model bug. Asking for ccc:75
# (a no_consume GRES, so it filters rather than reserves) keeps this off the old
# cards however the script is submitted. The app already passes its own --gres
# for 12g, which overrides this line; this is the guard for a direct sbatch.
# NOTE: this cluster rejects --exclude outright.
#SBATCH --gres=gpu:1,ccc:75
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --output=slurm/logs/rmg-eval-%j.out
#SBATCH --error=slurm/logs/rmg-eval-%j.err
#
# Unified, fully-configurable EVAL job. Like rmg/train.sbatch: every knob is an
# env var + OVERRIDES for arbitrary extra Hydra args. RUN_NAME is honoured (so
# the app can locate + pull the metrics this run writes under RUNS_ROOT/RUN_NAME).
#
#   CKPT=/path/to/latest.pt MODEL_PRESET=dit_base PRESET=rmg_base \
#   EVAL_SPLIT=test MAX_CLIPS=256 GUIDANCE_SCALES='[6.5]' RUN_NAME=eval-myrun \
#       sbatch slurm/rmg/eval.sbatch

set -euo pipefail
mkdir -p slurm/logs

CKPT="${CKPT:-${1:?set CKPT env var or pass as first positional arg}}"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
fi
IMAGE="${IMAGE:-${HOME}/rmg.sqsh}"
HF_CACHE="${HF_CACHE:-${HOME}/hf-cache}"
RUNS_ROOT="${RUNS_ROOT:-${REPO}/runs/rmg/eval}"
MODEL_PRESET="${MODEL_PRESET:-dit_base}"
PRESET="${PRESET:-rmg_base}"
TEXT_ENCODER="${TEXT_ENCODER:-qwen3}"
RUN_NAME="${RUN_NAME:-eval-$(basename "${CKPT}" .pt)-${SLURM_JOB_ID:-$(date +%H%M%S)}}"

EVAL_SPLIT="${EVAL_SPLIT:-test}"
MAX_CLIPS="${MAX_CLIPS:--1}"
GUIDANCE_SCALES="${GUIDANCE_SCALES:-[2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5]}"
MM_NUM_TEXTS="${MM_NUM_TEXTS:-30}"
MM_REPEATS="${MM_REPEATS:-10}"
SUBSET_FRACTION="${SUBSET_FRACTION:-1.0}"
SUBSET_SEED="${SUBSET_SEED:-0}"
NUM_SAMPLE_STEPS="${NUM_SAMPLE_STEPS:-50}"
USE_EMA="${USE_EMA:-true}"
OVERRIDES="${OVERRIDES:-}"

# Make sure the Guo evaluator assets (text-to-motion submodule + finest.tar)
# are in place before the container starts. Fast no-op once set up; the
# first run does the one-time clone (best on a data node — see the script).
bash "${REPO}/slurm/ensure_eval_assets.sh"

enroot start \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --env "HF_HOME=${HF_CACHE}" \
    --env "MGEN_RUNS_DIR=${RUNS_ROOT}" \
    --env "PYTHONPATH=${REPO}/src" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        cd ${REPO}
        pip install -e . --no-deps --quiet
        python -m spacy download en_core_web_sm --quiet || true
        python -u -m rmg.scripts.evaluate \\
            model=${MODEL_PRESET} \\
            train=${PRESET} \\
            text_encoder.type=${TEXT_ENCODER} \\
            data.subset_fraction=${SUBSET_FRACTION} \\
            data.subset_seed=${SUBSET_SEED} \\
            +eval.checkpoint=${CKPT} \\
            +eval.evaluator=real \\
            +eval.use_ema=${USE_EMA} \\
            +eval.split=${EVAL_SPLIT} \\
            +eval.max_clips=${MAX_CLIPS} \\
            +eval.mm_num_texts=${MM_NUM_TEXTS} \\
            +eval.mm_repeats=${MM_REPEATS} \\
            +eval.num_sample_steps=${NUM_SAMPLE_STEPS} \\
            +eval.guidance_scales='${GUIDANCE_SCALES}' \\
            run_name='${RUN_NAME}' \\
            ${OVERRIDES}
    "
