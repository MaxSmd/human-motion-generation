#!/bin/bash
#SBATCH --job-name=rmg-train
#SBATCH --partition=12g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=23:55:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm/logs/rmg-train-%j.out
#SBATCH --error=slurm/logs/rmg-train-%j.err
#
# Unified, fully-configurable TRAIN job (one of three scripts the interactive
# app drives; the others are rmg/eval.sbatch + rmg/viz.sbatch). Everything is an
# env var, and OVERRIDES carries arbitrary extra Hydra args (space-separated,
# simple key=value tokens), so any training config is reachable without editing
# this file. SBATCH resources (partition/time/gres) are overridden by the
# submitter via `sbatch --partition=… --time=… --gres=…` flags.
#
#   MODEL_PRESET=dit_base PRESET=rmg_base RUN_NAME=my-run \
#   OVERRIDES='data.subset_n=8 train.max_steps=15000 train.lr=1e-4' \
#       sbatch slurm/rmg/train.sbatch

set -euo pipefail
mkdir -p slurm/logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
fi
IMAGE="${IMAGE:-${HOME}/rmg.sqsh}"
HF_CACHE="${HF_CACHE:-${HOME}/hf-cache}"
RUNS_ROOT="${RUNS_ROOT:-${REPO}/runs/rmg/train}"
MODEL_PRESET="${MODEL_PRESET:-dit_base}"
PRESET="${PRESET:-rmg_base}"
TEXT_ENCODER="${TEXT_ENCODER:-qwen3}"
RUN_NAME="${RUN_NAME:-rmg-train-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
OVERRIDES="${OVERRIDES:-}"
: "${WANDB_MODE:=offline}"

enroot start \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --env "HF_HOME=${HF_CACHE}" \
    --env "WANDB_MODE=${WANDB_MODE}" \
    --env "MGEN_RUNS_DIR=${RUNS_ROOT}" \
    --env "PYTHONPATH=${REPO}/src" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        cd ${REPO}
        pip install -e . --no-deps --quiet
        python -u -m rmg.scripts.train \\
            model=${MODEL_PRESET} \\
            train=${PRESET} \\
            text_encoder.type=${TEXT_ENCODER} \\
            run_name='${RUN_NAME}' \\
            ${OVERRIDES}
    "
