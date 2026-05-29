#!/bin/bash
#SBATCH --job-name=rmg-full-subset
#SBATCH --partition=12g
#SBATCH --nodelist=frankfurt,hamburg,kassel,leipzig
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=23:55:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm/logs/rmg-full-subset-%j.out
#SBATCH --error=slurm/logs/rmg-full-subset-%j.err
#
# Train the FULL RMG configuration (24L/1024h DiT) on a 1% subset of the
# training data. The subset is deterministic via subset_seed — the same 112
# regular clips + their mirrors get picked every run, and the same set will
# be picked by MoMask and MARDM when they train with the same seed.
#
# Purpose: rapid iteration on architecture / hyperparams with paper-config
# model, comparable across all three reproductions.
#
# Env knobs:
#   MAX_STEPS  default 4000   (~16h on 12g, fits one overnight cycle)
#   RUN_NAME   default rmg-full-1pct-<jobid>
#   SUBSET     default 0.01
#
# Submit:
#   sbatch slurm/train_rmg_full_subset.sbatch
#   # or for a shorter daytime iteration:
#   MAX_STEPS=1500 sbatch slurm/train_rmg_full_subset.sbatch

set -euo pipefail
mkdir -p slurm/logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(dirname "${SCRIPT_DIR}")}"
fi

IMAGE="${IMAGE:-${HOME}/rmg.sqsh}"
RUNS_ROOT="${RUNS_ROOT:-${HOME}/rmg-runs}"
HF_CACHE="${HF_CACHE:-${HOME}/hf-cache}"
MAX_STEPS="${MAX_STEPS:-4000}"
SUBSET="${SUBSET:-0.01}"
SUBSET_SEED="${SUBSET_SEED:-0}"
RUN_NAME="${RUN_NAME:-rmg-full-1pct-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"

: "${WANDB_MODE:=offline}"

enroot start \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --env "HF_HOME=${HF_CACHE}" \
    --env "WANDB_MODE=${WANDB_MODE}" \
    --env "RMG_RUNS_DIR=${RUNS_ROOT}" \
    --env "PYTHONPATH=${REPO}/src" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        cd ${REPO}
        pip install -e . --no-deps --quiet
        python scripts/train.py \\
            model=dit_large \\
            train=rmg_large \\
            text_encoder.type=qwen3 \\
            data.subset_fraction=${SUBSET} \\
            data.subset_seed=${SUBSET_SEED} \\
            train.max_steps=${MAX_STEPS} \\
            train.ckpt_every=500 \\
            train.sample_every=500 \\
            train.log_every=20 \\
            run_name='${RUN_NAME}'
    "
