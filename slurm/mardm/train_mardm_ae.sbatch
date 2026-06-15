#!/bin/bash
#SBATCH --job-name=mardm-ae
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=23:55:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm/logs/mardm-ae-%j.out
#SBATCH --error=slurm/logs/mardm-ae-%j.err
#
# MARDM stage 1: train the motion AutoEncoder (mardm_mini, latent 512, ×4
# downsample). Computes the essential mean/std once (if missing) then trains.
# The AE is the cheap stage (a few hours); leave the bulk of the 1-day,
# per-model budget for the generation branch (slurm/mardm/train_mardm.sbatch).
#
# Defaults to the 24g partition (matches the documented 1-day-per-model GPU
# budget). 12g works equivalently — the train script auto-picks fp16 on
# Turing and bf16 on Ampere+, so precision doesn't need a manual override.
# To use 12g: `sbatch --partition=12g --nodelist=frankfurt,hamburg,kassel,leipzig ...`
#
# Submit:  sbatch slurm/mardm/train_mardm_ae.sbatch
# Resume:  just resubmit (auto-detects latest checkpoint under the run dir).
#
# Throughput probe (set train.max_steps for the budget): run once with
#   train.max_steps=500, read `steps_per_s` from the log, then set
#   train.max_steps ≈ steps_per_s * <budget_seconds> (e.g. ~4h = 14400 s).

set -euo pipefail
mkdir -p slurm/logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(dirname "${SCRIPT_DIR}")}"
fi
IMAGE="${IMAGE:-${HOME}/rmg.sqsh}"
RUN_NAME="${RUN_NAME:-mardm-ae-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
RUNS_ROOT="${RUNS_ROOT:-${HOME}/rmg-runs}"
HF_CACHE="${HF_CACHE:-${HOME}/hf-cache}"
DATA_ROOT="${RMG_DATA_ROOT:-${REPO}/external/data/humanml3d_packed}"
# Shared across AE / gen / eval — must be a writable path (the dataset mount is RO).
MARDM_STATS="${MARDM_STATS:-${RUNS_ROOT}/mardm_essential_stats.pt}"
# Extra Hydra overrides (space-separated key=value), e.g. from the interactive app.
OVERRIDES="${OVERRIDES:-}"
if [ -n "${WANDB_API_KEY:-}" ]; then
    : "${WANDB_MODE:=online}"
else
    : "${WANDB_MODE:=offline}"
fi

enroot start \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --env "HF_HOME=${HF_CACHE}" \
    --env "WANDB_MODE=${WANDB_MODE}" \
    --env "WANDB_API_KEY=${WANDB_API_KEY:-}" \
    --env "WANDB_ENTITY=${WANDB_ENTITY:-}" \
    --env "RMG_RUNS_DIR=${RUNS_ROOT}" \
    --env "RMG_DATA_ROOT=${DATA_ROOT}" \
    --env "MARDM_STATS=${MARDM_STATS}" \
    --env "PYTHONPATH=${REPO}/src" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        cd ${REPO}
        pip install -e . --no-deps --quiet
        [ -f '${MARDM_STATS}' ] || python -m mardm.scripts.compute_mardm_stats \\
            --data-root '${DATA_ROOT}' --out '${MARDM_STATS}'
        python -m mardm.scripts.train_mardm_ae run_name='${RUN_NAME}' data.preload=true ${OVERRIDES}
    "
