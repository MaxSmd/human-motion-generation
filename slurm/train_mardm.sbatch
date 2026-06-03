#!/bin/bash
#SBATCH --job-name=mardm-gen
#SBATCH --partition=12g
#SBATCH --nodelist=frankfurt,hamburg,kassel,leipzig
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=23:55:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm/logs/mardm-gen-%j.out
#SBATCH --error=slurm/logs/mardm-gen-%j.err
#
# MARDM stage 2: train the masked-autoregressive generation branch (mardm_mini)
# on the frozen AE's latents. Requires a stage-1 AE checkpoint whose `ae:` block
# matches configs/mardm/gen.yaml (paper-faithful AE by default).
#
# Submit:
#     AE_CKPT=/path/to/mardm-ae-XXXX/checkpoints/latest.pt sbatch slurm/train_mardm.sbatch
#     # or pass as the first positional arg
#     sbatch slurm/train_mardm.sbatch ${HOME}/rmg-runs/mardm-ae-XXXX/checkpoints/latest.pt
# Resume: just resubmit (auto-detects latest checkpoint under the run dir).
#
# Throughput probe: run with train.max_steps=500, read `steps_per_s`, then set
# train.max_steps ≈ steps_per_s * <budget_seconds> so AE + gen ≤ ~23h total.

set -euo pipefail
mkdir -p slurm/logs

AE_CKPT="${AE_CKPT:-${1:?set AE_CKPT env var or pass the AE checkpoint as the first positional arg}}"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(dirname "${SCRIPT_DIR}")}"
fi
IMAGE="${IMAGE:-${HOME}/rmg.sqsh}"
RUN_NAME="${RUN_NAME:-mardm-gen-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
RUNS_ROOT="${RUNS_ROOT:-${HOME}/rmg-runs}"
HF_CACHE="${HF_CACHE:-${HOME}/hf-cache}"
DATA_ROOT="${RMG_DATA_ROOT:-${REPO}/external/data/humanml3d_packed}"
MARDM_STATS="${MARDM_STATS:-${RUNS_ROOT}/mardm_essential_stats.pt}"
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
        [ -f '${MARDM_STATS}' ] || python scripts/compute_mardm_stats.py \\
            --data-root '${DATA_ROOT}' --out '${MARDM_STATS}'
        python scripts/train_mardm.py \\
            ae_checkpoint='${AE_CKPT}' \\
            text_encoder.type=qwen3 \\
            run_name='${RUN_NAME}'
    "
