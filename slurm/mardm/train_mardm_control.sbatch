#!/bin/bash
#SBATCH --job-name=mardm-control
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=23:55:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm/logs/mardm-control-%j.out
#SBATCH --error=slurm/logs/mardm-control-%j.err
#
# MARDM spatial control (phase 2): train the ControlNet-style condition
# regularizer against the FROZEN stage-1 AE + stage-2 generation checkpoint.
# Config: src/mardm/configs/control.yaml (MARDM-M dims).
#
# Submit:
#     AE_CKPT=.../mardm-ae-canon/checkpoints/latest.pt \
#     GEN_CKPT=.../mardm-gen-m-canon/checkpoints/latest.pt \
#     MARDM_STATS=.../mardm_essential_stats_canonical.pt \
#     CANONICAL_DIR=.../h3d-canonical/new_joint_vecs \
#     sbatch slurm/mardm/train_mardm_control.sbatch
# Resume: resubmit with the same RUN_NAME (auto-detects latest checkpoint).
#
# Throughput probe: OVERRIDES="train.max_steps=200" first, read steps_per_s,
# then size train.max_steps to the <=1-day budget.

set -euo pipefail
mkdir -p slurm/logs

AE_CKPT="${AE_CKPT:?set AE_CKPT to the stage-1 AE checkpoint}"
GEN_CKPT="${GEN_CKPT:?set GEN_CKPT to the stage-2 generation checkpoint}"
EXTRA_HYDRA_ARGS="$*"
OVERRIDES="${OVERRIDES:-}"
CONFIG_NAME="${CONFIG_NAME:-control}"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(dirname "$(dirname "${SCRIPT_DIR}")")}"
fi
IMAGE="${IMAGE:-${HOME}/rmg.sqsh}"
RUN_NAME="${RUN_NAME:-mardm-control-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
RUNS_ROOT="${RUNS_ROOT:-${HOME}/rmg-runs}"
HF_CACHE="${HF_CACHE:-${HOME}/hf-cache}"
DATA_ROOT="${RMG_DATA_ROOT:-${REPO}/external/data/humanml3d_packed}"
MARDM_STATS="${MARDM_STATS:-${RUNS_ROOT}/mardm_essential_stats.pt}"
CANONICAL_DIR="${CANONICAL_DIR:-}"
CANON_DATA_FLAG=""
[ -n "${CANONICAL_DIR}" ] && CANON_DATA_FLAG="data.canonical_dir='${CANONICAL_DIR}'"
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
        python -m mardm.scripts.train_mardm_control \\
            --config-name=${CONFIG_NAME} \\
            ae_checkpoint='${AE_CKPT}' \\
            gen_checkpoint='${GEN_CKPT}' \\
            text_encoder.type=qwen3 \\
            data.preload=true \\
            run_name='${RUN_NAME}' \\
            ${CANON_DATA_FLAG} ${EXTRA_HYDRA_ARGS} ${OVERRIDES}
    "
