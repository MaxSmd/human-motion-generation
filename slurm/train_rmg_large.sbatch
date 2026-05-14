#!/bin/bash
#SBATCH --job-name=rmg-large
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=23:55:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm/logs/rmg-large-%j.out
#SBATCH --error=slurm/logs/rmg-large-%j.err
#
# Train full RMG (24L/1024h) on a single 24g GPU. 600k steps; expect to
# resubmit several times (job time-limit cap is 72h). Resume is automatic.
#
# Note: the paper trains this on more GPUs; on a single 24g card the wall
# time for 600k optimizer steps is steep. Realistic primary target on a
# student account is RMG-base (150k); RMG-large is a stretch goal.

set -euo pipefail
mkdir -p slurm/logs

# Use SLURM_SUBMIT_DIR when run under sbatch (script is spooled, BASH_SOURCE
# doesn't resolve to the repo). Fall back to BASH_SOURCE for direct runs.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(dirname "${SCRIPT_DIR}")}"
fi
IMAGE="${IMAGE:-${HOME}/rmg.sqsh}"
RUN_NAME="${RUN_NAME:-rmg-large-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
RUNS_ROOT="${RUNS_ROOT:-${HOME}/rmg-runs}"
HF_CACHE="${HF_CACHE:-${HOME}/hf-cache}"

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
            +model=dit_large \\
            +train=rmg_large \\
            text_encoder.type=qwen3 \\
            run_name='${RUN_NAME}'
    "
