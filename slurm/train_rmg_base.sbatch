#!/bin/bash
#SBATCH --job-name=rmg-base
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm/logs/rmg-base-%j.out
#SBATCH --error=slurm/logs/rmg-base-%j.err
#
# Train RMG-base (6L/384h) end-to-end on a single 24g GPU.
# Effective BS = 32 * 8 = 256 (paper Table 7).
#
# Submit:
#     sbatch slurm/train_rmg_base.sbatch
#
# Resume: just resubmit. The training script auto-detects the latest
# checkpoint under ${output_dir}/checkpoints/.

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
RUN_NAME="${RUN_NAME:-rmg-base-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
RUNS_ROOT="${RUNS_ROOT:-${HOME}/rmg-runs}"
HF_CACHE="${HF_CACHE:-${HOME}/hf-cache}"

# WANDB_MODE=offline by default (cluster egress is uncertain).
# Override with `WANDB_MODE=online sbatch ...` if you've verified outbound.
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
            +model=dit_base \\
            +train=rmg_base \\
            text_encoder.type=qwen3 \\
            run_name='${RUN_NAME}'
    "
