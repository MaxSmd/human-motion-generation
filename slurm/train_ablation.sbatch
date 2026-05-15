#!/bin/bash
#SBATCH --job-name=rmg-ablation
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=23:55:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm/logs/rmg-ablation-%j.out
#SBATCH --error=slurm/logs/rmg-ablation-%j.err
#
# Train an RMG ablation (paper Figure 3a). Default: T+R (paper main result).
# Override REPRESENTATION to swap manifolds:
#
#     REPRESENTATION=t_plus_r       sbatch slurm/train_ablation.sbatch  # T+R
#     REPRESENTATION=t_plus_p       sbatch slurm/train_ablation.sbatch  # T+P
#     REPRESENTATION=t_plus_r_plus_p sbatch slurm/train_ablation.sbatch # T+R+P
#
# (dT/dR ablations are negative-result config stubs; see
#  configs/representation/{dt_plus_r,t_plus_dr}.yaml.)

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
HF_CACHE="${HF_CACHE:-${HOME}/hf-cache}"
RUNS_ROOT="${RUNS_ROOT:-${HOME}/rmg-runs}"
REPRESENTATION="${REPRESENTATION:-t_plus_r}"
RUN_NAME="${RUN_NAME:-rmg-base-${REPRESENTATION}-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"

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
            model=dit_base \\
            train=rmg_base \\
            representation=${REPRESENTATION} \\
            text_encoder.type=qwen3 \\
            run_name='${RUN_NAME}'
    "
