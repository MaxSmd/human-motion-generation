#!/bin/bash
#SBATCH --job-name=rmg-sanity-eval
#SBATCH --partition=12g
#SBATCH --nodelist=frankfurt,hamburg,kassel,leipzig
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/rmg-sanity-eval-%j.out
#SBATCH --error=slurm/logs/rmg-sanity-eval-%j.err
#
# No-train sanity check: real motions → decode → Guo evaluator → metrics.
# Reports R@1/R@2/R@3, diversity_real, MM-Dist, FID(real, real) and compares
# them to the HumanML3D paper's "GT row" values.
#
# If this FAILS, the eval pipeline is broken — fix it BEFORE more training.
#
# Usage:
#     sbatch slurm/sanity_eval.sbatch
#     # quick subset (a few minutes):
#     MAX_CLIPS=512 sbatch slurm/sanity_eval.sbatch

set -euo pipefail
mkdir -p slurm/logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(dirname "${SCRIPT_DIR}")}"
fi

IMAGE="${IMAGE:-${HOME}/rmg.sqsh}"
HF_CACHE="${HF_CACHE:-${HOME}/hf-cache}"
RUNS_ROOT="${RUNS_ROOT:-${HOME}/rmg-runs}"
MAX_CLIPS="${MAX_CLIPS:--1}"
RUN_NAME="${RUN_NAME:-sanity-eval-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"

enroot start \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --env "HF_HOME=${HF_CACHE}" \
    --env "RMG_RUNS_DIR=${RUNS_ROOT}" \
    --env "PYTHONPATH=${REPO}/src" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        cd ${REPO}
        pip install -e . --no-deps --quiet
        python -m spacy download en_core_web_sm --quiet || true
        python -u scripts/sanity_eval.py \\
            model=dit_base \\
            train=rmg_base \\
            text_encoder.type=random \\
            +eval.evaluator=real \\
            +eval.max_clips=${MAX_CLIPS} \\
            run_name='${RUN_NAME}'
    "
