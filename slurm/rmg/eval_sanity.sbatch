#!/bin/bash
#SBATCH --job-name=rmg-eval-sanity
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/rmg-eval-sanity-%j.out
#SBATCH --error=slurm/logs/rmg-eval-sanity-%j.err
#
# Eval-HARNESS sanity check (see rmg/scripts/eval_sanity.py): metrics on REAL
# motions only — real-vs-real FID ≈ 0 and real R-precision ≈ 0.51/0.70/0.80
# discriminate "harness broken" from "model broken" without touching a model.
#
#   MAX_CLIPS=512 sbatch slurm/rmg/eval_sanity.sbatch

set -euo pipefail
mkdir -p slurm/logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(dirname "$(dirname "${SCRIPT_DIR}")")}"
fi
IMAGE="${IMAGE:-${HOME}/rmg.sqsh}"
HF_CACHE="${HF_CACHE:-${HOME}/hf-cache}"
RUNS_ROOT="${RUNS_ROOT:-${REPO}/runs/rmg/eval}"
PRESET="${PRESET:-rmg_mid}"
RUN_NAME="${RUN_NAME:-eval-sanity-${SLURM_JOB_ID:-$(date +%H%M%S)}}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
MAX_CLIPS="${MAX_CLIPS:-512}"
OVERRIDES="${OVERRIDES:-}"

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
        python -u -m rmg.scripts.eval_sanity \\
            train=${PRESET} \\
            +eval.evaluator=real \\
            +eval.split=${EVAL_SPLIT} \\
            +eval.max_clips=${MAX_CLIPS} \\
            run_name=${RUN_NAME} \\
            ${OVERRIDES}
    "
