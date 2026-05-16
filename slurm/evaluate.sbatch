#!/bin/bash
#SBATCH --job-name=rmg-eval
#SBATCH --partition=12g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/rmg-eval-%j.out
#SBATCH --error=slurm/logs/rmg-eval-%j.err
#
# Evaluate an RMG checkpoint on the HumanML3D test split with a guidance
# sweep. Reproduces paper Table 4 / Figure 3a values (modulo the actual
# trained model).
#
# Usage:
#     CKPT=/path/to/checkpoint.pt sbatch slurm/evaluate.sbatch
#     # or
#     sbatch slurm/evaluate.sbatch /mnt/home/${USER}/rmg-runs/<run>/checkpoints/latest.pt

set -euo pipefail
mkdir -p slurm/logs

CKPT="${CKPT:-${1:?set CKPT env var or pass as first positional arg}}"
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
PRESET="${PRESET:-rmg_base}"
MODEL_PRESET="${MODEL_PRESET:-dit_base}"

enroot start \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --env "HF_HOME=${HF_CACHE}" \
    --env "PYTHONPATH=${REPO}/src" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        cd ${REPO}
        pip install -e . --no-deps --quiet
        python -m spacy download en_core_web_sm --quiet || true
        python scripts/evaluate.py \\
            model=${MODEL_PRESET} \\
            train=${PRESET} \\
            text_encoder.type=qwen3 \\
            eval.checkpoint=${CKPT} \\
            eval.evaluator=real \\
            eval.guidance_scales='[2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5]' \\
            run_name=eval-\$(basename ${CKPT} .pt)
    "
