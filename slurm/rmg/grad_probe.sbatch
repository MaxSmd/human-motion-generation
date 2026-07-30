#!/bin/bash
#SBATCH --job-name=rmg-grad-probe
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/rmg-grad-probe-%j.out
#SBATCH --error=slurm/logs/rmg-grad-probe-%j.err
#
# Diagnose non-finite gradients: runs real training steps on a checkpoint and
# reports, per step, whether the gradient is finite in the run's AMP precision
# vs fp32, plus which layers blow up. Read-only.
#
#   CKPT=/path/to/ckpt.pt MODEL_PRESET=dit_mid PRESET=rmg_mid \
#       sbatch slurm/rmg/grad_probe.sbatch

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
MODEL_PRESET="${MODEL_PRESET:-dit_mid}"
PRESET="${PRESET:-rmg_mid}"
TEXT_ENCODER="${TEXT_ENCODER:-qwen3}"
CKPT="${CKPT:-}"
STEPS="${STEPS:-20}"
OVERRIDES="${OVERRIDES:-}"

if [ -z "$CKPT" ]; then
    echo "ERROR: set CKPT=/path/to/ckpt.pt"; exit 1
fi

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
        python -u -m rmg.scripts.grad_probe \\
            model=${MODEL_PRESET} train=${PRESET} text_encoder.type=${TEXT_ENCODER} \\
            +probe_ckpt=${CKPT} +probe_steps=${STEPS} ${OVERRIDES}
    "
