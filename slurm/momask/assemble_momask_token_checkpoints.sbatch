#!/bin/bash
#SBATCH --job-name=momask-assemble
#SBATCH --partition=data
#SBATCH --qos=students_normal
#SBATCH --nodes=1
#SBATCH --output=slurm/logs/momask-assemble-%j.out
#SBATCH --error=slurm/logs/momask-assemble-%j.err
#
# CPU-only assembly of independently trained M- and R-Transformer checkpoints.

set -euo pipefail
mkdir -p slurm/logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
fi

IMAGE="${IMAGE:-${HOME}/rmg-momask.sqsh}"
MASKED_CKPT="${MASKED_CKPT:-runs/momask-canonical-mtrans-paperfaithful196-clip500e/checkpoints/tokens_best_val.pt}"
RESIDUAL_CKPT="${RESIDUAL_CKPT:-runs/momask-canonical-rtrans-paperfaithful196-clip500e/checkpoints/tokens_best_val.pt}"
OUTPUT="${OUTPUT:-runs/momask-canonical-tokens-paperfaithful196-clip500e/momask_smoke_latest.pt}"

echo "[momask-assemble] repo=${REPO}"
echo "[momask-assemble] masked=${MASKED_CKPT}"
echo "[momask-assemble] residual=${RESIDUAL_CKPT}"
echo "[momask-assemble] output=${OUTPUT}"

enroot start \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --mount "${REPO}:${REPO}" \
    --env "PYTHONPATH=${REPO}/src" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        cd '${REPO}'
        python -u -m momask.scripts.assemble_token_checkpoints \\
            --masked-checkpoint '${MASKED_CKPT}' \\
            --residual-checkpoint '${RESIDUAL_CKPT}' \\
            --output '${OUTPUT}'
    "
