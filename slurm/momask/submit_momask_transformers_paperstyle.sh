#!/bin/bash
# Submit independent paper-style M/R transformer jobs. The R job waits for M,
# then assembles both validated checkpoints before releasing its allocation.

set -euo pipefail

MASKED_RUN_NAME="${MASKED_RUN_NAME:-momask-canonical-mtrans-vq150k-maskfix196-clip500e}"
RESIDUAL_RUN_NAME="${RESIDUAL_RUN_NAME:-momask-canonical-rtrans-vq150k-maskfix196-clip500e}"
ASSEMBLED_RUN_NAME="${ASSEMBLED_RUN_NAME:-momask-canonical-tokens-vq150k-maskfix196-clip500e}"
VALIDATE_EVERY="${VALIDATE_EVERY:-5000}"

if [ "${VALIDATE_EVERY}" -le 0 ]; then
    echo "ERROR: the full launcher requires VALIDATE_EVERY > 0 so best checkpoints exist" >&2
    exit 2
fi
export VALIDATE_EVERY

masked_ckpt="runs/${MASKED_RUN_NAME}/checkpoints/tokens_best_val.pt"
assembled_ckpt="runs/${ASSEMBLED_RUN_NAME}/momask_smoke_latest.pt"

masked_submission="$(
    sbatch --parsable \
        --export="ALL,RUN_NAME=${MASKED_RUN_NAME}" \
        slurm/momask/train_momask_masked_paperstyle.sbatch
)"
masked_job_id="${masked_submission%%;*}"
echo "M-Transformer job: ${masked_job_id}"

residual_submission="$(
    sbatch --parsable \
        --dependency="afterok:${masked_job_id}" \
        --export="ALL,RUN_NAME=${RESIDUAL_RUN_NAME},ASSEMBLE_AFTER_TRAIN=1,ASSEMBLE_MASKED_CKPT=${masked_ckpt},ASSEMBLE_OUTPUT=${assembled_ckpt}" \
        slurm/momask/train_momask_residual_paperstyle.sbatch
)"
residual_job_id="${residual_submission%%;*}"

echo "R-Transformer job: ${residual_job_id} (after M; assembles on completion)"
echo "Final checkpoint:  ${assembled_ckpt}"
