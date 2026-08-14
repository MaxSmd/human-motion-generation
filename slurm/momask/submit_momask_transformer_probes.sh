#!/bin/bash
# Submit short stage-specific runs that report first-step CUDA peak memory.
# This script only submits jobs; it performs no training on the head node.

set -euo pipefail

PROBE_TAG="${PROBE_TAG:-$(date +%Y%m%d-%H%M%S)}"
PROBE_STEPS="${PROBE_STEPS:-2000}"
PROBE_MAX_CLIPS="${PROBE_MAX_CLIPS:-512}"

common_export="TOKEN_EPOCHS=0,TOKEN_STEPS=${PROBE_STEPS},MAX_CLIPS=${PROBE_MAX_CLIPS},VALIDATE_EVERY=${PROBE_STEPS},VAL_EVAL_BATCHES=4,SAVE_EVERY=${PROBE_STEPS},LOG_EVERY=200"

masked_run="momask-mtrans-probe-${PROBE_TAG}"
masked_submission="$(
    sbatch --parsable \
        --export="ALL,RUN_NAME=${masked_run},${common_export}" \
        slurm/momask/train_momask_masked_paperstyle.sbatch
)"
masked_job_id="${masked_submission%%;*}"
echo "M-Transformer probe: ${masked_job_id} (${masked_run})"

residual_run="momask-rtrans-probe-${PROBE_TAG}"
residual_submission="$(
    sbatch --parsable \
        --export="ALL,RUN_NAME=${residual_run},${common_export}" \
        slurm/momask/train_momask_residual_paperstyle.sbatch
)"
residual_job_id="${residual_submission%%;*}"
echo "R-Transformer probe: ${residual_job_id} (${residual_run})"

echo "Inspect:"
echo "  slurm/logs/momask-mtrans-paperstyle-${masked_job_id}.out"
echo "  slurm/logs/momask-rtrans-paperstyle-${residual_job_id}.out"
echo "Look for the [token cuda_memory] line before launching the full jobs."
