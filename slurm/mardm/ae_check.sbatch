#!/bin/bash
#SBATCH --job-name=mardm-aecheck
#SBATCH --partition=12g
#SBATCH --nodelist=frankfurt,hamburg,kassel,leipzig
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/mardm-aecheck-%j.out
#SBATCH --error=slurm/logs/mardm-aecheck-%j.err
#
# AE-only reconstruction check: render GT vs decode(encode(GT)) and report
# per-clip MPJPE (global + pose-only local). No generation branch. Use before
# committing GPU-days to the gen stage — a good AE is a precondition.
#
# For a canonically-trained AE, set CANONICAL_DIR + the matching MARDM_STATS so
# the GT fed in is the same distribution the AE learned.
#
# Usage:
#     AE_CKPT=.../mardm-ae-canon/checkpoints/latest.pt \
#     CANONICAL_DIR=.../h3d-canonical/new_joint_vecs \
#     MARDM_STATS=.../mardm_essential_stats_canonical.pt \
#     SPLIT=test NUM_CLIPS=12 \
#     sbatch slurm/mardm/ae_check.sbatch

set -euo pipefail
mkdir -p slurm/logs

AE_CKPT="${AE_CKPT:?set AE_CKPT to the stage-1 AE checkpoint}"
CONFIG_NAME="${CONFIG_NAME:-gen_m}"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
fi
IMAGE="${IMAGE:-${HOME}/rmg.sqsh}"
RUN_NAME="${RUN_NAME:-mardm-aecheck-${SLURM_JOB_ID:-$(date +%s)}}"
RUNS_ROOT="${RUNS_ROOT:-${HOME}/rmg-runs}"
HF_CACHE="${HF_CACHE:-${HOME}/hf-cache}"
DATA_ROOT="${RMG_DATA_ROOT:-${REPO}/external/data/humanml3d_packed}"
MARDM_STATS="${MARDM_STATS:-${RUNS_ROOT}/mardm_essential_stats.pt}"
CANONICAL_DIR="${CANONICAL_DIR:-}"
SPLIT="${SPLIT:-test}"
NUM_CLIPS="${NUM_CLIPS:-12}"
OVERRIDES="${OVERRIDES:-}"
CANON_FLAG=""
[ -n "${CANONICAL_DIR}" ] && CANON_FLAG="data.canonical_dir='${CANONICAL_DIR}'"

enroot start \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --env "HF_HOME=${HF_CACHE}" \
    --env "RMG_RUNS_DIR=${RUNS_ROOT}" \
    --env "RMG_DATA_ROOT=${DATA_ROOT}" \
    --env "MARDM_STATS=${MARDM_STATS}" \
    --env "PYTHONPATH=${REPO}/src" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        cd ${REPO}
        pip install -e . --no-deps --quiet
        python -u -m mardm.scripts.visualize \\
            --config-name=${CONFIG_NAME} \\
            ae_checkpoint='${AE_CKPT}' \\
            text_encoder.type=qwen3 \\
            +viz.ae_only=true \\
            +viz.split='${SPLIT}' \\
            +viz.num_clips=${NUM_CLIPS} \\
            run_name='${RUN_NAME}' \\
            ${CANON_FLAG} ${OVERRIDES}
    "
