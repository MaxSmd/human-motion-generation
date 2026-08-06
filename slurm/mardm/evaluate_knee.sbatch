#!/bin/bash
#SBATCH --job-name=mardm-knee
#SBATCH --partition=12g
#SBATCH --nodelist=frankfurt,hamburg,kassel,leipzig
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/mardm-knee-%j.out
#SBATCH --error=slurm/logs/mardm-knee-%j.err
#
# Joint-angle control check: can inference-time guidance stop a knee from
# bending? Samples unconstrained vs bend-limited on leg-heavy prompts, reports
# bend satisfaction + realism, and writes side-by-side GIFs. Zero training.
#
# Usage:
#     AE_CKPT=.../mardm-ae-canon/checkpoints/latest.pt \
#     GEN_CKPT=.../mardm-gen-m-canon/checkpoints/latest.pt \
#     MARDM_STATS=.../mardm_essential_stats_canonical.pt \
#     CANONICAL_DIR=.../h3d-canonical/new_joint_vecs \
#     SIDE=left \
#     sbatch slurm/mardm/evaluate_knee.sbatch

set -euo pipefail
mkdir -p slurm/logs

AE_CKPT="${AE_CKPT:?set AE_CKPT to the stage-1 AE checkpoint}"
GEN_CKPT="${GEN_CKPT:-${1:?set GEN_CKPT env var or pass the gen checkpoint as the first positional arg}}"
CONFIG_NAME="${CONFIG_NAME:-gen}"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(dirname "$(dirname "${SCRIPT_DIR}")")}"
fi
IMAGE="${IMAGE:-${HOME}/rmg.sqsh}"
RUNS_ROOT="${RUNS_ROOT:-${HOME}/rmg-runs}"
HF_CACHE="${HF_CACHE:-${HOME}/hf-cache}"
DATA_ROOT="${RMG_DATA_ROOT:-${REPO}/external/data/humanml3d_packed}"
MARDM_STATS="${MARDM_STATS:-${RUNS_ROOT}/mardm_essential_stats.pt}"
CANONICAL_DIR="${CANONICAL_DIR:-}"
SIDE="${SIDE:-left}"
MAX_DEG="${MAX_DEG:-10}"
WEIGHT="${WEIGHT:-1.0}"
DYN_WEIGHT="${DYN_WEIGHT:-1.0}"
SKATE_WEIGHT="${SKATE_WEIGHT:-1.0}"
GUIDANCE="${GUIDANCE:-3.0}"
INNER_ITERS="${INNER_ITERS:-30}"
RUN_NAME="${RUN_NAME:-mardm-knee-${SLURM_JOB_ID:-local}}"
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
        python -u -m mardm.scripts.evaluate_knee \\
            --config-name=${CONFIG_NAME} \\
            ae_checkpoint='${AE_CKPT}' \\
            text_encoder.type=qwen3 \\
            ${CANON_FLAG} \\
            +knee.checkpoint='${GEN_CKPT}' \\
            +knee.side='${SIDE}' \\
            +knee.max_deg='${MAX_DEG}' \\
            +knee.weight='${WEIGHT}' \\
            +knee.dyn_weight='${DYN_WEIGHT}' \\
            +knee.skate_weight='${SKATE_WEIGHT}' \\
            +knee.guidance='${GUIDANCE}' \\
            +knee.inner_iters='${INNER_ITERS}' \\
            +knee.render=true \\
            +knee.verbose=true \\
            run_name='${RUN_NAME}' \\
            ${OVERRIDES}
    "
