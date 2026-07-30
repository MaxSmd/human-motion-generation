#!/bin/bash
#SBATCH --job-name=mardm-viz
#SBATCH --partition=12g
#SBATCH --nodelist=frankfurt,hamburg,kassel,leipzig
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/mardm-viz-%j.out
#SBATCH --error=slurm/logs/mardm-viz-%j.err
#
# Overfit / memorization visual check for a MARDM checkpoint. For each clip in
# the deterministic train subset the model saw, renders GT vs prediction GIFs
# and reports per-clip MPJPE. Needs the frozen AE + the gen-branch checkpoint
# and the shared essential mean/std. Does NOT need external/HumanML3D (the
# joints come from recover_joints_from_ric, no IK).
#
# Usage (point SUBSET_FRAC at the value training used):
#     AE_CKPT=.../mardm-overfit-ae-XXXX/checkpoints/latest.pt \
#     GEN_CKPT=.../mardm-overfit-gen-XXXX/checkpoints/latest.pt \
#     SUBSET_FRAC=0.003 NUM_CLIPS=8 \
#     sbatch slurm/mardm/visualize_mardm.sbatch

set -euo pipefail
mkdir -p slurm/logs

AE_CKPT="${AE_CKPT:?set AE_CKPT to the stage-1 AE checkpoint}"
GEN_CKPT="${GEN_CKPT:-${1:?set GEN_CKPT env var or pass the gen checkpoint as the first positional arg}}"
CONFIG_NAME="${CONFIG_NAME:-gen}"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
fi
IMAGE="${IMAGE:-${HOME}/rmg.sqsh}"
RUNS_ROOT="${RUNS_ROOT:-${HOME}/rmg-runs}"
HF_CACHE="${HF_CACHE:-${HOME}/hf-cache}"
DATA_ROOT="${RMG_DATA_ROOT:-${REPO}/external/data/humanml3d_packed}"
MARDM_STATS="${MARDM_STATS:-${RUNS_ROOT}/mardm_essential_stats.pt}"

SUBSET_FRAC="${SUBSET_FRAC:-0.003}"
NUM_CLIPS="${NUM_CLIPS:-8}"
GUIDANCE="${GUIDANCE:-1.0}"   # 1.0 = pure conditional (correct for cond_drop_prob=0 overfit)
TIMESTEPS="${TIMESTEPS:-18}"
RUN_NAME="${RUN_NAME:-mardm-viz-$(basename "${GEN_CKPT}" .pt)}"
OVERRIDES="${OVERRIDES:-}"

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
        python -m spacy download en_core_web_sm --quiet || true
        python -u -m mardm.scripts.visualize_mardm \\
            --config-name=${CONFIG_NAME} \\
            ae_checkpoint='${AE_CKPT}' \\
            text_encoder.type=qwen3 \\
            subset_frac=${SUBSET_FRAC} \\
            +viz.checkpoint='${GEN_CKPT}' \\
            +viz.num_clips=${NUM_CLIPS} \\
            +viz.guidance=${GUIDANCE} \\
            +viz.timesteps=${TIMESTEPS} \\
            run_name='${RUN_NAME}' \\
            ${OVERRIDES}
    "
