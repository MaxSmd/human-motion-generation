#!/bin/bash
#SBATCH --job-name=mardm-eval
#SBATCH --partition=12g
#SBATCH --nodelist=frankfurt,hamburg,kassel,leipzig
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --output=slurm/logs/mardm-eval-%j.out
#SBATCH --error=slurm/logs/mardm-eval-%j.err
#
# Evaluate a MARDM checkpoint on the HumanML3D test split (HumanML3D format) via
# the Guo et al. evaluator — the row we want for Table 4. Needs both the frozen
# AE and the generation-branch checkpoint, the shared essential mean/std, and
# the external/HumanML3D submodule (for the essential->263 inverse kinematics).
#
# Usage:
#     AE_CKPT=.../mardm-ae-XXXX/checkpoints/latest.pt \
#     GEN_CKPT=.../mardm-gen-YYYY/checkpoints/latest.pt \
#     sbatch slurm/mardm/evaluate_mardm.sbatch

set -euo pipefail
mkdir -p slurm/logs

AE_CKPT="${AE_CKPT:?set AE_CKPT to the stage-1 AE checkpoint}"
GEN_CKPT="${GEN_CKPT:-${1:?set GEN_CKPT env var or pass the gen checkpoint as the first positional arg}}"
CONFIG_NAME="${CONFIG_NAME:-gen}"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(dirname "${SCRIPT_DIR}")}"
fi
IMAGE="${IMAGE:-${HOME}/rmg.sqsh}"
RUNS_ROOT="${RUNS_ROOT:-${HOME}/rmg-runs}"
HF_CACHE="${HF_CACHE:-${HOME}/hf-cache}"
DATA_ROOT="${RMG_DATA_ROOT:-${REPO}/external/data/humanml3d_packed}"
MARDM_STATS="${MARDM_STATS:-${RUNS_ROOT}/mardm_essential_stats.pt}"
GUIDANCE="${GUIDANCE:-[1.0,1.5,2.0,2.5,3.0]}"
TIMESTEPS="${TIMESTEPS:-18}"
# Run name (results land under RUNS_ROOT/RUN_NAME) — honoured so the app can
# locate + pull this eval's metrics. Defaults to the gen-checkpoint basename.
RUN_NAME="${RUN_NAME:-mardm-eval-$(basename "${GEN_CKPT}" .pt)}"
# Extra Hydra overrides (space-separated key=value), e.g. from the app.
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
        python -u -m mardm.scripts.evaluate_mardm \\
            --config-name=${CONFIG_NAME} \\
            ae_checkpoint='${AE_CKPT}' \\
            text_encoder.type=qwen3 \\
            +eval.checkpoint='${GEN_CKPT}' \\
            +eval.evaluator=real \\
            +eval.guidance_scales='${GUIDANCE}' \\
            +eval.timesteps='${TIMESTEPS}' \\
            run_name='${RUN_NAME}' \\
            ${OVERRIDES}
    "
