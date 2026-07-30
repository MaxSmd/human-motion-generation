#!/bin/bash
#SBATCH --job-name=mardm-vrf
#SBATCH --partition=12g
#SBATCH --nodelist=frankfurt,hamburg,kassel,leipzig
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/mardm-vrf-%j.out
#SBATCH --error=slurm/logs/mardm-vrf-%j.err
#
# Diagnose the eval 263-D feature space (isolation + per-block diff). See
# mardm/scripts/verify_h3d_features.py. Needs the Guo evaluator (external/
# text-to-motion) and HumanML3D new_joint_vecs (pass VECS=...).
#
# Submit:
#   AE_CKPT=.../mardm-ae-XXXX/checkpoints/latest.pt \
#   GEN_CKPT=.../mardm-gen-YYYY/checkpoints/latest.pt \
#   VECS=/path/to/HumanML3D/new_joint_vecs N_CLIPS=128 \
#   sbatch slurm/mardm/verify_h3d.sbatch

set -euo pipefail
mkdir -p slurm/logs

AE_CKPT="${AE_CKPT:?set AE_CKPT (unused for math, required by config)}"
GEN_CKPT="${GEN_CKPT:?set GEN_CKPT (unused for math, required by config)}"
CONFIG_NAME="${CONFIG_NAME:-gen_m}"
VECS="${VECS:-external/HumanML3D/HumanML3D/new_joint_vecs}"
N_CLIPS="${N_CLIPS:-128}"
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
        python -u -m mardm.scripts.verify_h3d_features \\
            --config-name=${CONFIG_NAME} \\
            ae_checkpoint='${AE_CKPT}' \\
            +eval.checkpoint='${GEN_CKPT}' \\
            +verify.h3d_vecs='${VECS}' \\
            +verify.n_clips=${N_CLIPS}
    "
