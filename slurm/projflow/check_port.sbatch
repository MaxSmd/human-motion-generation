#!/bin/bash
#SBATCH --job-name=projflow-check
#SBATCH --partition=24g
#SBATCH --nodelist=stuttgart,muenchen,heidelberg,koblenz,bonn
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/projflow-check-%j.out
#SBATCH --error=slurm/logs/projflow-check-%j.err
#
# Stage B validation steps 2-3 (projflow.scripts.check_port): our ported
# model / CLIP / sampler vs upstream on the same GPU, TF32 off. Uses the Stage A
# work dir's Python extras (OpenAI clip, timm 1.0.9) and the numpy/torch shims
# so upstream's own model can be built next to ours.
#
# Usage (from the repo root on the cluster):
#     sbatch slurm/projflow/check_port.sbatch

set -euo pipefail
mkdir -p slurm/logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(dirname "$(dirname "${SCRIPT_DIR}")")}"
fi
IMAGE="${IMAGE:-${HOME}/rmg.sqsh}"
HF_CACHE="${HF_CACHE:-${HOME}/hf-cache}"
WORK="${PROJFLOW_WORK:-${HOME}/projflow-work}"
RUNS_ROOT="${RUNS_ROOT:-${HOME}/rmg-runs}"
CANONICAL_DIR="${CANONICAL_DIR:-${RUNS_ROOT}/h3d-canonical/new_joint_vecs}"
CHECKPOINT="${CHECKPOINT:-${WORK}/checkpoints/t2m/ACMDM_Raw_Flow_S_PatchSize22/model/latest.tar}"
GOLDEN="${GOLDEN:-${REPO}/external/HumanML3D/HumanML3D/new_joints/012314.npy}"
OUT="${OUT:-${RUNS_ROOT}/projflow-stageB-check/check_port.json}"

enroot start \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --env "HF_HOME=${HF_CACHE}" \
    --env "PYTHONUSERBASE=${WORK}/pyuser" \
    --env "PYTHONPATH=${REPO}/src:${REPO}/slurm/projflow/compat" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        cd '${REPO}'
        python -u -m projflow.scripts.check_port \
            --upstream external/ProjFlow \
            --checkpoint '${CHECKPOINT}' \
            --canonical-dir '${CANONICAL_DIR}' \
            --golden-new-joints '${GOLDEN}' \
            --out '${OUT}'
    "
