#!/bin/bash
#SBATCH --job-name=rmg-roundtrip
#SBATCH --partition=12g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:RTX2080Ti:1
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/rmg-roundtrip-%j.out
#SBATCH --error=slurm/logs/rmg-roundtrip-%j.err
#
# Pack IK→FK round-trip error vs official HumanML3D files, on one clip present
# in both (see rmg/scripts/eval_roundtrip_check.py). Minutes, tiny GPU footprint.
#
#   sbatch slurm/rmg/roundtrip_check.sbatch

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
CLIP_ID="${CLIP_ID:-012314}"
PACK_ZIP="${PACK_ZIP:-${REPO}/external/data/humanml3d_packed/humanml3d.zip}"
OVERRIDES="${OVERRIDES:-}"

bash "${REPO}/slurm/ensure_eval_assets.sh"

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
        python -m spacy download en_core_web_sm --quiet || true
        python -u -m rmg.scripts.eval_roundtrip_check \\
            train=rmg_mid \\
            +check.clip_id=${CLIP_ID} \\
            +check.pack_zip=${PACK_ZIP} \\
            ${OVERRIDES}
    "
