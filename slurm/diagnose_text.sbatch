#!/bin/bash
#SBATCH --job-name=rmg-diag-text
#SBATCH --partition=12g
#SBATCH --nodelist=frankfurt,hamburg,kassel,leipzig
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:05:00
#SBATCH --output=slurm/logs/diag-text-%j.out
#SBATCH --error=slurm/logs/diag-text-%j.err
#
# Pin down the residual R@k / MM-Dist gap by checking:
#   1. Does texts.zip ship M-prefixed mirrored captions?
#   2. Do our packed M-clips have their own captions (or did we silently
#      fall back to the originals)?
#   3. How many caption lines are sub-clip vs whole-clip?
#   4. WordVectorizer vocab coverage on real test captions.
#
# No model, no GPU work — runs in seconds.

set -euo pipefail
mkdir -p slurm/logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(dirname "${SCRIPT_DIR}")}"
fi
IMAGE="${IMAGE:-${HOME}/rmg.sqsh}"

enroot start \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --env "PYTHONPATH=${REPO}/src" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        cd ${REPO}
        pip install -e . --no-deps --quiet
        python scripts/diagnose_text_pairing.py
    "
