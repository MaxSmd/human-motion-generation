#!/bin/bash
#SBATCH --job-name=rmg-diag-h3d
#SBATCH --partition=12g
#SBATCH --nodelist=frankfurt,hamburg,kassel,leipzig
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/diag-h3d-%j.out
#SBATCH --error=slurm/logs/diag-h3d-%j.err
#
# Compare our T+R → 263-D conversion to upstream `process_file` element-wise
# on a few real clips. Tells us which feature block is responsible for the
# diversity_real ≈ 4.05 (paper: 9.5) gap seen in evaluation.

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
        python scripts/diagnose_h3d_conversion.py --n 5 --split test
    "
