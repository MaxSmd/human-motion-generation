#!/bin/bash
#SBATCH --job-name=rmg-smoke
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:05:00
#SBATCH --output=slurm/logs/smoke-%j.out
#SBATCH --error=slurm/logs/smoke-%j.err
#
# 5-minute sanity job. Confirms that:
#   - the enroot image runs and sees the GPU
#   - our `rmg` package imports
#   - the test suite passes inside the container
#
# Edit the IMAGE / REPO paths below for your username before submitting.

set -euo pipefail

# Auto-derive REPO from this script's location (slurm/smoke.sbatch → <repo>).
# Override with `REPO=/path/to/repo sbatch slurm/smoke.sbatch` if needed.
# Use SLURM_SUBMIT_DIR when run under sbatch (script is spooled, BASH_SOURCE
# doesn't resolve to the repo). Fall back to BASH_SOURCE for direct runs.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(dirname "${SCRIPT_DIR}")}"
fi

IMAGE="${IMAGE:-${HOME}/rmg.sqsh}"
HF_CACHE="${HF_CACHE:-${HOME}/hf-cache}"

mkdir -p "${REPO}/slurm/logs"
cd "${REPO}"

enroot start \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --env "HF_HOME=${HF_CACHE}" \
    --env "PYTHONPATH=${REPO}/src" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        echo '--- nvidia-smi ---'
        nvidia-smi || true
        echo '--- torch ---'
        python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"no-gpu\")'
        echo '--- rmg import + tests ---'
        cd ${REPO}
        pip install -e . --no-deps --quiet
        pytest -q
    "
