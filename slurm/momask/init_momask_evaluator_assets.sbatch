#!/bin/bash
#SBATCH --job-name=momask-eval-assets
#SBATCH --partition=data
#SBATCH --nodes=1
#SBATCH --output=slurm/logs/momask-eval-assets-%j.out
#SBATCH --error=slurm/logs/momask-eval-assets-%j.err

set -euo pipefail
mkdir -p slurm/logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
fi

echo "[momask-eval-assets] host=$(hostname) repo=${REPO}"
REPO="${REPO}" DOWNLOAD_MOMASK_EVALUATOR=1 bash "${REPO}/slurm/ensure_momask_eval_assets.sh"
