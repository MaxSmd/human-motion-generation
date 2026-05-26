#!/bin/bash
#SBATCH --job-name=rmg-upstream-eval
#SBATCH --partition=12g
#SBATCH --nodelist=frankfurt,hamburg,kassel,leipzig
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/upstream-eval-%j.out
#SBATCH --error=slurm/logs/upstream-eval-%j.err
#
# Run UPSTREAM's exact eval pipeline (Text2MotionDatasetV2 +
# get_co_embeddings + matching-score loop) on OUR packed data, in upstream's
# expected file layout. Localizes whether the residual R@k / MM-Dist gap is
# from our sanity_eval wrapper or from our packed data itself.

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
        python -m spacy download en_core_web_sm --quiet || true
        python -u scripts/upstream_eval_on_our_data.py
    "
