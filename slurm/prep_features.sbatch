#!/bin/bash
#SBATCH --job-name=h3d-features
#SBATCH --partition=data
#SBATCH --qos=students_normal
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/features-%j.out
#SBATCH --error=slurm/logs/features-%j.err
#
# Stage 3: regenerate canonical HumanML3D `new_joint_vecs` 263-D features from
# the cached AMASS joints (joints_cache, Stage-1 output) — the features the Guo
# evaluator was trained on, whose originals were deleted from the share.
#
# CPU-only (no --gres): runs on the `data` partition without holding a GPU.
# Resume-safe: existing output files are skipped on resubmit.
#
# Ends with a golden-reference check: the one canonical file that survived on
# the share (012314.npy) is compared element-wise against our regeneration.

set -euo pipefail
mkdir -p slurm/logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(dirname "${SCRIPT_DIR}")}"
fi

IMAGE="${IMAGE:-${HOME}/rmg.sqsh}"
JOINTS_DIR="${JOINTS_DIR:-/mnt/projects/drl4cvb/human-motion-representation/data/data/joints_cache}"
OUT_DIR="${OUT_DIR:-${HOME}/rmg-runs/h3d-canonical}"
# This cluster forbids --cpus-per-task (CPUs follow GPU count); use whatever
# the allocation gives us.
WORKERS="${WORKERS:-${SLURM_CPUS_ON_NODE:-4}}"
GOLDEN="${GOLDEN:-/mnt/projects/drl4cvb/human-motion-representation/data/HumanML3D/HumanML3D/new_joint_vecs/012314.npy}"

cd "${REPO}"

echo "[features] repo=${REPO}  image=${IMAGE}"
echo "[features] joints=${JOINTS_DIR}  out=${OUT_DIR}  workers=${WORKERS}"

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

        echo
        echo '=== Stage 3: joints + index.csv -> canonical new_joint_vecs ==='
        python -m shared.data.prepare_humanml3d features \\
            --joints-root ${JOINTS_DIR} \\
            --humanml3d-repo external/HumanML3D \\
            --output-dir ${OUT_DIR} \\
            --workers ${WORKERS}

        echo
        echo '=== outputs ==='
        ls ${OUT_DIR}/new_joint_vecs | wc -l

        echo
        echo '=== golden-reference check (012314.npy survived the share wipe) ==='
        python - <<'EOF'
import numpy as np
ours = np.load('${OUT_DIR}/new_joint_vecs/012314.npy')
gold = np.load('${GOLDEN}')
print(f'ours {ours.shape}  golden {gold.shape}')
if ours.shape != gold.shape:
    print('SHAPE MISMATCH — frame windows differ; investigate before trusting the regeneration')
else:
    d = np.abs(ours - gold)
    print(f'max abs diff  {d.max():.6g}')
    print(f'mean abs diff {d.mean():.6g}')
    scale = np.abs(gold).mean()
    print(f'golden mean |x| {scale:.6g}  ->  rel err {d.mean()/scale:.3%}')
EOF
    "

echo "[features] DONE"
