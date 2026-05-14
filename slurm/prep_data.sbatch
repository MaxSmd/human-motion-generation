#!/bin/bash
#SBATCH --job-name=rmg-prep-data
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/prep-%j.out
#SBATCH --error=slurm/logs/prep-%j.err
#
# Build the packed HumanML3D dataset from AMASS + HumanAct12.
#
# Stage 1 (GPU): AMASS .npz → 22-joint positions .npy via SMPL+H FK +
#                trans_matrix Y-up swap + 20fps downsample. Also passes
#                HumanAct12 .npy files through (sliced to 22 joints).
# Stage 2 (CPU): index.csv-driven slicing → upstream IK on joint positions
#                → packed humanml3d.zip + splits.json + target_offsets.pt.
#
# Output lands in external/data/humanml3d_packed/ (git-ignored).
# Resubmitting is safe: stage 1 overwrites .npy files in /tmp/joints, stage 2
# overwrites the zip. Estimated total runtime: ~30-45 min on an A5000.

set -euo pipefail
mkdir -p slurm/logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(dirname "${SCRIPT_DIR}")}"
fi

IMAGE="${IMAGE:-${HOME}/rmg.sqsh}"
HF_CACHE="${HF_CACHE:-${HOME}/hf-cache}"
# Persistent default: survives across job submissions, so if Stage 2 ever
# fails we don't redo Stage 1's ~30 min GPU compute. Override with
# JOINTS_DIR=/tmp/... if you want speed and accept losing it on job exit.
JOINTS_DIR="${JOINTS_DIR:-${REPO}/external/data/joints_cache}"

cd "${REPO}"

echo "[prep] repo=${REPO}  image=${IMAGE}"
echo "[prep] joints intermediate: ${JOINTS_DIR}"

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

        echo
        echo '=== Stage 1: AMASS .npz -> joints .npy (SMPL+H FK on GPU) ==='
        python scripts/prepare_humanml3d.py raw-pose \\
            --amass-root external/data/amass_data \\
            --body-models external/data/body_models \\
            --out-dir ${JOINTS_DIR} \\
            --device cuda

        echo
        echo '=== Stage 2: joints + index.csv + texts -> packed humanml3d.zip ==='
        python scripts/prepare_humanml3d.py pack \\
            --joints-root ${JOINTS_DIR} \\
            --humanml3d-repo external/HumanML3D \\
            --output-dir external/data/humanml3d_packed

        echo
        echo '=== outputs ==='
        ls -lh external/data/humanml3d_packed/
        echo
        echo '=== sample clip ==='
        python -c \"
import zipfile, io, torch, json
zp = 'external/data/humanml3d_packed/humanml3d.zip'
with zipfile.ZipFile(zp) as zf:
    names = zf.namelist()
    print(f'total clips in zip: {len(names)}')
    sample_name = names[0]
    blob = torch.load(io.BytesIO(zf.read(sample_name)), weights_only=True)
    print(f'sample: {sample_name}')
    print(f'  translation: {blob[\\\"translation\\\"].shape}')
    print(f'  quats:       {blob[\\\"quats\\\"].shape}')
    print(f'  texts ({len(blob[\\\"texts\\\"])}): {blob[\\\"texts\\\"][0][:80]!r}')
splits = json.load(open('external/data/humanml3d_packed/splits.json'))
print(f'splits: train={len(splits[\\\"train\\\"])} val={len(splits[\\\"val\\\"])} test={len(splits[\\\"test\\\"])}')
\"
    "

echo "[prep] DONE"
