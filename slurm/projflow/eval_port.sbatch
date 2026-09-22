#!/bin/bash
#SBATCH --job-name=projflow-port
#SBATCH --partition=24g
#SBATCH --nodelist=stuttgart,muenchen,heidelberg,koblenz,bonn
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/logs/projflow-port-%j.out
#SBATCH --error=slurm/logs/projflow-port-%j.err
#
# Stage B — the ported ProjFlow sampler (src/projflow) on the 30-cell
# OmniControl grid, scored with our 263-D Guo evaluator
# (projflow.scripts.evaluate). One process, large sampling batches
# (BATCH_SIZE, default 1024) so a single job fills the GPU — QOS allows one
# running job / one GPU per user and cancels low-memory jobs.
#
# Paper target: supp. Table 8 (legacy metrics), ACMDM-S-PS22+ProjFlow.
#
# Usage (from the repo root on the cluster):
#     sbatch slurm/projflow/eval_port.sbatch                         # full grid
#     CELLS="0:1,0:100" MAX_ENTRIES=256 sbatch slurm/projflow/eval_port.sbatch   # smoke
#     EXTRA="--no-kinematic-metric" OUT_DIR=... sbatch ...           # Table 3 ablations
# Finished cells are skipped on resubmission (per-cell JSON under OUT_DIR/cells).

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
OFFSETS="${OFFSETS:-${REPO}/external/data/humanml3d_packed/target_offsets.pt}"
OUT_DIR="${OUT_DIR:-${RUNS_ROOT}/projflow-stageB-table8}"
CELLS="${CELLS:-all}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
MAX_ENTRIES="${MAX_ENTRIES:-0}"
EXTRA="${EXTRA:-}"

enroot start \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --env "HF_HOME=${HF_CACHE}" \
    --env "PYTHONPATH=${REPO}/src" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        cd '${REPO}'
        ( while sleep 900; do nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader \
            | sed 's/^/[gpu] /'; done ) &
        python -u -m projflow.scripts.evaluate \
            --canonical-dir '${CANONICAL_DIR}' \
            --checkpoint '${CHECKPOINT}' \
            --offsets '${OFFSETS}' \
            --out-dir '${OUT_DIR}' \
            --cells '${CELLS}' \
            --batch-size ${BATCH_SIZE} \
            --max-entries ${MAX_ENTRIES} \
            ${EXTRA}
    "
