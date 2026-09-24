#!/bin/bash
#SBATCH --job-name=projflow-eval
#SBATCH --partition=24g
#SBATCH --nodelist=stuttgart,muenchen,heidelberg,koblenz,bonn
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/logs/projflow-eval-%j.out
#SBATCH --error=slurm/logs/projflow-eval-%j.err
#
# Stage A — Table 1 of ProjFlow with the unmodified upstream evaluation
# (evaluation_ProjFlow.py, 67-D Meng et al. evaluator, ACMDM-S-PS22 prior).
#
# The grid is the OmniControl protocol: joints {pelvis 0, l-foot 10,
# r-foot 11, head 15, l-wrist 20, r-wrist 21} x density {1, 2, 5, 25%, 100%}
# = 30 cells. Upstream treats --intensity 1/2/5 as a keyframe count and
# anything else as a percentage of the clip, so 25/100 are the paper's
# "49 / 196 keyframes" on 196-frame clips. The paper's per-joint row is the
# mean over the five densities.
#
# Packing: QOS students_normal allows ONE running job per user (2 queued)
# with at most 1 GPU per job; the cluster assigns 4 CPUs / 24 GB and rejects
# explicit --cpus-per-task/--mem (and --exclude). It also cancels jobs whose
# GPU memory stays low, while a single cell uses only ~2 GB. So the one job
# keeps SLOTS_PER_GPU cells running on its GPU as a rolling pool
# (run_cells.sh) until every cell is done. A 24g job gets 24 GiB RAM (hard
# cgroup limit, OOM-kills) and 4 CPUs. Measured per cell (job 25775): 2.04 GB
# GPU memory and ~3.2 GB RAM (1.8 GB main process + 1.4 GB DataLoader worker;
# upstream's --num_workers 1 is kept, since num_workers changes the RNG stream
# that picks captions/crops). 10 slots OOM-killed cells (job 25774); 7 slots
# hold steady at ~22.6 GB RAM and 15.2 of 24 GB GPU at 100% util, ~63 s/batch
# per cell (~2.5 h/cell, ~10-11 h for the grid). On 12g use 5 slots (~10 of
# 11 GB on a 2080 Ti, ~16 GB RAM):
#     SLOTS_PER_GPU=5 sbatch --partition=12g \
#         --nodelist=dortmund,frankfurt,hamburg,leipzig,passau slurm/projflow/eval_upstream.sbatch
# (12g: kassel had an nvml failure; jena/ulm are Maxwell Titan X.)
#
# Prerequisite: sbatch slurm/projflow/setup_upstream.sbatch  (once)
#
# Usage:
#     sbatch slurm/projflow/eval_upstream.sbatch               # full 30-cell grid
#     CELLS="0 17" sbatch slurm/projflow/eval_upstream.sbatch  # rerun given cells
#     PROJFLOW=false sbatch ...                                # upstream's "ProjFlow off" switch
#
# Cell c = joint JOINTS[c / 5], density INTENSITIES[c % 5]. Per-cell logs land
# in ${OUT_DIR}; collect with
#     python -m projflow.scripts.collect_upstream <OUT_DIR>

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
PROJFLOW="${PROJFLOW:-true}"
CFG="${CFG:-3}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-7}"
NUM_GPUS=1
OUT_DIR="${OUT_DIR:-${RUNS_ROOT}/projflow-stageA-table1-projflow_${PROJFLOW}}"

JOINTS=(0 10 11 15 20 21)
INTENSITIES=(1 2 5 25 100)
NUM_CELLS=$(( ${#JOINTS[@]} * ${#INTENSITIES[@]} ))
CELLS="${CELLS:-$(seq -s " " 0 $((NUM_CELLS - 1)))}"

specs=()
for c in ${CELLS}; do
    [ "${c}" -lt "${NUM_CELLS}" ] || continue
    specs+=("${JOINTS[$((c / ${#INTENSITIES[@]}))]}:${INTENSITIES[$((c % ${#INTENSITIES[@]}))]}")
done
mkdir -p "${OUT_DIR}"

test -f "${WORK}/checkpoints/t2m/ACMDM_Raw_Flow_S_PatchSize22/model/latest.tar" \
    || { echo "missing ${WORK}; run slurm/projflow/setup_upstream.sbatch first"; exit 1; }

echo "[projflow-eval] cells=(${CELLS}) slots/gpu=${SLOTS_PER_GPU} projflow=${PROJFLOW} cfg=${CFG} -> ${OUT_DIR}"
echo "[projflow-eval] upstream commit $(git -C "${REPO}/external/ProjFlow" rev-parse --short HEAD 2>/dev/null || cat "${REPO}/external/ProjFlow.COMMIT" 2>/dev/null || echo unknown)"

enroot start \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --env "HF_HOME=${HF_CACHE}" \
    --env "PYTHONUSERBASE=${WORK}/pyuser" \
    --env "PYTHONPATH=${REPO}/slurm/projflow/compat" \
    "${IMAGE}" \
    bash "${REPO}/slurm/projflow/run_cells.sh" "${WORK}" "${OUT_DIR}" "${PROJFLOW}" "${CFG}" \
    "${NUM_GPUS}" "${SLOTS_PER_GPU}" "${specs[@]}"
