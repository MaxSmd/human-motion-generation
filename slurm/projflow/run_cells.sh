#!/bin/bash
# Run upstream ProjFlow eval cells as a rolling pool packed onto the node's
# GPUs. Called inside the container by eval_upstream.sbatch.
#
# Each cell is an independent, unmodified `evaluation_ProjFlow.py` process
# (batch 32, seed 3407) — packing only shares the card, it does not change
# what a cell computes. One process uses ~2 GB and ~1 CPU core, and the
# sampler is partly CPU-bound (single process: 77% GPU util), so co-located
# cells overlap well. Up to SLOTS_PER_GPU cells run per GPU; whenever one
# finishes, the next queued cell starts on the GPU it freed, so memory stays
# high for the whole job (the cluster cancels low-memory jobs).
#
# Usage: run_cells.sh <work_dir> <out_dir> <projflow> <cfg> <num_gpus> <slots_per_gpu> <joint:intensity>...
# Exit status: number of failed cells (0 = all finished).

set -uo pipefail

WORK="$1"; OUT_DIR="$2"; PROJFLOW="$3"; CFG="$4"; NUM_GPUS="$5"; SLOTS_PER_GPU="$6"; shift 6
queue=("$@")
cd "${WORK}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

declare -A pid_gpu pid_name
declare -a gpu_busy
for ((g = 0; g < NUM_GPUS; g++)); do gpu_busy[g]=0; done
failed=0
done_cells=0
total=${#queue[@]}

launch() {  # launch <gpu> <joint:intensity>
    local gpu="$1" joint="${2%%:*}" intensity="${2##*:}"
    local name="joint${joint}_intensity${intensity}"
    local log="${OUT_DIR}/${name}.log"
    (
        start=$(date +%s)
        CUDA_VISIBLE_DEVICES="${gpu}" python -u evaluation_ProjFlow.py \
            --name ACMDM_Raw_Flow_S_PatchSize22 \
            --model ACMDM-Raw-Flow-S-PatchSize22 \
            --dataset_name t2m \
            --cfg "${CFG}" \
            --index "${joint}" \
            --intensity "${intensity}" \
            --projflow "${PROJFLOW}" \
            > "${log}" 2>&1
        status=$?
        echo "[projflow-eval] gpu=${gpu} exit=${status} wall_seconds=$(( $(date +%s) - start ))" >> "${log}"
        exit "${status}"
    ) &
    pid_gpu[$!]="${gpu}"
    pid_name[$!]="${name}"
    gpu_busy[gpu]=$((gpu_busy[gpu] + 1))
    echo "[run_cells] $(date +%H:%M) start  ${name} on gpu${gpu}"
}

fill() {  # start queued cells on any GPU with a free slot
    local g
    while [ ${#queue[@]} -gt 0 ]; do
        local target=-1
        for ((g = 0; g < NUM_GPUS; g++)); do
            if [ "${gpu_busy[g]}" -lt "${SLOTS_PER_GPU}" ] \
               && { [ "${target}" -lt 0 ] || [ "${gpu_busy[g]}" -lt "${gpu_busy[target]}" ]; }; then
                target=$g
            fi
        done
        [ "${target}" -ge 0 ] || return
        launch "${target}" "${queue[0]}"
        queue=("${queue[@]:1}")
        sleep 20  # stagger startup: dataset load + CLIP init are CPU/IO heavy
    done
}

fill
# Every 15 min: GPU memory/util and the job cgroup's RAM (24 GiB hard limit).
(
    cg=$(find /sys/fs/cgroup -path "*job_${SLURM_JOB_ID:-none}/step_batch/user" -maxdepth 6 2>/dev/null | head -1)
    while sleep 900; do
        gpu=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader)
        ram=$( [ -n "${cg}" ] && awk '{printf "%.1f GiB", $1 / 2^30}' "${cg}/memory.current" || echo n/a)
        anon=$( [ -n "${cg}" ] && awk '$1 == "anon" {printf "%.1f GiB", $2 / 2^30}' "${cg}/memory.stat" || echo n/a)
        echo "[run_cells] $(date +%H:%M) gpu: ${gpu} | ram: ${ram} (anon ${anon})"
    done
) &
snapshot=$!

while [ "${done_cells}" -lt "${total}" ]; do
    wait -n -p finished "${!pid_gpu[@]}"
    status=$?
    gpu="${pid_gpu[$finished]}"
    name="${pid_name[$finished]}"
    unset "pid_gpu[$finished]" "pid_name[$finished]"
    gpu_busy[gpu]=$((gpu_busy[gpu] - 1))
    done_cells=$((done_cells + 1))
    if [ "${status}" -eq 0 ]; then
        echo "[run_cells] $(date +%H:%M) OK     ${name} (${done_cells}/${total})"
    else
        echo "[run_cells] $(date +%H:%M) FAILED ${name} exit=${status} (${done_cells}/${total}); see ${OUT_DIR}/${name}.log"
        failed=$((failed + 1))
    fi
    fill
done

kill "${snapshot}" 2>/dev/null
echo "[run_cells] finished: $((total - failed)) ok, ${failed} failed"
exit "${failed}"
