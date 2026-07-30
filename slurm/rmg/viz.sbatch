#!/bin/bash
#SBATCH --job-name=rmg-viz
#SBATCH --partition=24g
#SBATCH --qos=students_normal
# The 12g partition mixes GPU generations and `ccc` in each node's GRES is its
# compute capability x10: ulm/jena are TITANX (5.2) and dortmund is TITANXp (6.1),
# which this CUDA container has no kernels for — a job landing there dies with
# "no kernel image is available for execution on the device", and the traceback
# points into the text encoder so it reads like a model bug. Asking for ccc:75
# (a no_consume GRES, so it filters rather than reserves) keeps this off the old
# cards however the script is submitted. The app already passes its own --gres
# for 12g, which overrides this line; this is the guard for a direct sbatch.
# NOTE: this cluster rejects --exclude outright.
#SBATCH --gres=gpu:1,ccc:75
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/rmg-viz-%j.out
#SBATCH --error=slurm/logs/rmg-viz-%j.err
#
# Unified, fully-configurable VIZ job. MODE selects clip|prompt|compare|samples;
# all other knobs are env vars; OVERRIDES appends arbitrary extra Hydra args.
# Renders GIF (+ .npy joints) under RUNS_ROOT/RUN_NAME/viz/ which the app pulls.
#
#   MODE=clip CLIPS=000021,000019 sbatch slurm/rmg/viz.sbatch
#   MODE=prompt CKPT=/path PROMPTS='a person walks|a person sits' sbatch slurm/rmg/viz.sbatch

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
RUNS_ROOT="${RUNS_ROOT:-${REPO}/runs/rmg/viz}"

MODE="${MODE:-clip}"
CLIPS="${CLIPS:-000021,000019,000022,000026}"
SAMPLES_FILE="${SAMPLES_FILE:-}"
PROMPTS="${PROMPTS:-a person walks forward in a circle|a person sits down on the floor}"
CKPT="${CKPT:-}"
MODEL_PRESET="${MODEL_PRESET:-dit_base}"
PRESET="${PRESET:-rmg_base}"
TEXT_ENCODER="${TEXT_ENCODER:-qwen3}"
NUM_FRAMES="${NUM_FRAMES:-100}"
NUM_STEPS="${NUM_STEPS:-50}"
GUIDANCE="${GUIDANCE:-6.5}"
SEED="${SEED:-0}"
USE_EMA="${USE_EMA:-true}"
SUBSET_FRACTION="${SUBSET_FRACTION:-0.01}"
SUBSET_SEED="${SUBSET_SEED:-0}"
N_COMPARE="${N_COMPARE:-4}"
# Clip ids whose GT the app already has rendered (its GT registry) — mode=compare
# renders only the prediction for these. Comma-separated; empty renders both.
SKIP_GT="${SKIP_GT:-}"
RUN_NAME="${RUN_NAME:-rmg-viz-${MODE}-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
OVERRIDES="${OVERRIDES:-}"
# Optional sampling-time constraints (JSON); mode=prompt only. Read by
# visualize.py inside the container as RMG_CONSTRAINTS (fixed angles),
# RMG_RANGES (hinge limits) and RMG_SCENE (room/obstacles/spawn) + guidance.
CONSTRAINTS="${CONSTRAINTS:-}"
RANGES="${RANGES:-}"
SCENE="${SCENE:-}"
ROOM_GUIDANCE="${ROOM_GUIDANCE:-0.0}"
# Spatial (mask-control) trajectory targets: joints pinned to world positions at
# chosen frames. Read as RMG_TRAJECTORY — see flow.trajectory.
TRAJECTORY="${TRAJECTORY:-}"
# Fused mode=prompt batch (JSON list). The vars above apply to every prompt in the
# job; BATCH instead gives each its OWN prompt + constraints/ranges/scene/frames/
# seed, so the app can render a whole study in ONE job instead of queueing N. When
# set it supersedes PROMPTS/CONSTRAINTS/RANGES/SCENE.
BATCH="${BATCH:-}"

if { [ "$MODE" = "prompt" ] || [ "$MODE" = "compare" ]; } && [ -z "$CKPT" ]; then
    echo "ERROR: MODE=${MODE} requires CKPT=/path/to/checkpoint.pt"; exit 1
fi
if [ "$MODE" = "samples" ] && [ -z "$SAMPLES_FILE" ]; then
    echo "ERROR: MODE=samples requires SAMPLES_FILE=/path/to/samples"; exit 1
fi

enroot start \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --env "HF_HOME=${HF_CACHE}" \
    --env "MGEN_RUNS_DIR=${RUNS_ROOT}" \
    --env "PYTHONPATH=${REPO}/src" \
    --env "RMG_CONSTRAINTS=${CONSTRAINTS}" \
    --env "RMG_RANGES=${RANGES}" \
    --env "RMG_SCENE=${SCENE}" \
    --env "RMG_ROOM_GUIDANCE=${ROOM_GUIDANCE}" \
    --env "RMG_TRAJECTORY=${TRAJECTORY}" \
    --env "RMG_BATCH=${BATCH}" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        cd ${REPO}
        pip install -e . --no-deps --quiet
        if [ '${MODE}' = 'prompt' ]; then
            python -u -m rmg.scripts.visualize \\
                model=${MODEL_PRESET} train=${PRESET} text_encoder.type=${TEXT_ENCODER} \\
                run_name='${RUN_NAME}' \\
                +viz.mode=prompt +viz.checkpoint=${CKPT} +viz.use_ema=${USE_EMA} \\
                \"+viz.prompts='${PROMPTS}'\" \\
                +viz.num_frames=${NUM_FRAMES} +viz.num_sample_steps=${NUM_STEPS} \\
                +viz.guidance_scale=${GUIDANCE} +viz.seed=${SEED} ${OVERRIDES}
        elif [ '${MODE}' = 'compare' ]; then
            python -u -m rmg.scripts.visualize \\
                model=${MODEL_PRESET} train=${PRESET} text_encoder.type=${TEXT_ENCODER} \\
                run_name='${RUN_NAME}' \\
                +viz.mode=compare +viz.checkpoint=${CKPT} +viz.use_ema=${USE_EMA} \\
                +viz.subset_fraction=${SUBSET_FRACTION} +viz.subset_seed=${SUBSET_SEED} \\
                +viz.list_subset_n=${N_COMPARE} \\
                \"+viz.clips='${CLIPS}'\" \\
                \"+viz.skip_gt='${SKIP_GT}'\" \\
                +viz.num_sample_steps=${NUM_STEPS} +viz.guidance_scale=${GUIDANCE} ${OVERRIDES}
        elif [ '${MODE}' = 'samples' ]; then
            python -u -m rmg.scripts.visualize \\
                model=${MODEL_PRESET} train=${PRESET} text_encoder.type=${TEXT_ENCODER} \\
                run_name='${RUN_NAME}' \\
                +viz.mode=samples \"+viz.samples_file='${SAMPLES_FILE}'\" ${OVERRIDES}
        else
            python -u -m rmg.scripts.visualize \\
                model=${MODEL_PRESET} train=${PRESET} text_encoder.type=${TEXT_ENCODER} \\
                run_name='${RUN_NAME}' \\
                +viz.mode=clip \"+viz.clips='${CLIPS}'\" ${OVERRIDES}
        fi
    "
