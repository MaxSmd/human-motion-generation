#!/bin/bash
#SBATCH --job-name=rmg-viz
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/rmg-viz-%j.out
#SBATCH --error=slurm/logs/rmg-viz-%j.err
#
# Visualize motion as MP4 — real packed clips or text-conditioned model samples.
#
# Examples:
#
#   # 1. Render 4 real packed clips (no model needed)
#   MODE=clip CLIPS=000021,000019,000022,000026 sbatch slurm/visualize.sbatch
#
#   # 2. Prompt a trained checkpoint for 4 motions
#   MODE=prompt \
#       CKPT=/mnt/home/${USER}/rmg-runs/<run>/checkpoints/latest.pt \
#       MODEL_PRESET=dit_large \
#       PRESET=rmg_large \
#       PROMPTS='a person walks forward in a circle|a person sits down|a person waves left hand|a person jumps' \
#       sbatch slurm/visualize.sbatch
#
# Outputs land in ~/rmg-runs/<RUN_NAME>/viz/*.mp4

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
RUNS_ROOT="${RUNS_ROOT:-${HOME}/rmg-runs}"

MODE="${MODE:-clip}"
CLIPS="${CLIPS:-000021,000019,000022,000026}"
PROMPTS="${PROMPTS:-a person walks forward in a circle|a person sits down on the floor|a person waves their left hand|a person does jumping jacks}"
CKPT="${CKPT:-}"
MODEL_PRESET="${MODEL_PRESET:-dit_base}"
PRESET="${PRESET:-rmg_base}"
NUM_FRAMES="${NUM_FRAMES:-100}"
NUM_STEPS="${NUM_STEPS:-50}"
GUIDANCE="${GUIDANCE:-6.5}"
RUN_NAME="${RUN_NAME:-viz-${MODE}-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"

if [ "$MODE" = "prompt" ] && [ -z "$CKPT" ]; then
    echo "ERROR: MODE=prompt requires CKPT=/path/to/checkpoint.pt"
    exit 1
fi

# Hydra needs the value side of comma-bearing args quoted to disambiguate
# "string with commas" from "list literal". We pass literal single quotes
# through bash → inner bash → python argv via `\"…\"` escaping.

enroot start \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --env "HF_HOME=${HF_CACHE}" \
    --env "RMG_RUNS_DIR=${RUNS_ROOT}" \
    --env "PYTHONPATH=${REPO}/src" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        cd ${REPO}
        pip install -e . --no-deps --quiet
        # No ffmpeg needed — visualize.py outputs GIF via PillowWriter (Pillow
        # ships with matplotlib). Joints also dumped as .npy alongside the GIF.
        if [ '${MODE}' = 'prompt' ]; then
            python -u scripts/visualize.py \\
                model=${MODEL_PRESET} \\
                train=${PRESET} \\
                text_encoder.type=qwen3 \\
                run_name='${RUN_NAME}' \\
                +viz.mode=prompt \\
                +viz.checkpoint=${CKPT} \\
                \"+viz.prompts='${PROMPTS}'\" \\
                +viz.num_frames=${NUM_FRAMES} \\
                +viz.num_sample_steps=${NUM_STEPS} \\
                +viz.guidance_scale=${GUIDANCE}
        else
            python -u scripts/visualize.py \\
                model=${MODEL_PRESET} \\
                train=${PRESET} \\
                text_encoder.type=qwen3 \\
                run_name='${RUN_NAME}' \\
                +viz.mode=clip \\
                \"+viz.clips='${CLIPS}'\"
        fi
    "
