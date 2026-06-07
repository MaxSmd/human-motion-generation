#!/bin/bash
#SBATCH --job-name=rmg-overfit
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=02:30:00
#SBATCH --output=slurm/logs/rmg-overfit-%j.out
#SBATCH --error=slurm/logs/rmg-overfit-%j.err
#
# TINY-OVERFIT SANITY TEST — can the full RMG model memorize a handful of
# clips? This is the decisive implementation-correctness check: a correct
# representation / encode-decode / FK / flow-matching loss / sampler MUST be
# able to overfit a tiny fixed set. If it can't, there is a bug; if it can
# (loss → floor, generations reproduce the clips), the machinery is sound and
# weak full-runs are a data/compute story, not a code story.
#
# Setup: exactly N regular clips (no mirrors, no runtime mirror augmentation),
# full-batch gradient descent (micro-batch = N), fast-warming EMA so the
# periodic sample dumps are usable, frequent checkpoints + samples.
#
# Env knobs:
#   N           default 16    (>= micro-batch so the loader isn't empty under drop_last)
#   MAX_STEPS   default 3000
#   SUBSET_SEED default 0      (which N clips; deterministic)
#   MODEL_PRESET/PRESET        dit_large / rmg_large (the real model)
#
# Submit:
#   sbatch slurm/overfit_tiny.sbatch
#   # or a quicker pass on the smaller model:
#   MODEL_PRESET=dit_base PRESET=rmg_base MAX_STEPS=2000 sbatch slurm/overfit_tiny.sbatch

set -euo pipefail
mkdir -p slurm/logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(dirname "${SCRIPT_DIR}")}"
fi

IMAGE="${IMAGE:-${HOME}/rmg.sqsh}"
RUNS_ROOT="${RUNS_ROOT:-${HOME}/rmg-runs}"
HF_CACHE="${HF_CACHE:-${HOME}/hf-cache}"

N="${N:-16}"
MAX_STEPS="${MAX_STEPS:-3000}"
SUBSET_SEED="${SUBSET_SEED:-0}"
MODEL_PRESET="${MODEL_PRESET:-dit_large}"
PRESET="${PRESET:-rmg_large}"
EMA_DECAY="${EMA_DECAY:-0.99}"
CKPT_EVERY="${CKPT_EVERY:-500}"
SAMPLE_EVERY="${SAMPLE_EVERY:-250}"
PRECISION="${PRECISION:-bf16}"        # fp32 for a clean overfit (no bf16 rounding floor)
GRAD_CLIP="${GRAD_CLIP:-0.5}"         # raise (e.g. 1.0) so spiky steps aren't clipped ~100x
LR="${LR:-1e-4}"
RUN_NAME="${RUN_NAME:-rmg-overfit-${N}clip-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"

: "${WANDB_MODE:=offline}"

enroot start \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --env "HF_HOME=${HF_CACHE}" \
    --env "WANDB_MODE=${WANDB_MODE}" \
    --env "RMG_RUNS_DIR=${RUNS_ROOT}" \
    --env "PYTHONPATH=${REPO}/src" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        cd ${REPO}
        pip install -e . --no-deps --quiet
        python scripts/train.py \\
            model=${MODEL_PRESET} \\
            train=${PRESET} \\
            text_encoder.type=qwen3 \\
            data.subset_n=${N} \\
            data.subset_seed=${SUBSET_SEED} \\
            data.mirror_augment=false \\
            train.micro_batch_size=${N} \\
            train.grad_accum=1 \\
            train.max_steps=${MAX_STEPS} \\
            train.ema.decay=${EMA_DECAY} \\
            train.precision=${PRECISION} \\
            train.grad_clip=${GRAD_CLIP} \\
            train.optimizer.lr=${LR} \\
            train.ckpt_every=${CKPT_EVERY} \\
            train.sample_every=${SAMPLE_EVERY} \\
            train.log_every=10 \\
            run_name='${RUN_NAME}'
    "
