#!/bin/bash
#SBATCH --job-name=mardm-overfit
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/mardm-overfit-%j.out
#SBATCH --error=slurm/logs/mardm-overfit-%j.err
#
# MARDM overfit smoke: train AE then gen on a tiny deterministic subset of the
# train split (default 0.5%) and verify the pipeline can drive losses down. The
# strongest signal here is AE reconstruction L1 → ~0 on the subset; the gen
# branch's SiT velocity loss won't hit 0 (irreducible flow-matching variance
# across t / x0), but it should drop sharply and plateau low.
#
# Hyperparameters (vs. a normal run):
#   * subset_frac=0.005 (overridable via SUBSET_FRAC)
#   * data.preload=true         (encodes the ~56 clips into RAM once at init;
#                                the train script auto-forces num_workers=0
#                                because workers + preload is pickle-overhead
#                                without speedup)
#   * train.micro_batch_size=16 train.grad_accum=1 (BS=64 would yield 0 batches
#                                per epoch on a 56-clip subset with drop_last)
#   * model.cond_drop_prob=0 (no CFG dropout — let it memorize text→motion)
#   * model.dropout=0           (transformer dropout off)
#
# Submit:  sbatch slurm/mardm/overfit_mardm.sbatch
# Override the subset: SUBSET_FRAC=0.01 sbatch slurm/mardm/overfit_mardm.sbatch
# Override step counts: AE_STEPS=5000 GEN_STEPS=10000 sbatch ...

set -euo pipefail
mkdir -p slurm/logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
fi
IMAGE="${IMAGE:-${HOME}/rmg.sqsh}"
RUNS_ROOT="${RUNS_ROOT:-${HOME}/rmg-runs}"
HF_CACHE="${HF_CACHE:-${HOME}/hf-cache}"
DATA_ROOT="${RMG_DATA_ROOT:-${REPO}/external/data/humanml3d_packed}"
MARDM_STATS="${MARDM_STATS:-${RUNS_ROOT}/mardm_essential_stats.pt}"

SUBSET_FRAC="${SUBSET_FRAC:-0.005}"
AE_STEPS="${AE_STEPS:-3000}"
GEN_STEPS="${GEN_STEPS:-5000}"
# CFG dropout: must be >0 so the model learns a conditional-vs-unconditional
# contrast and generation can be driven by the caption with guidance>1. (0 only
# memorizes the context shortcut -> generation ignores text / collapses.)
COND_DROP="${COND_DROP:-0.1}"
# Min LR as a fraction of peak; raise (e.g. 0.3) for longer runs so the schedule
# doesn't floor out early and stall the loss.
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"

STAMP="${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}"
AE_RUN="mardm-overfit-ae-${STAMP}"
GEN_RUN="mardm-overfit-gen-${STAMP}"
AE_OUT="${RUNS_ROOT}/${AE_RUN}"
if [ -n "${WANDB_API_KEY:-}" ]; then
    : "${WANDB_MODE:=online}"
else
    : "${WANDB_MODE:=offline}"
fi

enroot start \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --env "HF_HOME=${HF_CACHE}" \
    --env "WANDB_MODE=${WANDB_MODE}" \
    --env "WANDB_API_KEY=${WANDB_API_KEY:-}" \
    --env "WANDB_ENTITY=${WANDB_ENTITY:-}" \
    --env "RMG_RUNS_DIR=${RUNS_ROOT}" \
    --env "RMG_DATA_ROOT=${DATA_ROOT}" \
    --env "MARDM_STATS=${MARDM_STATS}" \
    --env "PYTHONPATH=${REPO}/src" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        cd ${REPO}
        pip install -e . --no-deps --quiet

        [ -f '${MARDM_STATS}' ] || python -m mardm.scripts.compute_mardm_stats \\
            --data-root '${DATA_ROOT}' --out '${MARDM_STATS}'

        echo '=== overfit stage 1: AE on ${SUBSET_FRAC} of train split (${AE_STEPS} steps) ==='
        python -u -m mardm.scripts.train_mardm_ae \\
            run_name='${AE_RUN}' \\
            subset_frac=${SUBSET_FRAC} \\
            data.preload=true \\
            train.micro_batch_size=16 \\
            train.grad_accum=1 \\
            train.max_steps=${AE_STEPS} \\
            train.val_every=$((AE_STEPS / 4)) \\
            train.ckpt_every=${AE_STEPS} \\
            train.log_every=50

        echo '=== overfit stage 2: gen on the same subset (${GEN_STEPS} steps) ==='
        python -u -m mardm.scripts.train_mardm \\
            run_name='${GEN_RUN}' \\
            ae_checkpoint='${AE_OUT}/checkpoints/latest.pt' \\
            text_encoder.type=qwen3 \\
            subset_frac=${SUBSET_FRAC} \\
            data.preload=true \\
            train.micro_batch_size=16 \\
            train.grad_accum=1 \\
            model.cond_drop_prob=${COND_DROP} \\
            model.dropout=0 \\
            train.scheduler.min_lr_ratio=${MIN_LR_RATIO} \\
            train.max_steps=${GEN_STEPS} \\
            train.val_every=$((GEN_STEPS / 4)) \\
            train.ckpt_every=${GEN_STEPS} \\
            train.log_every=50

        echo '=== overfit smoke done ==='
        echo '  Look for: ae/l1 ~ 0 (AE memorized the subset) and gen/loss dropping'
        echo '  substantially from its initial value (SiT FM loss has irreducible'
        echo '  variance, so it plateaus low rather than hitting 0).'
    "
