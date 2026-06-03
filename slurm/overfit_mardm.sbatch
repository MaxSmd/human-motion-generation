#!/bin/bash
#SBATCH --job-name=mardm-overfit
#SBATCH --partition=12g
#SBATCH --nodelist=frankfurt,hamburg,kassel,leipzig
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
#   * data.mirror_augment=false (don't perturb the small set)
#   * data.num_workers=0        (multiproc dataloader hangs on tiny subsets due
#                                to fork+zipfile interaction; the subset is
#                                small enough that single-process loading is
#                                trivially fast — observed step 0 hang on
#                                kassel with workers=8, fixed by =0)
#   * data.persistent_workers=false (irrelevant when workers=0; explicit to
#                                avoid the DataLoader warning)
#   * train.micro_batch_size=16 train.grad_accum=1 (default 64 with drop_last=True
#                                yields 0 batches per epoch on a 56-clip subset;
#                                _infinite() then spins forever on an empty
#                                iterator and the job hangs at 100% CPU. BS=16
#                                gives 3 batches/epoch — plenty for overfit.)
#   * model.cond_drop_prob=0 (no CFG dropout — let it memorize text→motion)
#   * model.dropout=0           (transformer dropout off)
#
# Submit:  sbatch slurm/overfit_mardm.sbatch
# Override the subset: SUBSET_FRAC=0.01 sbatch slurm/overfit_mardm.sbatch
# Override step counts: AE_STEPS=5000 GEN_STEPS=10000 sbatch ...

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
DATA_ROOT="${RMG_DATA_ROOT:-${REPO}/external/data/humanml3d_packed}"
MARDM_STATS="${MARDM_STATS:-${RUNS_ROOT}/mardm_essential_stats.pt}"

SUBSET_FRAC="${SUBSET_FRAC:-0.005}"
AE_STEPS="${AE_STEPS:-3000}"
GEN_STEPS="${GEN_STEPS:-5000}"

STAMP="${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}"
AE_RUN="mardm-overfit-ae-${STAMP}"
GEN_RUN="mardm-overfit-gen-${STAMP}"
AE_OUT="${RUNS_ROOT}/${AE_RUN}"
: "${WANDB_MODE:=offline}"

enroot start \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --env "HF_HOME=${HF_CACHE}" \
    --env "WANDB_MODE=${WANDB_MODE}" \
    --env "RMG_RUNS_DIR=${RUNS_ROOT}" \
    --env "RMG_DATA_ROOT=${DATA_ROOT}" \
    --env "MARDM_STATS=${MARDM_STATS}" \
    --env "PYTHONPATH=${REPO}/src" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        cd ${REPO}
        pip install -e . --no-deps --quiet

        [ -f '${MARDM_STATS}' ] || python scripts/compute_mardm_stats.py \\
            --data-root '${DATA_ROOT}' --out '${MARDM_STATS}'

        echo '=== overfit stage 1: AE on ${SUBSET_FRAC} of train split (${AE_STEPS} steps) ==='
        python -u scripts/train_mardm_ae.py \\
            run_name='${AE_RUN}' \\
            subset_frac=${SUBSET_FRAC} \\
            data.mirror_augment=false \\
            data.num_workers=0 \\
            data.persistent_workers=false \\
            train.micro_batch_size=16 \\
            train.grad_accum=1 \\
            train.max_steps=${AE_STEPS} \\
            train.val_every=$((AE_STEPS / 4)) \\
            train.ckpt_every=${AE_STEPS} \\
            train.log_every=50

        echo '=== overfit stage 2: gen on the same subset (${GEN_STEPS} steps) ==='
        python -u scripts/train_mardm.py \\
            run_name='${GEN_RUN}' \\
            ae_checkpoint='${AE_OUT}/checkpoints/latest.pt' \\
            text_encoder.type=qwen3 \\
            subset_frac=${SUBSET_FRAC} \\
            data.mirror_augment=false \\
            data.num_workers=0 \\
            data.persistent_workers=false \\
            train.micro_batch_size=16 \\
            train.grad_accum=1 \\
            model.cond_drop_prob=0 \\
            model.dropout=0 \\
            train.max_steps=${GEN_STEPS} \\
            train.val_every=$((GEN_STEPS / 4)) \\
            train.ckpt_every=${GEN_STEPS} \\
            train.log_every=50

        echo '=== overfit smoke done ==='
        echo '  Look for: ae/l1 ~ 0 (AE memorized the subset) and gen/loss dropping'
        echo '  substantially from its initial value (SiT FM loss has irreducible'
        echo '  variance, so it plateaus low rather than hitting 0).'
    "
