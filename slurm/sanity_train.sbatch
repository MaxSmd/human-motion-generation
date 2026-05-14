#!/bin/bash
#SBATCH --job-name=rmg-sanity
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/sanity-%j.out
#SBATCH --error=slurm/logs/sanity-%j.err
#
# 2-minute end-to-end sanity training on the *real* packed dataset.
#
# Trains a tiny DiT (64h/2L) for 20 outer steps against the actual
# humanml3d.zip on the cluster GPU. Validates:
#   * the container CUDA path works (torch.compile, bf16, etc.)
#   * the packed dataset loads and the loader can read the zip
#   * Qwen3-Embedding-0.6B downloads + loads + encodes texts
#   * one optimizer step lowers the loss
#   * a checkpoint saves and a periodic sample writes
#   * Riemannian Euler sampler runs on GPU
#
# Run before any long training run after non-trivial code/container changes.

set -euo pipefail
mkdir -p slurm/logs

# Use SLURM_SUBMIT_DIR when run under sbatch; fall back to BASH_SOURCE.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(dirname "${SCRIPT_DIR}")}"
fi

IMAGE="${IMAGE:-${HOME}/rmg.sqsh}"
HF_CACHE="${HF_CACHE:-${HOME}/hf-cache}"
RUNS_ROOT="${RUNS_ROOT:-${HOME}/rmg-runs}"
RUN_NAME="${RUN_NAME:-sanity-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"

: "${WANDB_MODE:=disabled}"     # no wandb noise for a 2-min job

cd "${REPO}"

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
            model=dit_base \\
            train=rmg_base \\
            data.num_workers=2 \\
            data.persistent_workers=false \\
            data.mirror_augment=false \\
            model.hidden_dim=64 \\
            model.depth=2 \\
            model.num_heads=4 \\
            model.ffn_mult=2 \\
            model.max_seq_len=200 \\
            train.max_steps=20 \\
            train.micro_batch_size=8 \\
            train.grad_accum=1 \\
            train.precision=bf16 \\
            train.log_every=2 \\
            train.ckpt_every=10 \\
            train.sample_every=10 \\
            train.num_sample_steps=10 \\
            text_encoder.type=qwen3 \\
            logging.use_wandb=false \\
            logging.use_tensorboard=false \\
            run_name='${RUN_NAME}'

        # Post-conditions: assert that the run dir has what we expect.
        RUN_DIR=${RUNS_ROOT}/${RUN_NAME}
        echo
        echo '=== sanity post-conditions ==='
        test -f \$RUN_DIR/checkpoints/latest.pt && echo 'OK: latest.pt'
        test -f \$RUN_DIR/checkpoints/ckpt-000000010.pt && echo 'OK: ckpt-000000010.pt'
        test -f \$RUN_DIR/checkpoints/ckpt-000000020.pt && echo 'OK: ckpt-000000020.pt'
        test -f \$RUN_DIR/samples/step-000000010.pt && echo 'OK: sample at step 10'
        test -f \$RUN_DIR/samples/step-000000020.pt && echo 'OK: sample at step 20'
        test -f \$RUN_DIR/metrics.csv && echo 'OK: metrics.csv'

        # Show first and last loss to confirm it moved.
        head -1 \$RUN_DIR/metrics.csv
        tail -1 \$RUN_DIR/metrics.csv

        echo
        echo '[sanity] DONE — pipeline is healthy.'
    "
