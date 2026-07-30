#!/bin/bash
#SBATCH --job-name=rmg-forensics
#SBATCH --partition=12g
#SBATCH --qos=students_normal
# The 12g partition mixes GPU generations and `ccc` in each node's GRES is its
# compute capability x10: ulm/jena are TITANX (5.2) and dortmund is TITANXp
# (6.1), none of which this PyTorch build ships kernels for — a job landing
# there dies with "no kernel image is available for execution on the device".
# The RTX2080Ti and RTX5000 nodes publish ccc:75 and work. Requesting ccc:75
# (a no_consume GRES, so it filters rather than reserves) is what makes this
# reproducible instead of luck-of-the-scheduler. This cluster rejects --exclude.
#SBATCH --gres=gpu:1,ccc:75
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/rmg-forensics-%j.out
#SBATCH --error=slurm/logs/rmg-forensics-%j.err
#
# Generation forensics: generated vs real motion on quantities that do not pass
# through the Guo evaluator (quaternion norm, angular velocity, translation
# velocity, root displacement). Same wiring as rmg/eval.sbatch, different module.
#
#   CKPT=/path/to/ckpt.pt MODEL_PRESET=dit_mid PRESET=rmg_mid \
#   NUM_SAMPLE_STEPS=800 GUIDANCE_SCALES='[6.5]' MAX_CLIPS=512 \
#   RUN_NAME=forensics-mid300k-w65-ode800 sbatch slurm/rmg/forensics.sbatch

set -euo pipefail
mkdir -p slurm/logs

CKPT="${CKPT:-${1:?set CKPT env var or pass as first positional arg}}"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(dirname "${SCRIPT_DIR}")}"
fi
IMAGE="${IMAGE:-${HOME}/rmg.sqsh}"
HF_CACHE="${HF_CACHE:-${HOME}/hf-cache}"
RUNS_ROOT="${RUNS_ROOT:-${REPO}/runs/rmg/eval}"
MODEL_PRESET="${MODEL_PRESET:-dit_mid}"
PRESET="${PRESET:-rmg_mid}"
TEXT_ENCODER="${TEXT_ENCODER:-qwen3}"
RUN_NAME="${RUN_NAME:-forensics-$(basename "${CKPT}" .pt)-${SLURM_JOB_ID:-$(date +%H%M%S)}}"

EVAL_SPLIT="${EVAL_SPLIT:-test}"
MAX_CLIPS="${MAX_CLIPS:-512}"
GUIDANCE_SCALES="${GUIDANCE_SCALES:-[6.5]}"
SUBSET_FRACTION="${SUBSET_FRACTION:-1.0}"
SUBSET_SEED="${SUBSET_SEED:-0}"
NUM_SAMPLE_STEPS="${NUM_SAMPLE_STEPS:-800}"
USE_EMA="${USE_EMA:-true}"
OVERRIDES="${OVERRIDES:-}"

enroot start \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --env "HF_HOME=${HF_CACHE}" \
    --env "MGEN_RUNS_DIR=${RUNS_ROOT}" \
    --env "PYTHONPATH=${REPO}/src" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        cd ${REPO}
        pip install -e . --no-deps --quiet
        python -u -m rmg.scripts.forensics \\
            model=${MODEL_PRESET} \\
            train=${PRESET} \\
            text_encoder.type=${TEXT_ENCODER} \\
            data.subset_fraction=${SUBSET_FRACTION} \\
            data.subset_seed=${SUBSET_SEED} \\
            +eval.checkpoint=${CKPT} \\
            +eval.use_ema=${USE_EMA} \\
            +eval.split=${EVAL_SPLIT} \\
            +eval.max_clips=${MAX_CLIPS} \\
            +eval.num_sample_steps=${NUM_SAMPLE_STEPS} \\
            +eval.guidance_scales='${GUIDANCE_SCALES}' \\
            run_name='${RUN_NAME}' \\
            ${OVERRIDES}
    "
