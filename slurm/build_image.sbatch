#!/bin/bash
#SBATCH --job-name=rmg-build-image
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/build-%j.out
#SBATCH --error=slurm/logs/build-%j.err
#
# Build the RMG enroot image from `pytorch/pytorch:2.9.0-cuda13.0-cudnn9-devel`
# plus our deps from containers/requirements.txt. Runs on a compute node so it
# bypasses the head's ulimits (200 MB max file, 1200 s CPU, 50 MB RSS).
#
# The build is in two phases to keep the base image's CUDA-built torch intact:
#   Phase A: install requirements.txt
#            torch is NOT listed there → pip leaves the base image's
#            torch 2.9.0+cu130 alone, transformers/etc. install on top.
#   Phase B: install `human_body_prior` (AMASS preprocess only) with --no-deps,
#            so its setup.py doesn't get to demand a different torch.
#
# Re-run safe: enroot --force overwrites prior state.

set -euo pipefail
mkdir -p slurm/logs

REPO="${HOME}/riemann-motion-generation"
IMAGE_OUT="${HOME}/rmg.sqsh"

echo "[build] pulling upstream image ..."
enroot import -o /tmp/base.sqsh 'docker://pytorch/pytorch:2.9.0-cuda13.0-cudnn9-devel'

echo "[build] creating container ..."
enroot create --force --name rmg /tmp/base.sqsh

echo "[build] installing deps inside container ..."
enroot start --root --rw \
    --mount /mnt:mnt --mount /tmp:tmp \
    rmg \
    bash -lc "
        set -euo pipefail
        apt-get update && apt-get install -y --no-install-recommends git build-essential
        # Phase A: main requirements (no torch, so base torch survives)
        pip install --no-cache-dir -r ${REPO}/containers/requirements.txt
        # Phase B: AMASS preprocess pkg installed without its deps so it can't
        # downgrade torch / pin anything else.
        pip install --no-cache-dir --no-deps \
            git+https://github.com/nghorbani/human_body_prior.git
        # spaCy English model (eval-time text preprocessing for Guo evaluator)
        python -m spacy download en_core_web_sm
        # Validate. Use getattr fallback because some packages (roma, smplx)
        # don't expose __version__ — and we don't want a missing attribute to
        # kill the build right before enroot export.
        python - <<'PY'
def v(mod):
    return getattr(mod, '__version__', '?')

import torch, transformers, roma, spacy, smplx, human_body_prior
print('torch       ', v(torch), 'cuda?', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu         ', torch.cuda.get_device_name(0))
assert torch.cuda.is_available(), 'CUDA not available — base image torch was downgraded'
print('transformers', v(transformers))
print('roma        ', v(roma))
print('spacy       ', v(spacy))
print('smplx       ', v(smplx))
print('hbp         ', v(human_body_prior), 'at', human_body_prior.__file__)
PY
    "

echo "[build] exporting .sqsh ..."
enroot export --force --output "${IMAGE_OUT}" rmg

# Free the writable container; image is saved.
enroot remove --force rmg

ls -lh "${IMAGE_OUT}"
echo "[build] DONE"
