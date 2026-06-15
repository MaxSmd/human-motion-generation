# Training/eval container (shared across models) — builds an enroot-importable image.
#
# Two paths to use this on the cluster:
#   1. Enroot (recommended, primary path on this cluster — see BUILD.md). The
#      enroot recipe imports the upstream `pytorch/pytorch` image directly and
#      installs `requirements.txt` interactively, then exports a `.sqsh`.
#   2. Podman with this Dockerfile, then optionally convert to enroot via
#      `enroot import -o ~/rmg.sqsh podman://rmg`. Useful if you want a
#      reproducible build instead of the interactive flow.
FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/mnt/home/${USER}/hf-cache

# System packages: git for submodules, ssh client for any cluster-internal
# tooling, build-essential for any pip wheels that need to compile from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ssh wget curl build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Sanity at build time
RUN python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"

CMD ["bash", "-l"]
