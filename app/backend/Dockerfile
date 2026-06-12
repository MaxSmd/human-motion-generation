# RMG backend (FastAPI). Build context is the REPO ROOT so we can install the
# `rmg` package from src/ (configs live in src/rmg/configs).
#
#   docker build -f app/backend/Dockerfile -t rmg-backend .
#
# Default installs CPU torch wheels (laptop-friendly). For a GPU host, override:
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
# and run with `--gpus all`.
FROM python:3.11-slim

# ffmpeg → MP4 export; openssh-client + rsync → cluster control plane (SSH/sbatch
# + pulling rendered media); build tools for any wheels that need them.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg build-essential git openssh-client rsync \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

# Install deps first for better layer caching: torch, then the package's deps.
RUN pip install --index-url ${TORCH_INDEX_URL} torch

COPY pyproject.toml ./
COPY src ./src
COPY app/backend ./backend

# Editable install of the rmg package + backend-only extras.
RUN pip install -e . && pip install -r backend/requirements.txt

# Where runs/ (checkpoints + samples), the packed dataset, and rendered media
# live inside the container. Override via compose volumes / env.
ENV RMG_RUNS_DIR=/data/runs \
    RMG_DATA_ROOT=/data/humanml3d_packed \
    RMG_MEDIA_DIR=/data/media

EXPOSE 8000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
