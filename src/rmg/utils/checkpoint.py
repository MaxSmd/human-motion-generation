"""Atomic checkpoint save/load for the full training state.

The state captures everything needed to resume bit-identical: step counter,
model weights, EMA shadow, optimizer state, scheduler state, AMP scaler, and
RNG states (Python/numpy/torch CPU+CUDA). Saved as a single `.pt` file via
write-temp-then-rename to survive job preemption mid-write.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass
class TrainState:
    step: int
    model: dict          # state_dict
    ema: dict | None     # state_dict from EMA.state_dict()
    optimizer: dict      # state_dict
    scheduler: dict | None
    scaler: dict | None  # GradScaler.state_dict()
    rng: dict            # {"python": ..., "numpy": ..., "torch_cpu": ..., "torch_cuda": ...}
    extras: dict         # arbitrary extras (e.g. wandb_run_id)


def collect_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    """Restore RNG states from a checkpoint payload.

    Defensive against `torch.load(..., map_location='cuda')` moving the saved
    CPU ByteTensors to GPU — `torch.set_rng_state` and `torch.cuda.set_rng_state_all`
    both require the input to be a CPU ByteTensor.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])

    torch_cpu = state["torch_cpu"]
    if isinstance(torch_cpu, torch.Tensor):
        torch_cpu = torch_cpu.to(device="cpu", dtype=torch.uint8)
    torch.set_rng_state(torch_cpu)

    if torch.cuda.is_available() and "torch_cuda" in state:
        cuda_states = state["torch_cuda"]
        if isinstance(cuda_states, list):
            cuda_states = [
                t.to(device="cpu", dtype=torch.uint8) if isinstance(t, torch.Tensor) else t
                for t in cuda_states
            ]
        torch.cuda.set_rng_state_all(cuda_states)


def save_checkpoint(path: str | Path, state: TrainState) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "step": state.step,
        "model": state.model,
        "ema": state.ema,
        "optimizer": state.optimizer,
        "scheduler": state.scheduler,
        "scaler": state.scaler,
        "rng": state.rng,
        "extras": state.extras,
    }
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic on POSIX


def load_checkpoint(path: str | Path, map_location: Any = "cpu") -> TrainState:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    return TrainState(
        step=int(payload["step"]),
        model=payload["model"],
        ema=payload.get("ema"),
        optimizer=payload["optimizer"],
        scheduler=payload.get("scheduler"),
        scaler=payload.get("scaler"),
        rng=payload.get("rng", {}),
        extras=payload.get("extras", {}),
    )


def find_latest_checkpoint(run_dir: str | Path) -> Path | None:
    """Return the most-recent resume target under `run_dir/checkpoints/`.

    `latest.pt` is updated every ckpt_every/10 outer steps, so it is almost
    always more advanced than the highest-step `ckpt-*.pt`. We prefer it
    whenever its step count is >= the highest numbered checkpoint.
    """
    cdir = Path(run_dir) / "checkpoints"
    if not cdir.exists():
        return None
    candidates = list(cdir.glob("ckpt-*.pt"))
    latest = cdir / "latest.pt"

    if latest.exists() and not candidates:
        return latest

    if candidates:
        candidates.sort(key=lambda p: int(p.stem.split("-")[1]))
        newest_numbered = candidates[-1]
        newest_numbered_step = int(newest_numbered.stem.split("-")[1])
        if latest.exists():
            # Peek at the step in latest.pt without loading model weights.
            try:
                blob = torch.load(latest, map_location="cpu", weights_only=False)
                latest_step = int(blob.get("step", -1))
            except Exception:
                latest_step = -1
            return latest if latest_step >= newest_numbered_step else newest_numbered
        return newest_numbered

    return None
