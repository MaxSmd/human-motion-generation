"""Cosine learning-rate schedule with linear warmup (paper §4.1: 8% warmup ratio)."""

from __future__ import annotations

import math

import torch


def cosine_with_warmup(step: int, total_steps: int, warmup_steps: int, min_lr_ratio: float = 0.0) -> float:
    """Returns the LR multiplier in [min_lr_ratio, 1] for the given step.

    Linear from 0→1 over warmup, then cosine 1→min_lr_ratio over the remainder.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cos_factor


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.08,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        return cosine_with_warmup(step, total_steps, warmup_steps, min_lr_ratio)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
