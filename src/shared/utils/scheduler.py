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


def cosine_rewarm(
    step_abs: int, origin: int, max_steps: int, warmup_steps: int, peak_ratio: float
) -> float:
    """LR multiplier for a *continued-training* (re-warm) extension.

    A pure function of the ABSOLUTE step (not steps-since-resume): it warms
    0 → peak_ratio linearly over [origin, origin+warmup_steps], then cosine-
    anneals peak_ratio → 0 over [origin+warmup_steps, max_steps]. `origin` is the
    step at which the extension began (the old max_steps, e.g. 300k).

    Being absolute-step-based is what makes it resubmit-safe: every 24h walltime
    block rebuilds the identical curve and reads the LR for the current step, so
    the warm-up happens exactly ONCE at `origin`, never again per block.

    Used when resuming a model whose original schedule already decayed the LR to
    zero: continuing from 0 learns nothing, so we re-inject a modest LR
    (peak_ratio · base_lr, e.g. 0.3·1e-4 = 3e-5) and smoothly redecay — the
    standard continued-pretraining recipe. `peak_ratio < 1` keeps the re-warm
    gentle so it doesn't kick the converged weights out of their basin.
    """
    e = step_abs - origin
    if warmup_steps > 0 and e < warmup_steps:
        return peak_ratio * (max(0, e) / max(1, warmup_steps))
    progress = (step_abs - (origin + warmup_steps)) / max(1, max_steps - (origin + warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_ratio * cos_factor


def build_rewarm_scheduler(
    optimizer: torch.optim.Optimizer,
    origin: int,
    max_steps: int,
    warmup_steps: int,
    peak_ratio: float,
    base_lr: float,
    current_step: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Absolute-step LambdaLR for a re-warm extension (see `cosine_rewarm`).

    Built AFTER `opt.load_state_dict`, and the old scheduler state must NOT be
    restored — the lambda is a pure function of absolute step so no state needs
    carrying across resubmits. `base_lr` is pinned as each group's `initial_lr`
    so the peak is `peak_ratio · base_lr` regardless of the (near-zero) LR the
    loaded optimizer state carries. `current_step` aligns the scheduler's
    internal counter to the absolute training step on resume.
    """
    for g in optimizer.param_groups:
        g["initial_lr"] = base_lr

    def lr_lambda(step_abs: int) -> float:
        return cosine_rewarm(step_abs, origin, max_steps, warmup_steps, peak_ratio)

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_lambda, last_epoch=current_step - 1
    )
