"""Masking + scheduling helpers for MARDM's masked-autoregressive training.

Vendored/adapted from MARDM (neu-vi/MARDM, ``utils/train_utils.py``; see the
upstream LICENSE). These implement the cosine masking schedule from MoMask and
the BERT-style sub-masking used in the generation branch.
"""

from __future__ import annotations

import math
from functools import wraps

import torch
from torch import Tensor


def lengths_to_mask(lengths: Tensor, max_len: int) -> Tensor:
    """(B,) valid lengths -> (B, max_len) bool mask, True = valid frame."""
    return torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)


def get_mask_subset_prob(mask: Tensor, prob: float) -> Tensor:
    """Keep each True entry of `mask` independently with probability `prob`."""
    subset = (torch.rand_like(mask, dtype=torch.float) < prob) & mask
    return subset


def uniform(shape: tuple[int, ...], device: torch.device | None = None) -> Tensor:
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def cosine_schedule(t: Tensor) -> Tensor:
    """Mask-ratio schedule: cos(t · π/2), decreasing from 1 (t=0) to 0 (t=1)."""
    return torch.cos(t * math.pi * 0.5)


def eval_decorator(fn):
    """Run a method with the module in eval mode, restoring train state after."""

    @wraps(fn)
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out

    return inner
