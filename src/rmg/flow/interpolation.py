"""Geodesic interpolation and CFM target velocity (paper §3.3, eqs. 3-4).

Thin wrappers around the manifold methods so the trainer reads cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..manifolds import Manifold


@dataclass
class FlowMatchingBatch:
    """Per-step inputs to the network and targets for the loss."""

    x_t: Tensor  # (B, T, D) interpolation state on the manifold
    t: Tensor  # (B,) or (B, T) time in [0, 1]
    target: Tensor  # (B, T, D) tangent velocity at x_t — the CFM target


def build_cfm_batch(
    manifold: Manifold,
    x0: Tensor,
    x1: Tensor,
    t: Tensor,
) -> FlowMatchingBatch:
    """Compute (x_t, target_v) for Riemannian conditional flow matching.

    Inputs may have any leading shape; the manifold operates on the last dim.
    `t` is broadcast across the leading dims.
    """
    x_t = manifold.geodesic(x0, x1, t)
    target = manifold.cfm_target_velocity(x0, x1, t)
    return FlowMatchingBatch(x_t=x_t, t=t, target=target)


def sample_t(
    batch_size: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-3,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Uniform t in [eps, 1-eps]. Bounded away from endpoints to keep
    `Log_{x_t}(x_1)/(1-t)` and the geodesic itself numerically clean."""
    u = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
    return eps + (1.0 - 2.0 * eps) * u
