from __future__ import annotations

import torch
from torch import Tensor

from .base import Manifold, _broadcast_scalar


class Euclidean(Manifold):
    """R^D with the standard inner product. Every map is trivial; this manifold
    serves as a sanity baseline (Riemannian flow matching collapses to the
    standard Euclidean CFM here)."""

    def __init__(self, dim: int):
        self.ambient_dim = dim
        self.intrinsic_dim = dim

    def project_tangent(self, x: Tensor, u: Tensor) -> Tensor:
        return u

    def project_manifold(self, x: Tensor) -> Tensor:
        return x

    def exp(self, x: Tensor, v: Tensor) -> Tensor:
        return x + v

    def log(self, x: Tensor, y: Tensor) -> Tensor:
        return y - x

    def geodesic(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        tt = _broadcast_scalar(t, x0)
        return (1.0 - tt) * x0 + tt * x1

    def geodesic_velocity(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        # x1 - x0, broadcast to match interpolation shape
        v = x1 - x0
        return v.expand_as(self.geodesic(x0, x1, t))

    def cfm_target_velocity(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        # On R^D, (1/(1-t)) * (x1 - x_t) = x1 - x0, the standard linear-CFM target.
        return (x1 - x0).expand_as(self.geodesic(x0, x1, t))

    def sample_wrapped_gaussian(
        self,
        mu: Tensor,
        sigma: Tensor | float,
        shape: tuple[int, ...] = (),
        generator: torch.Generator | None = None,
    ) -> Tensor:
        full_shape = (*shape, *mu.shape) if shape else mu.shape
        noise = torch.randn(full_shape, dtype=mu.dtype, device=mu.device, generator=generator)
        if isinstance(sigma, Tensor):
            return mu + sigma * noise
        return mu + float(sigma) * noise

    def validate(self, x: Tensor, atol: float = 1e-4) -> Tensor:
        return torch.isfinite(x).all(dim=-1)
