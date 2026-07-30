"""Product of Riemannian manifolds.

Points and tangents are stored as flat tensors of shape (..., D_total) where
D_total = sum(factor.ambient_dim). All manifold operations dispatch factor-wise.
This matches the network input layout (flat 91-d for T+R) used in the paper.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import Manifold


class ProductManifold(Manifold):
    def __init__(self, factors: list[Manifold]):
        if not factors:
            raise ValueError("ProductManifold requires at least one factor")
        self.factors = factors
        self.ambient_dim = sum(f.ambient_dim for f in factors)
        self.intrinsic_dim = sum(f.intrinsic_dim for f in factors)
        self._slices: list[slice] = []
        offset = 0
        for f in factors:
            self._slices.append(slice(offset, offset + f.ambient_dim))
            offset += f.ambient_dim

    def _split(self, x: Tensor) -> list[Tensor]:
        return [x[..., s] for s in self._slices]

    def _join(self, parts: list[Tensor]) -> Tensor:
        return torch.cat(parts, dim=-1)

    def _apply(self, fn_name: str, *tensors: Tensor) -> Tensor:
        parts_per_arg = [self._split(t) for t in tensors]
        out_parts = []
        for i, f in enumerate(self.factors):
            args = [parts[i] for parts in parts_per_arg]
            out_parts.append(getattr(f, fn_name)(*args))
        return self._join(out_parts)

    def project_tangent(self, x: Tensor, u: Tensor) -> Tensor:
        return self._apply("project_tangent", x, u)

    def project_manifold(self, x: Tensor) -> Tensor:
        return self._apply("project_manifold", x)

    def exp(self, x: Tensor, v: Tensor) -> Tensor:
        return self._apply("exp", x, v)

    def log(self, x: Tensor, y: Tensor) -> Tensor:
        return self._apply("log", x, y)

    def geodesic(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        x0_parts = self._split(x0)
        x1_parts = self._split(x1)
        out = [f.geodesic(x0_parts[i], x1_parts[i], t) for i, f in enumerate(self.factors)]
        return self._join(out)

    def geodesic_velocity(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        x0_parts = self._split(x0)
        x1_parts = self._split(x1)
        out = [f.geodesic_velocity(x0_parts[i], x1_parts[i], t) for i, f in enumerate(self.factors)]
        return self._join(out)

    def cfm_target_velocity(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        x0_parts = self._split(x0)
        x1_parts = self._split(x1)
        out = [f.cfm_target_velocity(x0_parts[i], x1_parts[i], t) for i, f in enumerate(self.factors)]
        return self._join(out)

    def sample_wrapped_gaussian(
        self,
        mu: Tensor,
        sigma: Tensor | float,
        shape: tuple[int, ...] = (),
        generator: torch.Generator | None = None,
    ) -> Tensor:
        # Allow per-factor sigma (list/Tensor with len(factors)) or scalar.
        if isinstance(sigma, (list, tuple)):
            assert len(sigma) == len(self.factors)
            sigmas = list(sigma)
        else:
            sigmas = [sigma] * len(self.factors)
        mu_parts = self._split(mu)
        out = [
            f.sample_wrapped_gaussian(mu_parts[i], sigmas[i], shape=shape, generator=generator)
            for i, f in enumerate(self.factors)
        ]
        return self._join(out)

    def align_base_point(self, x0: Tensor, x1: Tensor) -> Tensor:
        return self._apply("align_base_point", x0, x1)

    def validate(self, x: Tensor, atol: float = 1e-4) -> Tensor:
        parts = self._split(x)
        ok = self.factors[0].validate(parts[0], atol=atol)
        for i in range(1, len(self.factors)):
            ok = ok & self.factors[i].validate(parts[i], atol=atol)
        return ok
