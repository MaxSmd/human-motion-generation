"""Manifold protocol for Riemannian flow matching.

Tensors carry an ambient dimension `D` in the last axis; arbitrary leading
batch / time dims (`*`) broadcast through every method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Manifold(ABC):
    ambient_dim: int
    intrinsic_dim: int

    @abstractmethod
    def project_tangent(self, x: Tensor, u: Tensor) -> Tensor:
        """Project an ambient vector `u` onto T_x M."""

    @abstractmethod
    def project_manifold(self, x: Tensor) -> Tensor:
        """Snap a near-manifold ambient point back onto M (e.g. renormalize)."""

    @abstractmethod
    def exp(self, x: Tensor, v: Tensor) -> Tensor:
        """Exponential map: T_x M -> M."""

    @abstractmethod
    def log(self, x: Tensor, y: Tensor) -> Tensor:
        """Logarithm map: M -> T_x M."""

    @abstractmethod
    def geodesic(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """Geodesic from x0 to x1 evaluated at t in [0, 1]."""

    @abstractmethod
    def geodesic_velocity(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """Velocity of the geodesic at gamma(t), as an ambient vector in T_{gamma(t)} M."""

    def cfm_target_velocity(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """Conditional flow-matching target v_t(x_t | x_1) = (1/(1-t)) Log_{x_t}(x_1).

        Default implementation routes through `log`; subclasses may override for
        a closed form that is more numerically stable. Paper eq. (4) target.
        """
        xt = self.geodesic(x0, x1, t)
        one_minus_t = (1.0 - t).clamp_min(1e-7)
        return self.log(xt, x1) / _broadcast_scalar(one_minus_t, xt)

    def align_base_point(self, x0: Tensor, x1: Tensor) -> Tensor:
        """Return the representative of `x1` best suited for a geodesic from `x0`.

        Default: `x1` unchanged. Manifolds with a discrete quotient symmetry
        (e.g. the quaternion double cover `q ~ -q` on S^3) override this to pick
        the representative on the same side as `x0`, keeping the conditional
        path off the cut locus where the geodesic / its velocity blow up. Used
        by the CFM batch builder before computing `x_t` and the target velocity.
        """
        return x1

    @abstractmethod
    def sample_wrapped_gaussian(
        self,
        mu: Tensor,
        sigma: Tensor | float,
        shape: tuple[int, ...] = (),
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Wrapped Gaussian: tangent-space N(0, sigma^2 I) at `mu`, then Exp."""

    @abstractmethod
    def validate(self, x: Tensor, atol: float = 1e-4) -> Tensor:
        """Boolean tensor (shape == x.shape[:-1]) of points that lie on M."""

    def squared_norm(self, x: Tensor, v: Tensor) -> Tensor:
        """<v, v>_g  at x. For the manifolds we use here, the metric is
        induced from the ambient Euclidean inner product, so this is just
        the squared L2 norm of the tangent-projected vector."""
        v = self.project_tangent(x, v)
        return (v * v).sum(dim=-1)


def _broadcast_scalar(t: Tensor, like: Tensor) -> Tensor:
    """Broadcast a scalar/per-batch t to match the trailing dims of `like`."""
    while t.dim() < like.dim():
        t = t.unsqueeze(-1)
    return t
