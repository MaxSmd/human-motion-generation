"""Riemannian wrapped Gaussian prior centered at a reference point (paper §3.2).

For RMG the reference is the rest pose: T = 0 ∈ R^3, q = [1,0,0,0] ∈ S^3 for
each joint. Tangent-space Gaussian noise is wrapped onto the manifold via Exp.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..manifolds import Euclidean, Manifold, ProductManifold, Sphere


class WrappedGaussianPrior:
    """Per-factor wrapped Gaussian prior on a (Product)Manifold.

    Args:
        manifold: the (product) manifold to sample on.
        mu: ambient reference point of shape (D_total,). For RMG, the rest pose.
        sigma: scalar or list per factor (passed through to the manifold).
    """

    def __init__(
        self,
        manifold: Manifold,
        mu: Tensor,
        sigma: float | list[float] | Tensor = 1.0,
    ) -> None:
        if mu.shape[-1] != manifold.ambient_dim:
            raise ValueError(
                f"mu last dim {mu.shape[-1]} != manifold.ambient_dim {manifold.ambient_dim}"
            )
        self.manifold = manifold
        self.register_buffer_name = None  # purely informational; kept stateless
        self.mu = mu
        self.sigma = sigma

    def sample(
        self,
        shape: tuple[int, ...],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Draw `shape`-many samples; output has shape (*shape, D_total)."""
        mu = self.mu.to(device=device, dtype=dtype) if (device or dtype) else self.mu
        return self.manifold.sample_wrapped_gaussian(
            mu, sigma=self.sigma, shape=shape, generator=generator
        )

    def to(self, *args, **kwargs) -> "WrappedGaussianPrior":
        self.mu = self.mu.to(*args, **kwargs)
        return self


def rest_pose_mu(num_joints: int, dtype: torch.dtype = torch.float32) -> Tensor:
    """Reference point for RMG: T = 0, q = [1,0,0,0] for each of the J joints
    (paper §3.2). Returns a (3 + 4*num_joints,) tensor."""
    rest_T = torch.zeros(3, dtype=dtype)
    rest_q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype)
    return torch.cat([rest_T] + [rest_q] * num_joints, dim=-1)


def rmg_manifold(num_joints: int) -> ProductManifold:
    """The paper's chosen manifold M_RMG = R^3 × (S^3)^J (eq. 3)."""
    return ProductManifold([Euclidean(3)] + [Sphere(3) for _ in range(num_joints)])
