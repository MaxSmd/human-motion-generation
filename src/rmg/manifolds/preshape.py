"""Kendall pre-shape space S^J_m (paper §3.1, App. A).

A pre-shape of J landmarks in R^m is a J×m matrix that is:
    1. **Centered** — column means are zero.
    2. **Unit Frobenius** — ||X||_F = 1.

Intrinsically the pre-shape space is a hypersphere of dimension
`J*m - m - 1` (m centering constraints, 1 norm constraint), so geodesics
collapse to slerp on the centered subspace. Used by the T+P / T+R+P
ablations (Figure 3a in the paper).

We store points as flat (..., J*m) tensors so they fit the existing
ProductManifold layout without surprises; the centering constraint is
re-applied internally as needed.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import Manifold, _broadcast_scalar
from .sphere import _safe_arccos, _safe_div_sin

_EPS = 1e-7


class PreShape(Manifold):
    """Kendall pre-shape space S^J_m, ambient = R^{J·m}, intrinsic = J·m - m - 1."""

    def __init__(self, num_landmarks: int, dim: int = 3):
        self.num_landmarks = num_landmarks
        self.dim = dim
        self.ambient_dim = num_landmarks * dim
        self.intrinsic_dim = num_landmarks * dim - dim - 1

    # ----------------------------------------------- shape juggling utilities

    def _as_matrix(self, x: Tensor) -> Tensor:
        return x.reshape(*x.shape[:-1], self.num_landmarks, self.dim)

    def _flatten(self, X: Tensor) -> Tensor:
        return X.reshape(*X.shape[:-2], self.num_landmarks * self.dim)

    def _center(self, X: Tensor) -> Tensor:
        """Subtract the column mean (over landmarks) from a (..., J, m) matrix."""
        return X - X.mean(dim=-2, keepdim=True)

    # ----------------------------------------------- core ops

    def project_tangent(self, x: Tensor, u: Tensor) -> Tensor:
        # 1) center the ambient vector (kill any direction that would shift the centroid)
        # 2) project off the radial direction at x (sphere tangent constraint)
        u_mat = self._as_matrix(u)
        u_centered = self._center(u_mat)
        u_flat = self._flatten(u_centered)
        inner = (x * u_flat).sum(dim=-1, keepdim=True)
        return u_flat - inner * x

    def project_manifold(self, x: Tensor) -> Tensor:
        X = self._center(self._as_matrix(x))
        flat = self._flatten(X)
        norm = torch.linalg.vector_norm(flat, dim=-1, keepdim=True).clamp_min(_EPS)
        return flat / norm

    def exp(self, x: Tensor, v: Tensor) -> Tensor:
        v = self.project_tangent(x, v)
        norm_v = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
        small = norm_v < 1e-7
        sinc = torch.where(small, torch.ones_like(norm_v), torch.sin(norm_v) / norm_v.clamp_min(_EPS))
        out = torch.cos(norm_v) * x + sinc * v
        return self.project_manifold(out)

    def log(self, x: Tensor, y: Tensor) -> Tensor:
        inner = (x * y).sum(dim=-1, keepdim=True).clamp(-1.0 + _EPS, 1.0 - _EPS)
        theta = _safe_arccos(inner)
        tangent_part = y - inner * x
        return _safe_div_sin(theta * tangent_part, theta)

    def _theta(self, x0: Tensor, x1: Tensor) -> Tensor:
        inner = (x0 * x1).sum(dim=-1, keepdim=True).clamp(-1.0 + _EPS, 1.0 - _EPS)
        return _safe_arccos(inner)

    def geodesic(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        tt = _broadcast_scalar(t, x0)
        theta = self._theta(x0, x1)
        a = _safe_div_sin(torch.sin((1.0 - tt) * theta), theta)
        b = _safe_div_sin(torch.sin(tt * theta), theta)
        return self.project_manifold(a * x0 + b * x1)

    def geodesic_velocity(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        # Same correction as Sphere.geodesic_velocity — paper App. A's `sin`
        # form is a typo; cos is the correct derivative.
        tt = _broadcast_scalar(t, x0)
        theta = self._theta(x0, x1)
        coef0 = _safe_div_sin(-torch.cos((1.0 - tt) * theta) * theta, theta)
        coef1 = _safe_div_sin(torch.cos(tt * theta) * theta, theta)
        return coef0 * x0 + coef1 * x1

    def cfm_target_velocity(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        return self.geodesic_velocity(x0, x1, t)

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
            noise = sigma * noise
        else:
            noise = float(sigma) * noise
        if shape:
            mu_b = mu.expand(full_shape)
        else:
            mu_b = mu
        v = self.project_tangent(mu_b, noise)
        return self.exp(mu_b, v)

    def validate(self, x: Tensor, atol: float = 1e-4) -> Tensor:
        X = self._as_matrix(x)
        col_means = X.mean(dim=-2)
        centered_ok = col_means.abs().max(dim=-1).values < atol
        norm = torch.linalg.vector_norm(x, dim=-1)
        norm_ok = (norm - 1.0).abs() < atol
        return centered_ok & norm_ok


# ---------------------------------------------------------------------------
# Helpers for building pre-shape representations from joint positions
# ---------------------------------------------------------------------------


def joints_to_preshape(joints: Tensor) -> Tensor:
    """Convert (..., J, m) joint positions into a pre-shape (..., J*m)."""
    centered = joints - joints.mean(dim=-2, keepdim=True)
    flat = centered.reshape(*joints.shape[:-2], joints.shape[-2] * joints.shape[-1])
    norm = torch.linalg.vector_norm(flat, dim=-1, keepdim=True).clamp_min(_EPS)
    return flat / norm
