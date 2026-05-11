"""Hypersphere S^d embedded in R^{d+1}.

Used for unit quaternions on S^3 (paper §3.1 / App. A). Geodesics are slerp,
with numerical guards near antipodal points and θ → 0.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import Manifold, _broadcast_scalar

_EPS = 1e-7


def _safe_arccos(x: Tensor) -> Tensor:
    return torch.arccos(x.clamp(-1.0 + _EPS, 1.0 - _EPS))


def _safe_div_sin(numer: Tensor, theta: Tensor) -> Tensor:
    """numer / sin(theta), Taylor-expanded near θ = 0 to numer * (1 + θ^2/6)."""
    sin_theta = torch.sin(theta)
    small = theta.abs() < 1e-4
    taylor = numer * (1.0 + theta * theta / 6.0)
    return torch.where(small, taylor, numer / sin_theta.clamp_min(_EPS))


class Sphere(Manifold):
    """Unit sphere S^d ⊂ R^{d+1}. `dim` is the *intrinsic* dimension d
    (so ambient_dim = d + 1)."""

    def __init__(self, dim: int):
        self.intrinsic_dim = dim
        self.ambient_dim = dim + 1

    def project_tangent(self, x: Tensor, u: Tensor) -> Tensor:
        # u - <x, u> x
        inner = (x * u).sum(dim=-1, keepdim=True)
        return u - inner * x

    def project_manifold(self, x: Tensor) -> Tensor:
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(_EPS)
        return x / norm

    def exp(self, x: Tensor, v: Tensor) -> Tensor:
        # v assumed in T_x S^d (or projected here defensively).
        v = self.project_tangent(x, v)
        norm_v = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
        small = norm_v < 1e-7
        # cos(|v|) x + sin(|v|) v / |v|
        cos_n = torch.cos(norm_v)
        sinc = torch.where(small, torch.ones_like(norm_v), torch.sin(norm_v) / norm_v.clamp_min(_EPS))
        out = cos_n * x + sinc * v
        return self.project_manifold(out)

    def log(self, x: Tensor, y: Tensor) -> Tensor:
        # theta * (y - cos(theta) x) / sin(theta)
        inner = (x * y).sum(dim=-1, keepdim=True).clamp(-1.0 + _EPS, 1.0 - _EPS)
        theta = _safe_arccos(inner)
        # tangent part of y at x
        tangent_part = y - inner * x
        return _safe_div_sin(theta * tangent_part, theta)

    # Geodesic & velocity from paper App. A (slerp form). Both are
    # mathematically equivalent to exp(x0, t * log(x0, x1)) but stay numerically
    # symmetric near t = 0 and t = 1.

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
        # Differentiating γ(t) = sin((1−t)θ)/sin(θ) x_0 + sin(tθ)/sin(θ) x_1 wrt t:
        #   γ̇(t) = θ/sin(θ) · ( −cos((1−t)θ) x_0 + cos(tθ) x_1 )
        # NB: paper App. A writes this with sin instead of cos, which is a typo
        # (their version isn't even tangent at t=0). We use the correct cos form,
        # which is what gives a unit-speed-up-to-θ geodesic and matches
        # log_{γ(t)}(x_1)/(1−t) exactly (verified algebraically).
        tt = _broadcast_scalar(t, x0)
        theta = self._theta(x0, x1)
        coef0 = _safe_div_sin(-torch.cos((1.0 - tt) * theta) * theta, theta)
        coef1 = _safe_div_sin(torch.cos(tt * theta) * theta, theta)
        return coef0 * x0 + coef1 * x1

    def cfm_target_velocity(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        # (1/(1−t)) * Log_{γ(t)}(x_1) reduces exactly to γ̇(t) for a
        # constant-speed geodesic. Avoids the geodesic→log roundtrip.
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
        # Project ambient noise onto T_mu M, then exp.
        if shape:
            mu_b = mu.expand(full_shape)
        else:
            mu_b = mu
        v = self.project_tangent(mu_b, noise)
        return self.exp(mu_b, v)

    def validate(self, x: Tensor, atol: float = 1e-4) -> Tensor:
        norm = torch.linalg.vector_norm(x, dim=-1)
        return (norm - 1.0).abs() < atol


# --- Quaternion helpers ----------------------------------------------------


def quat_to_upper_hemisphere(q: Tensor) -> Tensor:
    """Flip sign so that q_0 ≥ 0 (paper App. A: avoids the q ↔ -q ambiguity)."""
    sign = torch.where(q[..., :1] < 0, -torch.ones_like(q[..., :1]), torch.ones_like(q[..., :1]))
    return q * sign


def quat_continuity(q: Tensor, dim: int = -2) -> Tensor:
    """Propagate sign continuity across a sequence of quaternions (along `dim`).

    For adjacent frames q_t, q_{t+1}: if <q_t, q_{t+1}> < 0, flip q_{t+1}'s sign.
    Done as a python loop over the temporal axis (small, e.g. 196 frames).
    """
    if dim < 0:
        dim = q.dim() + dim
    if q.shape[dim] < 2:
        return q
    out = q.clone()
    # iterate from index 1 along `dim`
    prev = out.index_select(dim, torch.tensor([0], device=q.device)).squeeze(dim)
    pieces = [prev]
    for i in range(1, q.shape[dim]):
        cur = out.index_select(dim, torch.tensor([i], device=q.device)).squeeze(dim)
        dot = (prev * cur).sum(dim=-1, keepdim=True)
        cur = torch.where(dot < 0, -cur, cur)
        pieces.append(cur)
        prev = cur
    return torch.stack(pieces, dim=dim)
