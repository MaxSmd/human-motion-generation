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
    (so ambient_dim = d + 1).

    `antipodal_quotient` opts into the double-cover identification x ~ -x. Set
    it for unit quaternions on S^3, where q and -q are the *same* rotation: it
    lets `align_base_point` pick the target representative within 90° of the
    start, so geodesics never approach the antipodal cut locus (θ → π) where the
    velocity factor θ/sin(θ) explodes to Inf/NaN. Leave it False for a genuine
    sphere where antipodal points are distinct.
    """

    def __init__(self, dim: int, antipodal_quotient: bool = False):
        self.intrinsic_dim = dim
        self.ambient_dim = dim + 1
        self.antipodal_quotient = antipodal_quotient

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

    def align_base_point(self, x0: Tensor, x1: Tensor) -> Tensor:
        # Quaternion double cover: q and -q are the same rotation, so flip x1 to
        # whichever of {x1, -x1} lies in x0's hemisphere (<x0, x1> >= 0). This
        # caps the geodesic arc at θ <= π/2, keeping it off the antipodal cut
        # locus. No-op for a genuine sphere (antipodal_quotient=False).
        if not self.antipodal_quotient:
            return x1
        inner = (x0 * x1).sum(dim=-1, keepdim=True)
        flip = torch.where(inner < 0, -1.0, 1.0)
        return x1 * flip

    def validate(self, x: Tensor, atol: float = 1e-4) -> Tensor:
        norm = torch.linalg.vector_norm(x, dim=-1)
        return (norm - 1.0).abs() < atol


# --- Quaternion helpers ----------------------------------------------------
# These model-agnostic sign-convention helpers were lifted to
# `shared.geometry.quaternions` so mardm can reuse them. Re-exported here so
# existing `rmg.manifolds.sphere` imports keep working unchanged.
from shared.geometry.quaternions import (  # noqa: E402, F401
    quat_continuity,
    quat_to_upper_hemisphere,
)
