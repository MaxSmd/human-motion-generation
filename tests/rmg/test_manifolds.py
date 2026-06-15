"""Manifold correctness tests.

Strategy: small random tensors, double precision, generous-but-meaningful tolerances.
Every manifold must satisfy:
  * exp ∘ log = id  (when defined; for S^d, away from antipodes)
  * geodesic at t=0 is x0, at t=1 is x1
  * project_tangent is idempotent
  * sample_wrapped_gaussian → validate
  * cfm_target_velocity = (1/(1-t)) * log(x_t, x_1)
"""

from __future__ import annotations

import pytest
import torch

from rmg.manifolds import (
    Euclidean,
    PreShape,
    ProductManifold,
    Sphere,
    joints_to_preshape,
    quat_continuity,
    quat_to_upper_hemisphere,
)

torch.manual_seed(0)


def _rand_sphere(d: int, *batch: int, dtype=torch.float64) -> torch.Tensor:
    x = torch.randn(*batch, d + 1, dtype=dtype)
    return x / x.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Euclidean
# ---------------------------------------------------------------------------


def test_euclidean_exp_log_identity() -> None:
    M = Euclidean(5)
    x = torch.randn(4, 5, dtype=torch.float64)
    y = torch.randn(4, 5, dtype=torch.float64)
    assert torch.allclose(M.exp(x, M.log(x, y)), y)


def test_euclidean_geodesic_endpoints() -> None:
    M = Euclidean(3)
    x0 = torch.randn(2, 3, dtype=torch.float64)
    x1 = torch.randn(2, 3, dtype=torch.float64)
    t0 = torch.zeros(2, dtype=torch.float64)
    t1 = torch.ones(2, dtype=torch.float64)
    assert torch.allclose(M.geodesic(x0, x1, t0), x0)
    assert torch.allclose(M.geodesic(x0, x1, t1), x1)


def test_euclidean_cfm_target_is_x1_minus_x0() -> None:
    M = Euclidean(3)
    x0 = torch.randn(8, 3, dtype=torch.float64)
    x1 = torch.randn(8, 3, dtype=torch.float64)
    t = torch.rand(8, dtype=torch.float64) * 0.9  # avoid (1-t)→0
    target = M.cfm_target_velocity(x0, x1, t)
    assert torch.allclose(target, x1 - x0)


# ---------------------------------------------------------------------------
# Sphere S^d
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [3, 7])
def test_sphere_exp_log_identity(d: int) -> None:
    M = Sphere(d)
    x = _rand_sphere(d, 16)
    y = _rand_sphere(d, 16)
    # avoid antipodal pairs
    inner = (x * y).sum(-1)
    mask = inner > -0.99
    x, y = x[mask], y[mask]
    v = M.log(x, y)
    # log returns a tangent vector at x
    inner_xv = (x * v).sum(-1)
    assert inner_xv.abs().max() < 1e-8
    y_back = M.exp(x, v)
    assert torch.allclose(y_back, y, atol=1e-9)


def test_sphere_log_exp_identity_for_small_tangents() -> None:
    M = Sphere(3)
    x = _rand_sphere(3, 32)
    # pick random *small* tangent vectors at x
    raw = torch.randn(32, 4, dtype=torch.float64) * 0.1
    v = M.project_tangent(x, raw)
    y = M.exp(x, v)
    v_back = M.log(x, y)
    assert torch.allclose(v_back, v, atol=1e-9)


def test_sphere_project_tangent_is_orthogonal_and_idempotent() -> None:
    M = Sphere(3)
    x = _rand_sphere(3, 8)
    u = torch.randn(8, 4, dtype=torch.float64)
    pu = M.project_tangent(x, u)
    # orthogonality to x
    assert (x * pu).sum(-1).abs().max() < 1e-12
    # idempotence
    assert torch.allclose(M.project_tangent(x, pu), pu)


def test_sphere_geodesic_endpoints() -> None:
    M = Sphere(3)
    x0 = _rand_sphere(3, 6)
    x1 = _rand_sphere(3, 6)
    # avoid antipodes
    inner = (x0 * x1).sum(-1)
    keep = inner > -0.99
    x0, x1 = x0[keep], x1[keep]
    t0 = torch.zeros(x0.shape[0], dtype=torch.float64)
    t1 = torch.ones(x0.shape[0], dtype=torch.float64)
    assert torch.allclose(M.geodesic(x0, x1, t0), x0, atol=1e-10)
    assert torch.allclose(M.geodesic(x0, x1, t1), x1, atol=1e-10)


def test_sphere_geodesic_stays_on_manifold() -> None:
    M = Sphere(3)
    x0 = _rand_sphere(3, 8)
    x1 = _rand_sphere(3, 8)
    t = torch.linspace(0.05, 0.95, 8, dtype=torch.float64)
    xt = M.geodesic(x0, x1, t)
    assert M.validate(xt, atol=1e-9).all()


def test_sphere_cfm_target_matches_log_form() -> None:
    """v_t = (1/(1-t)) * Log_{x_t}(x_1) should equal γ̇(t)."""
    M = Sphere(3)
    x0 = _rand_sphere(3, 16)
    x1 = _rand_sphere(3, 16)
    inner = (x0 * x1).sum(-1)
    keep = inner > -0.99
    x0, x1 = x0[keep], x1[keep]
    t = torch.full((x0.shape[0],), 0.3, dtype=torch.float64)

    closed = M.cfm_target_velocity(x0, x1, t)
    # explicit log form
    xt = M.geodesic(x0, x1, t)
    log_form = M.log(xt, x1) / (1.0 - t).unsqueeze(-1)
    assert torch.allclose(closed, log_form, atol=1e-9)


def test_sphere_wrapped_gaussian_on_manifold() -> None:
    M = Sphere(3)
    mu = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    samples = M.sample_wrapped_gaussian(mu, sigma=0.3, shape=(2048,))
    assert samples.shape == (2048, 4)
    assert M.validate(samples, atol=1e-8).all()
    # mean direction near mu (small sigma)
    mean_dir = samples.mean(0)
    mean_dir = mean_dir / mean_dir.norm()
    assert (mean_dir * mu).sum() > 0.9


def test_quat_to_upper_hemisphere() -> None:
    q = torch.tensor(
        [[-0.5, 0.5, 0.5, 0.5], [0.5, -0.5, -0.5, -0.5], [0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    qh = quat_to_upper_hemisphere(q)
    assert (qh[..., 0] >= 0.0).all()
    # rotation represented is unchanged: q and -q are the same rotation
    assert torch.allclose(qh.abs(), q.abs())


def test_quat_continuity_resolves_sign_flips() -> None:
    # craft a sequence where adjacent quaternions are antipodal
    base = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    seq = torch.stack([base, -base, base, -base, base], dim=0)  # (T=5, 4)
    fixed = quat_continuity(seq, dim=0)
    # all adjacent dot products should be ≥ 0
    dots = (fixed[:-1] * fixed[1:]).sum(-1)
    assert (dots >= 0).all()


# ---------------------------------------------------------------------------
# PreShape S^J_m
# ---------------------------------------------------------------------------


def _rand_preshape(J: int, m: int, *batch: int) -> torch.Tensor:
    raw = torch.randn(*batch, J, m, dtype=torch.float64)
    return joints_to_preshape(raw)


def test_preshape_validate_after_construction() -> None:
    M = PreShape(num_landmarks=22, dim=3)
    x = _rand_preshape(22, 3, 16)
    assert M.validate(x, atol=1e-9).all()
    assert M.ambient_dim == 66
    assert M.intrinsic_dim == 22 * 3 - 3 - 1


def test_preshape_exp_log_identity() -> None:
    M = PreShape(num_landmarks=10, dim=3)
    x = _rand_preshape(10, 3, 32)
    y = _rand_preshape(10, 3, 32)
    inner = (x * y).sum(-1)
    keep = inner > -0.99
    x, y = x[keep], y[keep]
    y_back = M.exp(x, M.log(x, y))
    assert torch.allclose(y_back, y, atol=1e-9)


def test_preshape_geodesic_endpoints_and_on_manifold() -> None:
    M = PreShape(num_landmarks=8, dim=3)
    x0 = _rand_preshape(8, 3, 8)
    x1 = _rand_preshape(8, 3, 8)
    inner = (x0 * x1).sum(-1)
    keep = inner > -0.99
    x0, x1 = x0[keep], x1[keep]
    t = torch.linspace(0.05, 0.95, 8, dtype=torch.float64)
    xt = M.geodesic(x0, x1, t)
    assert M.validate(xt, atol=1e-7).all()
    # endpoints
    t0 = torch.zeros(x0.shape[0], dtype=torch.float64)
    t1 = torch.ones(x0.shape[0], dtype=torch.float64)
    assert torch.allclose(M.geodesic(x0, x1, t0), x0, atol=1e-9)
    assert torch.allclose(M.geodesic(x0, x1, t1), x1, atol=1e-9)


def test_preshape_cfm_target_matches_log_form() -> None:
    M = PreShape(num_landmarks=8, dim=3)
    x0 = _rand_preshape(8, 3, 16)
    x1 = _rand_preshape(8, 3, 16)
    inner = (x0 * x1).sum(-1)
    keep = inner > -0.99
    x0, x1 = x0[keep], x1[keep]
    t = torch.full((x0.shape[0],), 0.4, dtype=torch.float64)
    closed = M.cfm_target_velocity(x0, x1, t)
    xt = M.geodesic(x0, x1, t)
    log_form = M.log(xt, x1) / (1.0 - t).unsqueeze(-1)
    assert torch.allclose(closed, log_form, atol=1e-9)


def test_preshape_project_tangent_kills_centering_drift() -> None:
    """Tangent projection must zero the centering direction."""
    M = PreShape(num_landmarks=5, dim=3)
    x = _rand_preshape(5, 3, 1)[0]
    # Build an ambient u that has a uniform shift component (purely centering drift)
    drift = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64).repeat(5).reshape(15)
    pu = M.project_tangent(x, drift)
    # After projection, the per-landmark column means should be zero
    pu_mat = pu.reshape(5, 3)
    assert pu_mat.mean(dim=0).abs().max() < 1e-12


# ---------------------------------------------------------------------------
# ProductManifold
# ---------------------------------------------------------------------------


def _rmg_manifold(j: int = 22) -> ProductManifold:
    """T+R = R^3 × (S^3)^J — the paper's representation."""
    return ProductManifold([Euclidean(3)] + [Sphere(3) for _ in range(j)])


def _rand_rmg_point(M: ProductManifold, *batch: int) -> torch.Tensor:
    parts = []
    for f in M.factors:
        if isinstance(f, Euclidean):
            parts.append(torch.randn(*batch, f.ambient_dim, dtype=torch.float64))
        elif isinstance(f, Sphere):
            parts.append(_rand_sphere(f.intrinsic_dim, *batch))
        else:  # pragma: no cover
            raise AssertionError
    return torch.cat(parts, dim=-1)


def test_product_dims() -> None:
    M = _rmg_manifold(j=22)
    assert M.ambient_dim == 3 + 22 * 4
    assert M.intrinsic_dim == 3 + 22 * 3


def test_product_exp_log_identity() -> None:
    M = _rmg_manifold(j=4)
    x = _rand_rmg_point(M, 8)
    y = _rand_rmg_point(M, 8)
    # avoid antipodes per sphere factor
    inners = []
    for s in M._slices[1:]:
        inners.append(((x[..., s] * y[..., s]).sum(-1) > -0.99))
    keep = torch.stack(inners, dim=-1).all(-1)
    x, y = x[keep], y[keep]
    y_back = M.exp(x, M.log(x, y))
    assert torch.allclose(y_back, y, atol=1e-9)


def test_product_geodesic_endpoints() -> None:
    M = _rmg_manifold(j=4)
    x0 = _rand_rmg_point(M, 4)
    x1 = _rand_rmg_point(M, 4)
    t0 = torch.zeros(4, dtype=torch.float64)
    t1 = torch.ones(4, dtype=torch.float64)
    assert torch.allclose(M.geodesic(x0, x1, t0), x0, atol=1e-10)
    assert torch.allclose(M.geodesic(x0, x1, t1), x1, atol=1e-10)


def test_product_validate_after_sample() -> None:
    M = _rmg_manifold(j=4)
    # rest pose: T=0, q=[1,0,0,0]
    rest_T = torch.zeros(3, dtype=torch.float64)
    rest_q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    mu = torch.cat([rest_T] + [rest_q] * 4, dim=-1)
    samples = M.sample_wrapped_gaussian(mu, sigma=0.2, shape=(64,))
    assert samples.shape == (64, M.ambient_dim)
    assert M.validate(samples, atol=1e-8).all()


def test_product_cfm_target_factor_consistency() -> None:
    """Target velocity on the product equals concatenation of per-factor targets."""
    M = _rmg_manifold(j=2)
    x0 = _rand_rmg_point(M, 4)
    x1 = _rand_rmg_point(M, 4)
    t = torch.full((4,), 0.4, dtype=torch.float64)
    full = M.cfm_target_velocity(x0, x1, t)
    parts = []
    for i, f in enumerate(M.factors):
        s = M._slices[i]
        parts.append(f.cfm_target_velocity(x0[..., s], x1[..., s], t))
    factorwise = torch.cat(parts, dim=-1)
    assert torch.allclose(full, factorwise, atol=1e-12)
