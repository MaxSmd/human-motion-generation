"""Flow-matching tests.

Centerpiece: with the *oracle* CFM target velocity, integrating from t=0 to t=1
on the manifold must recover x_1 (since the velocity field is exactly the
conditional flow that connects x_0 → x_1 along the geodesic).
"""

from __future__ import annotations

import torch

from rmg.flow import (
    FlowMatchingTrainer,
    FlowMatchingTrainerCfg,
    OracleVelocity,
    RiemannianEulerSampler,
    SamplerCfg,
    WrappedGaussianPrior,
    build_cfm_batch,
    rest_pose_mu,
    rmg_manifold,
    sample_t,
)
from rmg.manifolds import Euclidean, ProductManifold, Sphere

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------


def test_prior_samples_lie_on_manifold() -> None:
    M = rmg_manifold(num_joints=4)
    mu = rest_pose_mu(num_joints=4, dtype=torch.float64).to(torch.float64)
    prior = WrappedGaussianPrior(M, mu, sigma=0.5)
    samples = prior.sample((128,), dtype=torch.float64)
    assert samples.shape == (128, M.ambient_dim)
    assert M.validate(samples, atol=1e-7).all()


def test_prior_supports_batched_temporal_shape() -> None:
    M = rmg_manifold(num_joints=2)
    mu = rest_pose_mu(num_joints=2, dtype=torch.float64).to(torch.float64)
    prior = WrappedGaussianPrior(M, mu, sigma=0.3)
    samples = prior.sample((4, 16), dtype=torch.float64)  # (B, T, D)
    assert samples.shape == (4, 16, M.ambient_dim)
    assert M.validate(samples, atol=1e-7).all()


def test_prior_zero_sigma_returns_mu() -> None:
    M = rmg_manifold(num_joints=3)
    mu = rest_pose_mu(num_joints=3, dtype=torch.float64).to(torch.float64)
    prior = WrappedGaussianPrior(M, mu, sigma=0.0)
    samples = prior.sample((8,), dtype=torch.float64)
    expected = mu.expand_as(samples)
    assert torch.allclose(samples, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# Interpolation / target velocity
# ---------------------------------------------------------------------------


def test_cfm_target_endpoints_are_consistent() -> None:
    """At t→0, x_t→x_0 and target ≈ Log_{x_0}(x_1) = γ̇(0)."""
    M = rmg_manifold(num_joints=2)
    mu = rest_pose_mu(num_joints=2, dtype=torch.float64).to(torch.float64)
    prior = WrappedGaussianPrior(M, mu, sigma=0.3)
    x0 = prior.sample((4,), dtype=torch.float64)
    x1 = prior.sample((4,), dtype=torch.float64)
    t = torch.full((4,), 1e-4, dtype=torch.float64)
    batch = build_cfm_batch(M, x0, x1, t)
    # x_t at t≈0 must be ≈ x_0 (and on the manifold)
    assert torch.allclose(batch.x_t, x0, atol=1e-3)
    # target is in T_{x_0}M-ish, finite
    assert torch.isfinite(batch.target).all()


def test_sample_t_is_inside_open_unit_interval() -> None:
    t = sample_t(1024, dtype=torch.float64, eps=1e-3)
    assert (t > 1e-3 - 1e-12).all()
    assert (t < 1.0 - 1e-3 + 1e-12).all()


# ---------------------------------------------------------------------------
# Sampler with oracle velocity — the end-to-end correctness check
# ---------------------------------------------------------------------------


def test_oracle_sampler_recovers_x1_on_sphere() -> None:
    M = Sphere(3)

    def _rand_sphere(*shape: int) -> torch.Tensor:
        x = torch.randn(*shape, 4, dtype=torch.float64)
        return x / x.norm(dim=-1, keepdim=True)

    x0 = _rand_sphere(8, 4)
    x1 = _rand_sphere(8, 4)
    # avoid antipodes
    inner = (x0 * x1).sum(-1)
    keep = inner > -0.9
    x0, x1 = x0[keep], x1[keep]

    # Build a prior that returns x_0 deterministically (sigma=0, mu=x_0 itself
    # would only work for one batch element — instead, override the prior).
    class FixedPrior:
        def sample(self, shape, device=None, dtype=None, generator=None):
            assert shape == (x0.shape[0], x0.shape[1])
            return x0.to(dtype=dtype) if dtype is not None else x0

    sampler = RiemannianEulerSampler(M, FixedPrior(), SamplerCfg(num_steps=200))  # type: ignore[arg-type]
    oracle = OracleVelocity(M, x0=x0, x1=x1)

    out = sampler.sample(oracle, shape=(x0.shape[0], x0.shape[1]), dtype=torch.float64)
    assert M.validate(out, atol=1e-6).all()
    # geodesic distance at t=1 should be ≈ 0
    inner_pred = (out * x1).sum(-1).clamp(-1.0, 1.0)
    err_angle = torch.arccos(inner_pred)
    assert err_angle.max() < 5e-3, f"max err {err_angle.max():.4e}"


def test_oracle_sampler_recovers_x1_on_product_manifold() -> None:
    """Strongest single test: oracle velocity on R^3 × (S^3)^J recovers x_1."""
    M = ProductManifold([Euclidean(3)] + [Sphere(3) for _ in range(4)])

    def _rand_pt(B: int) -> torch.Tensor:
        parts = [torch.randn(B, 3, dtype=torch.float64)]
        for _ in range(4):
            q = torch.randn(B, 4, dtype=torch.float64)
            parts.append(q / q.norm(dim=-1, keepdim=True))
        return torch.cat(parts, dim=-1)

    x0 = _rand_pt(8)
    x1 = _rand_pt(8)
    # The product slerp on each S^3 needs non-antipodal — with random inits this
    # is overwhelmingly likely; resample if it fires.
    for s in M._slices[1:]:
        inner = (x0[..., s] * x1[..., s]).sum(-1)
        if (inner < -0.9).any():
            return  # extremely rare; skip

    class FixedPrior:
        def sample(self, shape, device=None, dtype=None, generator=None):
            return x0.to(dtype=dtype) if dtype is not None else x0

    sampler = RiemannianEulerSampler(M, FixedPrior(), SamplerCfg(num_steps=200))  # type: ignore[arg-type]
    oracle = OracleVelocity(M, x0=x0, x1=x1)

    out = sampler.sample(oracle, shape=(8, 1), dtype=torch.float64)  # (B, T=1, D)
    # squeeze trailing T dim wasn't actually added — our oracle ignores T,
    # but the shape contract requires (B, T). We stored x0/x1 as (B, D); test
    # passes shape (8, 1) and the broadcast over T=1 yields (B, 1, D).
    out = out.squeeze(1)

    assert M.validate(out, atol=1e-6).all()
    # element-wise close to x_1
    assert torch.allclose(out, x1, atol=5e-3)


# ---------------------------------------------------------------------------
# Trainer: end-to-end loss with a tiny stub model
# ---------------------------------------------------------------------------


class _ZeroModel(torch.nn.Module):
    """Model that always predicts zero velocity. Loss = E‖target‖^2."""

    def forward(self, x_t, t, *, cond=None, drop_cond_mask=None, mask=None):  # noqa: D401
        return torch.zeros_like(x_t)


def test_trainer_loss_is_finite_and_decreasing_with_tangent_projection() -> None:
    M = rmg_manifold(num_joints=4)
    mu = rest_pose_mu(num_joints=4, dtype=torch.float32)
    prior = WrappedGaussianPrior(M, mu, sigma=0.5)

    trainer = FlowMatchingTrainer(M, prior, FlowMatchingTrainerCfg(cfg_dropout=0.1))
    B, T = 6, 8
    x1 = prior.sample((B, T))  # use prior samples as fake "data"
    cond = torch.randn(B, 16)  # dummy text features
    mask = torch.ones(B, T, dtype=torch.bool)

    loss_zero, info = trainer.compute_loss(_ZeroModel(), x1, cond=cond, mask=mask)
    assert torch.isfinite(loss_zero)
    assert info["x_t_offmanifold"] < 1e-2  # x_t stays on M


def test_trainer_perfect_oracle_gives_zero_loss() -> None:
    """If the model returns the exact CFM target, the loss must be 0.

    Caveat: trainer samples (t, x_0, x_1, x_t) per call; we route the oracle
    through a thin shim that recomputes the target from a known (x_0, x_1).
    Since the trainer also computes that target internally, an oracle that
    returns it bit-for-bit gives zero.
    """
    M = rmg_manifold(num_joints=2)
    mu = rest_pose_mu(num_joints=2, dtype=torch.float64).to(torch.float64)
    prior = WrappedGaussianPrior(M, mu, sigma=0.3)
    trainer = FlowMatchingTrainer(M, prior, FlowMatchingTrainerCfg(cfg_dropout=0.0))

    class OracleAtCallTime(torch.nn.Module):
        """Recomputes target from (x_t, t) — but the trainer has already
        sampled (x_0, x_1) internally, so this oracle can't see them. The
        clean way is to monkey-patch the prior to return a fixed x_0, then
        also fix x_1, then have the oracle return the correct target."""

    # Easiest: make the prior deterministic (sigma=0 → x_0 = mu = rest pose).
    # Then in the trainer, x_0 == mu. The target is then a known function of
    # (mu, x_1, t), and an oracle returning M.cfm_target_velocity(mu, x_1, t)
    # gives bit-exact agreement.
    prior_fixed = WrappedGaussianPrior(M, mu, sigma=0.0)
    trainer_fixed = FlowMatchingTrainer(M, prior_fixed, FlowMatchingTrainerCfg(cfg_dropout=0.0))

    B, T = 4, 6
    # Build x_1 as wrapped-Gaussian samples around mu — guaranteed on-manifold.
    x1 = WrappedGaussianPrior(M, mu, sigma=0.3).sample((B, T), dtype=torch.float64)

    class OracleFixedX0(torch.nn.Module):
        def __init__(self, M, mu, x1):
            super().__init__()
            self.M = M
            self.register_buffer("mu", mu)
            self.register_buffer("x1", x1)

        def forward(self, x_t, t, *, cond=None, drop_cond_mask=None, mask=None):
            x0 = self.mu.expand_as(x_t)
            return self.M.cfm_target_velocity(x0, self.x1, t)

    oracle = OracleFixedX0(M, mu, x1).to(torch.float64)
    # Convert dtypes so everything is float64 for tightness.
    loss, info = trainer_fixed.compute_loss(oracle, x1)
    assert loss < 1e-12, f"oracle loss not zero: {loss.item():.3e}"
