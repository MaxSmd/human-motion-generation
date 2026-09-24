"""The ProjFlow port against the pinned upstream code (external/ProjFlow).

Upstream's projflow_helpers.py and its transport Sampler are pure torch, so the
port is checked numerically against them on random inputs: the metric, the
pseudo-observation model, the batched projection (upstream: one dense system
per sample), and the full sampling loop with a toy velocity field. Skipped when
the submodule is not checked out.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from projflow.sampler import KinematicMetric, ProjFlowConfig, project_clean_endpoint, projflow_sample
from projflow.sampler import observations as obs
from projflow.sampler.metric import skeleton_laplacian

UPSTREAM = Path(__file__).resolve().parents[2] / "external" / "ProjFlow"
HELPERS = UPSTREAM / "diffusions" / "transport" / "projflow_helpers.py"
pytestmark = pytest.mark.skipif(not HELPERS.exists(), reason="external/ProjFlow submodule not checked out")

B, D, L, J = 3, 3, 40, 22
ATOL = 1e-5


# Importing upstream modules must not leave __pycache__ inside the submodule.
sys.dont_write_bytecode = True


@pytest.fixture(scope="module")
def up():
    spec = importlib.util.spec_from_file_location("upstream_projflow_helpers", HELPERS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _up_metric(up, kinematic: bool = True):
    lap = up.build_skeleton_laplacian(J, device="cpu", dtype=torch.float64)
    if kinematic:
        return up.build_kinematic_metric(J=J, w_kin=10.0, ridge=1.0, L_kin=lap)
    ones = torch.ones(J, dtype=torch.float64)
    return up.KinematicMetric(apply_Rinv=lambda b: b, diag_Rinv_joint=ones, joint_weights_q=ones, L_kin=lap)


def _keyframes(seed: int, density: float = 0.05):
    g = torch.Generator().manual_seed(seed)
    mask_lj = torch.rand(B, 1, L, J, generator=g) < density
    mask_lj[:, :, 5, 0] = True                                  # every sample has a keyframe
    mask = mask_lj.expand(B, D, L, J).to(torch.float64)
    value = torch.randn(B, D, L, J, generator=g, dtype=torch.float64) * mask
    return mask, value


def test_laplacian_and_metric_match_upstream(up):
    ours = KinematicMetric.kinematic(10.0, 1.0, dtype=torch.float64)
    theirs = _up_metric(up)
    assert torch.equal(skeleton_laplacian(dtype=torch.float64), theirs.L_kin)
    b = torch.randn(B, D, L, J, dtype=torch.float64)
    torch.testing.assert_close(ours.apply_rinv(b), theirs.apply_Rinv(b), atol=1e-10, rtol=0)
    torch.testing.assert_close(ours.diag_rinv_joint, theirs.diag_Rinv_joint, atol=1e-10, rtol=0)
    torch.testing.assert_close(ours.joint_weights_q, theirs.joint_weights_q, rtol=1e-8, atol=0)


@pytest.mark.parametrize("radius", [3.0, 6.5, 10.0])
def test_pseudo_observations_match_upstream(up, radius):
    mask, value = _keyframes(0)
    y_ours, halo_ours = obs.pseudo_observations(mask, value, radius)
    y_up, halo_up = up.build_pseudo_observations(hard_mask=mask, hard_value=value, halo_radius=radius)
    torch.testing.assert_close(y_ours, y_up, atol=1e-12, rtol=0)
    assert torch.equal(halo_ours, halo_up)


def test_trust_and_variance_match_upstream(up):
    mask, value = _keyframes(1)
    x1 = torch.randn(B, D, L, J, dtype=torch.float64)
    ours, theirs = KinematicMetric.kinematic(10.0, 1.0, dtype=torch.float64), _up_metric(up)
    sched = obs.TrustSchedule()

    curv = obs.curvature(x1, ours)
    torch.testing.assert_close(curv, up.curvature_per_frame(x1_hat=x1, metric=theirs, w_kin=10.0, ridge=1.0))
    pi = obs.frame_trust(0.37, curv, sched)
    torch.testing.assert_close(pi, up.frame_trust_schedule(
        t_scalar=0.37, curvature=curv, pi_min=0.02, pi_max=1.0, tau_min=0.1, c0=3.0, lambda_s=1.0, p=2.0))

    _, halo = obs.pseudo_observations(mask, value, 6.0)
    halo_any = halo[:, 0] > 0.5
    pj = obs.joint_trust(pi, halo_any, ours.joint_weights_q, sched)
    torch.testing.assert_close(pj, up.distribute_trust_over_halo_joints(
        pi_frame=pi, M_halo_any=halo_any, q_joint=theirs.joint_weights_q, pi_min=0.02, pi_max=1.0))

    selected = ((mask > 0.5) | (halo > 0.5)).to(torch.float64)
    torch.testing.assert_close(
        obs.trust_to_variance(pj, ours, mask, selected),
        up.trust_to_variance(pi_joint=pj, metric=theirs, hard_mask=mask, M_all=selected))


@pytest.mark.parametrize("kinematic", [True, False])
@pytest.mark.parametrize("soft", [True, False])
def test_projection_matches_upstream(up, kinematic, soft):
    mask, value = _keyframes(2, density=0.08)
    x1 = torch.randn(B, D, L, J, dtype=torch.float64)
    build = KinematicMetric.kinematic if kinematic else KinematicMetric.euclidean
    ours, theirs = build(10.0, 1.0, dtype=torch.float64), _up_metric(up, kinematic)

    y_src, halo = obs.pseudo_observations(mask, value, 5.0)
    if soft:
        hard = mask > 0.5
        selected = (hard | (halo > 0.5)).to(torch.float64)
        targets = torch.where(hard, value, torch.where(halo > 0.5, y_src, value))
        sigma2 = torch.rand(B, 1, L, J, dtype=torch.float64).expand(B, D, L, J) * (halo > 0.5)
    else:
        selected, targets, sigma2 = mask, value, None

    got = project_clean_endpoint(x1, selected, targets, ours, sigma2)
    want = up.metric_project_clean_endpoint(
        x1_hat=x1, selector=selected, targets=targets, apply_Rinv=theirs.apply_Rinv, sigma2=sigma2)
    torch.testing.assert_close(got, want, atol=1e-9, rtol=0)
    if not soft:  # hard constraints are met exactly
        hard = mask > 0.5
        torch.testing.assert_close(got[hard], value[hard], atol=1e-6, rtol=0)


def _toy_velocity():
    g = torch.Generator().manual_seed(7)
    w = torch.randn(J, J, generator=g) * 0.1

    def velocity(x, t):
        return torch.einsum("bdlj,jk->bdlk", x, w) * (1.0 + t.view(-1, 1, 1, 1)) - 0.3 * x

    return velocity


@pytest.mark.parametrize("use_projflow", [True, False])
def test_full_sampler_matches_upstream_loop(use_projflow):
    pytest.importorskip("torchdiffeq")
    sys.path.insert(0, str(UPSTREAM))
    try:
        from diffusions.transport import Sampler, create_transport
    finally:
        sys.path.remove(str(UPSTREAM))

    mask, value = (t.float() for t in _keyframes(3))
    x0 = torch.randn(B, D, L, J, generator=torch.Generator().manual_seed(11))
    velocity = _toy_velocity()

    torch.manual_seed(123)
    want = Sampler(create_transport()).sample_projflow(num_steps=20, use_projflow=use_projflow)(
        x0, lambda x, t, **_: velocity(x, t), A=mask, y=value)[-1]

    cfg = ProjFlowConfig(num_steps=20) if use_projflow else ProjFlowConfig.upstream_off(num_steps=20)
    torch.manual_seed(123)
    got = projflow_sample(velocity, x0, mask, value, cfg)

    torch.testing.assert_close(got, want, atol=ATOL, rtol=0)
    hard = mask > 0.5
    torch.testing.assert_close(got[hard], value[hard], atol=1e-5, rtol=0)


@pytest.mark.parametrize("switch", ["use_kinematic_metric", "use_pseudo_obs", "use_noise_mixing"])
def test_each_ablation_keeps_constraints_exact(switch):
    mask, value = (t.float() for t in _keyframes(4))
    x0 = torch.randn(B, D, L, J, generator=torch.Generator().manual_seed(5))
    cfg = ProjFlowConfig(num_steps=10, **{switch: False})
    out = projflow_sample(_toy_velocity(), x0, mask, value, cfg)
    hard = mask > 0.5
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[hard], value[hard], atol=1e-5, rtol=0)
