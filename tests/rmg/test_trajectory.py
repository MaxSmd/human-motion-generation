"""Trajectory (spatial mask-control) constraint tests.

The claims worth pinning down, since they are what makes RMG's answer to the
mask-control task different from the diffusion baselines:

  * pelvis control is EXACT — translation is the world pelvis position;
  * any single non-root joint is exact too, absorbed by the root, with the
    POSE left untouched;
  * `axes` restricts which world axes are touched;
  * the blend field is exact at keyframes in all three modes and continuous
    under "interp";
  * the guidance channel (the literature's mechanism) reduces the error on its
    own, and the metrics match the published definitions.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rmg.flow import (
    OracleVelocity,
    RiemannianEulerSampler,
    SamplerCfg,
    TrajectoryConstraint,
    TrajectoryControl,
    WrappedGaussianPrior,
    build_trajectory_energy_fn,
    build_trajectory_projector,
    compose_energies,
    compose_projectors,
    flat_to_joints,
    merge_metrics,
    parse_trajectory,
    resample_path,
    rest_pose_mu,
    rmg_manifold,
    sample_control_signal,
    sample_keyframes,
    stack_controls,
    trajectory_metrics,
)
from rmg.flow import apply_path_facing
from rmg.flow.trajectory import _apply_root_yaw, _yaw_between, body_forward
from shared.geometry.skeleton import NUM_JOINTS, Skeleton

torch.manual_seed(0)

J = NUM_JOINTS


def _toy_skeleton() -> Skeleton:
    """Non-degenerate offsets — enough for FK to place joints away from the root."""
    offs = torch.zeros(J, 3)
    offs[:, 1] = 0.12
    offs[:, 0] = 0.03
    return Skeleton(offsets=offs)


def _body_skeleton() -> Skeleton:
    """A skeleton with real LEFT/RIGHT separation at hips and shoulders.

    `_toy_skeleton` stacks every joint along one axis, so hips and shoulders
    coincide and `body_forward`'s "across" vector is degenerate — fine for
    position tests, useless for facing ones. With identity quaternions this body
    faces +Z, matching the dataset's canonical orientation.
    """
    offs = torch.zeros(J, 3)
    offs[:, 1] = 0.12
    offs[1] = torch.tensor([0.10, -0.05, 0.0])    # L_Hip  → +X
    offs[2] = torch.tensor([-0.10, -0.05, 0.0])   # R_Hip  → −X
    offs[13] = torch.tensor([0.05, 0.05, 0.0])    # L_Collar
    offs[14] = torch.tensor([-0.05, 0.05, 0.0])   # R_Collar
    offs[16] = torch.tensor([0.15, 0.0, 0.0])     # L_Shoulder → +X
    offs[17] = torch.tensor([-0.15, 0.0, 0.0])    # R_Shoulder → −X
    return Skeleton(offsets=offs)


def _random_state(B: int, T: int, seed: int = 0) -> torch.Tensor:
    """A valid flat tr state: arbitrary translation + unit quaternions."""
    g = torch.Generator().manual_seed(seed)
    trans = torch.randn(B, T, 3, generator=g)
    q = torch.randn(B, T, J, 4, generator=g)
    q = q / q.norm(dim=-1, keepdim=True)
    return torch.cat([trans, q.reshape(B, T, 4 * J)], dim=-1)


def _control(joint, frames, targets, T, **kw) -> TrajectoryControl:
    pos = torch.zeros(T, 3)
    mask = torch.zeros(T, dtype=torch.bool)
    for f, p in zip(frames, targets):
        pos[f] = torch.as_tensor(p, dtype=torch.float32)
        mask[f] = True
    c = TrajectoryConstraint(joint=joint, positions=pos, mask=mask,
                             axes=kw.pop("axes", "xyz"))
    return TrajectoryControl(constraints=[c], **kw)


# --------------------------------------------------------------------------- path / keyframes


def test_resample_path_is_constant_speed() -> None:
    # An L-shaped path: equal arc-length spacing means equal step sizes.
    pts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    out = resample_path(pts, 21)
    assert out.shape == (21, 3)
    torch.testing.assert_close(out[0], torch.tensor([0.0, 0.0, 0.0]))
    torch.testing.assert_close(out[-1], torch.tensor([1.0, 0.0, 1.0]))
    steps = torch.linalg.vector_norm(out[1:] - out[:-1], dim=-1)
    assert steps.max() - steps.min() < 1e-5, "resampling is not constant-speed"
    assert torch.allclose(out[:, 1], torch.zeros(21)), "2-D sketch must stay on the floor"


def test_resample_path_degenerate() -> None:
    assert resample_path([[1.0, 2.0, 3.0]], 5).shape == (5, 3)
    # all-coincident points: no arc length to distribute along
    out = resample_path([[0.0, 0.0], [0.0, 0.0]], 4)
    assert torch.allclose(out, torch.zeros(4, 3))


def test_sample_keyframes_densities() -> None:
    rng = np.random.default_rng(0)
    for k in (1, 2, 5, 49):
        f = sample_keyframes(60, k, rng)
        assert f.numel() == k
        assert torch.equal(f, f.sort().values), "keyframes must be sorted"
        assert f.min() >= 0 and f.max() < 60
    # density >= length and "all" both degenerate to every frame
    assert torch.equal(sample_keyframes(30, 49, rng), torch.arange(30))
    assert torch.equal(sample_keyframes(30, "all", rng), torch.arange(30))
    assert sample_keyframes(0, 5, rng).numel() == 0


# --------------------------------------------------------------------------- exactness


@pytest.mark.parametrize("blend", ["none", "local", "interp"])
def test_pelvis_projection_is_exact(blend: str) -> None:
    """Pelvis targets are hit to machine precision in ONE projection, in every
    blend mode — the R^3 factor of the manifold *is* the world pelvis position."""
    T = 40
    skel = _toy_skeleton()
    x = _random_state(2, T, seed=1)
    frames = [3, 17, 33]
    targets = [[1.0, 0.9, -2.0], [-0.5, 1.0, 0.25], [2.0, 0.8, 3.0]]
    ctrl = _control("pelvis", frames, targets, T, mode="project", blend=blend)
    proj = build_trajectory_projector(ctrl, skel, T)
    out = proj(x)

    joints = flat_to_joints(out, skel)
    for f, tgt in zip(frames, targets):
        got = joints[:, f, 0, :]
        torch.testing.assert_close(got, torch.tensor(tgt).expand_as(got), atol=1e-5, rtol=0)
    # quaternions untouched — only the Euclidean factor moved
    torch.testing.assert_close(out[..., 3:], x[..., 3:])


@pytest.mark.parametrize("joint", ["L_Wrist", "Head", "R_Foot"])
def test_single_nonroot_joint_is_exact_via_root(joint: str) -> None:
    """A target on any ONE joint is met exactly by shifting the root — p_j =
    τ + f_j(q), so τ ← target − f_j(q) lands it with the pose untouched."""
    T = 24
    skel = _toy_skeleton()
    x = _random_state(3, T, seed=2)
    frames = [5, 19]
    targets = [[0.7, 1.4, -0.3], [-1.2, 0.9, 2.1]]
    ctrl = _control(joint, frames, targets, T, mode="project", blend="none")
    out = build_trajectory_projector(ctrl, skel, T)(x)

    joints_before = flat_to_joints(x, skel)
    joints_after = flat_to_joints(out, skel)
    jidx = ctrl.constraints[0].joint_idx
    for f, tgt in zip(frames, targets):
        got = joints_after[:, f, jidx, :]
        torch.testing.assert_close(got, torch.tensor(tgt).expand_as(got), atol=1e-5, rtol=0)
    # POSE is preserved: every joint moved by the SAME rigid offset in a
    # controlled frame (the root absorbed the correction, nothing bent).
    for f in frames:
        off = joints_after[:, f] - joints_before[:, f]          # (B, J, 3)
        torch.testing.assert_close(off, off[:, :1].expand_as(off), atol=1e-5, rtol=0)


def test_axes_restriction_leaves_other_axes_alone() -> None:
    """An xz floor path must not decide the body's height."""
    T = 16
    skel = _toy_skeleton()
    x = _random_state(1, T, seed=3)
    ctrl = _control("pelvis", [4], [[1.0, 99.0, -1.0]], T,
                    axes="xz", mode="project", blend="none")
    out = build_trajectory_projector(ctrl, skel, T)(x)
    j = flat_to_joints(out, skel)
    assert abs(float(j[0, 4, 0, 0]) - 1.0) < 1e-5
    assert abs(float(j[0, 4, 0, 2]) + 1.0) < 1e-5
    # y untouched despite the absurd y target
    torch.testing.assert_close(out[..., 1], x[..., 1])


def test_two_joints_share_the_root_shift() -> None:
    """With two joints controlled in one frame the root can only take their
    MEAN residual — the leftover is a pose property (what guidance is for)."""
    T = 12
    skel = _toy_skeleton()
    x = _random_state(1, T, seed=4)
    pos_a = torch.zeros(T, 3); mask = torch.zeros(T, dtype=torch.bool)
    pos_b = torch.zeros(T, 3)
    mask[6] = True
    pos_a[6] = torch.tensor([1.0, 0.0, 0.0])
    pos_b[6] = torch.tensor([-1.0, 0.0, 0.0])
    ctrl = TrajectoryControl(
        constraints=[
            TrajectoryConstraint(joint="L_Wrist", positions=pos_a, mask=mask.clone()),
            TrajectoryConstraint(joint="R_Wrist", positions=pos_b, mask=mask.clone()),
        ],
        mode="project", blend="none",
    )
    before = flat_to_joints(x, skel)
    after = flat_to_joints(build_trajectory_projector(ctrl, skel, T)(x), skel)
    ea = (after[0, 6, 20] - pos_a[6]).norm()
    eb = (after[0, 6, 21] - pos_b[6]).norm()
    ba = (before[0, 6, 20] - pos_a[6]).norm()
    bb = (before[0, 6, 21] - pos_b[6]).norm()
    # Both improve, neither is exact — the shift is the shared component.
    assert ea < ba and eb < bb
    assert ea > 1e-4 and eb > 1e-4


# --------------------------------------------------------------------------- blending


def test_interp_blend_is_continuous_and_exact_at_keys() -> None:
    """The whole root path is warped through the control points: exact at the
    keyframes, and no velocity spike at them (unlike blend="none")."""
    T = 60
    skel = _toy_skeleton()
    # A SMOOTH base root path (a steady walk along +Z) — the point of the test
    # is the discontinuity the edit introduces, which white-noise translation
    # would drown out.
    x = _random_state(1, T, seed=5)
    x = x.clone()
    t = torch.arange(T, dtype=torch.float32)
    x[0, :, 0] = 0.0
    x[0, :, 1] = 1.0
    x[0, :, 2] = 0.05 * t
    frames = [10, 30, 50]
    targets = [[1.0, 1.0, 0.0], [2.0, 1.0, 1.0], [3.0, 1.0, 2.0]]

    def root_speed(state):
        tr = state[0, :, :3]
        return torch.linalg.vector_norm(tr[1:] - tr[:-1], dim=-1)

    base = root_speed(x).max()
    out_i = build_trajectory_projector(
        _control("pelvis", frames, targets, T, mode="project", blend="interp"), skel, T)(x)
    out_n = build_trajectory_projector(
        _control("pelvis", frames, targets, T, mode="project", blend="none"), skel, T)(x)

    for f, tgt in zip(frames, targets):     # exact in both
        torch.testing.assert_close(out_i[0, f, :3], torch.tensor(tgt), atol=1e-5, rtol=0)
        torch.testing.assert_close(out_n[0, f, :3], torch.tensor(tgt), atol=1e-5, rtol=0)

    # "none" teleports the root in and out of each keyframe; "interp" does not.
    assert root_speed(out_n).max() > 5 * root_speed(out_i).max()
    assert root_speed(out_i).max() < base + 1.0


def test_single_keyframe_interp_is_a_rigid_shift() -> None:
    T = 20
    skel = _toy_skeleton()
    x = _random_state(1, T, seed=6)
    ctrl = _control("pelvis", [7], [[5.0, 1.0, -5.0]], T, mode="project", blend="interp")
    out = build_trajectory_projector(ctrl, skel, T)(x)
    delta = out[0, :, :3] - x[0, :, :3]
    torch.testing.assert_close(delta, delta[0:1].expand_as(delta), atol=1e-5, rtol=0)


def test_local_blend_decays_to_zero_outside_its_radius() -> None:
    T = 60
    skel = _toy_skeleton()
    x = _random_state(1, T, seed=7)
    ctrl = _control("pelvis", [30], [[4.0, 1.0, 4.0]], T,
                    mode="project", blend="local", blend_frames=8)
    out = build_trajectory_projector(ctrl, skel, T)(x)
    delta = out[0, :, :3] - x[0, :, :3]
    assert delta[30].norm() > 1.0                    # exact at the key
    assert delta[:22].abs().max() < 1e-6             # untouched far away
    assert delta[39:].abs().max() < 1e-6


# --------------------------------------------------------------------------- guidance channel


def test_guidance_energy_gradient_reduces_error() -> None:
    """The soft channel on its own (the mechanism the baselines use) pulls the
    controlled joint toward its target."""
    T = 12
    skel = _toy_skeleton()
    x = _random_state(1, T, seed=8).requires_grad_(True)
    ctrl = _control("L_Wrist", [4, 9], [[1.5, 1.0, 0.0], [-1.0, 1.2, 1.0]], T, mode="guide")
    energy_fn = build_trajectory_energy_fn(ctrl, skel)
    e0 = energy_fn(x)
    (g,) = torch.autograd.grad(e0, x)
    with torch.no_grad():
        x2 = x - 0.05 * g
    e1 = float(energy_fn(x2).detach())
    e0f = float(e0.detach())
    assert e1 < e0f, f"guidance step raised the energy: {e0f} → {e1}"


def test_guidance_only_mode_has_no_projector() -> None:
    T = 10
    ctrl = _control("pelvis", [2], [[1.0, 1.0, 1.0]], T, mode="guide")
    assert build_trajectory_projector(ctrl, _toy_skeleton(), T) is None
    assert build_trajectory_energy_fn(ctrl, _toy_skeleton()) is not None
    ctrl_p = _control("pelvis", [2], [[1.0, 1.0, 1.0]], T, mode="project")
    assert build_trajectory_energy_fn(ctrl_p, _toy_skeleton()) is None


def test_sampler_end_to_end_hits_the_control_signal() -> None:
    """Through the real ODE loop: the projector runs after every Euler step, so
    the returned sample satisfies the control signal."""
    T = 16
    M = rmg_manifold(num_joints=J)
    mu = rest_pose_mu(num_joints=J, dtype=torch.float64).to(torch.float64)
    prior = WrappedGaussianPrior(M, mu, sigma=0.5)
    x0 = prior.sample((1, T), dtype=torch.float64)
    x1 = prior.sample((1, T), dtype=torch.float64)
    model = OracleVelocity(M, x0, x1).double()
    sampler = RiemannianEulerSampler(M, prior, SamplerCfg(num_steps=20, guidance_scale=1.0))
    skel = _toy_skeleton()

    frames, targets = [2, 8, 14], [[1.0, 1.0, 0.0], [2.0, 1.0, 1.0], [0.0, 1.0, 2.0]]
    ctrl = _control("pelvis", frames, targets, T, mode="project", blend="interp")
    proj = build_trajectory_projector(ctrl, skel, T, dtype=torch.float64)

    free = sampler.sample(model, shape=(1, T), num_steps=20, dtype=torch.float64)
    held = sampler.sample(model, shape=(1, T), num_steps=20, dtype=torch.float64, project_fn=proj)

    m_free = trajectory_metrics(flat_to_joints(free, skel), ctrl)
    m_held = trajectory_metrics(flat_to_joints(held, skel), ctrl)
    assert m_held["avg_err"] < 1e-4, f"projected sample missed its targets: {m_held}"
    assert m_free["avg_err"] > m_held["avg_err"]


def test_compose_projectors_and_energies() -> None:
    T = 8
    skel = _toy_skeleton()
    x = _random_state(1, T, seed=9)
    ctrl = _control("pelvis", [3], [[1.0, 1.0, 1.0]], T, mode="project")
    p = build_trajectory_projector(ctrl, skel, T)

    def shift_y(state):
        out = state.clone()
        out[..., 1] = out[..., 1] + 10.0
        return out

    # Order matters: the trajectory projector runs last and absorbs the shift.
    out = compose_projectors(shift_y, p)(x)
    torch.testing.assert_close(out[0, 3, :3], torch.tensor([1.0, 1.0, 1.0]), atol=1e-5, rtol=0)
    assert compose_projectors(None, None) is None
    assert compose_projectors(None, p) is p

    e = build_trajectory_energy_fn(_control("pelvis", [3], [[1.0, 1.0, 1.0]], T, mode="guide"), skel)
    total = compose_energies(e, e)(x)
    torch.testing.assert_close(total, 2 * e(x))


# --------------------------------------------------------------------------- metrics


def test_metrics_match_published_definitions() -> None:
    """loc_err = share of missed LOCATIONS, traj_err = share of failed CLIPS."""
    T, B = 6, 4
    pos = torch.zeros(T, 3)
    mask = torch.zeros(T, dtype=torch.bool)
    mask[[1, 4]] = True
    ctrl = TrajectoryControl(
        constraints=[TrajectoryConstraint(joint="pelvis", positions=pos, mask=mask)],
        mode="project",
    )
    joints = torch.zeros(B, T, J, 3)
    # clip 0: both keys perfect. clip 1: one key off by 1 m. clips 2,3: perfect.
    joints[1, 1, 0, 0] = 1.0
    m = trajectory_metrics(joints, ctrl)
    assert m["n_locations"] == 8 and m["n_clips"] == 4
    assert m["avg_err"] == pytest.approx(1.0 / 8, abs=1e-6)
    assert m["loc_err_0.5"] == pytest.approx(1 / 8)      # 1 of 8 locations missed
    assert m["traj_err_0.5"] == pytest.approx(1 / 4)     # 1 of 4 clips failed
    assert m["max_err"] == pytest.approx(1.0, abs=1e-5)


def test_metrics_respect_clip_lengths() -> None:
    """Keyframes past a clip's true length are padding and must not be scored."""
    T = 8
    pos = torch.zeros(T, 3)
    mask = torch.zeros(T, dtype=torch.bool)
    mask[[1, 6]] = True
    ctrl = TrajectoryControl(
        constraints=[TrajectoryConstraint(joint="pelvis", positions=pos, mask=mask)],
        mode="project")
    joints = torch.zeros(1, T, J, 3)
    joints[0, 6, 0, 0] = 5.0                     # huge error, but beyond length
    m = trajectory_metrics(joints, ctrl, lengths=torch.tensor([4]))
    assert m["n_locations"] == 1
    assert m["avg_err"] == pytest.approx(0.0, abs=1e-6)


def test_merge_metrics_weights_by_slot_count() -> None:
    a = {"n_locations": 10, "n_clips": 2, "avg_err": 0.1, "max_err": 0.4,
         "loc_err_0.5": 0.0, "traj_err_0.5": 0.0}
    b = {"n_locations": 30, "n_clips": 6, "avg_err": 0.5, "max_err": 0.9,
         "loc_err_0.5": 0.4, "traj_err_0.5": 0.5}
    m = merge_metrics([a, b])
    assert m["n_locations"] == 40 and m["n_clips"] == 8
    assert m["avg_err"] == pytest.approx((0.1 * 10 + 0.5 * 30) / 40)
    assert m["loc_err_0.5"] == pytest.approx((0.0 * 10 + 0.4 * 30) / 40)
    assert m["traj_err_0.5"] == pytest.approx((0.0 * 2 + 0.5 * 6) / 8)
    assert m["max_err"] == pytest.approx(0.9)
    assert merge_metrics([])["n_locations"] == 0


# --------------------------------------------------------------------------- protocol / parsing


def test_sample_control_signal_from_ground_truth() -> None:
    T = 50
    gt = torch.randn(T, J, 3)
    rng = np.random.default_rng(0)
    ctrl = sample_control_signal(gt, length=40, density=5, joint_set="pelvis", rng=rng)
    assert len(ctrl.constraints) == 1
    c = ctrl.constraints[0]
    assert c.joint_idx == 0 and c.num_controlled == 5
    # targets are literally the reference motion at those frames
    keys = torch.nonzero(c.mask[0]).squeeze(-1)
    assert int(keys.max()) < 40, "keyframes must stay inside the valid length"
    torch.testing.assert_close(c.positions[0, keys], gt[keys, 0, :])

    # cross-combination: one joint drawn per clip, from the published set
    cross = sample_control_signal(gt, 40, 2, joint_set="cross", rng=rng)
    assert len(cross.constraints) == 1
    assert cross.constraints[0].joint_idx in (0, 10, 11, 15, 20, 21)

    # explicit multi-joint set
    multi = sample_control_signal(gt, 40, 2, joint_set=(20, 21), rng=rng, pick_one=False)
    assert {c.joint_idx for c in multi.constraints} == {20, 21}

    with pytest.raises(ValueError):
        sample_control_signal(gt, 40, 2, joint_set="nope", rng=rng)


def test_parse_trajectory_wire_formats() -> None:
    T = 20
    sparse = parse_trajectory({"constraints": [
        {"joint": "pelvis", "frames": [0, 10], "points": [[0, 1, 0], [1, 1, 1]]}]}, T)
    assert sparse.constraints[0].num_controlled == 2
    assert sparse.mode == "hybrid" and sparse.blend == "interp"

    dense = parse_trajectory({"mode": "guide", "constraints": [
        {"joint": "Head", "positions": [[0, 1, 0]] * T}]}, T)
    assert dense.constraints[0].num_controlled == T
    assert dense.mode == "guide"

    path = parse_trajectory({"constraints": [
        {"joint": "pelvis", "axes": "xz", "path": [[0, 0], [2, 0]], "every": 5}]}, T)
    c = path.constraints[0]
    assert c.axes == "xz"
    # every=5 over 20 frames ⇒ 0,5,10,15 — plus the path endpoint (19), which is
    # always kept so a sub-sampled path still reaches where it was drawn to.
    assert c.num_controlled == 5
    assert bool(c.mask[0, 19])
    assert float(c.positions[0, T - 1, 0]) == pytest.approx(2.0)

    assert parse_trajectory(None, T) is None
    assert parse_trajectory({}, T) is None
    assert parse_trajectory({"constraints": [{"joint": "pelvis", "frames": [], "points": []}]}, T) is None
    # out-of-range frames are dropped, not clamped onto a valid frame
    assert parse_trajectory({"constraints": [
        {"joint": "pelvis", "frames": [999], "points": [[0, 0, 0]]}]}, T) is None


def test_invalid_specs_are_rejected() -> None:
    T = 5
    pos, mask = torch.zeros(T, 3), torch.ones(T, dtype=torch.bool)
    with pytest.raises(ValueError):
        TrajectoryConstraint(joint="pelvis", positions=pos, mask=torch.ones(3, dtype=torch.bool))
    with pytest.raises(ValueError):
        TrajectoryConstraint(joint="pelvis", positions=pos, mask=mask, axes="q")
    with pytest.raises(ValueError):
        TrajectoryConstraint(joint="pelvis", positions=pos, mask=mask, weight=0.0)
    with pytest.raises(ValueError):
        TrajectoryControl([TrajectoryConstraint("pelvis", pos, mask)], mode="nope")
    with pytest.raises(ValueError):
        TrajectoryControl([TrajectoryConstraint("pelvis", pos, mask)], blend="nope")


def test_per_row_control_signals_are_independent() -> None:
    """The eval protocol gives every clip its OWN targets and keyframes; the
    projector must honour them row by row, not share one signal."""
    T, B = 30, 3
    skel = _toy_skeleton()
    x = _random_state(B, T, seed=11)
    pos = torch.zeros(B, T, 3)
    mask = torch.zeros(B, T, dtype=torch.bool)
    keys = [[5], [12, 20], [2, 9, 27]]
    for b, ks in enumerate(keys):
        for k in ks:
            mask[b, k] = True
            pos[b, k] = torch.tensor([float(b), 1.0, float(k)])
    ctrl = TrajectoryControl(
        constraints=[TrajectoryConstraint(joint="pelvis", positions=pos, mask=mask)],
        mode="project", blend="interp")
    out = build_trajectory_projector(ctrl, skel, T)(x)
    for b, ks in enumerate(keys):
        for k in ks:
            torch.testing.assert_close(
                out[b, k, :3], torch.tensor([float(b), 1.0, float(k)]), atol=1e-5, rtol=0)
    m = trajectory_metrics(flat_to_joints(out, skel), ctrl)
    assert m["n_locations"] == 6 and m["avg_err"] < 1e-5


def test_stack_controls_merges_per_clip_signals() -> None:
    """Cross-combination: clips pick different joints, and stacking keeps each
    clip scored only on the joint it actually chose."""
    T = 20
    gt = torch.randn(T, J, 3)
    a = sample_control_signal(gt, T, 3, joint_set=(20,), rng=np.random.default_rng(0))
    b = sample_control_signal(gt, T, 2, joint_set=(15,), rng=np.random.default_rng(1))
    merged = stack_controls([a, b])
    assert {c.joint_idx for c in merged.constraints} == {20, 15}
    for c in merged.constraints:
        assert c.mask.shape == (2, T)
        # exactly one row controls each joint
        assert int((c.mask.sum(dim=1) > 0).sum()) == 1
    joints = torch.zeros(2, T, J, 3)
    m = trajectory_metrics(joints, merged)
    assert m["n_locations"] == 5, m           # 3 from clip A + 2 from clip B
    assert stack_controls([]) is None


def test_stacked_control_survives_a_shorter_batch() -> None:
    """A batch cropped to fewer frames than the signal must truncate, not crash."""
    T = 24
    skel = _toy_skeleton()
    gt = torch.randn(T, J, 3)
    ctrl = stack_controls([
        sample_control_signal(gt, T, "all", joint_set="pelvis", rng=np.random.default_rng(k))
        for k in range(2)
    ])
    short = 10
    x = _random_state(2, short, seed=12)
    out = build_trajectory_projector(ctrl, skel, short)(x)
    m = trajectory_metrics(flat_to_joints(out, skel), ctrl)
    assert m["n_locations"] == 2 * short
    assert m["avg_err"] < 1e-5


def test_eval_protocol_end_to_end() -> None:
    """The exact flow `rmg.scripts.evaluate` runs: read targets off reference
    motions, stack them per clip, project, then score control fidelity and
    physical cost together."""
    from shared.eval import motion_quality

    B, T = 4, 40
    skel = _toy_skeleton()
    lengths = torch.tensor([40, 32, 25, 40])
    reference = _random_state(B, T, seed=21)
    gt_joints = flat_to_joints(reference, skel)

    rng = np.random.default_rng(0)
    ctrl = stack_controls([
        sample_control_signal(gt_joints[i], length=int(lengths[i]), density=5,
                              joint_set="pelvis", rng=rng, mode="project", blend="interp")
        for i in range(B)
    ])
    assert ctrl.constraints[0].mask.shape == (B, T)

    gen = _random_state(B, T, seed=22)
    held = build_trajectory_projector(ctrl, skel, T)(gen)
    joints = flat_to_joints(held, skel)

    m = trajectory_metrics(joints, ctrl, lengths=lengths)
    assert m["n_locations"] == B * 5
    assert m["avg_err"] < 1e-5
    assert m["loc_err_0.5"] == 0.0 and m["traj_err_0.5"] == 0.0
    # And the free sample is genuinely far off, so the metric has range.
    assert trajectory_metrics(flat_to_joints(gen, skel), ctrl, lengths=lengths)["avg_err"] > 0.1

    q = motion_quality(joints, lengths=lengths)
    assert set(q) == {"foot_skate_ratio", "jerk", "root_speed"}


def test_blend_trades_smoothness_against_locality() -> None:
    """The claim the evaluation is built to test: enforcing sparse targets
    exactly costs smoothness, and `blend` is the dial. "none" writes the root
    only at keyframes and kinks the trajectory; "interp" warps the whole path
    through them and stays smooth."""
    from shared.eval import jerk

    T = 60
    skel = _toy_skeleton()
    # A smooth reference clip: steady walk, fixed pose.
    x = _random_state(1, T, seed=23).clone()
    t = torch.arange(T, dtype=torch.float32)
    x[0, :, 0], x[0, :, 1], x[0, :, 2] = 0.0, 1.0, 0.05 * t
    x[0, :, 3:] = x[0, 0:1, 3:]                      # freeze the pose
    frames = [10, 25, 40, 55]
    targets = [[0.6, 1.0, 0.5], [-0.4, 1.0, 1.2], [0.8, 1.0, 2.0], [0.0, 1.0, 2.8]]

    out = {}
    for blend in ("none", "interp"):
        ctrl = _control("pelvis", frames, targets, T, mode="project", blend=blend)
        held = build_trajectory_projector(ctrl, skel, T)(x)
        joints = flat_to_joints(held, skel)
        out[blend] = jerk(joints)
        # both are exact — they differ only in what happens between keyframes
        assert trajectory_metrics(joints, ctrl)["avg_err"] < 1e-5

    assert out["none"] > 5 * out["interp"], (
        f"expected 'none' to kink the trajectory: {out}"
    )


def _heading_vs_travel_deg(joints, min_step=0.005):
    """Mean angle between where the body faces and where it is going."""
    out = []
    for t in range(joints.shape[0] - 1):
        v = joints[t + 1, 0] - joints[t, 0]
        v[1] = 0.0
        s = float(v.norm())
        if s < min_step:
            continue
        f = body_forward(joints[t])
        cos = float((v / s * f).sum().clamp(-1, 1))
        out.append(np.degrees(np.arccos(cos)))
    return float(np.mean(out)) if out else 0.0


def test_body_forward_matches_the_dataset_construction() -> None:
    """A body whose hips/shoulders straddle the X axis faces +Z — the same
    convention `humanml3d_io._canonicalize_first_frame` normalises to."""
    j = torch.zeros(J, 3)
    r_hip, l_hip, sdr_r, sdr_l = 2, 1, 17, 16
    j[r_hip] = torch.tensor([-0.1, 0.9, 0.0])     # right hip at −X
    j[l_hip] = torch.tensor([0.1, 0.9, 0.0])
    j[sdr_r] = torch.tensor([-0.2, 1.4, 0.0])
    j[sdr_l] = torch.tensor([0.2, 1.4, 0.0])
    torch.testing.assert_close(body_forward(j), torch.tensor([0.0, 0.0, 1.0]), atol=1e-6, rtol=0)


def test_yaw_helpers_round_trip() -> None:
    f = torch.tensor([0.0, 0.0, 1.0])
    g = torch.tensor([1.0, 0.0, 0.0])
    yaw = _yaw_between(f, g)
    assert float(yaw) == pytest.approx(np.pi / 2, abs=1e-6)
    # Applying that yaw to an identity root must rotate +Z onto +X.
    q = _apply_root_yaw(torch.tensor([1.0, 0.0, 0.0, 0.0]), yaw)
    from shared.geometry.skeleton import quat_rotate
    torch.testing.assert_close(quat_rotate(q, f), g, atol=1e-6, rtol=0)


def test_face_path_turns_the_body_along_the_route() -> None:
    """The gap this closes: moving the root's POSITION without its ORIENTATION
    drags the body sideways/backwards along the path."""
    T = 80
    skel = _body_skeleton()
    # A body facing +Z, walking a quarter-circle that ends heading +X.
    x = _random_state(1, T, seed=31).clone()
    q = torch.zeros(J, 4); q[:, 0] = 1.0            # identity pose, faces +Z
    x[0, :, 3:] = q.reshape(-1)
    ang = torch.linspace(0, np.pi / 2, T)
    x[0, :, 0], x[0, :, 1], x[0, :, 2] = 2 * torch.sin(ang), 1.0, 2 * torch.cos(ang)

    frames = list(range(0, T, 10))
    targets = [[float(2 * np.sin(a)), 1.0, float(2 * np.cos(a))]
               for a in np.linspace(0, np.pi / 2, T)[frames]]

    off = _control("pelvis", frames, targets, T, mode="project", blend="interp")
    on = _control("pelvis", frames, targets, T, mode="project", blend="interp",
                  face_path=True)

    p_off, p_on = (build_trajectory_projector(c, skel, T) for c in (off, on))
    j_off = flat_to_joints(p_off(x), skel)[0]
    j_on = flat_to_joints(apply_path_facing(p_on(x), on, skel, project_fn=p_on), skel)[0]

    a_off = _heading_vs_travel_deg(j_off)
    a_on = _heading_vs_travel_deg(j_on)
    assert a_on < 15.0, f"face_path did not align the body: {a_on:.1f}°"
    assert a_on < a_off - 20.0, f"no improvement: off={a_off:.1f}° on={a_on:.1f}°"

    # …and the position constraint is still met exactly (the yaw runs BEFORE the
    # translation absorption precisely so this stays true).
    assert trajectory_metrics(j_on.unsqueeze(0), on)["avg_err"] < 1e-5


def test_contact_blend_moves_the_root_during_swing() -> None:
    """`interp` spends the root correction uniformly in time, including while a
    foot is planted — which is why the feet slide however slow the path is.
    `contact` spends it while the feet are in the air instead, and is still
    exact at the keyframes."""
    from rmg.flow import swing_weight

    T, skel = 40, _body_skeleton()
    x = _random_state(1, T, seed=41).clone()
    q = torch.zeros(J, 4); q[:, 0] = 1.0
    x[0, :, 3:] = q.reshape(-1)
    x[0, :, :3] = torch.tensor([0.0, 1.0, 0.0])

    # Feet planted for the first half, airborne for the second.
    joints = flat_to_joints(x, skel)
    free = torch.zeros(1, T)
    free[0, T // 2:] = 1.0

    delta = torch.zeros(1, T, 3)
    has = torch.zeros(1, T, dtype=torch.bool)
    has[0, 0] = True; has[0, T - 1] = True
    delta[0, T - 1] = torch.tensor([2.0, 0.0, 0.0])       # 2 m to make up

    from rmg.flow.trajectory import _spread
    f_interp = _spread(delta, has, "interp", 1)
    f_contact = _spread(delta, has, "contact", 1, free=free)

    # Both land the full correction at the last keyframe…
    torch.testing.assert_close(f_interp[0, -1], f_contact[0, -1], atol=1e-5, rtol=0)
    # …but only `interp` moves the root while the feet are planted.
    assert f_interp[0, T // 2 - 1, 0] > 0.4, "interp should have moved by mid-clip"
    assert f_contact[0, T // 2 - 1, 0] < 0.05, (
        f"contact moved the root under a planted foot: {float(f_contact[0, T//2-1, 0]):.3f}"
    )
    assert swing_weight(joints).shape == (1, T)


def test_retime_preserves_a_standing_lead_in() -> None:
    """A generated clip stands still before it walks. Constant-speed targets
    demand motion from frame 0 and drag the standing body; `retime` reads the
    target off the body's own progress, so standing frames stay put."""
    T, skel = 60, _body_skeleton()
    x = _random_state(1, T, seed=51).clone()
    q = torch.zeros(J, 4); q[:, 0] = 1.0
    x[0, :, 3:] = q.reshape(-1)
    # Stand for 30 frames, then walk +Z. Slightly off the path in x, so there IS
    # a correction to make — just not one that should move the standing frames.
    trans = torch.zeros(T, 3); trans[:, 1] = 1.0; trans[:, 0] = 0.3
    trans[30:, 2] = torch.linspace(0, 2.0, T - 30)
    x[0, :, :3] = trans

    frames = [0, 20, 40, T - 1]
    targets = [[0.0, 1.0, 0.0], [0.0, 1.0, 0.67], [0.0, 1.0, 1.33], [0.0, 1.0, 2.0]]

    fixed = _control("pelvis", frames, targets, T, mode="project", blend="interp")
    retimed = _control("pelvis", frames, targets, T, mode="project", blend="interp",
                       retime=True)
    j_fixed = flat_to_joints(build_trajectory_projector(fixed, skel, T)(x), skel)[0]
    j_ret = flat_to_joints(build_trajectory_projector(retimed, skel, T)(x), skel)[0]

    def travel(j, lo, hi):
        return float(torch.linalg.vector_norm(
            j[lo + 1:hi, 0, [0, 2]] - j[lo:hi - 1, 0, [0, 2]], dim=-1).sum())

    # Constant-speed targets march the body forward through the standing phase;
    # retiming leaves it where it stood.
    assert travel(j_fixed, 0, 30) > 0.4, "expected the fixed schedule to drag"
    assert travel(j_ret, 0, 30) < 0.05, (
        f"retime moved a standing body: {travel(j_ret, 0, 30):.3f} m"
    )
    # Both still end up at the end of the path.
    assert float(torch.linalg.vector_norm(j_ret[-1, 0, [0, 2]]
                                          - torch.tensor([0.0, 2.0]))) < 0.05


def test_contact_blend_is_still_exact_at_keyframes() -> None:
    T, skel = 48, _body_skeleton()
    x = _random_state(2, T, seed=42)
    frames = [0, 16, 32, 47]
    targets = [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [2.0, 1.0, 1.5], [3.0, 1.0, 2.0]]
    ctrl = _control("pelvis", frames, targets, T, mode="project", blend="contact")
    out = build_trajectory_projector(ctrl, skel, T)(x)
    assert trajectory_metrics(flat_to_joints(out, skel), ctrl)["avg_err"] < 1e-5


def test_face_smooth_tames_a_corner_snap() -> None:
    """A right-angled path turns 90° between two frames. Tracking that literally
    snaps the root and spikes the jerk (measured 30× on the real model); the
    heading is averaged in time so the body turns the way a body can."""
    from shared.eval import jerk as jerk_metric

    T = 120
    skel = _body_skeleton()
    x = _random_state(1, T, seed=35).clone()
    q = torch.zeros(J, 4); q[:, 0] = 1.0
    x[0, :, 3:] = q.reshape(-1)
    # An L: straight along +Z, then a right-angle turn along +X.
    frames = list(range(0, T, 20)) + [T - 1]
    pts = []
    for f in frames:
        s = (f / (T - 1)) * 5.0
        pts.append([0.0, 1.0, s] if s <= 3 else [s - 3, 1.0, 3.0])
    dense = torch.zeros(T, 3)
    for i in range(T):
        s = (i / (T - 1)) * 5.0
        dense[i] = torch.tensor([0.0, 1.0, s] if s <= 3 else [s - 3, 1.0, 3.0])
    x[0, :, :3] = dense

    def run(sm):
        c = _control("pelvis", frames, pts, T, mode="project", blend="interp",
                     face_path=True, face_smooth=sm)
        pr = build_trajectory_projector(c, skel, T)
        return flat_to_joints(apply_path_facing(pr(x), c, skel, project_fn=pr), skel)

    snap = jerk_metric(run(0))
    smooth = jerk_metric(run(15))
    assert smooth < 0.5 * snap, f"smoothing did not tame the corner: {snap:.0f} -> {smooth:.0f}"
    # …and the position constraint is untouched by the smoothing.
    c = _control("pelvis", frames, pts, T, mode="project", blend="interp",
                 face_path=True, face_smooth=15)
    assert trajectory_metrics(run(15), c)["avg_err"] < 1e-5


def test_face_path_ignored_for_non_root_joints() -> None:
    """A wrist target says nothing about which way the body faces."""
    T = 20
    skel = _body_skeleton()
    x = _random_state(1, T, seed=32)
    ctrl = _control("L_Wrist", [4, 12], [[1.0, 1.2, 0.0], [2.0, 1.2, 1.0]], T,
                    mode="project", face_path=True)
    pr = build_trajectory_projector(ctrl, skel, T)
    out = apply_path_facing(pr(x), ctrl, skel, project_fn=pr)
    # root quaternion untouched
    torch.testing.assert_close(out[..., 3:7], pr(x)[..., 3:7])


def test_face_path_leaves_stationary_frames_alone() -> None:
    """No travel ⇒ no heading to follow; don't snap to a noise direction."""
    T = 24
    skel = _body_skeleton()
    x = _random_state(1, T, seed=33)
    ctrl = _control("pelvis", [0, 12, 23], [[1.0, 1.0, 1.0]] * 3, T,
                    mode="project", blend="interp", face_path=True)
    pr = build_trajectory_projector(ctrl, skel, T)
    out = apply_path_facing(pr(x), ctrl, skel, project_fn=pr)
    torch.testing.assert_close(out[..., 3:7], pr(x)[..., 3:7], atol=1e-6, rtol=0)


def test_face_strength_scales_the_correction() -> None:
    T = 40
    skel = _body_skeleton()
    x = _random_state(1, T, seed=34).clone()
    q = torch.zeros(J, 4); q[:, 0] = 1.0
    x[0, :, 3:] = q.reshape(-1)
    t = torch.arange(T, dtype=torch.float32)
    x[0, :, 0], x[0, :, 1], x[0, :, 2] = 0.0, 1.0, 0.05 * t
    frames = [0, 20, 39]
    targets = [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [2.0, 1.0, 2.0]]   # heads +X+Z

    def resid(s):
        c = _control("pelvis", frames, targets, T, mode="project", blend="interp",
                     face_path=True, face_strength=s)
        pr = build_trajectory_projector(c, skel, T)
        return _heading_vs_travel_deg(flat_to_joints(apply_path_facing(pr(x), c, skel,
                                                                      project_fn=pr), skel)[0])

    full, half, none_ = resid(1.0), resid(0.5), resid(0.0)
    assert full < half < none_, f"strength not monotone: {full:.1f} {half:.1f} {none_:.1f}"


def test_flat_to_joints_rejects_short_state() -> None:
    with pytest.raises(ValueError, match="representation"):
        flat_to_joints(torch.zeros(1, 4, 10), _toy_skeleton())
