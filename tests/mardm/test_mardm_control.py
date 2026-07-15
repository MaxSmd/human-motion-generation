"""Control-guidance tests: differentiable loss path, metrics, guided sampling.

Tiny configs so the per-token euler ODE sampling runs quickly on CPU. Uses
random weights; DiffMLPs zero-inits its final layer (zero Jacobian w.r.t. z),
so the gradient-path test nudges those weights to emulate a trained head.
"""

from __future__ import annotations

import torch

from mardm.control import (
    ControlMARDM,
    ControlSignal,
    GuidanceConfig,
    control_forward_loss,
    control_loss,
    control_metrics,
    control_signal_features,
    generate_guided,
    latents_to_joints,
)
from mardm.models import AE, MARDM, AEConfig, MARDMConfig


def _tiny() -> tuple[AE, MARDM]:
    ae = AE(AEConfig(input_width=67, output_emb_width=16, width=32, depth=2))
    mardm = MARDM(MARDMConfig(
        ae_dim=16, text_dim=32, latent_dim=64, ff_size=128,
        num_layers=2, num_heads=4, diffmlps_width=64, diffmlps_depth=2, diffmlps_batch_mul=2,
    ))
    return ae.eval(), mardm.eval()


def _stats() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(67), torch.ones(67)


def _unzero_diff_head(mardm: MARDM) -> None:
    with torch.no_grad():
        mardm.DiffMLPs.net.final_layer.linear.weight.normal_(0, 0.02)
        for blk in mardm.DiffMLPs.net.res_blocks:
            blk.adaLN_modulation[-1].weight.normal_(0, 0.02)


def test_control_metrics_handcrafted() -> None:
    joints = torch.zeros(2, 10, 22, 3)
    targets = torch.zeros(2, 10, 22, 3)
    mask = torch.zeros(2, 10, 22, dtype=torch.bool)
    mask[0, 0, 0] = True   # off by 1.0m -> traj+loc failure
    mask[0, 5, 0] = True   # off by 0.1m -> ok
    mask[1, 3, 5] = True   # exact -> ok, seq 1 succeeds
    targets[0, 0, 0, 0] = 1.0
    targets[0, 5, 0, 0] = 0.1
    m = control_metrics(joints, ControlSignal(targets, mask), threshold=0.5)
    assert m["traj_err"] == 0.5                  # seq 0 fails, seq 1 ok
    assert abs(m["loc_err"] - 1 / 3) < 1e-6      # 1 of 3 cells beyond 0.5m
    assert abs(m["avg_err"] - (1.0 + 0.1) / 3) < 1e-6

    loss = control_loss(joints, ControlSignal(targets, mask))
    assert torch.isclose(loss, torch.tensor((1.0 + 0.1) / 3))


def test_control_loss_length_mismatch() -> None:
    # Signal built on GT length > decoded length must still work (and vice versa).
    joints = torch.zeros(1, 8, 22, 3)
    sig = ControlSignal(torch.zeros(1, 12, 22, 3), torch.ones(1, 12, 22, dtype=torch.bool))
    assert torch.isfinite(control_loss(joints, sig))
    assert torch.isfinite(torch.tensor(control_metrics(joints, sig)["avg_err"]))


def test_gradient_flows_z_to_control_loss() -> None:
    torch.manual_seed(0)
    ae, mardm = _tiny()
    _unzero_diff_head(mardm)
    mean, std = _stats()

    b, l = 1, 6
    z = torch.randn(b * l, mardm.latent_dim, requires_grad=True)
    noise = torch.randn(b * l, mardm.ae_dim)
    sample_fn = mardm.DiffMLPs.gen_transport.sample_ode(sampling_method="euler", num_steps=4)
    x = sample_fn(noise, mardm.DiffMLPs.net, c=z)[-1]
    joints = latents_to_joints(x.reshape(b, l, -1), ae, mean, std)

    mask = torch.zeros(b, joints.shape[1], 22, dtype=torch.bool)
    mask[:, ::5, 0] = True
    loss = control_loss(joints, ControlSignal(torch.randn_like(joints), mask))
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0


def test_generate_guided_unguided_baseline() -> None:
    torch.manual_seed(0)
    ae, mardm = _tiny()
    mean, std = _stats()
    m_lens = torch.tensor([6, 4])
    T = 6 * ae.downsample_rate
    control = ControlSignal(torch.zeros(2, T, 22, 3), torch.zeros(2, T, 22, dtype=torch.bool))
    latents = generate_guided(
        mardm, ae, torch.randn(2, 32), m_lens, control, mean, std,
        timesteps=2, cond_scale=2.0, guidance=GuidanceConfig(inner_iters=0, post_iters=0,
                                                             ode_steps_final=4),
    )
    assert latents.shape == (2, 16, 6)
    assert torch.isfinite(latents).all()
    assert (latents[1, :, 4:] == 0).all()        # padded positions stay zero


def test_generate_guided_with_optimization() -> None:
    torch.manual_seed(0)
    ae, mardm = _tiny()
    _unzero_diff_head(mardm)
    mean, std = _stats()
    m_lens = torch.tensor([5])
    T = 5 * ae.downsample_rate
    mask = torch.zeros(1, T, 22, dtype=torch.bool)
    mask[0, ::7, 0] = True
    control = ControlSignal(torch.randn(1, T, 22, 3) * 0.1, mask)
    latents = generate_guided(
        mardm, ae, torch.randn(1, 32), m_lens, control, mean, std,
        timesteps=2, cond_scale=1.0,
        guidance=GuidanceConfig(inner_iters=2, lr=0.05, ode_steps_guidance=3,
                                ode_steps_final=4, post_iters=2, post_lr=0.01),
    )
    assert latents.shape == (1, 16, 5)
    assert torch.isfinite(latents).all()


def test_generate_guided_with_repair_rounds() -> None:
    torch.manual_seed(0)
    ae, mardm = _tiny()
    _unzero_diff_head(mardm)
    mean, std = _stats()
    m_lens = torch.tensor([6, 3])
    T = 6 * ae.downsample_rate
    mask = torch.zeros(2, T, 22, dtype=torch.bool)
    mask[:, ::9, 0] = True
    control = ControlSignal(torch.randn(2, T, 22, 3) * 0.1, mask)
    latents = generate_guided(
        mardm, ae, torch.randn(2, 32), m_lens, control, mean, std,
        timesteps=2, cond_scale=2.0,
        guidance=GuidanceConfig(inner_iters=2, lr=0.05, ode_steps_guidance=3,
                                ode_steps_final=4, repair_rounds=2, repair_frac=0.5,
                                repair_iters=1),
    )
    assert latents.shape == (2, 16, 6)
    assert torch.isfinite(latents).all()
    assert (latents[1, :, 3:] == 0).all()        # padding survives repair remasking


def _control(b: int, t: int, k_every: int = 7) -> ControlSignal:
    mask = torch.zeros(b, t, 22, dtype=torch.bool)
    mask[:, ::k_every, 0] = True
    return ControlSignal(torch.randn(b, t, 22, 3) * 0.1, mask)


def test_regularizer_identity_at_init() -> None:
    # Zero-init joins: base(latents) == base(latents, residuals) at init.
    torch.manual_seed(0)
    ae, mardm = _tiny()
    reg = ControlMARDM(mardm, frames_per_latent=ae.downsample_rate).eval()
    b, l = 2, 6
    latents = torch.randn(b, l, 16)
    cond = torch.randn(b, 32)
    padding = torch.zeros(b, l, dtype=torch.bool)
    control = _control(b, l * ae.downsample_rate)
    feats = control_signal_features(control, l, ae.downsample_rate)
    res = reg.residuals(latents, cond, padding, feats)
    z_base = mardm.forward(latents, cond, padding)
    z_reg = mardm.forward(latents, cond, padding, control_residuals=res)
    assert torch.allclose(z_base, z_reg, atol=1e-6)
    assert all(float(r.detach().abs().max()) == 0.0 for r in res)


def test_regularizer_state_dict_excludes_base() -> None:
    ae, mardm = _tiny()
    reg = ControlMARDM(mardm, frames_per_latent=ae.downsample_rate)
    keys = list(reg.state_dict().keys())
    assert keys and all(k.split(".")[0] in ("copy_blocks", "control_in", "zero_proj")
                        for k in keys)
    # round-trip: a fresh instance loads the checkpoint standalone
    reg2 = ControlMARDM(mardm, frames_per_latent=ae.downsample_rate)
    reg2.load_state_dict(reg.state_dict())


def test_control_forward_loss_trains_regularizer_only() -> None:
    torch.manual_seed(0)
    ae, mardm = _tiny()
    _unzero_diff_head(mardm)
    for p in mardm.parameters():
        p.requires_grad_(False)
    for p in ae.parameters():
        p.requires_grad_(False)
    reg = ControlMARDM(mardm, frames_per_latent=ae.downsample_rate).train()
    mean, std = _stats()

    latents = ae.encode(torch.randn(2, 32, 67))            # (2, 16, 8)
    m_lens = torch.tensor([8, 6])
    control = _control(2, 8 * ae.downsample_rate)
    l_diff, l_s = control_forward_loss(reg, ae, latents, torch.randn(2, 32),
                                       m_lens, control, mean, std)
    assert torch.isfinite(l_diff) and torch.isfinite(l_s)
    (0.1 * l_diff + 0.9 * l_s).backward()
    grads = [p.grad for p in reg.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)
    assert all(p.grad is None for p in mardm.parameters())
    assert all(p.grad is None for p in ae.parameters())


def test_generate_guided_with_regularizer() -> None:
    torch.manual_seed(0)
    ae, mardm = _tiny()
    reg = ControlMARDM(mardm, frames_per_latent=ae.downsample_rate).eval()
    for p in reg.parameters():
        p.requires_grad_(False)
    mean, std = _stats()
    m_lens = torch.tensor([5, 3])
    control = _control(2, 5 * ae.downsample_rate)
    latents = generate_guided(
        mardm, ae, torch.randn(2, 32), m_lens, control, mean, std,
        timesteps=2, cond_scale=2.0, regularizer=reg,
        guidance=GuidanceConfig(inner_iters=0, post_iters=0, ode_steps_final=4),
    )
    assert latents.shape == (2, 16, 5)
    assert torch.isfinite(latents).all()
