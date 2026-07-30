"""MARDM model tests: forward_loss (train) and generate->decode (sampling).

Tiny configs so the SiT ODE sampling runs quickly on CPU. The 263-D eval bridge
(needs the HumanML3D submodule) is covered separately and on the cluster.
"""

from __future__ import annotations

import torch

from mardm.models import AE, MARDM, AEConfig, MARDMConfig


def _tiny() -> tuple[AE, MARDM]:
    ae = AE(AEConfig(input_width=67, output_emb_width=16, width=32, depth=2))
    mardm = MARDM(MARDMConfig(
        ae_dim=16, text_dim=32, latent_dim=64, ff_size=128,
        num_layers=2, num_heads=4, diffmlps_width=64, diffmlps_depth=2, diffmlps_batch_mul=2,
    ))
    return ae, mardm


def test_forward_loss_backward() -> None:
    ae, mardm = _tiny()
    latents = ae.encode(torch.randn(2, 32, 67))        # (2, 16, 8)
    loss = mardm.forward_loss(latents, torch.randn(2, 32), torch.tensor([8, 6]))
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in mardm.parameters())


def test_generate_and_decode_shapes() -> None:
    ae, mardm = _tiny()
    cond = torch.randn(3, 32)
    latent_lens = torch.tensor([8, 6, 4])
    latents = mardm.generate(cond, m_lens=latent_lens, timesteps=2, cond_scale=2.0)
    assert latents.shape == (3, 16, 8)                 # (B, ae_dim, max latent len)
    essential = ae.decode(latents)
    assert essential.shape == (3, 8 * ae.downsample_rate, 67)
    assert torch.isfinite(essential).all()


def test_generate_no_cfg() -> None:
    ae, mardm = _tiny()
    latents = mardm.generate(torch.randn(2, 32), m_lens=torch.tensor([5, 5]),
                             timesteps=2, cond_scale=1.0)
    assert latents.shape == (2, 16, 5)
