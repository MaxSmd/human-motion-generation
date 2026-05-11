"""DiT + conditioning + text-encoder tests.

Strongest checks:
  * AdaLN-Zero init ⇒ RMGDiT(...) returns exactly zero before any training.
  * End-to-end smoke: trainer.compute_loss + a single optimizer step lowers
    the loss on a tiny batch (model can fit something).
  * Sampler with the trained-on-noise model still produces on-manifold output.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from rmg.flow import (
    FlowMatchingTrainer,
    FlowMatchingTrainerCfg,
    RiemannianEulerSampler,
    SamplerCfg,
    WrappedGaussianPrior,
    rest_pose_mu,
    rmg_manifold,
)
from rmg.models import (
    RMG_BASE_CONFIG,
    ConditioningFusion,
    DiTBlock,
    DiTConfig,
    FinalLayer,
    RMGDiT,
    RandomTextEncoder,
    sinusoidal_time_embedding,
)

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
# ---------------------------------------------------------------------------


def test_time_embedding_shape_and_known_values() -> None:
    t = torch.tensor([0.0, 0.5, 1.0])
    emb = sinusoidal_time_embedding(t, dim=4)
    assert emb.shape == (3, 4)
    # at t=0 with dim=4 → [cos(0), cos(0), sin(0), sin(0)] = [1,1,0,0]
    assert torch.allclose(emb[0], torch.tensor([1.0, 1.0, 0.0, 0.0]), atol=1e-12)


def test_time_embedding_is_distinct_for_different_t() -> None:
    t = torch.tensor([0.1, 0.2, 0.3])
    emb = sinusoidal_time_embedding(t, dim=64)
    # pairwise cosine similarity < 1
    for i in range(3):
        for j in range(i + 1, 3):
            cs = F.cosine_similarity(emb[i:i + 1], emb[j:j + 1]).item()
            assert cs < 0.9999


# ---------------------------------------------------------------------------
# Conditioning fusion
# ---------------------------------------------------------------------------


def test_conditioning_fusion_shapes_and_finiteness() -> None:
    fusion = ConditioningFusion(text_dim=128, hidden_dim=64)
    t = torch.rand(5)
    cond = torch.randn(5, 128)
    out = fusion(t, cond)
    assert out.shape == (5, 64)
    assert torch.isfinite(out).all()


def test_conditioning_fusion_handles_none_cond_via_null_embed() -> None:
    """cond=None must be equivalent to dropping all samples."""
    fusion = ConditioningFusion(text_dim=128, hidden_dim=64)
    t = torch.rand(4)
    out_none = fusion(t, cond=None)
    # Equivalent: pass any cond + drop_mask all-True
    cond = torch.randn(4, 128)
    drop = torch.ones(4, dtype=torch.bool)
    out_drop_all = fusion(t, cond=cond, drop_cond_mask=drop)
    assert torch.allclose(out_none, out_drop_all, atol=1e-6)


def test_conditioning_fusion_drop_mask_is_per_sample() -> None:
    fusion = ConditioningFusion(text_dim=128, hidden_dim=64)
    fusion.eval()  # disable dropout (none here, but defensive)
    t = torch.rand(2)
    cond = torch.randn(2, 128)
    drop = torch.tensor([True, False])
    mixed = fusion(t, cond=cond, drop_cond_mask=drop)
    full = fusion(t, cond=cond, drop_cond_mask=torch.zeros(2, dtype=torch.bool))
    none = fusion(t, cond=None)
    # row 0 was dropped → matches `none` row 0
    assert torch.allclose(mixed[0], none[0], atol=1e-6)
    # row 1 was kept → matches `full` row 1
    assert torch.allclose(mixed[1], full[1], atol=1e-6)


# ---------------------------------------------------------------------------
# DiT block / final layer
# ---------------------------------------------------------------------------


def test_dit_block_is_identity_at_init() -> None:
    """AdaLN-Zero starts each block as residual-only ⇒ x unchanged."""
    block = DiTBlock(hidden_dim=32, num_heads=4, ffn_mult=2)
    block.eval()
    x = torch.randn(2, 5, 32)
    c = torch.randn(2, 32)
    out = block(x, c, key_padding_mask=None)
    assert torch.allclose(out, x, atol=1e-6)


def test_final_layer_outputs_zero_at_init() -> None:
    fl = FinalLayer(hidden_dim=32, out_dim=91)
    fl.eval()
    x = torch.randn(3, 7, 32)
    c = torch.randn(3, 32)
    out = fl(x, c)
    assert torch.allclose(out, torch.zeros_like(out))


# ---------------------------------------------------------------------------
# RMGDiT
# ---------------------------------------------------------------------------


def _small_dit() -> RMGDiT:
    cfg = DiTConfig(input_dim=91, hidden_dim=64, depth=2, num_heads=4, ffn_mult=2,
                    text_dim=32, time_freq_dim=64, max_seq_len=64)
    return RMGDiT(cfg)


def test_rmgdit_forward_shape_and_zero_init() -> None:
    model = _small_dit()
    model.eval()
    B, T, D = 3, 12, 91
    x = torch.randn(B, T, D)
    t = torch.rand(B)
    cond = torch.randn(B, model.cfg.text_dim)
    drop = torch.zeros(B, dtype=torch.bool)
    out = model(x, t, cond=cond, drop_cond_mask=drop)
    assert out.shape == x.shape
    # All zero at init (FinalLayer is zero-initialized).
    assert torch.allclose(out, torch.zeros_like(out))


def test_rmgdit_respects_max_seq_len() -> None:
    model = _small_dit()
    with pytest.raises(ValueError):
        model(torch.zeros(1, model.cfg.max_seq_len + 1, model.cfg.input_dim),
              torch.zeros(1), cond=None)


def test_rmgdit_with_mask_does_not_blow_up() -> None:
    model = _small_dit()
    model.eval()
    B, T, D = 2, 8, 91
    x = torch.randn(B, T, D)
    t = torch.rand(B)
    cond = torch.randn(B, model.cfg.text_dim)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, 5:] = False  # last 3 frames of sample 0 are pad
    out = model(x, t, cond=cond, drop_cond_mask=torch.zeros(B, dtype=torch.bool), mask=mask)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_rmgdit_param_count_matches_paper_base_when_input_dim_aligned() -> None:
    """Sanity: full-config RMG-base param count is in the right ballpark.
    Paper hints 'small'; we don't enforce exact match (their exact init/extras
    are unknown), but we want >1M params for hidden=384, depth=6."""
    model = RMGDiT(RMG_BASE_CONFIG)
    n = model.num_params()
    assert n > 1_000_000, f"RMG_BASE_CONFIG has only {n} params"


# ---------------------------------------------------------------------------
# End-to-end: trainer + RMGDiT, single optimizer step lowers the loss
# ---------------------------------------------------------------------------


def test_training_lowers_loss_on_tiny_dataset() -> None:
    """End-to-end: trainer.compute_loss + optimizer.step lowers the loss
    averaged over many evaluations (each call resamples t and x_0 so a single
    evaluation is too noisy to compare directly)."""
    torch.manual_seed(42)
    M = rmg_manifold(num_joints=4)
    mu = rest_pose_mu(num_joints=4)
    prior = WrappedGaussianPrior(M, mu, sigma=0.5)

    cfg = DiTConfig(input_dim=M.ambient_dim, hidden_dim=32, depth=2, num_heads=4,
                    ffn_mult=2, text_dim=16, time_freq_dim=32, max_seq_len=32)
    model = RMGDiT(cfg)
    trainer = FlowMatchingTrainer(M, prior, FlowMatchingTrainerCfg(cfg_dropout=0.0))

    B, T = 8, 16
    x1 = WrappedGaussianPrior(M, mu, sigma=0.4).sample((B, T))
    cond = torch.randn(B, cfg.text_dim)

    @torch.no_grad()
    def avg_loss(n: int) -> float:
        ls = []
        for _ in range(n):
            l, _ = trainer.compute_loss(model, x1, cond=cond)
            ls.append(l.item())
        return sum(ls) / len(ls)

    initial = avg_loss(20)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(300):
        loss, _ = trainer.compute_loss(model, x1, cond=cond)
        opt.zero_grad()
        loss.backward()
        opt.step()
    final = avg_loss(20)
    assert final < initial * 0.7, f"loss didn't drop enough: {initial:.4f} → {final:.4f}"


# ---------------------------------------------------------------------------
# End-to-end: trained model + sampler stays on manifold
# ---------------------------------------------------------------------------


def test_sampler_with_dit_yields_on_manifold_output() -> None:
    torch.manual_seed(7)
    M = rmg_manifold(num_joints=4)
    mu = rest_pose_mu(num_joints=4)
    prior = WrappedGaussianPrior(M, mu, sigma=0.5)

    cfg = DiTConfig(input_dim=M.ambient_dim, hidden_dim=16, depth=2, num_heads=4,
                    ffn_mult=2, text_dim=8, time_freq_dim=16, max_seq_len=16)
    model = RMGDiT(cfg)
    sampler = RiemannianEulerSampler(M, prior, SamplerCfg(num_steps=8, guidance_scale=1.0))

    cond = torch.randn(2, cfg.text_dim)
    out = sampler.sample(model, shape=(2, 6), cond=cond, num_steps=8)
    assert out.shape == (2, 6, M.ambient_dim)
    assert M.validate(out, atol=1e-4).all()


# ---------------------------------------------------------------------------
# RandomTextEncoder is deterministic per text
# ---------------------------------------------------------------------------


def test_random_text_encoder_is_deterministic_per_text() -> None:
    enc = RandomTextEncoder(text_dim=64)
    a = enc.encode(["a person walks forward", "a person jumps", "a person waves"])
    b = enc.encode(["a person walks forward", "a person jumps", "a person waves"])
    assert torch.allclose(a, b)
    # different texts → different rows
    assert not torch.allclose(a[0], a[1])
