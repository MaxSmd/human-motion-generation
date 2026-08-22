"""Tests for `shared.eval.metrics` on synthetic features.

We can't exercise the real Guo evaluator without ~hundreds of MB of
checkpoints, but the metric formulas are isolated and have known limits:
    * FID(p, p) = 0
    * R@1 = 1 when text/motion features are paired-identical
    * MM-Dist = 0 when paired
    * Diversity ≈ √2·D_eff for unit Gaussians (D-dim)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from shared.eval import (
    RandomGuoEvaluator,
    diversity,
    fid,
    mm_distance,
    multimodality,
    r_precision,
)
from shared.eval.guo_evaluator import _resolve_evaluator_assets, _upstream_align_indices


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------


def test_fid_is_zero_for_same_distribution() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((1024, 32))
    b = rng.standard_normal((1024, 32))
    val = fid(a, b)
    # both i.i.d. unit Gaussian — FID should be small (sample noise only)
    assert val < 1.0, f"FID between two unit Gaussians too large: {val:.4f}"


def test_fid_grows_with_mean_shift() -> None:
    rng = np.random.default_rng(0)
    real = rng.standard_normal((512, 32))
    shifted_a = rng.standard_normal((512, 32)) + 1.0
    shifted_b = rng.standard_normal((512, 32)) + 5.0
    fid_small = fid(real, shifted_a)
    fid_large = fid(real, shifted_b)
    assert fid_large > fid_small > 0.0


# ---------------------------------------------------------------------------
# R@k
# ---------------------------------------------------------------------------


def test_r_precision_perfect_alignment() -> None:
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((64, 16)).astype(np.float32)
    rs = r_precision(feats.copy(), feats.copy(), top_k=3, batch_size=32)
    # Identical features paired by index — every retrieval is correct
    assert rs[0] == pytest.approx(1.0)
    assert rs[1] == pytest.approx(1.0)
    assert rs[2] == pytest.approx(1.0)


def test_r_precision_random_features_is_near_chance() -> None:
    rng = np.random.default_rng(0)
    text = rng.standard_normal((320, 32))
    motion = rng.standard_normal((320, 32))
    rs = r_precision(text, motion, top_k=3, batch_size=32, rng=rng)
    # Chance R@1 over batches of 32 is 1/32 ≈ 0.031 — should be in that ballpark
    assert rs[0] < 0.15


def test_r_precision_is_monotone_in_k() -> None:
    rng = np.random.default_rng(1)
    text = rng.standard_normal((320, 32))
    motion = rng.standard_normal((320, 32))
    rs = r_precision(text, motion, top_k=5, batch_size=32)
    assert all(rs[k] <= rs[k + 1] + 1e-12 for k in range(len(rs) - 1))


# ---------------------------------------------------------------------------
# MM-Dist
# ---------------------------------------------------------------------------


def test_mm_distance_zero_when_paired_identical() -> None:
    feats = np.random.randn(20, 16)
    assert mm_distance(feats, feats) == pytest.approx(0.0)


def test_mm_distance_grows_with_perturbation() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((50, 16))
    a_close = a + 0.1 * rng.standard_normal((50, 16))
    a_far = a + 1.0 * rng.standard_normal((50, 16))
    assert mm_distance(a, a_close) < mm_distance(a, a_far)


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------


def test_diversity_for_unit_gaussian_known_scale() -> None:
    """For X, Y ~ N(0, I_D) independent, E‖X − Y‖ ≈ √2 √D for large D."""
    D = 64
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((1024, D))
    div = diversity(feats, diversity_times=512, rng=rng)
    # Theoretical mean ≈ √2 · √(D-0.5); allow generous slack.
    expected = np.sqrt(2 * D)
    assert abs(div - expected) / expected < 0.05


def test_diversity_zero_when_all_features_identical() -> None:
    feats = np.ones((100, 16))
    assert diversity(feats, diversity_times=10) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# MultiModality
# ---------------------------------------------------------------------------


def test_multimodality_zero_when_per_text_samples_are_identical() -> None:
    base = np.random.randn(8, 1, 16)            # T=8, K=1 — duplicated to K=10
    mm_input = np.repeat(base, 10, axis=1)      # all K samples identical per text
    val = multimodality(mm_input, multimodality_times=4)
    assert val == pytest.approx(0.0)


def test_multimodality_grows_with_per_text_noise() -> None:
    rng = np.random.default_rng(0)
    # T=20 distinct texts, K=10 generations each, dim=32. Higher noise → bigger MM.
    T, K, D = 20, 10, 32
    base = rng.standard_normal((T, 1, D))
    low = base + 0.1 * rng.standard_normal((T, K, D))
    high = base + 1.0 * rng.standard_normal((T, K, D))
    assert multimodality(high) > multimodality(low) > 0.0


# ---------------------------------------------------------------------------
# RandomGuoEvaluator (smoke / shape contract)
# ---------------------------------------------------------------------------


def test_random_guo_evaluator_shapes() -> None:
    ev = RandomGuoEvaluator(motion_dim=128, text_dim=128)
    motions = torch.randn(4, 16, 263)
    lengths = torch.tensor([15, 14, 13, 12])
    me = ev.encode_motion(motions, lengths)
    te = ev.encode_text_from_strings(["a person walks"] * 4)
    assert me.shape == (4, 128)
    assert te.shape == (4, 128)


def test_random_guo_evaluator_text_is_deterministic_per_string() -> None:
    ev = RandomGuoEvaluator()
    a = ev.encode_text_from_strings(["a person walks", "a person runs"])
    b = ev.encode_text_from_strings(["a person walks", "a person runs"])
    assert torch.allclose(a, b)


# ---------------------------------------------------------------------------
# Upstream length-ordering (the tie-group trap)
# ---------------------------------------------------------------------------


def test_upstream_align_indices_round_trip_under_ties() -> None:
    # Eval caps long clips at a common length, so ties are the norm. Reversed
    # argsort flips each tie group, so treating the permutation as a no-op
    # mis-pairs tied motions with other clips' captions.
    m_lens = torch.tensor([195, 195, 195, 195, 120, 80, 80, 40])
    align, inv = _upstream_align_indices(m_lens)

    assert not np.array_equal(align, np.arange(m_lens.numel())), (
        "reversed-argsort must not be assumed to be the identity under ties"
    )
    # lengths must land descending — pack_padded_sequence requires it
    assert np.all(np.diff(m_lens.numpy()[align]) <= 0)
    # applying inv to align-ordered rows restores the caller's original order
    rows = np.arange(m_lens.numel())
    assert np.array_equal(rows[align][inv], rows)


def test_upstream_align_indices_is_identity_without_ties() -> None:
    # All-distinct lengths are why the tie bug hid: here the permutation really
    # is a no-op, so any test using distinct lengths passes either way.
    m_lens = torch.tensor([200, 150, 100, 50])
    align, inv = _upstream_align_indices(m_lens)
    assert np.array_equal(align, np.arange(m_lens.numel()))
    assert np.array_equal(inv, np.arange(m_lens.numel()))


def test_resolve_explicit_evaluator_assets_keeps_checkpoint_and_stats_paired(tmp_path) -> None:
    repo = tmp_path / "text-to-motion"
    checkpoint_root = tmp_path / "momask-official"
    checkpoint = checkpoint_root / "t2m" / "text_mot_match" / "model" / "finest.tar"
    meta = checkpoint_root / "t2m" / "Comp_v6_KLD005" / "meta"
    checkpoint.parent.mkdir(parents=True)
    meta.mkdir(parents=True)
    checkpoint.touch()
    (meta / "mean.npy").touch()
    (meta / "std.npy").touch()

    root, mean, std = _resolve_evaluator_assets(
        text_to_motion_repo=repo,
        humanml3d_repo=tmp_path / "HumanML3D",
        checkpoints_dir=checkpoint_root,
        normalization_name="Comp_v6_KLD005",
    )

    assert root == checkpoint_root.resolve()
    assert mean == meta / "mean.npy"
    assert std == meta / "std.npy"


def test_resolve_explicit_evaluator_assets_does_not_fallback(tmp_path) -> None:
    checkpoint_root = tmp_path / "momask-official"
    checkpoint = checkpoint_root / "t2m" / "text_mot_match" / "model" / "finest.tar"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    h3d = tmp_path / "HumanML3D" / "HumanML3D"
    h3d.mkdir(parents=True)
    (h3d / "Mean.npy").touch()
    (h3d / "Std.npy").touch()

    with pytest.raises(FileNotFoundError, match="Comp_v6_KLD005"):
        _resolve_evaluator_assets(
            text_to_motion_repo=tmp_path / "text-to-motion",
            humanml3d_repo=tmp_path / "HumanML3D",
            checkpoints_dir=checkpoint_root,
            normalization_name="Comp_v6_KLD005",
        )
