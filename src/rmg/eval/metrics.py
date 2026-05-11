"""HumanML3D-standard evaluation metrics.

Identical formulas to the upstream `text-to-motion/utils/metrics.py`:
  - FID (Fréchet distance between feature distributions)
  - R@1, R@2, R@3 (top-k text-motion retrieval accuracy)
  - MM-Dist (mean Euclidean distance between paired text/motion features)
  - Diversity (pairwise distance among random motion-feature pairs)
  - MultiModality (within-text variability of generated motions)

All inputs are *features* (e.g. 512-d outputs of the Guo et al. encoders).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import linalg


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------


def calculate_activation_statistics(features: NDArray) -> tuple[NDArray, NDArray]:
    """Return (mean, covariance) for an (N, D) feature matrix."""
    mu = features.mean(axis=0)
    cov = np.cov(features, rowvar=False)
    return mu, cov


def calculate_fid(
    mu1: NDArray, sigma1: NDArray, mu2: NDArray, sigma2: NDArray, eps: float = 1e-6
) -> float:
    """Fréchet Inception Distance between two Gaussian distributions.

    FID = ||μ1 - μ2||² + Tr(Σ1 + Σ2 - 2 (Σ1 Σ2)^{1/2})
    """
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if isinstance(covmean, tuple):                # older SciPy returned (M, errest)
        covmean = covmean[0]
    if not np.isfinite(covmean).all():
        # Numerical hack from FID reference implementation: add a small
        # multiple of identity to both covariances and retry.
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
        if isinstance(covmean, tuple):
            covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)


def fid(real_features: NDArray, gen_features: NDArray) -> float:
    """Convenience: FID from raw (N, D) feature matrices."""
    mu_r, sig_r = calculate_activation_statistics(real_features)
    mu_g, sig_g = calculate_activation_statistics(gen_features)
    return calculate_fid(mu_r, sig_r, mu_g, sig_g)


# ---------------------------------------------------------------------------
# R@k retrieval
# ---------------------------------------------------------------------------


def r_precision_batch(
    text_features: NDArray, motion_features: NDArray, top_k: int = 3
) -> NDArray:
    """For one batch of paired (text, motion), compute R@1..R@top_k.

    Inputs of shape (B, D); paired by index. Returns (top_k,) — sample-mean
    accuracies; element k is the fraction of texts whose paired motion ranks
    in the top (k+1).

    Uses Euclidean distance (matches HumanML3D upstream `euclidean_distance_matrix`).
    """
    B = text_features.shape[0]
    # Pairwise euclidean distance between texts and motions: (B, B)
    diff = text_features[:, None, :] - motion_features[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    # For each text i, rank motions by ascending distance. Position of motion i.
    order = np.argsort(dist, axis=1)  # (B, B)
    correct = (order == np.arange(B)[:, None])  # (B, B)
    # cumulative top-k: any of first k matches
    top = np.zeros(top_k, dtype=np.float64)
    for k in range(top_k):
        top[k] = correct[:, : k + 1].any(axis=1).mean()
    return top


def r_precision(
    text_features: NDArray,
    motion_features: NDArray,
    top_k: int = 3,
    batch_size: int = 32,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """HumanML3D R@k convention: split into random batches of `batch_size`,
    compute R@k per batch, average. Returns (top_k,)."""
    rng = rng or np.random.default_rng(0)
    N = text_features.shape[0]
    perm = rng.permutation(N)
    text_features = text_features[perm]
    motion_features = motion_features[perm]
    n_batches = N // batch_size
    if n_batches == 0:
        return r_precision_batch(text_features, motion_features, top_k)
    accs = np.zeros((n_batches, top_k), dtype=np.float64)
    for b in range(n_batches):
        s = slice(b * batch_size, (b + 1) * batch_size)
        accs[b] = r_precision_batch(text_features[s], motion_features[s], top_k)
    return accs.mean(axis=0)


# ---------------------------------------------------------------------------
# MM-Dist (matching distance between paired text + motion features)
# ---------------------------------------------------------------------------


def mm_distance(text_features: NDArray, motion_features: NDArray) -> float:
    return float(np.linalg.norm(text_features - motion_features, axis=-1).mean())


# ---------------------------------------------------------------------------
# Diversity (intra-set pairwise distance, random pairs)
# ---------------------------------------------------------------------------


def diversity(
    motion_features: NDArray,
    diversity_times: int = 300,
    rng: np.random.Generator | None = None,
) -> float:
    """Mean pairwise L2 distance among `diversity_times` random pairs."""
    rng = rng or np.random.default_rng(0)
    N = motion_features.shape[0]
    if N < 2:
        return 0.0
    n = min(diversity_times, N // 2)
    idx_a = rng.choice(N, n, replace=False)
    idx_b = rng.choice(N, n, replace=False)
    return float(np.linalg.norm(motion_features[idx_a] - motion_features[idx_b], axis=-1).mean())


# ---------------------------------------------------------------------------
# MultiModality (within-text diversity across multiple generations per text)
# ---------------------------------------------------------------------------


def multimodality(
    motion_features_per_text: NDArray,
    multimodality_times: int = 10,
    rng: np.random.Generator | None = None,
) -> float:
    """Per-text generated diversity.

    `motion_features_per_text` has shape (T, K, D) where K is the number of
    independent generations per text and T is the number of distinct texts.
    Returns mean pairwise L2 distance between two random generations per text,
    averaged over texts and `multimodality_times` random pair samples.
    """
    rng = rng or np.random.default_rng(0)
    T, K, _ = motion_features_per_text.shape
    if K < 2:
        return 0.0
    n = min(multimodality_times, K // 2)
    idx_a = rng.choice(K, n, replace=False)
    idx_b = rng.choice(K, n, replace=False)
    a = motion_features_per_text[:, idx_a]  # (T, n, D)
    b = motion_features_per_text[:, idx_b]
    return float(np.linalg.norm(a - b, axis=-1).mean())
