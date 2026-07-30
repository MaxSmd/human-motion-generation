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

    Implementation note: we only need *trace* of the matrix square root, not
    the matrix itself. For PSD Σ1, Σ2 the eigenvalues of Σ1·Σ2 are real and
    non-negative (it's similar to the symmetric PSD Σ1^{1/2}·Σ2·Σ1^{1/2}), so
    tr((Σ1·Σ2)^{1/2}) = Σ √λᵢ(Σ1·Σ2). This is O(n³) but bounded — `eigvals`
    is one LAPACK call (≈50 ms for n=512), unlike `scipy.linalg.sqrtm` whose
    iterative Schur algorithm can grind for *hours* on ill-conditioned products
    (which is exactly what we get when the model's generated-feature
    distribution sits outside the evaluator's training support).
    """
    diff = mu1 - mu2
    prod = sigma1 @ sigma2
    # Small ridge if needed for numerical sanity; cheap insurance.
    eigvals = np.linalg.eigvals(prod)
    real_eigs = eigvals.real
    if not np.isfinite(real_eigs).all() or real_eigs.min() < -eps:
        # Retry with a tiny identity offset on both covariances.
        offset = np.eye(sigma1.shape[0]) * eps
        eigvals = np.linalg.eigvals((sigma1 + offset) @ (sigma2 + offset))
        real_eigs = eigvals.real
    sqrt_eigs = np.sqrt(np.clip(real_eigs, 0.0, None))
    tr_covmean = float(sqrt_eigs.sum())
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)


def _drop_bad_rows(name: str, x: NDArray) -> NDArray:
    """Drop rows containing NaN/inf. Reports if any were removed.

    Generated motions sometimes contain degenerate bones (zero-length →
    NaN in upstream IK's `v / ||v||`), which propagates into the Guo
    motion-encoder embeddings. Without this guard, downstream `eigvals` /
    `cov` hit NaN and either crash or hang.
    """
    bad = ~np.isfinite(x).all(axis=-1)
    if bad.any():
        print(
            f"[fid] WARN: dropping {int(bad.sum())}/{x.shape[0]} "
            f"{name} rows with NaN/inf",
            flush=True,
        )
    return x[~bad]


def crop_to_unit_length(
    feats, rng: np.random.Generator, unit: int = 4, jitter: bool = True
):
    """Crop a (T, 263) feature clip to a multiple of `unit` frames, mirroring
    upstream text-to-motion's eval protocol ("Crop the motions in to times of
    4, and introduce small variations", data/dataset.py). The Guo movement
    encoder is a stride-`unit` conv whose output the motion GRU reads in
    m_length//unit steps — non-multiple lengths misalign the co-embedding and
    depress R-precision/mm_dist. With `jitter`, ~1/3 of clips lose one extra
    unit and the crop offset is random (upstream's 'double' coin + random
    start); without, the crop is deterministic (first m frames)."""
    L = int(feats.shape[0])
    m = (L // unit) * unit
    if jitter and m > unit and rng.random() < (1.0 / 3.0):
        m -= unit
    if m <= 0:
        return feats
    off = int(rng.integers(0, L - m + 1)) if jitter else 0
    return feats[off:off + m]


def fid(real_features: NDArray, gen_features: NDArray) -> float:
    """Convenience: FID from raw (N, D) feature matrices."""
    real_features = _drop_bad_rows("real", real_features)
    gen_features = _drop_bad_rows("gen", gen_features)
    if len(real_features) < 2 or len(gen_features) < 2:
        return float("nan")
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
    # Drop paired rows where *either* side has NaN/inf so the pairing stays valid.
    good = np.isfinite(text_features).all(-1) & np.isfinite(motion_features).all(-1)
    if not good.all():
        print(f"[r_precision] WARN: dropping {int((~good).sum())}/{good.size} bad pairs", flush=True)
        text_features = text_features[good]
        motion_features = motion_features[good]
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
    good = np.isfinite(text_features).all(-1) & np.isfinite(motion_features).all(-1)
    if not good.all():
        text_features = text_features[good]
        motion_features = motion_features[good]
    if len(text_features) == 0:
        return float("nan")
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
    motion_features = motion_features[np.isfinite(motion_features).all(-1)]
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
