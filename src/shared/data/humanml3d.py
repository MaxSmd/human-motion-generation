"""Model-agnostic HumanML3D packed-dataset machinery.

Every model trains on the same packed dataset (built by
`shared.data.prepare_humanml3d`): one zip of per-clip `.pt` blobs

    {"translation": (T,3), "quats": (T,22,4), "texts": list[str]}

plus `splits.json` and `target_offsets.pt`. The loading / augmentation logic
here is shared; each model composes it and supplies its own *encode* step (rmg →
manifold T+R; momask/mardm → 263-D features). Keep new shared dataset code here,
not duplicated per model.
"""

from __future__ import annotations

import io
import random
import zipfile

import torch
from torch import Tensor

# SMPL 22-joint left/right pairs (HumanML3D-standard mirror).
LR_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21),
)


def mirror_motion(translation: Tensor, quats: Tensor) -> tuple[Tensor, Tensor]:
    """Mirror an SMPL motion across the YZ plane (X-flip): translation x→-x;
    per-quaternion (w,x,y,z)→(w,x,-y,-z); swap left/right joint indices."""
    t = translation.clone()
    t[..., 0] = -t[..., 0]
    q = quats.clone()
    q[..., 2] = -q[..., 2]
    q[..., 3] = -q[..., 3]
    for l, r in LR_PAIRS:
        q[..., [l, r], :] = q[..., [r, l], :].clone()
    return t, q


def select_clip_ids(
    clip_ids: list[str], split: str, *,
    subset_n: int = 0, subset_fraction: float = 1.0, subset_seed: int = 0,
) -> list[str]:
    """Deterministic train-subset selection (non-train splits returned as-is).

    `subset_n` keeps exactly N regular clips, no mirrors (tiny-overfit checks).
    `subset_fraction` keeps a fraction of regular clips *plus their `M`-mirrors*.
    Same seed → same clips, so methods compare on identical data.
    """
    if split != "train":
        return list(clip_ids)
    if subset_n > 0:
        rng = random.Random(subset_seed)
        regular = [c for c in clip_ids if not c.startswith("M")]
        ids = sorted(rng.sample(regular, k=min(subset_n, len(regular))))
        print(f"[HumanML3DDataset] subset_n={subset_n} subset_seed={subset_seed} "
              f"→ {len(ids)} train clips (exact, no mirrors): {ids}", flush=True)
        return ids
    if subset_fraction < 1.0:
        rng = random.Random(subset_seed)
        regular = [c for c in clip_ids if not c.startswith("M")]
        n_keep_reg = max(1, int(round(len(regular) * subset_fraction)))
        keep_reg = set(rng.sample(regular, k=n_keep_reg))
        all_keep = keep_reg | {f"M{c}" for c in keep_reg}
        ids = sorted(c for c in clip_ids if c in all_keep)
        print(f"[HumanML3DDataset] subset_fraction={subset_fraction} subset_seed={subset_seed} "
              f"→ {len(ids)} train clips ({n_keep_reg} regular + their mirrors)", flush=True)
        return ids
    return list(clip_ids)


def read_clip(zf: zipfile.ZipFile, clip_id: str) -> tuple[Tensor, Tensor, list[str]]:
    """Read a packed clip → (translation (T,3), quats (T,22,4), texts)."""
    try:
        raw = zf.read(f"{clip_id}.pt")
    except KeyError as e:
        raise KeyError(f"clip {clip_id!r} listed in splits but not in zip") from e
    # weights_only=False: blob mixes tensors + list[str]; we produce these files ourselves.
    blob = torch.load(io.BytesIO(raw), weights_only=False)
    return blob["translation"], blob["quats"], blob["texts"]


def random_crop(translation: Tensor, quats: Tensor, max_seq_len: int) -> tuple[Tensor, Tensor, int]:
    """Random temporal crop to at most `max_seq_len`. Returns (translation, quats, T)."""
    T = translation.shape[0]
    if T > max_seq_len:
        start = random.randint(0, T - max_seq_len)
        translation = translation[start : start + max_seq_len]
        quats = quats[start : start + max_seq_len]
        T = max_seq_len
    return translation, quats, T


def pad_batch(xs: list[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    """Pad a list of (T_i, D) tensors → (x (B,Tmax,D), mask (B,Tmax) bool, lengths (B,))."""
    B = len(xs)
    Tmax = max(x.shape[0] for x in xs)
    D = xs[0].shape[-1]
    out = torch.zeros(B, Tmax, D, dtype=xs[0].dtype)
    mask = torch.zeros(B, Tmax, dtype=torch.bool)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, x in enumerate(xs):
        L = x.shape[0]
        out[i, :L] = x
        mask[i, :L] = True
        lengths[i] = L
    return out, mask, lengths
