"""MARDM dataset: 67-D essential features over the packed HumanML3D dataset.

Thin wrapper around `rmg`'s `HumanML3DDataset` — reusing its zip reader, split
handling, random crop, and mirror augmentation — but with an
`EssentialRepresentation` so each sample's `x1` is the 67-D essential feature.

Two roles, selected by `window_size`:
  * `window_size=None` → full variable-length clips (generation branch);
  * `window_size=W`    → a random W-frame window per clip (AutoEncoder stage,
    mirroring upstream MARDM's fixed-window AE training).

The 67-D feature is one frame shorter than the input clip (velocity diff), so
this wrapper reports `length` from the *encoded* feature rather than the raw
frame count — fixing the off-by-one that `rmg`'s dataset would otherwise carry.
"""

from __future__ import annotations

import random
from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset

from rmg.data.humanml3d import HumanML3DDataset, HumanML3DSample

from ..representation import EssentialRepresentation


class EssentialDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        *,
        mean: Tensor | None = None,
        std: Tensor | None = None,
        window_size: int | None = None,
        mirror_augment: bool = True,
        max_seq_len: int = 196,
        min_seq_len: int = 40,
        subset_frac: float | None = None,
        limit_clips: int | None = None,
        zip_name: str = "humanml3d.zip",
        splits_name: str = "splits.json",
        offsets_name: str = "target_offsets.pt",
    ) -> None:
        self.window_size = window_size
        self._rep = EssentialRepresentation(mean=mean, std=std)
        self.inner = HumanML3DDataset(
            root=root,
            split=split,
            max_seq_len=max_seq_len,
            min_seq_len=min_seq_len,
            mirror_augment=mirror_augment,
            zip_name=zip_name,
            splits_name=splits_name,
            offsets_name=offsets_name,
            representation=self._rep,
        )
        # Deterministic subset (overfit smoke / debugging). subset_frac takes
        # precedence; both keep the first N clip ids so runs are reproducible.
        n_full = len(self.inner.clip_ids)
        if subset_frac is not None:
            n = max(1, round(subset_frac * n_full))
            self.inner.clip_ids = self.inner.clip_ids[:n]
        elif limit_clips is not None:
            self.inner.clip_ids = self.inner.clip_ids[: max(1, limit_clips)]

    def set_stats(self, mean: Tensor, std: Tensor) -> None:
        self._rep.set_stats(mean, std)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> HumanML3DSample:
        s = self.inner[idx]
        x = s.x1                       # (L, 67), L = (cropped T) - 1
        length = x.shape[0]
        if self.window_size is not None:
            if length < self.window_size:
                return self.__getitem__((idx + 1) % len(self))
            start = random.randint(0, length - self.window_size)
            x = x[start : start + self.window_size]
            length = self.window_size
        return HumanML3DSample(x1=x, text=s.text, length=length, clip_id=s.clip_id)
