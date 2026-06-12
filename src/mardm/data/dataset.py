"""MARDM dataset: 67-D essential features over the packed HumanML3D dataset.

Thin wrapper around the shared `HumanML3DDataset` (`shared.data`) — reusing its
zip reader, split handling, random crop, and mirror augmentation — but with an
`EssentialRepresentation` so each sample's `x1` is the 67-D essential feature.

Two roles, selected by `window_size`:
  * `window_size=None` → full variable-length clips (generation branch);
  * `window_size=W`    → a random W-frame window per clip (AutoEncoder stage,
    mirroring upstream MARDM's fixed-window AE training).

The 67-D feature is one frame shorter than the input clip (velocity diff), so
this wrapper reports `length` from the *encoded* feature rather than the raw
frame count — fixing the off-by-one that the shared dataset would otherwise carry.

Set `preload=True` to encode every retained clip once at init and serve from a
RAM cache afterwards. Removes the encode pipeline (zip read + torch.load + quat
math + FK) from the dataloader hot path. Necessary on CAMP cluster runs where
the auto-cancel policy (<5% GPU util for ~2h) trips when the data pipeline is
the bottleneck. Memory cost: ~600 MB (or ~1.2 GB if mirror_augment is on, since
we cache both mirrored and unmirrored features).
"""

from __future__ import annotations

import io
import random
import zipfile
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from shared.data import HumanML3DDataset, HumanML3DSample, mirror_motion
from shared.geometry import make_continuous, normalize_quaternions

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
        preload: bool = False,
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

        self.preload = preload
        self._preload_cache: dict[str, dict] | None = None
        if preload:
            self._build_preload_cache()

    def _build_preload_cache(self) -> None:
        # Encode every retained clip once and stash (x1, x1_mirrored, texts).
        # __getitem__ becomes a RAM slice; no zip/torch.load/FK in the hot path.
        cache: dict[str, dict] = {}
        skeleton = self.inner._skeleton
        with zipfile.ZipFile(self.inner.zip_path, mode="r") as zf:
            for clip_id in self.inner.clip_ids:
                try:
                    raw = zf.read(f"{clip_id}.pt")
                except KeyError:
                    continue
                blob = torch.load(io.BytesIO(raw), weights_only=False)
                translation: Tensor = blob["translation"]
                quats: Tensor = blob["quats"]
                texts: list[str] = blob["texts"]

                q = make_continuous(normalize_quaternions(quats), time_dim=0)
                x1 = self._rep.encode_clip(translation, q, skeleton=skeleton).float()
                entry: dict = {"x1": x1, "texts": texts}

                if self.inner.mirror_augment:
                    tm, qm = mirror_motion(translation, quats)
                    qm = make_continuous(normalize_quaternions(qm), time_dim=0)
                    entry["x1_mirrored"] = self._rep.encode_clip(tm, qm, skeleton=skeleton).float()

                cache[clip_id] = entry
        self._preload_cache = cache

    def set_stats(self, mean: Tensor, std: Tensor) -> None:
        # Preload caches the *normalized* features; changing stats after the
        # fact would silently produce stale samples.
        if self.preload and self._preload_cache is not None:
            raise RuntimeError(
                "set_stats() called after preload cache was built; stats must be "
                "fixed before EssentialDataset(preload=True) construction."
            )
        self._rep.set_stats(mean, std)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> HumanML3DSample:
        if self.preload and self._preload_cache is not None:
            return self._getitem_preloaded(idx)
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

    def _getitem_preloaded(self, idx: int) -> HumanML3DSample:
        clip_id = self.inner.clip_ids[idx]
        entry = self._preload_cache[clip_id]

        if self.inner.mirror_augment and "x1_mirrored" in entry and random.random() < 0.5:
            x = entry["x1_mirrored"]
        else:
            x = entry["x1"]

        # max_seq_len crop. Original cropped the clip then encoded → max_seq_len-1
        # features; slicing the full encoded features [start:start+max_seq_len-1]
        # gives the same result (feature[i] depends only on frames i, i+1).
        max_feat_len = self.inner.max_seq_len - 1
        if x.shape[0] > max_feat_len:
            s = random.randint(0, x.shape[0] - max_feat_len)
            x = x[s : s + max_feat_len]

        if x.shape[0] < self.inner.min_seq_len:
            return self.__getitem__((idx + 1) % len(self))

        length = x.shape[0]
        if self.window_size is not None:
            if length < self.window_size:
                return self.__getitem__((idx + 1) % len(self))
            s = random.randint(0, length - self.window_size)
            x = x[s : s + self.window_size]
            length = self.window_size

        text = random.choice(entry["texts"])
        return HumanML3DSample(x1=x, text=text, length=length, clip_id=clip_id)
