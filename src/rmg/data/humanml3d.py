"""HumanML3D dataset reader for rmg — composes the shared dataset machinery
(`shared.data`) and adds rmg's manifold encoding.

The packed format and all loading / subset / mirror / crop / pad logic are shared
(see `shared.data.humanml3d`). The only rmg-specific step is encoding each clip's
(translation, quaternions) into the flat T+R manifold tensor `x1` the model trains
on; other models compose the same helpers with their own encode.
"""

from __future__ import annotations

import json
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from shared.data import (
    mirror_motion,
    pad_batch,
    random_crop,
    read_clip,
    select_clip_ids,
)

from ..representation import (
    Representation,
    Skeleton,
    TRRepresentation,
    make_continuous,
    normalize_quaternions,
)


@dataclass
class HumanML3DSample:
    x1: Tensor          # (T, D=91) flat T+R encoding on the manifold
    text: str           # one caption sampled uniformly from the clip's pool
    length: int         # T (true sequence length, before padding)
    clip_id: str


class HumanML3DDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        max_seq_len: int = 196,
        min_seq_len: int = 40,
        mirror_augment: bool = True,
        zip_name: str = "humanml3d.zip",
        splits_name: str = "splits.json",
        offsets_name: str = "target_offsets.pt",
        representation: Representation | None = None,
        subset_fraction: float = 1.0,
        subset_seed: int = 0,
        subset_n: int = 0,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be one of train/val/test, got {split}")
        self.root = Path(root)
        self.zip_path = self.root / zip_name
        self.split = split
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.mirror_augment = mirror_augment and (split == "train")

        if not self.zip_path.exists():
            raise FileNotFoundError(
                f"Packed dataset zip not found at {self.zip_path}. Run "
                "`python -m shared.data.prepare_humanml3d` to build it from AMASS, or "
                "set `data.root` to a directory that contains it."
            )

        with open(self.root / splits_name) as f:
            splits = json.load(f)
        self.clip_ids: list[str] = select_clip_ids(
            list(splits[split]), split,
            subset_n=subset_n, subset_fraction=subset_fraction, subset_seed=subset_seed,
        )

        # Per-worker zip handle (lazily opened on first __getitem__).
        self._zip: zipfile.ZipFile | None = None

        # Reference offsets (built once; needed by representations that go through
        # forward kinematics, e.g. T+P / T+R+P for the pre-shape factor).
        offs_path = self.root / offsets_name
        self.target_offsets: Tensor | None = (
            torch.load(offs_path, weights_only=True) if offs_path.exists() else None
        )
        self._skeleton: Skeleton | None = (
            Skeleton(offsets=self.target_offsets) if self.target_offsets is not None else None
        )

        # Representation defaults to T+R (paper main result).
        self.representation: Representation = representation or TRRepresentation()

    def _open_zip(self) -> zipfile.ZipFile:
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.zip_path, mode="r")
        return self._zip

    def __len__(self) -> int:
        return len(self.clip_ids)

    def __getitem__(self, idx: int) -> HumanML3DSample:
        clip_id = self.clip_ids[idx]
        translation, quats, texts = read_clip(self._open_zip(), clip_id)

        translation, quats, T = random_crop(translation, quats, self.max_seq_len)
        if T < self.min_seq_len:
            # Should not happen for a well-built dataset; skip to the next index.
            return self.__getitem__((idx + 1) % len(self))

        if self.mirror_augment and random.random() < 0.5:
            translation, quats = mirror_motion(translation, quats)

        # Normalize + temporal sign continuity, then encode via the chosen
        # Representation (T+R main result; T+P / T+R+P for ablations).
        quats = normalize_quaternions(quats)
        quats = make_continuous(quats, time_dim=0)
        x1 = self.representation.encode_clip(translation, quats, skeleton=self._skeleton)

        return HumanML3DSample(x1=x1.float(), text=random.choice(texts), length=T, clip_id=clip_id)


@dataclass
class CollatedBatch:
    x1: Tensor          # (B, T, D)
    mask: Tensor        # (B, T) bool, True = valid
    texts: list[str]
    lengths: Tensor     # (B,) long
    clip_ids: list[str]


def collate(samples: list[HumanML3DSample]) -> CollatedBatch:
    x1, mask, lengths = pad_batch([s.x1 for s in samples])
    return CollatedBatch(
        x1=x1, mask=mask,
        texts=[s.text for s in samples],
        lengths=lengths,
        clip_ids=[s.clip_id for s in samples],
    )
