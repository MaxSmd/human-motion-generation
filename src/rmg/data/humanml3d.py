"""HumanML3D dataset reader for rmg and MoMask-compatible 263-D features.

The packed format and loading / subset / mirror / crop / pad logic live in
`shared.data.humanml3d`. This module only provides rmg's default T+R
representation and a compatibility `output_mode="h3d_263"` path used by
MoMask.
"""

from __future__ import annotations

import json
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from shared.data import (
    ClipRepresentation,
    CollatedBatch,
    HumanML3DSample,
    collate,
    read_clip,
    select_clip_ids,
)
from shared.data import HumanML3DDataset as _BaseHumanML3DDataset
from shared.geometry import Skeleton

from ..representation import H3D_FEATURE_DIM, TRRepresentation, tplusr_to_h3d_features_with_quats

__all__ = ["HumanML3DDataset", "CanonicalHumanML3DDataset", "HumanML3DSample", "CollatedBatch", "collate"]


class _HumanML3D263Representation:
    """Adapter that converts packed T+R clips into standard HumanML3D 263-D features."""

    def encode_clip(self, translation: Tensor, quaternions: Tensor, skeleton: Skeleton | None = None) -> Tensor:
        if skeleton is None:
            raise FileNotFoundError(
                "output_mode='h3d_263' requires target_offsets.pt so clips can be "
                "converted into standard HumanML3D 263-D features."
            )
        x = tplusr_to_h3d_features_with_quats(translation, quaternions, skeleton)
        if x.shape[-1] != H3D_FEATURE_DIM:
            raise AssertionError(f"expected {H3D_FEATURE_DIM}-D H3D features, got {x.shape[-1]}")
        return x


class HumanML3DDataset(_BaseHumanML3DDataset):
    """Shared packed loader with rmg's T+R representation as the default.

    Set `output_mode="h3d_263"` for token-based baselines such as MoMask.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        max_seq_len: int = 196,
        min_seq_len: int = 40,
        mirror_augment: bool = False,
        zip_name: str = "humanml3d.zip",
        splits_name: str = "splits.json",
        offsets_name: str = "target_offsets.pt",
        representation: ClipRepresentation | None = None,
        output_mode: str = "representation",
        subset_fraction: float = 1.0,
        subset_seed: int = 0,
        subset_n: int = 0,
    ) -> None:
        if output_mode not in ("representation", "h3d_263"):
            raise ValueError("output_mode must be 'representation' or 'h3d_263'")
        if output_mode == "h3d_263":
            if representation is not None:
                raise ValueError("pass either representation or output_mode='h3d_263', not both")
            representation = _HumanML3D263Representation()
        else:
            representation = representation or TRRepresentation()

        super().__init__(
            root,
            split=split,
            max_seq_len=max_seq_len,
            min_seq_len=min_seq_len,
            zip_name=zip_name,
            splits_name=splits_name,
            offsets_name=offsets_name,
            mirror_augment=mirror_augment,
            representation=representation,
            subset_fraction=subset_fraction,
            subset_seed=subset_seed,
            subset_n=subset_n,
        )


class CanonicalHumanML3DDataset(Dataset):
    """HumanML3D loader whose motion features come from canonical new_joint_vecs.

    The packed dataset is still used for split ids and captions, but `x1` is
    loaded directly from `canonical_dir/<clip_id>.npy`, i.e. the official 263-D
    feature distribution used by the Guo evaluator.
    """

    def __init__(
        self,
        root: str | Path,
        canonical_dir: str | Path,
        split: str = "train",
        max_seq_len: int = 196,
        min_seq_len: int = 40,
        zip_name: str = "humanml3d.zip",
        splits_name: str = "splits.json",
        subset_fraction: float = 1.0,
        subset_seed: int = 0,
        subset_n: int = 0,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be one of train/val/test, got {split}")
        self.root = Path(root)
        self.canonical_dir = Path(canonical_dir)
        self.zip_path = self.root / zip_name
        self.split = split
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        if not self.zip_path.exists():
            raise FileNotFoundError(f"packed HumanML3D zip not found: {self.zip_path}")
        if not self.canonical_dir.exists():
            raise FileNotFoundError(f"canonical new_joint_vecs dir not found: {self.canonical_dir}")
        with open(self.root / splits_name) as f:
            splits = json.load(f)
        self.clip_ids = select_clip_ids(
            list(splits[split]),
            split,
            subset_n=subset_n,
            subset_fraction=subset_fraction,
            subset_seed=subset_seed,
        )
        self._zip: zipfile.ZipFile | None = None

    def _open_zip(self) -> zipfile.ZipFile:
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.zip_path, mode="r")
        return self._zip

    def __len__(self) -> int:
        return len(self.clip_ids)

    def _motion_path(self, clip_id: str) -> Path:
        path = self.canonical_dir / f"{clip_id}.npy"
        if not path.exists():
            raise FileNotFoundError(f"canonical HumanML3D feature not found for {clip_id} in {self.canonical_dir}")
        return path

    def __getitem__(self, idx: int) -> HumanML3DSample:
        for offset in range(len(self.clip_ids)):
            clip_id = self.clip_ids[(idx + offset) % len(self.clip_ids)]
            path = self._motion_path(clip_id)
            arr = np.load(path).astype(np.float32)
            if arr.ndim != 2 or arr.shape[1] != H3D_FEATURE_DIM:
                raise ValueError(f"{path} must have shape (T, {H3D_FEATURE_DIM}), got {arr.shape}")
            if arr.shape[0] > self.max_seq_len:
                start = random.randint(0, arr.shape[0] - self.max_seq_len)
                arr = arr[start : start + self.max_seq_len]
            if arr.shape[0] < self.min_seq_len:
                continue
            _translation, _quats, texts = read_clip(self._open_zip(), clip_id)
            return HumanML3DSample(
                x1=torch.from_numpy(arr).float(),
                text=random.choice(texts),
                length=arr.shape[0],
                clip_id=clip_id,
            )
        raise RuntimeError(f"no clips with length >= {self.min_seq_len} in split={self.split!r}")
