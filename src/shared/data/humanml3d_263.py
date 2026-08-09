"""Model-agnostic loaders that yield standard 263-D HumanML3D features.

`shared.data.HumanML3DDataset` is representation-parameterised: each model
package supplies the encoding it wants (rmg → T+R on the manifold). Token-based
baselines such as MoMask instead want the *standard* 263-D HumanML3D feature
vector, so the two loaders here provide it without any model package involved:

- `H3D263Dataset` reads the packed clips and converts `{translation, quats}` to
  263-D features on the fly (our own conversion, `process_file` parity).
- `CanonicalHumanML3DDataset` skips the conversion and reads the official
  `new_joint_vecs/*.npy` directly — the exact feature distribution the Guo
  evaluator was trained on, so FID is comparable to published numbers.

Both live in `shared` because nothing here is specific to one model.
"""

from __future__ import annotations

import json
import random
import zipfile
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from shared.geometry import H3D_FEATURE_DIM, Skeleton, tplusr_to_h3d_features_with_quats

from .humanml3d import HumanML3DDataset, HumanML3DSample, read_clip, select_clip_ids

__all__ = [
    "H3D263Representation",
    "H3D263Dataset",
    "CanonicalHumanML3DDataset",
    "CanonicalHumanML3DWindowDataset",
]


class H3D263Representation:
    """`ClipRepresentation` adapter: packed T+R clip → standard 263-D features."""

    def encode_clip(self, translation: Tensor, quaternions: Tensor, skeleton: Skeleton | None = None) -> Tensor:
        if skeleton is None:
            raise FileNotFoundError(
                "263-D HumanML3D features require target_offsets.pt so packed clips "
                "can be converted; none was found next to the packed zip."
            )
        x = tplusr_to_h3d_features_with_quats(translation, quaternions, skeleton)
        if x.shape[-1] != H3D_FEATURE_DIM:
            raise AssertionError(f"expected {H3D_FEATURE_DIM}-D H3D features, got {x.shape[-1]}")
        return x


class H3D263Dataset(HumanML3DDataset):
    """Packed HumanML3D loader that yields `(T-1, 263)` standard H3D features.

    Same public signature as the shared base loader, with the representation
    fixed to `H3D263Representation()`. Used by token-based models (MoMask).
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
        subset_fraction: float = 1.0,
        subset_seed: int = 0,
        subset_n: int = 0,
    ) -> None:
        super().__init__(
            root,
            split=split,
            max_seq_len=max_seq_len,
            min_seq_len=min_seq_len,
            zip_name=zip_name,
            splits_name=splits_name,
            offsets_name=offsets_name,
            mirror_augment=mirror_augment,
            representation=H3D263Representation(),
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


class CanonicalHumanML3DWindowDataset(Dataset):
    """Fixed-window canonical HumanML3D features for RVQ-VAE training.

    MoMask trains its RVQ tokenizer on 64-frame motion windows rather than one
    item per full clip. This dataset indexes every valid sliding window, so
    longer motions contribute proportionally more snippets and every returned
    sample has exactly `window_size` frames.
    """

    def __init__(
        self,
        root: str | Path,
        canonical_dir: str | Path,
        split: str = "train",
        window_size: int = 64,
        window_stride: int = 1,
        splits_name: str = "splits.json",
        subset_fraction: float = 1.0,
        subset_seed: int = 0,
        subset_n: int = 0,
        preload: bool = True,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be one of train/val/test, got {split}")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if window_stride <= 0:
            raise ValueError("window_stride must be positive")
        self.root = Path(root)
        self.canonical_dir = Path(canonical_dir)
        self.split = split
        self.window_size = window_size
        self.window_stride = window_stride
        self.preload = preload
        if not self.canonical_dir.exists():
            raise FileNotFoundError(f"canonical new_joint_vecs dir not found: {self.canonical_dir}")
        with open(self.root / splits_name) as f:
            splits = json.load(f)
        split_ids = list(splits[split])
        effective_subset_n = subset_n if 0 < subset_n < len(split_ids) else 0
        clip_ids = select_clip_ids(
            split_ids,
            split,
            subset_n=effective_subset_n,
            subset_fraction=subset_fraction,
            subset_seed=subset_seed,
        )
        self.clip_ids: list[str] = []
        self._motions: list[np.ndarray | Path] = []
        self._window_ends: list[int] = []
        self.num_skipped_short = 0
        total_windows = 0
        for clip_id in clip_ids:
            path = self.canonical_dir / f"{clip_id}.npy"
            if not path.exists():
                raise FileNotFoundError(f"canonical HumanML3D feature not found for {clip_id} in {self.canonical_dir}")
            arr = np.load(path, mmap_mode="r")
            if arr.ndim != 2 or arr.shape[1] != H3D_FEATURE_DIM:
                raise ValueError(f"{path} must have shape (T, {H3D_FEATURE_DIM}), got {arr.shape}")
            length = int(arr.shape[0])
            if length < window_size:
                self.num_skipped_short += 1
                continue
            count = (length - window_size) // window_stride + 1
            self.clip_ids.append(clip_id)
            self._motions.append(np.array(arr, dtype=np.float32, copy=True) if preload else path)
            total_windows += count
            self._window_ends.append(total_windows)
        self.num_clips = len(self.clip_ids)
        self.num_windows = total_windows
        if self.num_windows == 0:
            raise RuntimeError(
                f"no {window_size}-frame canonical windows found in split={split!r} under {self.canonical_dir}"
            )

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> HumanML3DSample:
        if idx < 0:
            idx += self.num_windows
        if idx < 0 or idx >= self.num_windows:
            raise IndexError(idx)
        motion_idx = bisect_right(self._window_ends, idx)
        previous_end = 0 if motion_idx == 0 else self._window_ends[motion_idx - 1]
        start = (idx - previous_end) * self.window_stride
        clip_id = self.clip_ids[motion_idx]
        stored = self._motions[motion_idx]
        arr = np.load(stored, mmap_mode="r") if isinstance(stored, Path) else stored
        window = np.array(arr[start : start + self.window_size], dtype=np.float32, copy=True)
        return HumanML3DSample(
            x1=torch.from_numpy(window).float(),
            text="",
            length=self.window_size,
            clip_id=f"{clip_id}:{start}",
        )
