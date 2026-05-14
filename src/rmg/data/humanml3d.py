"""HumanML3D dataset reader (RMG packed format).

The packed dataset is a single zip of per-clip `.pt` files plus a small index.
This avoids unpacking 14k+ files (the cluster has a 100k-file quota per
folder). Layout:

    <data.root>/
        humanml3d.zip       # zip of per-clip <clip_id>.pt
        splits.json         # {"train": [...], "val": [...], "test": [...]}
        target_offsets.pt   # (22, 3) reference T-pose offsets (used by FK / H3D)
        meta.json           # {"fps": 20, "num_clips": ..., "version": "..."}

Each `<clip_id>.pt` (loaded via `torch.load(BytesIO(zip.read(...)))`) is:

    {
        "translation": Tensor(T, 3),     # root position in HumanML3D coords (Y-up, post-trans_matrix)
        "quats":       Tensor(T, 22, 4), # per-joint local quaternions [w, x, y, z]
        "texts":       list[str],        # ≥1 captions
    }

Two layouts are supported via `data.layout`:
  - submodule_zip: zip lives in this repo (default while AMASS isn't on the cluster)
  - mounted_zip : zip is exposed at `data.root` by `/mnt/datasets/tools/mount_dataset.py`
Both behave identically — `data.root` is the only knob the dataset class sees.
"""

from __future__ import annotations

import io
import json
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..representation import (
    NUM_JOINTS,
    Representation,
    Skeleton,
    TPlusR,
    TRRepresentation,
    encode,
    make_continuous,
    normalize_quaternions,
)


# ---------------------------------------------------------------------------
# Mirror augmentation
# ---------------------------------------------------------------------------

# SMPL 22-joint left/right pairs (HumanML3D-standard mirror).
LR_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 2),    # L_Hip / R_Hip
    (4, 5),    # L_Knee / R_Knee
    (7, 8),    # L_Ankle / R_Ankle
    (10, 11),  # L_Foot / R_Foot
    (13, 14),  # L_Collar / R_Collar
    (16, 17),  # L_Shoulder / R_Shoulder
    (18, 19),  # L_Elbow / R_Elbow
    (20, 21),  # L_Wrist / R_Wrist
)


def mirror_motion(translation: Tensor, quats: Tensor) -> tuple[Tensor, Tensor]:
    """Mirror an SMPL motion across the YZ plane (X-flip).

    Effects:
      - translation: x → -x
      - per-quaternion: (w, x, y, z) → (w, x, -y, -z)
        (Reflection across YZ conjugates a rotation by diag(-1, 1, 1); on the
         quaternion this negates the y and z components.)
      - swap left/right joint indices.
    """
    t = translation.clone()
    t[..., 0] = -t[..., 0]

    q = quats.clone()
    q[..., 2] = -q[..., 2]
    q[..., 3] = -q[..., 3]

    for l, r in LR_PAIRS:
        q[..., [l, r], :] = q[..., [r, l], :].clone()

    return t, q


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


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
                "`scripts/prepare_humanml3d.py` to build it from AMASS, or "
                "set `data.root` to a directory that contains it."
            )

        with open(self.root / splits_name) as f:
            splits = json.load(f)
        self.clip_ids: list[str] = list(splits[split])

        # Per-worker zip handle (lazily opened on first __getitem__).
        self._zip: zipfile.ZipFile | None = None

        # Reference offsets (built once; needed by representations that go
        # through forward kinematics, e.g. T+P / T+R+P for the pre-shape factor).
        offs_path = self.root / offsets_name
        self.target_offsets: Tensor | None = (
            torch.load(offs_path, weights_only=True) if offs_path.exists() else None
        )
        self._skeleton: Skeleton | None = (
            Skeleton(offsets=self.target_offsets) if self.target_offsets is not None else None
        )

        # Representation defaults to T+R (paper main result) so existing
        # callers that pass nothing keep working.
        self.representation: Representation = representation or TRRepresentation()

    # ------------------------------------------------------------------ utils

    def _open_zip(self) -> zipfile.ZipFile:
        if self._zip is None:
            # 'r' is fine for concurrent readers from multiple workers, since
            # each Dataset replica opens its own handle.
            self._zip = zipfile.ZipFile(self.zip_path, mode="r")
        return self._zip

    def __len__(self) -> int:
        return len(self.clip_ids)

    def __getitem__(self, idx: int) -> HumanML3DSample:
        clip_id = self.clip_ids[idx]
        zf = self._open_zip()
        try:
            raw = zf.read(f"{clip_id}.pt")
        except KeyError as e:
            raise KeyError(f"clip {clip_id!r} listed in splits but not in zip") from e
        # weights_only=False because the blob is a dict with both tensors AND
        # a list[str] for texts; torch>=2.6's strict weights_only mode rejects
        # the tensor-storage persistent-ids in mixed blobs. Safe here because
        # we produce these files ourselves in `prepare_humanml3d.py pack`.
        blob = torch.load(io.BytesIO(raw), weights_only=False)
        translation: Tensor = blob["translation"]    # (T, 3) float
        quats: Tensor = blob["quats"]                # (T, 22, 4)
        texts: list[str] = blob["texts"]

        # Random crop if longer than max_seq_len
        T = translation.shape[0]
        if T > self.max_seq_len:
            start = random.randint(0, T - self.max_seq_len)
            translation = translation[start : start + self.max_seq_len]
            quats = quats[start : start + self.max_seq_len]
            T = self.max_seq_len

        if T < self.min_seq_len:
            # Should not happen if the dataset was built correctly; skip via
            # next index. Tests build clips long enough.
            return self.__getitem__((idx + 1) % len(self))

        # Mirror augmentation (train only)
        if self.mirror_augment and random.random() < 0.5:
            translation, quats = mirror_motion(translation, quats)

        # Normalize + temporal sign continuity, then encode via the chosen
        # Representation (T+R for the paper's main result; T+P / T+R+P for ablations).
        quats = normalize_quaternions(quats)
        quats = make_continuous(quats, time_dim=0)
        x1 = self.representation.encode_clip(translation, quats, skeleton=self._skeleton)

        text = random.choice(texts)
        return HumanML3DSample(x1=x1.float(), text=text, length=T, clip_id=clip_id)


# ---------------------------------------------------------------------------
# Collate: pad to max-T-in-batch, build a boolean mask
# ---------------------------------------------------------------------------


@dataclass
class CollatedBatch:
    x1: Tensor          # (B, T, D)
    mask: Tensor        # (B, T) bool, True = valid
    texts: list[str]
    lengths: Tensor     # (B,) long
    clip_ids: list[str]


def collate(samples: list[HumanML3DSample]) -> CollatedBatch:
    B = len(samples)
    Tmax = max(s.length for s in samples)
    D = samples[0].x1.shape[-1]
    x1 = torch.zeros(B, Tmax, D, dtype=samples[0].x1.dtype)
    mask = torch.zeros(B, Tmax, dtype=torch.bool)
    lengths = torch.zeros(B, dtype=torch.long)
    texts: list[str] = []
    clip_ids: list[str] = []
    for i, s in enumerate(samples):
        x1[i, : s.length] = s.x1
        mask[i, : s.length] = True
        lengths[i] = s.length
        texts.append(s.text)
        clip_ids.append(s.clip_id)
    return CollatedBatch(x1=x1, mask=mask, texts=texts, lengths=lengths, clip_ids=clip_ids)
