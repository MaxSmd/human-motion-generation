"""HumanML3D test set for ProjFlow evaluation, built from our canonical data.

Follows upstream's evaluation protocol (external/ProjFlow utils/datasets.py,
`Text2MotionDataset(evaluation=True)`, itself Guo et al.'s):

  * every test clip with 40 ≤ frames < 200 is an entry; a caption with a
    #start#end tag (seconds, 20 fps) becomes its own sub-segment entry;
  * per draw: one random caption, length rounded down to a multiple of 4
    (with probability 1/3 one unit shorter), and a random crop of that length.

Joint positions are recover_from_ric(new_joint_vecs) of the *whole* clip, then
sliced — exactly how HumanML3D's `new_joints` relate to `new_joint_vecs`, so
this matches upstream's input without the rebuilt `new_joints` directory. The
same crop of `new_joint_vecs` is kept as the real-motion 263-D feature for the
Guo evaluator. Clips without a canonical feature file (the 171 mirrored
HumanAct12 clips, see src/projflow/reports/stage-a-notes.md) are skipped and
counted.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from shared.geometry.humanml3d_io import recover_joints_from_ric

FPS = 20
MIN_LEN, MAX_LEN_EXCL = 40, 200
UNIT_LENGTH = 4
MAX_MOTION_LENGTH = 196


@dataclass(frozen=True)
class Caption:
    text: str
    tokens: tuple[str, ...]   # HumanML3D word/POS tokens (with *_VIP tags)


@dataclass(frozen=True)
class Entry:
    name: str
    joints: np.ndarray        # (T, 22, 3) world positions
    vecs: np.ndarray          # (T, 263) canonical features
    captions: tuple[Caption, ...]


@dataclass(frozen=True)
class Draw:
    """One evaluation sample: a caption and a crop of an entry."""

    name: str
    caption: Caption
    length: int
    joints: np.ndarray        # (length, 22, 3)
    vecs: np.ndarray          # (length, 263)


def _parse_captions(lines: list[str]) -> list[tuple[Caption, float, float]]:
    out = []
    for line in lines:
        parts = line.strip().split("#")
        if len(parts) < 4:
            continue
        f_tag, to_tag = float(parts[2]), float(parts[3])
        f_tag = 0.0 if np.isnan(f_tag) else f_tag
        to_tag = 0.0 if np.isnan(to_tag) else to_tag
        out.append((Caption(parts[0], tuple(parts[1].split(" "))), f_tag, to_tag))
    return out


class ControlTestSet:
    def __init__(self, canonical_dir: str | Path, split_file: str | Path, texts_zip: str | Path):
        canonical_dir = Path(canonical_dir)
        ids = [line.strip() for line in Path(split_file).read_text().splitlines() if line.strip()]
        self.missing: list[str] = []
        entries: list[Entry] = []
        with zipfile.ZipFile(texts_zip) as zf:
            text_members = {Path(n).stem: n for n in zf.namelist() if n.endswith(".txt")}
            for clip_id in ids:
                path = canonical_dir / f"{clip_id}.npy"
                if not path.exists() or clip_id not in text_members:
                    self.missing.append(clip_id)
                    continue
                vecs = np.load(path).astype(np.float32)
                if not MIN_LEN <= len(vecs) < MAX_LEN_EXCL:
                    continue
                joints = recover_joints_from_ric(torch.from_numpy(vecs)).numpy()
                whole: list[Caption] = []
                lines = zf.read(text_members[clip_id]).decode("utf-8").splitlines()
                for k, (cap, f_tag, to_tag) in enumerate(_parse_captions(lines)):
                    if f_tag == 0.0 and to_tag == 0.0:
                        whole.append(cap)
                        continue
                    a, b = int(f_tag * FPS), int(to_tag * FPS)
                    if MIN_LEN <= len(vecs[a:b]) < MAX_LEN_EXCL:
                        entries.append(Entry(f"{clip_id}#{k}", joints[a:b], vecs[a:b], (cap,)))
                if whole:
                    entries.append(Entry(clip_id, joints, vecs, tuple(whole)))
        self.entries = sorted(entries, key=lambda e: len(e.vecs))

    def __len__(self) -> int:
        return len(self.entries)

    def draw(self, index: int, rng: np.random.Generator) -> Draw:
        entry = self.entries[index]
        caption = entry.captions[rng.integers(len(entry.captions))]
        n = len(entry.vecs)
        units = n // UNIT_LENGTH - (1 if rng.integers(3) == 2 else 0)   # 'double' with prob. 1/3
        length = units * UNIT_LENGTH
        start = int(rng.integers(0, n - length + 1))
        return Draw(entry.name, caption, length, entry.joints[start:start + length], entry.vecs[start:start + length])


class JointNormalizer:
    """Upstream's 22×3 z-normalisation (utils/22x3_mean_std/t2m)."""

    def __init__(self, stats_dir: str | Path):
        stats_dir = Path(stats_dir)
        self.mean = torch.from_numpy(np.load(stats_dir / "22x3_mean.npy")).float()   # (22, 3)
        self.std = torch.from_numpy(np.load(stats_dir / "22x3_std.npy")).float()

    def normalize(self, joints: Tensor) -> Tensor:
        return (joints - self.mean.to(joints)) / self.std.to(joints)

    def denormalize(self, x: Tensor) -> Tensor:
        return x * self.std.to(x) + self.mean.to(x)


def batch_draws(draws: list[Draw], normalizer: JointNormalizer,
                max_len: int = MAX_MOTION_LENGTH) -> tuple[Tensor, Tensor]:
    """-> control (B, 3, max_len, 22) normalised and zero-padded, lengths (B,)."""
    control = torch.zeros(len(draws), max_len, 22, 3)
    for i, d in enumerate(draws):
        control[i, : d.length] = normalizer.normalize(torch.from_numpy(d.joints))
    lengths = torch.tensor([d.length for d in draws])
    return control.permute(0, 3, 1, 2).contiguous(), lengths
