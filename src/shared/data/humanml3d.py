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
import json
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor
from torch.utils.data import Dataset

from shared.geometry import Skeleton, make_continuous, normalize_quaternions

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


@runtime_checkable
class ClipRepresentation(Protocol):
    """Structural interface a model supplies to turn a packed clip into features.

    Keeps the dataset model-agnostic: it never imports a model package, it just
    calls `encode_clip` on whatever representation is passed (rmg's manifold T+R,
    mardm's 67-D essential, …).
    """

    def encode_clip(
        self, translation: Tensor, quaternions: Tensor, skeleton: "Skeleton | None" = None
    ) -> Tensor:
        ...


@dataclass
class HumanML3DSample:
    x1: Tensor          # (T, D) model-specific encoded features for one clip
    text: str           # one caption sampled uniformly from the clip's pool
    length: int         # T (true sequence length, before padding)
    clip_id: str


class HumanML3DDataset(Dataset):
    """Packed HumanML3D loader, parameterised by a `ClipRepresentation`.

    Reads packed clips, applies the standard normalize + temporal sign-continuity
    pass, then delegates the model-specific encoding to `representation.encode_clip`.
    `representation` is required here; model packages provide a thin subclass that
    defaults it (e.g. `rmg.data.HumanML3DDataset` → T+R).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        max_seq_len: int = 196,
        min_seq_len: int = 40,
        zip_name: str = "humanml3d.zip",
        splits_name: str = "splits.json",
        offsets_name: str = "target_offsets.pt",
        mirror_augment: bool = False,
        representation: ClipRepresentation | None = None,
        subset_fraction: float = 1.0,
        subset_seed: int = 0,
        subset_n: int = 0,
        preload: bool = False,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be one of train/val/test, got {split}")
        if representation is None:
            raise ValueError(
                "HumanML3DDataset requires a `representation`. Use a model subclass "
                "(e.g. rmg.data.HumanML3DDataset) or pass one explicitly."
            )
        self.root = Path(root)
        self.zip_path = self.root / zip_name
        self.split = split
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len

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

        # Runtime mirror augmentation (train split only), opt-in. Off by default:
        # rmg leaves it off because its packed data already ships baked-in
        # mirrors; mardm enables it explicitly for its pipeline.
        self.mirror_augment = mirror_augment and (split == "train")

        self.representation: ClipRepresentation = representation

        # Optional RAM cache of the raw clips (translation, quats, texts). Reading
        # + decompressing from the zip every step is the main CPU cost and stalls
        # the GPU at grad-accum boundaries; caching kills it (the crop/mirror/encode
        # still run per step, so augmentation is unchanged). Built once in the
        # parent process; DataLoader workers inherit it via fork (copy-on-write, so
        # no per-worker duplication). ~0.6GB unmirrored / ~1.2GB mirrored.
        self._cache: dict[str, tuple[Tensor, Tensor, list[str]]] | None = None
        if preload:
            zf = self._open_zip()
            self._cache = {cid: read_clip(zf, cid) for cid in self.clip_ids}
            # Drop the open zip handle: everything's in RAM now, and an open
            # BufferedReader isn't picklable (breaks 'spawn' DataLoader workers).
            self._zip.close()
            self._zip = None

    def _open_zip(self) -> zipfile.ZipFile:
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.zip_path, mode="r")
        return self._zip

    def __len__(self) -> int:
        return len(self.clip_ids)

    def __getitem__(self, idx: int) -> HumanML3DSample:
        clip_id = self.clip_ids[idx]
        if self._cache is not None:
            # Clone so the per-step crop/mirror/normalize never mutate the shared
            # cache (cheap — a clip is only a few hundred frames).
            tr, q, texts = self._cache[clip_id]
            translation, quats = tr.clone(), q.clone()
        else:
            translation, quats, texts = read_clip(self._open_zip(), clip_id)

        translation, quats, T = random_crop(translation, quats, self.max_seq_len)
        if T < self.min_seq_len:
            # Should not happen for a well-built dataset; skip to the next index.
            return self.__getitem__((idx + 1) % len(self))

        # Optional runtime mirror augmentation (train only, opt-in). rmg leaves
        # this off — its packed data already ships every clip in both
        # orientations (`<id>` + `M<id>`, both listed in the splits), so a
        # runtime flip would re-mirror without swapping the text. mardm enables
        # it (`mirror_augment=True`) to reproduce its own pipeline.
        if self.mirror_augment and random.random() < 0.5:
            translation, quats = mirror_motion(translation, quats)

        # Normalize + temporal sign continuity, then encode via the chosen
        # representation (rmg T+R main result; mardm essential; …).
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
