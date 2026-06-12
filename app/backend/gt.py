"""Ground-truth clip access for the GT browser.

Small, dependency-light reimplementations of the bits of `src/rmg/scripts/visualize.py`
that read the packed dataset, so the backend doesn't import the Hydra CLI
script. Reads clips straight out of `humanml3d.zip`.
"""

from __future__ import annotations

import io
import json
import random as _random
import zipfile
from pathlib import Path

import numpy as np
import torch

from shared.geometry import Skeleton, forward_kinematics


def subset_train_ids(
    data_root: Path,
    splits_name: str = "splits.json",
    *,
    subset_n: int = 0,
    subset_fraction: float = 0.01,
    subset_seed: int = 0,
) -> list[str]:
    """Replay HumanML3DDataset's train-subset selection → sorted regular clip ids.

    `subset_n > 0` keeps exactly that many regular clips (no mirrors), matching
    the exact-count overfit path; otherwise keep `subset_fraction` of them.
    """
    with open(Path(data_root) / splits_name) as f:
        splits = json.load(f)
    regular = [c for c in splits["train"] if not c.startswith("M")]
    rng = _random.Random(subset_seed)
    if subset_n > 0:
        n_keep = min(subset_n, len(regular))
        return sorted(rng.sample(regular, k=n_keep))
    n_keep = max(1, int(round(len(regular) * subset_fraction)))
    return sorted(rng.sample(regular, k=n_keep))


def load_clip(data_root: Path, clip_id: str) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Return (translation (T,3), quats (T,22,4), captions) for a packed clip."""
    zip_path = Path(data_root) / "humanml3d.zip"
    with zipfile.ZipFile(zip_path) as zf:
        blob = torch.load(io.BytesIO(zf.read(f"{clip_id}.pt")), weights_only=False)
    return blob["translation"].float(), blob["quats"].float(), list(blob["texts"])


def list_captions(data_root: Path, clip_ids: list[str]) -> list[dict]:
    """[{cid, caption}] — first caption per clip; skips clips missing from the zip."""
    zip_path = Path(data_root) / "humanml3d.zip"
    out: list[dict] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for cid in clip_ids:
            if f"{cid}.pt" not in names:
                continue
            blob = torch.load(io.BytesIO(zf.read(f"{cid}.pt")), weights_only=False)
            caps = blob.get("texts") or [""]
            out.append({"cid": cid, "caption": caps[0]})
    return out


def clip_joints(data_root: Path, clip_id: str, skeleton: Skeleton) -> tuple[np.ndarray, str]:
    """Decode a GT clip to (T,22,3) world joints + its first caption."""
    translation, quats, caps = load_clip(data_root, clip_id)
    joints = forward_kinematics(skeleton, quats, translation).cpu().numpy().astype(np.float32)
    return joints, (caps[0] if caps else "")
