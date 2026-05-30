"""Tests for MARDM's EssentialDataset (windowing, collate, stats) + AE fwd/bwd.

Builds a tiny synthetic packed dataset with non-degenerate skeleton offsets (so
the 67-D essential features are meaningful) and exercises the data path the way
`scripts/train_mardm_ae.py` does.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mardm.data import EssentialDataset, collate
from mardm.models import AE, AEConfig
from mardm.representation import ESSENTIAL_DIM, compute_essential_stats
from rmg.representation import NUM_JOINTS


def _offsets() -> torch.Tensor:
    torch.manual_seed(0)
    o = torch.randn(NUM_JOINTS, 3) * 0.1
    o[1, 0], o[2, 0] = 0.10, -0.10     # L_Hip / R_Hip x-spread
    o[16, 0], o[17, 0] = 0.12, -0.12   # L_Shoulder / R_Shoulder
    return o


def _clip(T: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    translation = torch.randn(T, 3, generator=g) * 0.05
    quats = torch.randn(T, NUM_JOINTS, 4, generator=g)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    return {"translation": translation, "quats": quats,
            "texts": [f"a person does action {seed}", f"motion {seed}"]}


def _build(root: Path, n_train: int = 6) -> None:
    root.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    with zipfile.ZipFile(root / "humanml3d.zip", "w", compression=zipfile.ZIP_STORED) as zf:
        idx = 0
        for split, n in (("train", n_train), ("val", 2), ("test", 2)):
            for _ in range(n):
                cid = f"{idx:06d}"
                buf = io.BytesIO()
                torch.save(_clip(80, idx), buf)
                zf.writestr(f"{cid}.pt", buf.getvalue())
                splits[split].append(cid)
                idx += 1
    (root / "splits.json").write_text(json.dumps(splits))
    torch.save(_offsets(), root / "target_offsets.pt")


def test_windowed_dataset_and_collate(tmp_path: Path) -> None:
    _build(tmp_path)
    ds = EssentialDataset(tmp_path, "train", window_size=32, mirror_augment=False, min_seq_len=10)
    s = ds[0]
    assert s.x1.shape == (32, ESSENTIAL_DIM)
    assert s.length == 32
    assert torch.isfinite(s.x1).all()
    loader = DataLoader(ds, batch_size=4, collate_fn=collate, num_workers=0, drop_last=True)
    batch = next(iter(loader))
    assert batch.x1.shape == (4, 32, ESSENTIAL_DIM)
    assert batch.mask.all()  # fixed windows → every frame valid


def test_full_clip_length_matches_feature(tmp_path: Path) -> None:
    _build(tmp_path)
    ds = EssentialDataset(tmp_path, "train", window_size=None, mirror_augment=False,
                          min_seq_len=10, max_seq_len=200)
    s = ds[0]
    # 67-D feature is T-1 frames; the off-by-one fix sets length == x1 rows.
    assert s.length == s.x1.shape[0]
    assert s.x1.shape[-1] == ESSENTIAL_DIM


def test_stats_and_normalized_dataset(tmp_path: Path) -> None:
    _build(tmp_path)
    raw_ds = EssentialDataset(tmp_path, "train", window_size=None, mirror_augment=False, min_seq_len=10)
    mean, std = compute_essential_stats(raw_ds)
    assert mean.shape == (ESSENTIAL_DIM,)
    assert (std > 0).all()
    norm_ds = EssentialDataset(tmp_path, "train", mean=mean, std=std,
                               window_size=None, mirror_augment=False, min_seq_len=10)
    xs = torch.cat([norm_ds[i].x1 for i in range(len(norm_ds))], dim=0)
    assert torch.isfinite(xs).all()


def test_subset_truncation(tmp_path: Path) -> None:
    _build(tmp_path, n_train=10)
    full = EssentialDataset(tmp_path, "train", window_size=None, mirror_augment=False, min_seq_len=10)
    assert len(full) == 10
    # subset_frac rounds: 0.3 * 10 = 3
    sub_frac = EssentialDataset(tmp_path, "train", window_size=None, mirror_augment=False,
                                min_seq_len=10, subset_frac=0.3)
    assert len(sub_frac) == 3
    # First clips kept (deterministic)
    full_ids = [full[i].clip_id for i in range(3)]
    sub_ids = [sub_frac[i].clip_id for i in range(3)]
    assert full_ids == sub_ids
    # limit_clips takes effect when subset_frac is None
    sub_lim = EssentialDataset(tmp_path, "train", window_size=None, mirror_augment=False,
                               min_seq_len=10, limit_clips=2)
    assert len(sub_lim) == 2
    # subset_frac wins over limit_clips
    both = EssentialDataset(tmp_path, "train", window_size=None, mirror_augment=False,
                            min_seq_len=10, subset_frac=0.5, limit_clips=2)
    assert len(both) == 5


def test_ae_forward_backward_on_windows(tmp_path: Path) -> None:
    _build(tmp_path)
    ds = EssentialDataset(tmp_path, "train", window_size=32, mirror_augment=False, min_seq_len=10)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate, num_workers=0, drop_last=True)
    batch = next(iter(loader))
    ae = AE(AEConfig(input_width=ESSENTIAL_DIM, output_emb_width=16, width=32, depth=2))
    z = ae.encode(batch.x1)
    assert z.shape == (4, 16, 32 // ae.downsample_rate)
    recon = ae(batch.x1)
    assert recon.shape == batch.x1.shape
    loss = (recon - batch.x1).abs().mean()
    loss.backward()
    assert torch.isfinite(loss)
