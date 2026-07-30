"""HumanML3D dataset / collate / mirror tests.

Builds a tiny synthetic packed dataset (3 clips, varying length, dummy texts)
in a tmp dir and exercises the loader end-to-end.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from rmg.data import HumanML3DDataset, collate
from shared.data import LR_PAIRS, mirror_motion
from rmg.representation import NUM_JOINTS, decode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_clip(T: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    translation = torch.randn(T, 3, generator=g) * 0.05
    quats = torch.randn(T, NUM_JOINTS, 4, generator=g)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    return {
        "translation": translation,
        "quats": quats,
        "texts": [
            f"a person performs action {seed} at a slow pace",
            f"slow {seed}-action",
        ],
    }


def _build_synthetic_dataset(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    clips = {f"{i:06d}": _make_synthetic_clip(T=60 + 10 * i, seed=i) for i in range(4)}
    with zipfile.ZipFile(root / "humanml3d.zip", "w", compression=zipfile.ZIP_STORED) as zf:
        for cid, blob in clips.items():
            buf = io.BytesIO()
            torch.save(blob, buf)
            zf.writestr(f"{cid}.pt", buf.getvalue())
    splits = {
        "train": ["000000", "000001"],
        "val":   ["000002"],
        "test":  ["000003"],
    }
    (root / "splits.json").write_text(json.dumps(splits))
    torch.save(torch.zeros(NUM_JOINTS, 3), root / "target_offsets.pt")


# ---------------------------------------------------------------------------
# Mirror augmentation
# ---------------------------------------------------------------------------


def test_mirror_is_involutive() -> None:
    """Applying mirror twice returns the original."""
    T = 8
    trans = torch.randn(T, 3)
    quats = torch.randn(T, NUM_JOINTS, 4)
    quats = quats / quats.norm(dim=-1, keepdim=True)

    t1, q1 = mirror_motion(trans, quats)
    t2, q2 = mirror_motion(t1, q1)
    assert torch.allclose(t2, trans, atol=1e-6)
    assert torch.allclose(q2, quats, atol=1e-6)


def test_mirror_swaps_left_right_pairs() -> None:
    """L_Hip ↔ R_Hip and friends should swap joint slots."""
    T = 4
    trans = torch.zeros(T, 3)
    quats = torch.zeros(T, NUM_JOINTS, 4)
    # Mark each joint with a distinct quaternion so we can detect swaps
    for j in range(NUM_JOINTS):
        quats[:, j, 0] = 1.0
        quats[:, j, 1] = float(j)
    _, qm = mirror_motion(trans, quats)
    for l, r in LR_PAIRS:
        assert torch.allclose(qm[:, l, 1], torch.full((T,), float(r)))
        assert torch.allclose(qm[:, r, 1], torch.full((T,), float(l)))


def test_mirror_x_flips_translation_x_axis() -> None:
    trans = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    quats = torch.zeros(2, NUM_JOINTS, 4)
    quats[..., 0] = 1.0
    tm, _ = mirror_motion(trans, quats)
    assert tm[0, 0].item() == -1.0
    assert tm[0, 1].item() == 2.0
    assert tm[1, 0].item() == -4.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def test_dataset_round_trip(tmp_path: Path) -> None:
    _build_synthetic_dataset(tmp_path)
    ds = HumanML3DDataset(tmp_path, split="train", min_seq_len=10, max_seq_len=200)
    assert len(ds) == 2
    sample = ds[0]
    # x1 is encoded T+R: (T, 3 + 4*22)
    assert sample.x1.shape == (sample.length, 3 + 4 * NUM_JOINTS)
    # decode and check quats are unit-norm
    tpr = decode(sample.x1)
    norms = tpr.quaternions.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    # text is one of the captions
    assert isinstance(sample.text, str) and sample.text


def test_dataset_split_filtering(tmp_path: Path) -> None:
    _build_synthetic_dataset(tmp_path)
    ds_train = HumanML3DDataset(tmp_path, split="train", min_seq_len=10)
    ds_val = HumanML3DDataset(tmp_path, split="val", min_seq_len=10)
    ds_test = HumanML3DDataset(tmp_path, split="test", min_seq_len=10)
    assert len(ds_train) == 2
    assert len(ds_val) == 1
    assert len(ds_test) == 1
    train_ids = {ds_train[i].clip_id for i in range(len(ds_train))}
    assert train_ids == {"000000", "000001"}


def test_dataset_max_seq_len_random_crop(tmp_path: Path) -> None:
    _build_synthetic_dataset(tmp_path)
    ds = HumanML3DDataset(tmp_path, split="train", min_seq_len=10, max_seq_len=20)
    sample = ds[1]  # clip 000001 has T=70 — must be cropped to 20
    assert sample.length == 20


def test_collate_pads_and_masks(tmp_path: Path) -> None:
    _build_synthetic_dataset(tmp_path)
    ds = HumanML3DDataset(tmp_path, split="train", min_seq_len=10, max_seq_len=200)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate, num_workers=0)
    batch = next(iter(loader))
    B = batch.x1.shape[0]
    Tmax = batch.x1.shape[1]
    assert B == 2
    # the longer clip in a batch sets Tmax; the shorter is zero-padded outside its mask
    for i in range(B):
        L = batch.lengths[i].item()
        assert batch.mask[i, :L].all()
        if L < Tmax:
            assert not batch.mask[i, L:].any()
            assert (batch.x1[i, L:] == 0).all()
    assert isinstance(batch.texts[0], str)


# ---------------------------------------------------------------------------
# Per-crop SE(2) canonicalization
# ---------------------------------------------------------------------------


def _yaw_quat(deg: float) -> torch.Tensor:
    half = torch.deg2rad(torch.tensor(deg)) / 2
    return torch.stack([torch.cos(half), torch.tensor(0.0), torch.sin(half), torch.tensor(0.0)])


def test_canonicalize_crop_origin_and_heading() -> None:
    from shared.data.humanml3d import canonicalize_crop
    from shared.geometry.skeleton import quat_rotate

    T = 8
    # Walk along +X, starting away from the origin, body yawed 90° (facing +X).
    translation = torch.zeros(T, 3)
    translation[:, 0] = torch.linspace(2.0, 3.4, T)   # +X march
    translation[:, 1] = 0.9                            # pelvis height
    translation[:, 2] = -1.7
    quats = torch.zeros(T, NUM_JOINTS, 4)
    quats[..., 0] = 1.0
    quats[:, 0] = _yaw_quat(90.0)                      # root faces +X

    t_c, q_c = canonicalize_crop(translation, quats)

    # First frame at XZ origin, height untouched.
    assert torch.allclose(t_c[0, [0, 2]], torch.zeros(2), atol=1e-6)
    assert torch.allclose(t_c[:, 1], translation[:, 1], atol=1e-6)
    # Heading now +Z: root-forward maps to +Z, and the +X march became +Z.
    fwd = quat_rotate(q_c[0, 0], torch.tensor([0.0, 0.0, 1.0]))
    assert fwd[2] > 0.999 and abs(fwd[0]) < 1e-4
    assert t_c[-1, 2] > 1.0 and abs(t_c[-1, 0]) < 1e-4
    # Rigid: frame-to-frame displacement magnitudes preserved.
    assert torch.allclose(
        (t_c[1:] - t_c[:-1]).norm(dim=-1),
        (translation[1:] - translation[:-1]).norm(dim=-1),
        atol=1e-5,
    )
    # Local (non-root) joint quaternions untouched.
    assert torch.equal(q_c[:, 1:], quats[:, 1:])


def test_canonicalize_crop_vertical_heading_falls_back_to_shift() -> None:
    from shared.data.humanml3d import canonicalize_crop

    T = 4
    translation = torch.randn(T, 3)
    quats = torch.zeros(T, NUM_JOINTS, 4)
    # Root pitched 90° about X: canonical +Z forward now points straight up.
    half = torch.deg2rad(torch.tensor(90.0)) / 2
    quats[:, 0, 0] = torch.cos(half)
    quats[:, 0, 1] = torch.sin(half)
    quats[:, 1:, 0] = 1.0

    t_c, q_c = canonicalize_crop(translation, quats)
    assert torch.allclose(t_c[0, [0, 2]], torch.zeros(2), atol=1e-6)
    assert torch.equal(q_c, quats)  # rotation skipped, quats untouched


def test_dataset_canonicalize_crops_flag(tmp_path: Path) -> None:
    _build_synthetic_dataset(tmp_path)
    ds = HumanML3DDataset(tmp_path, split="train", min_seq_len=10, canonicalize_crops=True)
    sample = ds[0]
    tpr = decode(sample.x1)
    assert float(tpr.translation[0, [0, 2]].abs().max()) < 1e-5
