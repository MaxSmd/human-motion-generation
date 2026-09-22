"""Evaluation plumbing for the ProjFlow port: test-set protocol and metric helpers."""

from __future__ import annotations

import zipfile

import numpy as np
import torch

from projflow.control import keyframes_for_length, sample_keyframe_mask
from projflow.data import ControlTestSet
from projflow.scripts.evaluate import cell_list, control_errors, skating_ratio_upstream
from shared.geometry.humanml3d_io import recover_joints_from_ric


def _fixture(tmp_path):
    vec_dir = tmp_path / "vecs"
    vec_dir.mkdir()
    rng = np.random.default_rng(0)
    lengths = {"000001": 120, "000002": 30, "000003": 199}      # 000002 too short
    for cid, n in lengths.items():
        np.save(vec_dir / f"{cid}.npy", (rng.standard_normal((n, 263)) * 0.01).astype(np.float32))
    (tmp_path / "test.txt").write_text("000001\n000002\n000003\nM000004\n")   # M000004 has no features
    texts = {
        "000001": "a person walks#a/DET person/NOUN walk/VERB#0.0#0.0\n"
                  "a person waves#a/DET person/NOUN wave/VERB#1.0#4.0\n",   # 60-frame sub-segment
        "000002": "short#short/ADJ#0.0#0.0\n",
        "000003": "jumps#jump/VERB#0.0#0.0\nspins#spin/VERB#0.0#0.0\n",
        "M000004": "mirrored#mirror/VERB#0.0#0.0\n",
    }
    with zipfile.ZipFile(tmp_path / "texts.zip", "w") as zf:
        for cid, body in texts.items():
            zf.writestr(f"texts/{cid}.txt", body)
    return ControlTestSet(vec_dir, tmp_path / "test.txt", tmp_path / "texts.zip")


def test_testset_follows_guo_protocol(tmp_path):
    ts = _fixture(tmp_path)
    names = sorted(e.name for e in ts.entries)
    assert names == ["000001", "000001#1", "000003"]
    assert ts.missing == ["M000004"]
    sub = next(e for e in ts.entries if e.name == "000001#1")
    whole = next(e for e in ts.entries if e.name == "000001")
    assert len(sub.vecs) == 60
    np.testing.assert_allclose(sub.joints, whole.joints[20:80])          # sliced after recovery
    np.testing.assert_allclose(whole.joints, recover_joints_from_ric(torch.from_numpy(whole.vecs)).numpy())

    rng = np.random.default_rng(1)
    for _ in range(50):
        d = ts.draw(ts.entries.index(next(e for e in ts.entries if e.name == "000003")), rng)
        assert d.length % 4 == 0 and d.length in (192, 196)
        assert d.joints.shape == (d.length, 22, 3) and d.vecs.shape == (d.length, 263)
        assert d.caption.text in ("jumps", "spins")


def test_keyframe_counts_and_mask():
    assert [keyframes_for_length(196, d) for d in (1, 2, 5, 25, 100)] == [1, 2, 5, 49, 196]
    mask = sample_keyframe_mask(torch.tensor([196, 60]), [0, 20], 25, 196)
    assert mask.shape == (2, 3, 196, 22)
    assert mask[0, 0, :, 0].sum() == 49 and mask[1, 0, :, 20].sum() == 15
    assert mask[1, :, 60:].sum() == 0 and mask[:, :, :, 5].sum() == 0


def test_control_errors_and_skating():
    gt = np.zeros((10, 22, 3))
    gen = gt.copy()
    mask = np.zeros((10, 22), bool)
    mask[[2, 7], 0] = True
    assert control_errors(gen, gt, mask).tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]
    gen[7, 0, 0] = 0.6
    traj02, traj05, loc02, loc05, avg = control_errors(gen, gt, mask)
    assert (traj02, traj05, loc02, loc05) == (1.0, 1.0, 0.5, 0.5) and np.isclose(avg, 0.3)

    still = np.zeros((1, 30, 22, 3))
    assert skating_ratio_upstream(still)[0] == 0.0
    sliding = still.copy()
    sliding[0, :, 10, 0] = np.arange(30) * 0.1                            # left foot slides 2 m/s on the floor
    assert skating_ratio_upstream(sliding)[0] > 0.5


def test_cell_list():
    assert len(cell_list("all")) == 30
    assert cell_list("0:1,21:100") == [(0, 1), (21, 100)]
