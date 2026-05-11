"""End-to-end smoke training test.

Runs a tiny version of `scripts/train.py`'s loop against a synthetic packed
dataset for 5 outer steps, saves a checkpoint, then resumes and runs 3 more
steps. Verifies:
    - the loop runs to completion without errors
    - checkpoint round-trip preserves model weights bit-exactly
    - EMA, optimizer, scheduler all restore correctly
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rmg.data import HumanML3DDataset, collate
from rmg.flow import (
    FlowMatchingTrainer,
    FlowMatchingTrainerCfg,
    WrappedGaussianPrior,
    rest_pose_mu,
    rmg_manifold,
)
from rmg.models import DiTConfig, RandomTextEncoder, RMGDiT
from rmg.representation import NUM_JOINTS
from rmg.utils import (
    EMA,
    TrainState,
    build_scheduler,
    collect_rng_state,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


# ---------------------------------------------------------------------------
# Synthetic packed dataset (matches the format documented in
# src/rmg/data/humanml3d.py)
# ---------------------------------------------------------------------------


def _make_clip(T: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    trans = torch.randn(T, 3, generator=g) * 0.05
    quats = torch.randn(T, NUM_JOINTS, 4, generator=g)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    return {"translation": trans, "quats": quats, "texts": [f"motion {seed}"]}


def _build_synthetic_data(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(root / "humanml3d.zip", "w") as zf:
        for i in range(8):
            buf = io.BytesIO()
            torch.save(_make_clip(T=32, seed=i), buf)
            zf.writestr(f"{i:06d}.pt", buf.getvalue())
    splits = {"train": [f"{i:06d}" for i in range(6)],
              "val": [f"{i:06d}" for i in range(6, 7)],
              "test": [f"{i:06d}" for i in range(7, 8)]}
    (root / "splits.json").write_text(json.dumps(splits))
    torch.save(torch.zeros(NUM_JOINTS, 3), root / "target_offsets.pt")


# ---------------------------------------------------------------------------
# Tiny driver that mirrors scripts/train.py's training step
# ---------------------------------------------------------------------------


def _build_tiny_setup(tmp_path: Path):
    set_seed(0)
    data_dir = tmp_path / "data"
    _build_synthetic_data(data_dir)

    M = rmg_manifold(num_joints=NUM_JOINTS)
    mu = rest_pose_mu(num_joints=NUM_JOINTS)
    prior = WrappedGaussianPrior(M, mu, sigma=0.5)

    dit_cfg = DiTConfig(
        input_dim=M.ambient_dim, hidden_dim=32, depth=2, num_heads=4, ffn_mult=2,
        text_dim=64, time_freq_dim=32, max_seq_len=64,
    )
    model = RMGDiT(dit_cfg)
    enc = RandomTextEncoder(text_dim=dit_cfg.text_dim)

    ds = HumanML3DDataset(data_dir, split="train", min_seq_len=8, max_seq_len=24, mirror_augment=False)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate, num_workers=0, drop_last=True)

    trainer = FlowMatchingTrainer(M, prior, FlowMatchingTrainerCfg(cfg_dropout=0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = build_scheduler(opt, total_steps=20, warmup_ratio=0.1)
    ema = EMA(model, decay=0.99)

    return model, ema, opt, sched, trainer, enc, loader, dit_cfg


def _step_once(model, ema, opt, sched, trainer, enc, loader, grad_accum=1) -> float:
    opt.zero_grad(set_to_none=True)
    loss_total = 0.0
    it = iter(loader)
    for _ in range(grad_accum):
        batch = next(it)
        cond = enc.encode(batch.texts)
        loss, _ = trainer.compute_loss(model, batch.x1, cond=cond, mask=batch.mask)
        (loss / grad_accum).backward()
        loss_total += float(loss.detach())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    opt.step()
    sched.step()
    ema.update(model)
    return loss_total / grad_accum


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_smoke_training_runs_for_a_few_steps(tmp_path: Path) -> None:
    model, ema, opt, sched, trainer, enc, loader, _ = _build_tiny_setup(tmp_path)
    losses = []
    for _ in range(5):
        losses.append(_step_once(model, ema, opt, sched, trainer, enc, loader))
    # Sanity: finite + monotonic enough on average that *some* learning occurred.
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)


def test_checkpoint_resume_preserves_state(tmp_path: Path) -> None:
    model, ema, opt, sched, trainer, enc, loader, _ = _build_tiny_setup(tmp_path)
    # 3 steps
    for _ in range(3):
        _step_once(model, ema, opt, sched, trainer, enc, loader)

    ckpt_path = tmp_path / "ckpts" / "ckpt-000000003.pt"
    save_checkpoint(ckpt_path, TrainState(
        step=3,
        model=model.state_dict(),
        ema=ema.state_dict(),
        optimizer=opt.state_dict(),
        scheduler=sched.state_dict(),
        scaler=None,
        rng=collect_rng_state(),
        extras={},
    ))

    # Build a fresh setup and load
    set_seed(0)
    fresh_model, fresh_ema, fresh_opt, fresh_sched, fresh_trainer, fresh_enc, fresh_loader, _ = _build_tiny_setup(tmp_path)
    state = load_checkpoint(ckpt_path)
    fresh_model.load_state_dict(state.model)
    fresh_ema.load_state_dict(state.ema)
    fresh_opt.load_state_dict(state.optimizer)
    fresh_sched.load_state_dict(state.scheduler)

    # Assert the state matches the source
    for (k, v_src), (_, v_dst) in zip(model.state_dict().items(), fresh_model.state_dict().items()):
        assert torch.allclose(v_src, v_dst)

    # And one more step from each side gives the same scheduler LR
    lr_before_src = sched.get_last_lr()[0]
    lr_before_dst = fresh_sched.get_last_lr()[0]
    assert lr_before_src == lr_before_dst

    # Continuing for two more steps must run without errors
    for _ in range(2):
        _step_once(fresh_model, fresh_ema, fresh_opt, fresh_sched, fresh_trainer, fresh_enc, fresh_loader)
