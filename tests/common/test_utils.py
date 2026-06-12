"""Training-utility tests: EMA, scheduler, checkpoint, seeding, logger."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from common.utils import (
    EMA,
    Logger,
    LoggerConfig,
    TrainState,
    build_scheduler,
    collect_rng_state,
    cosine_with_warmup,
    find_latest_checkpoint,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    set_seed,
)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


def test_ema_update_matches_closed_form() -> None:
    model = nn.Linear(4, 4)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1.0)
    ema = EMA(model, decay=0.9)
    # set parameters to 0, then run one update — shadow should be 0.9*1 + 0.1*0 = 0.9
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0.0)
    ema.update(model)
    for k, v in ema.shadow.items():
        assert torch.allclose(v, torch.full_like(v, 0.9), atol=1e-7)


def test_ema_swapped_context_restores_live_weights() -> None:
    model = nn.Linear(2, 2)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(2.0)
    ema = EMA(model, decay=0.9)
    # mutate live weights post-init
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(7.0)
    # snapshot live weights for comparison
    live_before = {k: v.clone() for k, v in model.state_dict().items()}
    with ema.swapped(model) as m:
        for k, v in m.state_dict().items():
            if v.is_floating_point():
                assert torch.allclose(v, torch.full_like(v, 2.0))
    # back to live after exit
    for k, v in model.state_dict().items():
        if v.is_floating_point():
            assert torch.allclose(v, live_before[k])


def test_ema_state_dict_roundtrip() -> None:
    model = nn.Linear(2, 2)
    ema = EMA(model, decay=0.99)
    sd = ema.state_dict()

    other = nn.Linear(2, 2)
    ema2 = EMA(other, decay=0.5)
    ema2.load_state_dict(sd)
    assert ema2.decay == 0.99
    for k in ema.shadow:
        assert torch.allclose(ema2.shadow[k], ema.shadow[k])


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def test_cosine_warmup_linear_then_cosine() -> None:
    total = 1000
    warmup = 100

    # Warmup: linear in [0, 1]
    assert cosine_with_warmup(0, total, warmup) == 0.0
    assert cosine_with_warmup(50, total, warmup) == pytest.approx(0.5, abs=1e-9)
    assert cosine_with_warmup(100, total, warmup) == pytest.approx(1.0, abs=1e-9)

    # End of training: cosine should hit ~min_lr_ratio
    end = cosine_with_warmup(total, total, warmup, min_lr_ratio=0.1)
    assert end == pytest.approx(0.1, abs=1e-9)

    # Mid-cosine: at progress=0.5 → cos(pi/2) = 0 → multiplier = min + 0.5*(1-min)
    mid_step = warmup + (total - warmup) // 2
    mult = cosine_with_warmup(mid_step, total, warmup, min_lr_ratio=0.0)
    assert mult == pytest.approx(0.5, abs=5e-3)


def test_build_scheduler_steps_correctly() -> None:
    model = nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    sched = build_scheduler(opt, total_steps=100, warmup_ratio=0.1)
    # Step 0 (before any sched.step call): lr at index 0
    assert opt.param_groups[0]["lr"] == 0.0  # warmup starts at 0
    # Take a real opt step first so torch doesn't warn about reverse order.
    model(torch.zeros(1, 2)).sum().backward()
    opt.step()
    sched.step()
    # After 1 step, fraction = 1/10 = 0.1
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1 * 1e-3, abs=1e-12)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = nn.Linear(3, 3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = build_scheduler(opt, total_steps=10, warmup_ratio=0.1)
    ema = EMA(model, decay=0.999)

    # take a step so optimizer state is non-trivial
    x = torch.randn(4, 3)
    (model(x).sum()).backward()
    opt.step()
    sched.step()
    ema.update(model)

    state = TrainState(
        step=42,
        model=model.state_dict(),
        ema=ema.state_dict(),
        optimizer=opt.state_dict(),
        scheduler=sched.state_dict(),
        scaler=None,
        rng=collect_rng_state(),
        extras={"wandb_run_id": "abc123"},
    )
    save_checkpoint(tmp_path / "ckpt-000000042.pt", state)

    loaded = load_checkpoint(tmp_path / "ckpt-000000042.pt")
    assert loaded.step == 42
    assert loaded.extras["wandb_run_id"] == "abc123"

    # restore into a fresh model — weights and EMA must match
    fresh = nn.Linear(3, 3)
    fresh.load_state_dict(loaded.model)
    for k, v in model.state_dict().items():
        assert torch.allclose(fresh.state_dict()[k], v)
    fresh_ema = EMA(fresh, decay=0.5)
    fresh_ema.load_state_dict(loaded.ema)
    for k in ema.shadow:
        assert torch.allclose(fresh_ema.shadow[k], ema.shadow[k])


def test_find_latest_checkpoint(tmp_path: Path) -> None:
    cdir = tmp_path / "checkpoints"
    cdir.mkdir()
    (cdir / "ckpt-000000005.pt").write_bytes(b"")
    (cdir / "ckpt-000000020.pt").write_bytes(b"")
    (cdir / "ckpt-000000010.pt").write_bytes(b"")
    found = find_latest_checkpoint(tmp_path)
    assert found is not None
    assert found.name == "ckpt-000000020.pt"


def test_find_latest_checkpoint_empty(tmp_path: Path) -> None:
    assert find_latest_checkpoint(tmp_path) is None


# ---------------------------------------------------------------------------
# Seeding / RNG roundtrip
# ---------------------------------------------------------------------------


def test_set_seed_reproducible() -> None:
    set_seed(123)
    a = (random.random(), float(np.random.rand()), torch.randn(1).item())
    set_seed(123)
    b = (random.random(), float(np.random.rand()), torch.randn(1).item())
    assert a == b


def test_rng_state_roundtrip() -> None:
    set_seed(7)
    state = collect_rng_state()
    a = (random.random(), float(np.random.rand()), torch.randn(1).item())
    restore_rng_state(state)
    b = (random.random(), float(np.random.rand()), torch.randn(1).item())
    assert a == b


# ---------------------------------------------------------------------------
# Logger smoke (no wandb installed in CI is fine — graceful fallback)
# ---------------------------------------------------------------------------


def test_logger_writes_csv(tmp_path: Path) -> None:
    cfg = LoggerConfig(
        run_dir=tmp_path,
        run_name="t",
        use_wandb=False,
        use_tensorboard=False,
        config={"x": 1},
    )
    with Logger(cfg) as logger:
        logger.log({"loss": 1.5, "acc": 0.7}, step=1)
        logger.log({"loss": 1.2, "acc": 0.8}, step=2)
    csv = (tmp_path / "metrics.csv").read_text().strip().splitlines()
    assert csv[0].split(",")[:3] == ["loss", "acc", "step"]
    assert "1.5" in csv[1]
    assert "1.2" in csv[2]
    assert (tmp_path / "config.json").exists()
