"""Pluggable logger: wandb (offline-by-default) + tensorboard + CSV.

All three write to the run directory. wandb-online is an opt-in via
`WANDB_MODE=online`; the default is offline so jobs work without internet
egress (cluster note in the project plan).
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LoggerConfig:
    run_dir: Path
    run_name: str
    use_wandb: bool = True
    use_tensorboard: bool = True
    wandb_project: str = "rmg"
    wandb_mode: str = "offline"      # offline | online | disabled
    wandb_entity: str | None = None
    config: dict = field(default_factory=dict)


class Logger:
    def __init__(self, cfg: LoggerConfig) -> None:
        self.cfg = cfg
        self.run_dir = Path(cfg.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # CSV — always on, simplest fallback
        self._csv_path = self.run_dir / "metrics.csv"
        self._csv_keys: list[str] | None = None
        self._csv_fh = None

        # TensorBoard — optional, lazy
        self._tb_writer = None
        if cfg.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb_writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))
            except ImportError:
                self._tb_writer = None

        # wandb — optional, lazy. Honors env override.
        self._wandb = None
        if cfg.use_wandb and os.environ.get("WANDB_MODE", "").lower() != "disabled":
            try:
                import wandb

                wandb.init(
                    project=cfg.wandb_project,
                    name=cfg.run_name,
                    dir=str(self.run_dir),
                    mode=os.environ.get("WANDB_MODE", cfg.wandb_mode),
                    entity=cfg.wandb_entity,
                    config=cfg.config,
                    resume="allow",
                )
                self._wandb = wandb
            except (ImportError, Exception):  # pragma: no cover
                self._wandb = None

        # Persist run config snapshot for offline inspection
        (self.run_dir / "config.json").write_text(json.dumps(cfg.config, indent=2, default=str))

    @property
    def wandb_run_id(self) -> str | None:
        if self._wandb is not None and self._wandb.run is not None:
            return self._wandb.run.id
        return None

    # ---------------------------------------------------------------- log

    def log(self, metrics: dict[str, Any], step: int) -> None:
        flat = {k: float(v) if hasattr(v, "item") else v for k, v in metrics.items()}
        flat["step"] = step

        # CSV — append on resume rather than truncate.
        if self._csv_keys is None:
            self._csv_keys = list(flat.keys())
            resuming = self._csv_path.exists() and self._csv_path.stat().st_size > 0
            mode = "a" if resuming else "w"
            self._csv_fh = open(self._csv_path, mode, newline="")
            self._csv_writer = csv.DictWriter(self._csv_fh, fieldnames=self._csv_keys)
            if not resuming:
                self._csv_writer.writeheader()
        else:
            for k in flat:
                if k not in self._csv_keys:
                    self._csv_keys.append(k)
        # subset write — DictWriter ignores extras with extrasaction='ignore'
        self._csv_writer.writerow({k: flat.get(k, "") for k in self._csv_keys})
        self._csv_fh.flush()

        # TensorBoard
        if self._tb_writer is not None:
            for k, v in flat.items():
                if k == "step":
                    continue
                try:
                    self._tb_writer.add_scalar(k, float(v), step)
                except (TypeError, ValueError):
                    pass

        # wandb
        if self._wandb is not None:
            self._wandb.log(flat, step=step)

    def close(self) -> None:
        if self._csv_fh is not None:
            self._csv_fh.close()
        if self._tb_writer is not None:
            self._tb_writer.close()
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:  # pragma: no cover
                pass

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
