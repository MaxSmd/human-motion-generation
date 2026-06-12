#!/usr/bin/env python
"""Plot RMG training metrics from one or more runs' ``metrics.csv``.

No GPU, no container, no pandas — just stdlib + matplotlib. Run it locally
after copying a run down, or on the head node directly:

    # single run → one panel per metric, written to <run>/metrics.png
    python scripts/plot_metrics.py ~/rmg-runs/rmg-full-1pct-9337

    # overlay several runs on shared axes (great for base-vs-subset slides)
    python scripts/plot_metrics.py ~/rmg-runs/runA ~/rmg-runs/runB --out compare.png

    # tweak the loss view
    python scripts/plot_metrics.py <run> --log-loss --smooth 0.9

You can pass either a run directory (we look for ``metrics.csv`` inside) or the
``metrics.csv`` path directly.

The trainer historically logged no ``step`` column; if it's absent we
reconstruct the x-axis from ``config.json``'s ``train.log_every`` (the trainer
logs at step 1 then every ``log_every`` steps). Override with ``--log-every``.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Metrics where "lower is better" / a log y-axis usually reads better.
_LOG_FRIENDLY = {"loss", "grad_norm"}


def _find_csv(p: Path) -> Path:
    if p.is_file():
        return p
    cand = p / "metrics.csv"
    if cand.exists():
        return cand
    raise FileNotFoundError(f"no metrics.csv found at {p} (or {cand})")


def _log_every_from_config(run_dir: Path) -> int | None:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return None
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        return None
    # config is the resolved training config; log_every lives under train.
    train = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    le = train.get("log_every") if isinstance(train, dict) else None
    try:
        return int(le) if le is not None else None
    except (TypeError, ValueError):
        return None


def _load(csv_path: Path, log_every_override: int | None) -> tuple[list[float], dict[str, list[float]]]:
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{csv_path} is empty")

    cols = list(rows[0].keys())
    data: dict[str, list[float]] = {c: [] for c in cols}
    for r in rows:
        for c in cols:
            try:
                data[c].append(float(r[c]))
            except (ValueError, TypeError):
                data[c].append(float("nan"))

    if "step" in data and any(v == v for v in data["step"]):  # has a real step column
        steps = data.pop("step")
    else:
        data.pop("step", None)
        le = log_every_override or _log_every_from_config(csv_path.parent) or 1
        # Trainer logs at step 1, then at every multiple of log_every.
        steps = [1.0] + [float(i * le) for i in range(1, len(rows))]
        print(f"[plot_metrics] no 'step' column — reconstructed x-axis with "
              f"log_every={le} (override with --log-every)")
    return steps, data


def _ema(ys: list[float], alpha: float) -> list[float]:
    out: list[float] = []
    m: float | None = None
    for y in ys:
        if y != y:  # NaN
            out.append(float("nan"))
            continue
        m = y if m is None else alpha * m + (1.0 - alpha) * y
        out.append(m)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path,
                    help="run dir(s) or metrics.csv path(s); >1 overlays them")
    ap.add_argument("--out", type=Path, default=None,
                    help="output PNG (default: <first-run>/metrics.png)")
    ap.add_argument("--log-every", type=int, default=None,
                    help="override step spacing when CSV lacks a 'step' column")
    ap.add_argument("--smooth", type=float, default=0.9,
                    help="EMA factor for a smoothed loss overlay (0 disables)")
    ap.add_argument("--log-loss", action="store_true",
                    help="use a log y-axis for loss / grad_norm panels")
    args = ap.parse_args()

    series: list[tuple[str, list[float], dict[str, list[float]]]] = []
    for rp in args.runs:
        csv_path = _find_csv(rp)
        steps, data = _load(csv_path, args.log_every)
        if rp.is_dir():
            name = rp.name
        elif rp.stem == "metrics":      # a run-dir's metrics.csv → label by run
            name = rp.parent.name
        else:                            # a bare/renamed csv → label by file stem
            name = rp.stem
        series.append((name, steps, data))
        print(f"[plot_metrics] {name}: {len(steps)} points, "
              f"metrics={list(data.keys())}")

    # Stable union of metric columns across runs.
    metric_cols: list[str] = []
    for _, _, d in series:
        for c in d:
            if c not in metric_cols:
                metric_cols.append(c)

    n = len(metric_cols)
    ncol = 2 if n > 1 else 1
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.4 * ncol, 3.2 * nrow), squeeze=False)
    overlay = len(series) > 1

    for idx, col in enumerate(metric_cols):
        ax = axes[idx // ncol][idx % ncol]
        smoothing = (col == "loss") and args.smooth and args.smooth > 0.0
        for name, steps, data in series:
            if col not in data:
                continue
            ys = data[col]
            ax.plot(steps, ys, lw=1.0, alpha=0.35 if smoothing else 0.9,
                    label=(name if not smoothing else f"{name} (raw)"))
            if smoothing:
                ax.plot(steps, _ema(ys, args.smooth), lw=1.9,
                        label=f"{name} (ema{args.smooth:g})")
        ax.set_title(col)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        if args.log_loss and col in _LOG_FRIENDLY:
            ax.set_yscale("log")
        if overlay or smoothing:
            ax.legend(fontsize=7)

    for j in range(n, nrow * ncol):  # hide unused panels
        axes[j // ncol][j % ncol].axis("off")

    fig.suptitle(" vs ".join(s[0] for s in series), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    first = args.runs[0]
    default_dir = first if first.is_dir() else first.parent
    out = Path(args.out) if args.out else (default_dir / "metrics.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"[plot_metrics] wrote {out}")


if __name__ == "__main__":
    main()
