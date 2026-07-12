"""Lightweight job-progress markers for long cluster jobs.

Eval and viz jobs write a `.progress.json` sidecar into their output dir as they
run; the backend `job_progress` reader cats it in one SSH round-trip so the app
can draw a live bar (mirrors train.py's bare-integer `.progress` step marker).
The write is atomic (temp file + os.replace) so the backend never reads a
half-written file, and best-effort — a failed write never interrupts the job.

Schema (all fields optional; the backend normalises what it finds):
    stage    : short phase label, e.g. "sample" | "render" | "multimodality"
    outer    : {"i": int, "n": int, "label": str}  # e.g. guidance-level sweep
    inner    : {"i": int, "n": int}                 # e.g. batch within a level
    ts       : unix seconds (auto-filled)
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path


def write_progress(out_dir, **fields) -> None:
    """Atomically write `<out_dir>/.progress.json` with the given fields."""
    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fields.setdefault("ts", time.time())
        fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=".progress.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(fields, f, default=float)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, out_dir / ".progress.json")
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception:
        pass  # progress is best-effort — never break the job on a marker write
