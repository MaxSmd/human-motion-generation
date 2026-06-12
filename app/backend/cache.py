"""Params-hash → rendered-media cache.

Renders are deterministic in their inputs (text/guidance/steps/seed/ckpt for
generation; clip-id for GT; run/step for training samples), so we key the
output filename on a stable hash of those inputs. A cache hit just re-serves the
existing file under `/media/...`; the model never runs again.

Media files live in a single flat directory (RMG_MEDIA_DIR, default `<repo>/.media`)
which the FastAPI app mounts at `/media`.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .config import REPO


def media_dir() -> Path:
    d = Path(os.environ.get("MGEN_MEDIA_DIR", os.environ.get("RMG_MEDIA_DIR", REPO / ".media")))
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_key(**params) -> str:
    """Stable short hash of the render parameters (order-independent)."""
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def media_path(key: str, ext: str) -> Path:
    """Absolute path to the media file for `key` (no extension dot in `ext`)."""
    return media_dir() / f"{key}.{ext}"


def media_url(path: Path) -> str:
    """Public URL the frontend uses to fetch a rendered file."""
    return f"/media/{Path(path).name}"


def find_cached(key: str) -> Path | None:
    """Return an existing rendered media file for `key`, if any (mp4 preferred)."""
    d = media_dir()
    for ext in ("mp4", "gif"):
        p = d / f"{key}.{ext}"
        if p.exists():
            return p
    return None
