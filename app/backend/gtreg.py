"""Rendered-ground-truth registry: clip_id → the GT clip's media, rendered once.

A GT clip's render is a pure function of its id (the stored motion never changes),
so rendering it a second time burns a GPU job for a byte-identical GIF. Every viz
job that produces `kind="gt"` media deposits it here, keyed by clip id; later jobs
that need the same GT skip the render (`SKIP_GT` → `visualize.py`) and get the
stored copy spliced back into their outputs.

Files are COPIED out of the per-job media dir into a flat `<media>/gt/` store, so
a GT render outlives the job that happened to produce it (jobs are deletable
history; the GT store is a permanent asset). The index is JSON next to jobs.json.

Entries are the same output shape the frontend renders:
    {media_url, npy_url, caption, kind: "gt", clip_id, label, from_registry}
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

from . import cache, config as cfgmod

_LOCK = threading.RLock()


def store_dir() -> Path:
    d = cache.media_dir() / "gt"
    d.mkdir(parents=True, exist_ok=True)
    return d


def index_path() -> Path:
    return cfgmod.jobs_state_path().parent / "gt_registry.json"


def _load() -> dict[str, dict]:
    p = index_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else {}
    except (ValueError, OSError):
        return {}


def _save(index: dict[str, dict]) -> None:
    p = index_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(index, indent=0))
    tmp.replace(p)


def _entry_ok(e: dict) -> bool:
    """An index entry is only usable while its media file is still on disk."""
    name = str(e.get("media_url", "")).rsplit("/", 1)[-1]
    return bool(name) and (store_dir() / name).exists()


def get(clip_id: str) -> dict | None:
    """The stored GT render for `clip_id`, or None if absent/evicted."""
    with _LOCK:
        e = _load().get(str(clip_id))
    return e if e and _entry_ok(e) else None


def known(clip_ids: list[str]) -> list[str]:
    """Subset of `clip_ids` that already have a usable GT render."""
    with _LOCK:
        index = _load()
    return [c for c in clip_ids if (e := index.get(str(c))) and _entry_ok(e)]


def outputs(clip_ids: list[str]) -> list[dict]:
    """Registry entries for `clip_ids`, in the job-output shape (missing ids skipped)."""
    return [e for c in clip_ids if (e := get(c)) is not None]


def put(clip_id: str, media: Path, npy: Path | None, caption: str) -> dict:
    """Copy a rendered GT clip into the store and index it. Re-registering an id
    refreshes the stored copy (a re-render is byte-identical, but this keeps the
    store self-healing if a file was deleted underneath us)."""
    cid = str(clip_id)
    media = Path(media)
    dest = store_dir() / f"{cid}{media.suffix}"
    shutil.copy2(media, dest)
    npy_url = None
    if npy is not None and Path(npy).exists():
        npy_dest = store_dir() / f"{cid}.npy"
        shutil.copy2(npy, npy_dest)
        npy_url = f"/media/gt/{npy_dest.name}"
    entry = {
        "media_url": f"/media/gt/{dest.name}",
        "npy_url": npy_url,
        "caption": caption,
        "label": f"real-{cid}",
        "kind": "gt",
        "clip_id": cid,
        "step": None,
        "from_registry": True,
        "at": time.time(),
    }
    with _LOCK:
        index = _load()
        index[cid] = entry
        _save(index)
    return entry


def harvest(outputs_list: list[dict], job_dir: Path) -> int:
    """Register every GT clip a finished job rendered. `job_dir` is the job's local
    media dir (where the pulled files live). Returns how many were newly stored."""
    n = 0
    for o in outputs_list:
        cid = o.get("clip_id")
        if o.get("kind") != "gt" or not cid or o.get("from_registry"):
            continue
        media = job_dir / str(o["media_url"]).rsplit("/", 1)[-1]
        if not media.exists():
            continue
        npy_name = str(o["npy_url"]).rsplit("/", 1)[-1] if o.get("npy_url") else None
        npy = job_dir / npy_name if npy_name else None
        try:
            put(cid, media, npy, o.get("caption") or "")
            n += 1
        except OSError:
            continue  # a copy failure must never fail the job's pull
    return n


def splice(outputs_list: list[dict]) -> list[dict]:
    """Ensure every predicted clip is preceded by its GT tile.

    A job that skipped GT rendering (because the registry already had it) comes
    back with PRED tiles only; pull the stored GT back in so the viewer always
    shows the pair. Ordering mirrors the pull's sort: GT then PRED, per clip.
    """
    have_gt = {o.get("clip_id") for o in outputs_list if o.get("kind") == "gt"}
    out: list[dict] = []
    for o in outputs_list:
        cid = o.get("clip_id")
        if o.get("kind") == "pred" and cid and cid not in have_gt:
            gt = get(cid)
            if gt:
                out.append(dict(gt))
                have_gt.add(cid)
        out.append(o)
    return out


def stats() -> dict:
    """Registry size + ids (for the UI's "GT cached" hint)."""
    with _LOCK:
        index = _load()
    ids = sorted(c for c, e in index.items() if _entry_ok(e))
    return {"count": len(ids), "clip_ids": ids}
