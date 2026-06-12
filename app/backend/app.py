"""FastAPI app for the RMG interactive frontend.

Single warm process. Endpoints render motion to a media file on disk (cached by
a hash of their inputs) and return its `/media/...` URL. See `plan.md` §2.

Run:  uvicorn backend.app:app --app-dir app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from contextlib import asynccontextmanager

from . import cache, config as cfgmod, gt as gtmod
from .cluster import jobs as jobsmod
from .cluster.routes import router as cluster_router
from .cluster import ssh as clusterssh
from .render import decode_to_joints, render_joints, resolve_format
from .state import get_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the SLURM job poller when cluster mode is on; tear down the shared
    # SSH ControlMaster on shutdown (temporary footprint — see cluster-plan.md).
    if cfgmod.cluster_mode():
        jobsmod.get_manager().start()
    yield
    if cfgmod.cluster_mode():
        jobsmod.get_manager().stop()
        clusterssh.close_master()


app = FastAPI(title="RMG Interactive Backend", version="1.0", lifespan=lifespan)

# Vite dev server + any localhost origin during the demo.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve rendered media files. Mounted before any request so the dir exists.
app.mount("/media", StaticFiles(directory=str(cache.media_dir())), name="media")

# Cluster control plane (/cluster/*).
app.include_router(cluster_router)


# --------------------------------------------------------------------------- models


class GenerateRequest(BaseModel):
    text: str
    guidance: float = 6.5
    num_steps: int = 50
    seed: int = 0
    num_frames: int = 100
    checkpoint: str | None = None  # explicit ckpt path; None → server default
    fmt: str = "auto"              # auto | mp4 | gif


# --------------------------------------------------------------------------- health


@app.get("/health")
def health() -> dict:
    st = get_state()
    ckpt = cfgmod.default_checkpoint()
    offs = cfgmod.offsets_path(st.default_cfg)
    return {
        "status": "ok",
        "device": str(st.device),
        "representation": str(st.default_cfg.representation.name),
        "ambient_dim": 3 + 4 * int(st.default_cfg.representation.get("num_joints", 22)),
        "default_checkpoint": str(ckpt) if ckpt else None,
        "data_root": str(cfgmod.data_root(st.default_cfg)),
        "runs_dir": str(cfgmod.runs_dir()),
        "offsets_available": offs.exists(),
        "ffmpeg": resolve_format("mp4") == "mp4",
        "media_format": resolve_format("auto"),
        "cluster_mode": cfgmod.cluster_mode(),
        "cluster_host": cfgmod.cluster_host(),
    }


@app.get("/checkpoints")
def checkpoints() -> list[dict]:
    """All resume-able checkpoints under RMG_RUNS_DIR, newest first."""
    rd = cfgmod.runs_dir()
    if not rd.exists():
        return []
    out = []
    for p in sorted(rd.glob("*/checkpoints/*.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
        out.append({
            "path": str(p),
            "run": p.parent.parent.name,
            "name": p.name,
            "mtime": p.stat().st_mtime,
        })
    return out


# --------------------------------------------------------------------------- generate


@app.post("/generate")
def generate(req: GenerateRequest) -> dict:
    st = get_state()
    ckpt_path = req.checkpoint or (
        str(cfgmod.default_checkpoint()) if cfgmod.default_checkpoint() else None
    )
    if not ckpt_path:
        raise HTTPException(
            503,
            "No checkpoint available. Set RMG_CHECKPOINT or copy a run with "
            "checkpoints/ down from the cluster into RMG_RUNS_DIR.",
        )

    key = cache.cache_key(
        kind="generate", text=req.text, guidance=req.guidance, num_steps=req.num_steps,
        seed=req.seed, num_frames=req.num_frames, ckpt=ckpt_path, fmt=resolve_format(req.fmt),
    )
    hit = cache.find_cached(key)
    if hit:
        return {
            "media_url": cache.media_url(hit),
            "joints_npy_url": cache.media_url(hit.with_suffix(".npy")),
            "cached": True, "text": req.text,
        }

    try:
        bundle = st.load_bundle(ckpt_path)
        sample = st.generate(
            bundle, text=req.text, num_frames=req.num_frames,
            guidance=req.guidance, num_steps=req.num_steps, seed=req.seed,
        )
        joints = decode_to_joints(sample, st.skeleton(), bundle.representation_name)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e
    except NotImplementedError as e:
        raise HTTPException(501, str(e)) from e

    media, npy = render_joints(
        joints, cache.media_path(key, "media"), title=req.text,
        fps=int(st.default_cfg.data.get("fps", 20)), fmt=req.fmt,
    )
    return {
        "media_url": cache.media_url(media),
        "joints_npy_url": cache.media_url(npy),
        "cached": False, "text": req.text, "step": bundle.step,
    }


# --------------------------------------------------------------------------- GT browser


@app.get("/gt")
def gt_list(
    subset_n: int = Query(0, ge=0),
    subset_fraction: float = Query(0.01, gt=0, le=1.0),
    subset_seed: int = Query(0),
    limit: int = Query(60, ge=1, le=500),
) -> list[dict]:
    st = get_state()
    root = cfgmod.data_root(st.default_cfg)
    if not (root / "humanml3d.zip").exists():
        raise HTTPException(
            503,
            f"Packed dataset not found at {root}/humanml3d.zip. Copy the GT samples "
            "down from the cluster and set RMG_DATA_ROOT.",
        )
    ids = gtmod.subset_train_ids(
        root, str(st.default_cfg.data.splits_name),
        subset_n=subset_n, subset_fraction=subset_fraction, subset_seed=subset_seed,
    )[:limit]
    return gtmod.list_captions(root, ids)


@app.get("/gt/{cid}")
def gt_clip(cid: str, fmt: str = "auto") -> dict:
    st = get_state()
    root = cfgmod.data_root(st.default_cfg)
    if not (root / "humanml3d.zip").exists():
        raise HTTPException(503, f"Packed dataset not found at {root}/humanml3d.zip.")

    key = cache.cache_key(kind="gt", cid=cid, fmt=resolve_format(fmt))
    hit = cache.find_cached(key)
    if hit:
        return {"media_url": cache.media_url(hit), "cid": cid, "cached": True}

    try:
        joints, caption = gtmod.clip_joints(root, cid, st.skeleton())
    except KeyError as e:
        raise HTTPException(404, f"clip {cid!r} not in packed dataset") from e
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e

    media, _ = render_joints(
        joints, cache.media_path(key, "media"), title=f"[{cid}] {caption[:60]}",
        fps=int(st.default_cfg.data.get("fps", 20)), fmt=fmt,
    )
    return {"media_url": cache.media_url(media), "cid": cid, "caption": caption, "cached": False}


# --------------------------------------------------------------------------- training viewer


@app.get("/runs")
def runs() -> list[str]:
    rd = cfgmod.runs_dir()
    if not rd.exists():
        return []
    return sorted(
        p.parent.name for p in rd.glob("*/samples") if any(p.glob("step-*.pt"))
    )


@app.get("/runs/{run_id}/steps")
def run_steps(run_id: str) -> list[int]:
    sdir = cfgmod.runs_dir() / run_id / "samples"
    if not sdir.exists():
        raise HTTPException(404, f"run {run_id!r} has no samples/ dir")
    steps = []
    for p in sdir.glob("step-*.pt"):
        try:
            steps.append(int(p.stem.split("-")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(steps)


@app.get("/runs/{run_id}/sample/{step}")
def run_sample(run_id: str, step: int, fmt: str = "auto") -> dict:
    """Render the fixed-prompt clips dumped at `step` (decode → FK → render)."""
    st = get_state()
    sdir = cfgmod.runs_dir() / run_id / "samples"
    pt = sdir / f"step-{step:09d}.pt"
    if not pt.exists():
        # tolerate non-zero-padded callers
        cands = list(sdir.glob(f"step-*{step}.pt"))
        if not cands:
            raise HTTPException(404, f"no sample dump for run {run_id!r} step {step}")
        pt = cands[0]

    # Representation of the run that produced these samples (dims → decode path).
    run_cfg = cfgmod.load_run_config(cfgmod.runs_dir() / run_id)
    rep_name = str(run_cfg.representation.name) if run_cfg else "tr"

    blob = torch.load(pt, map_location="cpu", weights_only=False)
    texts = list(blob["texts"])
    samples = blob["samples"]  # (B, T, ambient_dim)

    clips = []
    for i, text in enumerate(texts):
        key = cache.cache_key(kind="sample", run=run_id, step=int(step), idx=i, fmt=resolve_format(fmt))
        hit = cache.find_cached(key)
        if hit:
            clips.append({"text": text, "media_url": cache.media_url(hit), "cached": True})
            continue
        try:
            joints = decode_to_joints(samples[i], st.skeleton(), rep_name)
        except FileNotFoundError as e:
            raise HTTPException(503, str(e)) from e
        except NotImplementedError as e:
            raise HTTPException(501, str(e)) from e
        media, _ = render_joints(
            joints, cache.media_path(key, "media"),
            title=f"[step {step}] {text[:55]}",
            fps=int(st.default_cfg.data.get("fps", 20)), fmt=fmt,
        )
        clips.append({"text": text, "media_url": cache.media_url(media), "cached": False})

    return {"run": run_id, "step": step, "clips": clips}
