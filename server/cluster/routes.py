"""FastAPI routes for the cluster control plane (mounted under /cluster)."""

from __future__ import annotations

import shlex

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import config as cfgmod
from . import jobs as jobsmod
from . import squeue as squeuemod
from . import ssh, status, submit

router = APIRouter(prefix="/cluster", tags=["cluster"])


def _require_mode() -> None:
    if not cfgmod.cluster_mode():
        raise HTTPException(503, "Cluster mode is off (set RMG_CLUSTER_MODE=1).")


def _require_online() -> None:
    _require_mode()
    st = status.probe()
    if st.state != "online":
        raise HTTPException(503, st.detail or f"cluster {st.state}")


# --------------------------------------------------------------------------- status


@router.get("/status")
def cluster_status() -> dict:
    return status.probe(force=True).to_dict()


@router.get("/squeue")
def cluster_squeue() -> list[dict]:
    _require_online()
    return squeuemod.squeue_me()


@router.get("/runs")
def cluster_runs() -> list[dict]:
    _require_online()
    return squeuemod.list_runs()


@router.get("/checkpoints")
def cluster_checkpoints(run: str) -> list[str]:
    _require_online()
    return squeuemod.list_checkpoints(run)


@router.post("/cancel/{slurm_id}")
def cluster_cancel(slurm_id: str) -> dict:
    _require_online()
    if not slurm_id.isdigit():
        raise HTTPException(400, "slurm_id must be numeric")
    ssh.run(f"scancel {shlex.quote(slurm_id)}", timeout=15, check=False)
    return {"cancelled": slurm_id}


# --------------------------------------------------------------------------- jobs


class VizRequest(BaseModel):
    mode: str = "clip"               # clip | prompt | compare | samples
    clips: str | None = None
    prompts: str | None = None
    checkpoint: str | None = None
    model_preset: str = "dit_base"
    train_preset: str = "rmg_base"
    num_frames: int = 100
    num_steps: int = 50
    guidance: float = 6.5
    use_ema: bool = True
    subset_fraction: float = 0.01
    subset_seed: int = 0
    run: str | None = None
    samples_file: str | None = None


class TrainRequest(BaseModel):
    model_preset: str = "dit_base"   # dit_base | dit_large
    train_preset: str = "rmg_base"   # rmg_base | rmg_large
    representation: str | None = None
    run_name: str | None = None
    max_steps: int | None = None
    subset_n: int | None = None
    subset_fraction: float | None = None
    subset_seed: int | None = None
    lr: float | None = None
    guidance: float | None = None
    precision: str | None = None
    overrides: str | None = None     # extra free-form hydra args


class EvalRequest(BaseModel):
    run: str | None = None
    checkpoint: str               # eval keys off the checkpoint
    model_preset: str = "dit_base"
    train_preset: str = "rmg_base"
    eval_split: str | None = None
    max_clips: int | None = None
    guidance_scales: str | None = None   # e.g. "[6.5]"
    subset_fraction: float | None = None
    subset_seed: int | None = None
    num_sample_steps: int | None = None
    use_ema: bool | None = None
    overrides: str | None = None


@router.post("/jobs/viz")
def submit_viz(req: VizRequest) -> dict:
    _require_online()
    try:
        job = jobsmod.get_manager().submit("viz", submit.build_viz, req.model_dump())
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e)) from e
    return job.to_dict()


@router.post("/jobs/train/preview")
def preview_train(req: TrainRequest) -> dict:
    _require_mode()
    try:
        script, env, job_name, run_name = submit.build_train(req.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"command": submit.render_command(script, env, job_name), "run_name": run_name}


@router.post("/jobs/train")
def submit_train(req: TrainRequest) -> dict:
    _require_online()
    try:
        job = jobsmod.get_manager().submit("train", submit.build_train, req.model_dump())
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e)) from e
    return job.to_dict()


@router.post("/jobs/eval")
def submit_eval(req: EvalRequest) -> dict:
    _require_online()
    try:
        job = jobsmod.get_manager().submit("eval", submit.build_eval, req.model_dump())
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e)) from e
    return job.to_dict()


@router.get("/jobs")
def list_jobs() -> list[dict]:
    return [j.to_dict() for j in jobsmod.get_manager().list()]


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = jobsmod.get_manager().get(job_id)
    if not job:
        raise HTTPException(404, f"no job {job_id}")
    return job.to_dict()


@router.get("/jobs/{job_id}/log")
def job_log(job_id: str, lines: int = 200) -> dict:
    _require_online()
    job = jobsmod.get_manager().get(job_id)
    if not job:
        raise HTTPException(404, f"no job {job_id}")
    return {"job_id": job_id, "log": jobsmod.get_manager().log_tail(job_id, lines)}
