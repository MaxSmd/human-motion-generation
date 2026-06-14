"""FastAPI routes for the cluster control plane (mounted under /cluster)."""

from __future__ import annotations

import shlex

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import config as cfgmod
from ..analysis import curves as curvesmod
from ..analysis import eval_tables
from ..analysis import joints as jointsmod
from . import gtbrowse
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


# --------------------------------------------------------------------------- model


@router.get("/model")
def get_model() -> dict:
    """Active model + the set of models the app can drive, plus which tasks each
    supports (so the UI can gate Generate/Visualize when a model has no viz)."""
    return {
        "active": cfgmod.cluster_model(),
        "models": list(cfgmod.MODELS),
        "tasks": {m: sorted(submit._BUILDERS.get(m, {})) for m in cfgmod.MODELS},
    }


class ModelRequest(BaseModel):
    model: str


@router.post("/model")
def set_model(req: ModelRequest) -> dict:
    """Set the global active model (rmg | mardm). Persisted across restarts."""
    try:
        active = cfgmod.set_cluster_model(req.model)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"active": active, "models": list(cfgmod.MODELS)}


# --------------------------------------------------------------------------- status


@router.get("/status")
def cluster_status() -> dict:
    # Cached (≈5 s TTL): the gate + page + cluster tab all poll this, so a forced
    # probe per request would hammer SSH. The TTL keeps it to ~one probe / 5 s.
    return {**status.probe().to_dict(), "model": cfgmod.cluster_model()}


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


@router.get("/run-sample-steps")
def cluster_run_sample_steps(run: str) -> list[dict]:
    """Saved sample steps for a run → [{step, path}], so the Visualize tab can
    let the user pick which steps to render instead of the whole directory."""
    _require_online()
    return squeuemod.list_sample_steps(run)


@router.get("/gt-clips")
def cluster_gt_clips(
    split: str = "train",
    subset_fraction: float = 1.0,
    subset_seed: int = 0,
    subset_n: int = 0,
    limit: int = 60,
    tag_seen: bool = False,
    q: str = "",
) -> list[dict]:
    """Available GT clips (id + assigned caption) for the Visualize tab. Plain
    browse by default; with `tag_seen` each clip is tagged `seen` for the run's
    training subset (fraction/seed/n) vs unseen. `q` searches captions across the
    whole dataset. Cached, SSH-light."""
    _require_online()
    return gtbrowse.gt_clips(
        split,
        subset_fraction=subset_fraction,
        subset_seed=subset_seed,
        subset_n=subset_n,
        limit=limit,
        tag_seen=tag_seen,
        q=q,
    )


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
    steps: list[int] | None = None   # mode=samples: render only these steps
    # mode=prompt: sampling-time constraints forwarded to visualize.py.
    constraints: list[dict] | None = None   # fixed joint angles (inpaint)
    ranges: list[dict] | None = None         # hinge limits (projection)
    scene: dict | None = None                # euclidean room/obstacles/spawn
    room_guidance: float = 0.0               # room/obstacle guidance weight


class TrainRequest(BaseModel):
    model_preset: str = "dit_base"   # dit_base | dit_large
    train_preset: str = "rmg_base"   # rmg_base | rmg_large
    representation: str | None = None
    run_name: str | None = None
    max_steps: int | None = None
    sample_every: int | None = None   # steps between periodic sample dumps
    ckpt_every: int | None = None     # steps between checkpoint saves
    subset_n: int | None = None
    subset_fraction: float | None = None
    subset_seed: int | None = None
    lr: float | None = None
    guidance: float | None = None
    precision: str | None = None
    overrides: str | None = None     # extra free-form hydra args
    # MARDM-only: two-stage training. stage ∈ {ae, gen}; gen needs an AE ckpt.
    stage: str | None = None
    ae_checkpoint: str | None = None


class EvalRequest(BaseModel):
    run: str | None = None
    checkpoint: str               # eval keys off the checkpoint (gen ckpt for MARDM)
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
    ae_checkpoint: str | None = None     # MARDM-only: the stage-1 AE checkpoint


@router.get("/queue")
def queue_summary() -> dict:
    """Cheap, SSH-free busy/queue snapshot for the global indicator (telemetry)."""
    return jobsmod.get_manager().summary()


@router.post("/jobs/viz")
def submit_viz(req: VizRequest) -> dict:
    _require_online()
    try:
        job = jobsmod.get_manager().enqueue("viz", req.model_dump())
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
        job = jobsmod.get_manager().enqueue("train", req.model_dump())
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e)) from e
    return job.to_dict()


@router.post("/jobs/eval")
def submit_eval(req: EvalRequest) -> dict:
    _require_online()
    try:
        job = jobsmod.get_manager().enqueue("eval", req.model_dump())
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


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    """Cancel one of our jobs — drops it from the local queue if not yet started,
    or scancels it if it's on the cluster."""
    ok = jobsmod.get_manager().cancel(job_id)
    if not ok:
        raise HTTPException(404, f"no cancellable job {job_id}")
    return {"cancelled": job_id}


@router.get("/jobs/{job_id}/log")
def job_log(job_id: str, lines: int = 300) -> dict:
    _require_online()
    job = jobsmod.get_manager().get(job_id)
    if not job:
        raise HTTPException(404, f"no job {job_id}")
    return {"job_id": job_id, "log": jobsmod.get_manager().log_tail(job_id, lines)}


# --------------------------------------------------------------------------- eval / analysis


@router.get("/eval-runs")
def eval_runs() -> list[dict]:
    """Runs that have eval/results.json (across both runs roots)."""
    _require_online()
    return eval_tables.list_eval_runs()


@router.get("/eval/{run}")
def eval_results(run: str) -> dict:
    """Full per-guidance metrics for one run."""
    _require_online()
    try:
        per = eval_tables.fetch_results(run)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    # JSON keys must be strings.
    return {"run": run, "results": {str(k): v for k, v in per.items()}}


@router.get("/analysis/table")
def analysis_table(runs: str) -> dict:
    """Run-comparison rows + a copy-ready LaTeX tabular. `runs` = comma-separated."""
    _require_online()
    run_list = [r.strip() for r in runs.split(",") if r.strip()]
    if not run_list:
        raise HTTPException(400, "pass ?runs=a,b,c")
    return eval_tables.comparison(run_list)


@router.get("/metrics")
def run_metrics(run: str) -> dict:
    """Parsed training-curve data from a run's metrics.csv."""
    _require_online()
    try:
        return curvesmod.fetch_metrics(run)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e


@router.get("/run-info")
def run_info(run: str) -> dict:
    """A run's saved config.json + approx wall-clock duration."""
    _require_online()
    try:
        return curvesmod.fetch_run_info(run)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e


@router.get("/analysis/npy")
def analysis_npy(job: str, name: str) -> dict:
    """Trajectory/jitter/speed/foot-height series from a pulled job .npy (local)."""
    _require_mode()
    try:
        return jointsmod.analyze(jointsmod.resolve_npy(job, name))
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(404, str(e)) from e
