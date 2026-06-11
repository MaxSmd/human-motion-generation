"""Async SLURM job registry: submit → poll (sacct) → rsync media → serve.

A daemon thread polls every active job's state via a single batched `sacct` call.
On COMPLETED it rsyncs the job's `viz/` dir down into the media cache and registers
the pulled clips as outputs. The registry is persisted to JSON so jobs survive a
backend restart (we reconcile against sacct on the next poll).
"""

from __future__ import annotations

import json
import shlex
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .. import cache, config as cfgmod
from . import squeue, ssh, submit

POLL_INTERVAL = 5.0

# our state ← SLURM state
_ACTIVE = {"submitting", "pending", "running", "pulling"}
_SLURM_RUNNING = {"RUNNING", "COMPLETING", "CONFIGURING"}
_SLURM_PENDING = {"PENDING", "REQUEUED", "RESIZING"}
_SLURM_OK = {"COMPLETED"}
_SLURM_FAIL = {"FAILED", "NODE_FAIL", "BOOT_FAIL", "OUT_OF_MEMORY", "DEADLINE"}
_SLURM_CANCEL = {"CANCELLED", "TIMEOUT", "PREEMPTED", "SUSPENDED"}


@dataclass
class ClusterJob:
    id: str
    kind: str                  # viz | train | eval
    mode: str = ""             # viz mode, etc.
    slurm_id: str | None = None
    run_name: str | None = None
    job_name: str | None = None
    state: str = "submitting"  # submitting|pending|running|pulling|done|failed|cancelled
    params: dict = field(default_factory=dict)
    command: str | None = None
    outputs: list = field(default_factory=list)  # [{media_url, npy_url, caption}]
    error: str | None = None
    submitted_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, ClusterJob] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._load()

    # ----------------------------------------------------------------- lifecycle

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="rmg-job-poller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ----------------------------------------------------------------- persistence

    def _load(self) -> None:
        p = cfgmod.jobs_state_path()
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text())
            for d in data:
                self._jobs[d["id"]] = ClusterJob(**d)
        except Exception:
            pass

    def _save(self) -> None:
        p = cfgmod.jobs_state_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = [j.to_dict() for j in self._jobs.values()]
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=0))
        tmp.replace(p)

    # ----------------------------------------------------------------- submit / query

    def submit(self, kind: str, builder, params: dict) -> ClusterJob:
        import secrets
        job = ClusterJob(
            id=secrets.token_hex(6), kind=kind, mode=params.get("mode", ""),
            params=params, submitted_at=time.time(), updated_at=time.time(),
        )
        with self._lock:
            self._jobs[job.id] = job
        try:
            info = submit.submit(builder, params)
            job.slurm_id = info["slurm_id"]
            job.run_name = info["run_name"]
            job.job_name = info["job_name"]
            job.command = info["command"]
            job.state = "pending"
        except Exception as e:
            job.state = "failed"
            job.error = str(e)
        job.updated_at = time.time()
        self._save()
        return job

    def get(self, job_id: str) -> ClusterJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[ClusterJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.submitted_at, reverse=True)

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if not job or not job.slurm_id:
            return False
        ssh.run(f"scancel {shlex.quote(job.slurm_id)}", timeout=15, check=False)
        job.state = "cancelled"
        job.updated_at = time.time()
        self._save()
        return True

    def log_tail(self, job_id: str, lines: int = 200) -> str:
        job = self.get(job_id)
        if not job or not job.slurm_id:
            return ""
        proj = ssh.abs_remote(cfgmod.cluster_project_dir())
        # visualize/train sbatch write slurm/logs/<name>-<jobid>.out
        glob = shlex.quote(f"{proj}/slurm/logs/*{job.slurm_id}.out")
        res = ssh.run(f"tail -n {int(lines)} {glob} 2>/dev/null", timeout=20, check=False)
        return res.stdout

    # ----------------------------------------------------------------- poll loop

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception:
                pass
            self._stop.wait(POLL_INTERVAL)

    def poll_once(self) -> None:
        with self._lock:
            active = [j for j in self._jobs.values() if j.state in _ACTIVE and j.slurm_id]
        if not active:
            return

        ids = ",".join(j.slurm_id for j in active)  # batched sacct
        try:
            res = ssh.run(
                f"sacct -j {shlex.quote(ids)} -X -n -P -o JobID,State", timeout=20, check=False
            )
        except ssh.SSHError:
            return  # transient VPN drop — leave states as-is
        states: dict[str, str] = {}
        for line in res.stdout.splitlines():
            jid, _, st = line.partition("|")
            if jid.strip():
                states[jid.strip()] = st.strip().split()[0] if st.strip() else ""

        changed = False
        for job in active:
            slurm_state = states.get(job.slurm_id)
            if not slurm_state:
                continue
            if slurm_state in _SLURM_PENDING and job.state != "pending":
                job.state, changed = "pending", True
            elif slurm_state in _SLURM_RUNNING and job.state != "running":
                job.state, changed = "running", True
            elif slurm_state in _SLURM_OK and job.state not in ("pulling", "done"):
                if job.kind == "viz":
                    job.state = "pulling"
                    self._pull(job)  # blocking rsync (we're on the poller thread)
                else:
                    # train/eval don't emit viz media — completion is enough.
                    # (Eval-metric pulling is a separate, later step.)
                    job.state = "done"
                changed = True
            elif slurm_state in _SLURM_FAIL:
                job.state, job.error, changed = "failed", f"slurm: {slurm_state}", True
            elif slurm_state in _SLURM_CANCEL:
                job.state, job.error, changed = "cancelled", f"slurm: {slurm_state}", True
            job.updated_at = time.time()
        if changed:
            self._save()

    # ----------------------------------------------------------------- pull media

    def _pull(self, job: ClusterJob) -> None:
        """rsync the job's viz/ dir down and register pulled clips as outputs."""
        if not job.run_name:
            job.state, job.error = "failed", "no run_name to pull"
            return
        remote = f"{ssh.abs_remote(cfgmod.cluster_runs_dir())}/{job.run_name}/viz/"
        local = cache.media_dir() / "jobs" / job.id
        local.mkdir(parents=True, exist_ok=True)
        try:
            ssh.rsync_pull(remote, str(local), timeout=180)
        except ssh.SSHError as e:
            job.state, job.error = "failed", f"rsync: {e}"
            return

        outputs = []
        media_files = sorted([*local.glob("*.gif"), *local.glob("*.mp4")])
        for m in media_files:
            npy = m.with_suffix(".npy")
            outputs.append({
                "media_url": f"/media/jobs/{job.id}/{m.name}",
                "npy_url": f"/media/jobs/{job.id}/{npy.name}" if npy.exists() else None,
                "caption": m.stem,
            })
        job.outputs = outputs
        job.state = "done" if outputs else "failed"
        if not outputs:
            job.error = "job completed but no media was produced"


# module singleton
MANAGER: JobManager | None = None


def get_manager() -> JobManager:
    global MANAGER
    if MANAGER is None:
        MANAGER = JobManager()
    return MANAGER
