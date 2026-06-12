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

# The cluster allows ONE job at a time. Extra jobs wait in a LOCAL queue (state
# "queued", not yet sbatch'd); the queue-runner promotes the next only when the
# single on-cluster slot is free, so we never have >1 job submitted on the cluster.
_ON_CLUSTER = {"submitting", "pending", "running", "pulling"}  # occupies the slot
_ACTIVE = _ON_CLUSTER | {"queued"}  # still "live" from the user's perspective (polled)

# our state ← SLURM state
_SLURM_RUNNING = {"RUNNING", "COMPLETING", "CONFIGURING"}
_SLURM_PENDING = {"PENDING", "REQUEUED", "RESIZING"}
_SLURM_OK = {"COMPLETED"}
_SLURM_FAIL = {"FAILED", "NODE_FAIL", "BOOT_FAIL", "OUT_OF_MEMORY", "DEADLINE"}
_SLURM_CANCEL = {"CANCELLED", "TIMEOUT", "PREEMPTED", "SUSPENDED"}

# kind → builder (validates params + resolves the sbatch command at enqueue time)
_BUILDERS = {"viz": submit.build_viz, "train": submit.build_train, "eval": submit.build_eval}


@dataclass
class ClusterJob:
    id: str
    kind: str                  # viz | train | eval
    mode: str = ""             # viz mode, etc.
    slurm_id: str | None = None
    run_name: str | None = None
    job_name: str | None = None
    state: str = "queued"      # queued|submitting|pending|running|pulling|done|failed|cancelled
    params: dict = field(default_factory=dict)
    command: str | None = None
    # Built at enqueue (validates params + resolves paths); deferred sbatch uses it.
    script: str | None = None
    env: dict = field(default_factory=dict)
    outputs: list = field(default_factory=list)  # [{media_url, npy_url, caption}]
    error: str | None = None
    queue_pos: int | None = None  # 1-based position among queued jobs (computed on read)
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

    def on_cluster_job(self) -> ClusterJob | None:
        """The one job currently occupying the cluster slot (if any)."""
        with self._lock:
            for j in self._jobs.values():
                if j.state in _ON_CLUSTER:
                    return j
        return None

    def _queued(self) -> list[ClusterJob]:
        return sorted(
            (j for j in self._jobs.values() if j.state == "queued"),
            key=lambda j: j.submitted_at,
        )

    def summary(self) -> dict:
        """Cheap, SSH-free snapshot for the global busy indicator (telemetry)."""
        with self._lock:
            active = next((j for j in self._jobs.values() if j.state in _ON_CLUSTER), None)
            queued = self._queued()
        return {
            "active": active.to_dict() if active else None,
            "queued": len(queued),
            "busy": active is not None or bool(queued),
        }

    def enqueue(self, kind: str, params: dict) -> ClusterJob:
        """Validate + build the job, add it to the LOCAL queue, and start it now if
        the cluster slot is free. Never blocks: extra jobs wait locally (not on the
        cluster). Raises ValueError/KeyError on bad params (→ 400)."""
        import secrets
        builder = _BUILDERS.get(kind)
        if builder is None:
            raise ValueError(f"unknown job kind {kind!r}")
        # Build now → validates params + resolves remote paths (raises on error).
        script, env, job_name, run_name = builder(params)
        job = ClusterJob(
            id=secrets.token_hex(6), kind=kind, mode=params.get("mode", ""),
            params=params, script=script, env=env, job_name=job_name, run_name=run_name,
            command=submit.render_command(script, env, job_name),
            state="queued", submitted_at=time.time(), updated_at=time.time(),
        )
        with self._lock:
            self._jobs[job.id] = job
        self._save()
        self._maybe_start_next()
        return job

    def _maybe_start_next(self) -> None:
        """If the cluster slot is free, sbatch the oldest queued job. Claims the job
        under the lock (→ 'submitting') so concurrent callers can't double-start."""
        with self._lock:
            if any(j.state in _ON_CLUSTER for j in self._jobs.values()):
                return
            nxt = min(self._queued(), key=lambda j: j.submitted_at, default=None)
            if nxt is None:
                return
            nxt.state = "submitting"  # claim the single slot
            nxt.updated_at = time.time()
        self._start(nxt)

    def _start(self, job: ClusterJob) -> None:
        """sbatch a claimed job (its command was built at enqueue)."""
        try:
            job.slurm_id = submit._submit(job.script, job.env, job.job_name)
            job.state = "pending"
        except Exception as e:  # noqa: BLE001
            job.state, job.error = "failed", str(e)
        job.updated_at = time.time()
        self._save()
        if job.state == "failed":
            self._maybe_start_next()  # couldn't even submit → let the queue advance

    def get(self, job_id: str) -> ClusterJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[ClusterJob]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.submitted_at, reverse=True)
            order = self._queued()
        pos = {j.id: i + 1 for i, j in enumerate(order)}
        for j in jobs:
            j.queue_pos = pos.get(j.id) if j.state == "queued" else None
        return jobs

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if not job:
            return False
        if job.state == "queued":  # local-only; nothing on the cluster to scancel
            job.state, job.updated_at = "cancelled", time.time()
            self._save()
            return True
        if not job.slurm_id:
            return False
        ssh.run(f"scancel {shlex.quote(job.slurm_id)}", timeout=15, check=False)
        job.state = "cancelled"
        job.updated_at = time.time()
        self._save()
        self._maybe_start_next()  # slot freed → start the next queued job
        return True

    def log_tail(self, job_id: str, lines: int = 300) -> str:
        """Tail the job's SLURM stdout AND stderr (rmg_*.sbatch write both under
        slurm/logs/<name>-<jobid>.{out,err}). `tail -v` prints `==> file <==`
        banners so the UI can tell the two streams apart."""
        job = self.get(job_id)
        if not job or not job.slurm_id:
            return ""
        sid = str(job.slurm_id)
        if not sid.isdigit():
            return "(invalid job id)"
        # Quote only the directory — the `*<jobid>` glob must stay UNQUOTED so the
        # remote shell expands it (a quoted `*` would be a literal filename and
        # match nothing). %j (job id), not --job-name, is what's in the filename.
        base = shlex.quote(f"{ssh.abs_remote(cfgmod.cluster_project_dir())}/slurm/logs")
        res = ssh.run(
            f"tail -v -n {int(lines)} {base}/*{sid}.out {base}/*{sid}.err 2>/dev/null",
            timeout=25, check=False,
        )
        return res.stdout or "(no log output yet)"

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
            active = [j for j in self._jobs.values() if j.state in _ON_CLUSTER and j.slurm_id]
        if not active:
            self._maybe_start_next()  # slot free → promote a queued job
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
            # A job may have just finished (done/failed/cancelled) → free the slot.
            self._maybe_start_next()

    # ----------------------------------------------------------------- pull media

    def _pull(self, job: ClusterJob) -> None:
        """rsync the job's viz/ dir down and register pulled clips as outputs."""
        if not job.run_name:
            job.state, job.error = "failed", "no run_name to pull"
            return
        remote = f"{ssh.abs_remote(cfgmod.cluster_runs_dir())}/{cfgmod.cluster_model()}/viz/{job.run_name}/viz/"
        local = cache.media_dir() / "jobs" / job.id
        local.mkdir(parents=True, exist_ok=True)
        try:
            ssh.rsync_pull(remote, str(local), timeout=180)
        except ssh.SSHError as e:
            job.state, job.error = "failed", f"rsync: {e}"
            return

        # The viz job writes a manifest.json mapping each rendered file to its
        # TRUE caption / clip id / kind. Read it so the viewer labels clips from
        # data, not from filename guessing — this is what keeps GT vs PRED tiles
        # aligned with the captions they were actually conditioned on.
        meta: dict[str, dict] = {}
        manifest_path = local / "manifest.json"
        if manifest_path.exists():
            try:
                for e in json.loads(manifest_path.read_text()):
                    f = e.get("file")
                    if f:
                        meta[f] = e
            except (ValueError, OSError):
                pass

        # Kind order so compare GT/PRED of the same clip sit next to each other:
        # sort by (clip_id, kind-rank), gt before pred. Files without manifest
        # metadata fall back to stem sorting.
        _KIND_RANK = {"gt": 0, "pred": 1}
        media_files = [*local.glob("*.gif"), *local.glob("*.mp4")]

        def _sort_key(m):
            e = meta.get(m.name, {})
            step = e.get("step")
            return (
                int(step) if isinstance(step, int) else -1,  # numeric, not lexical
                str(e.get("clip_id") or m.stem),
                _KIND_RANK.get(e.get("kind"), 9),
                m.name,
            )

        outputs = []
        for m in sorted(media_files, key=_sort_key):
            npy = m.with_suffix(".npy")
            e = meta.get(m.name, {})
            outputs.append({
                "media_url": f"/media/jobs/{job.id}/{m.name}",
                "npy_url": f"/media/jobs/{job.id}/{npy.name}" if npy.exists() else None,
                # caption = the real text (or stem fallback); label/kind/clip_id
                # let the UI render a proper GT/PRED badge + id + caption.
                "caption": e.get("caption") or m.stem,
                "label": m.stem,
                "kind": e.get("kind"),
                "clip_id": e.get("clip_id"),
                "step": e.get("step"),
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
