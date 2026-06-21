"""Async SLURM job registry: submit → poll (sacct) → rsync media → serve.

A daemon thread polls every active job's state via a single batched `sacct` call.
On COMPLETED it rsyncs the job's `viz/` dir down into the media cache and registers
the pulled clips as outputs. The registry is persisted to JSON so jobs survive a
backend restart (we reconcile against sacct on the next poll).
"""

from __future__ import annotations

import json
import os
import shlex
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .. import cache, config as cfgmod
from . import squeue, ssh, submit

POLL_INTERVAL = 5.0

# Safety cap on how many times a single train job auto-resubmits across walltime
# cycles (a multi-day 300k-step run needs dozens). Beyond this we stop and surface
# the job as failed rather than looping forever.
MAX_RESUBMITS = int(os.environ.get("RMG_MAX_RESUBMITS", "60"))

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

# (model, kind) → builder is resolved at enqueue time via `submit.builder_for`
# against the active model (cfgmod.cluster_model()), so the same kinds dispatch
# to rmg or MARDM scripts depending on the global toggle.


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
    # SBATCH CLI overrides (partition/walltime); reused on every (re)submit.
    sbatch_flags: list = field(default_factory=list)
    # Auto-resubmit across the cluster's 24h walltime (train jobs only). The job
    # keeps the SAME run_name → train.py resumes from latest.pt each cycle, until
    # it writes the `.complete` marker.
    auto_resubmit: bool = False
    resubmit_count: int = 0
    last_resubmit_step: int = -1  # progress step at the last resubmit (loop guard)

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
        # Resolve against the active model so the same kind dispatches to the
        # right rmg/MARDM script (raises ValueError on an unsupported task).
        builder = submit.builder_for(cfgmod.cluster_model(), kind)
        # Build now → validates params + resolves remote paths (raises on error).
        script, env, job_name, run_name = builder(params)
        # Train jobs carry partition/walltime overrides and opt into auto-resubmit
        # so a multi-day run survives the cluster's 24h walltime untended.
        sbatch_flags = submit.sbatch_flags_for_train(params) if kind == "train" else []
        auto_resubmit = kind == "train" and params.get("auto_resubmit", True)
        job = ClusterJob(
            id=secrets.token_hex(6), kind=kind, mode=params.get("mode", ""),
            params=params, script=script, env=env, job_name=job_name, run_name=run_name,
            command=submit.render_command(script, env, job_name, sbatch_flags),
            sbatch_flags=sbatch_flags, auto_resubmit=auto_resubmit,
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
            job.slurm_id = submit._submit(job.script, job.env, job.job_name, job.sbatch_flags)
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
        # queued (never sbatch'd) and paused (already scancelled) have nothing
        # live on the cluster — just retire them locally.
        if job.state in ("queued", "paused"):
            job.state, job.updated_at = "cancelled", time.time()
            self._save()
            self._maybe_start_next()
            return True
        if not job.slurm_id:
            return False
        ssh.run(f"scancel {shlex.quote(job.slurm_id)}", timeout=15, check=False)
        job.state = "cancelled"
        job.updated_at = time.time()
        self._save()
        self._maybe_start_next()  # slot freed → start the next queued job
        return True

    def delete(self, job_id: str) -> bool:
        """Remove a TERMINAL job (done/failed/cancelled) from the registry — pure
        history cleanup. Refuses to delete a live job (cancel it first), so it can
        never orphan something still on the cluster."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.state not in ("done", "failed", "cancelled"):
                return False
            del self._jobs[job_id]
        self._save()
        return True

    # ------------------------------------------------------------ pause / resume

    def pause(self, job_id: str) -> bool:
        """Interrupt a running/pending TRAIN job and free the single cluster slot
        (for ad-hoc eval/viz) WITHOUT losing the run: scancel it but keep its
        identity (run_name, resubmit counters) so `resume()` can pick up from
        latest.pt. Auto-resubmit will NOT relaunch it while paused."""
        job = self.get(job_id)
        if not job or job.kind != "train":
            return False
        if job.state not in ("pending", "running"):
            return False
        # Order matters: flip to 'paused' FIRST so the poller's on-cluster guard
        # skips this job and can't race the imminent CANCELLED sacct state into a
        # resubmit. Then scancel (sends SIGTERM → train.py flushes a final
        # checkpoint), then let any queued job take the freed slot.
        job.state = "paused"
        job.updated_at = time.time()
        if job.slurm_id:
            ssh.run(f"scancel {shlex.quote(job.slurm_id)}", timeout=15, check=False)
        self._save()
        self._maybe_start_next()
        return True

    def resume(self, job_id: str) -> bool:
        """Re-queue a paused train job. It re-submits under the SAME run_name, so
        train.py auto-resumes from latest.pt; resubmit counters are preserved.
        Starts immediately if the slot is free, else waits in the local queue."""
        job = self.get(job_id)
        if not job or job.state != "paused":
            return False
        job.slurm_id = None      # a fresh sbatch id is assigned on (re)start
        job.state = "queued"
        job.error = None
        job.updated_at = time.time()
        self._save()
        self._maybe_start_next()
        return True

    def train_progress(self, job_id: str) -> dict | None:
        """Live progress for a train job: checkpointed step + completion marker +
        max_steps (from the run's saved config), in one SSH round-trip. Returns
        None if the job is unknown or not a train job."""
        job = self.get(job_id)
        if not job or job.kind != "train":
            return None
        out = {
            "id": job.id, "run_name": job.run_name, "state": job.state,
            "slurm_id": job.slurm_id, "resubmit_count": job.resubmit_count,
            "step": None, "complete": False, "max_steps": None,
        }
        if not job.run_name:
            return out
        rundir = shlex.quote(submit.train_run_dir(job.run_name))
        sep = "\x1e"
        cmd = (
            f"cat {rundir}/.progress 2>/dev/null; printf '{sep}'; "
            f"test -f {rundir}/.complete && printf DONE; printf '{sep}'; "
            f"cat {rundir}/config.json 2>/dev/null"
        )
        try:
            raw = ssh.run(cmd, timeout=20, check=False).stdout
        except ssh.SSHError:
            return out  # transient — caller keeps last known values
        parts = (raw.split(sep) + ["", "", ""])[:3]
        prog, comp, cfg_text = parts
        if prog.strip().isdigit():
            out["step"] = int(prog.strip())
        out["complete"] = "DONE" in comp
        # max_steps: prefer the value the user submitted, else the saved config.
        ms = job.params.get("max_steps")
        if ms in (None, "", 0):
            try:
                ms = json.loads(cfg_text)["train"]["max_steps"] if cfg_text.strip() else None
            except (ValueError, KeyError, TypeError):
                ms = None
        out["max_steps"] = int(ms) if ms not in (None, "", 0) else None
        return out

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
            # The snapshot above may be stale: a job can leave the on-cluster set
            # between snapshot and here (e.g. the user paused/cancelled it). Skip
            # it so we never resubmit or re-classify a job that's no longer ours.
            if job.state not in _ON_CLUSTER:
                continue
            slurm_state = states.get(job.slurm_id)
            if not slurm_state:
                continue
            if slurm_state in _SLURM_PENDING and job.state != "pending":
                job.state, changed = "pending", True
            elif slurm_state in _SLURM_RUNNING and job.state != "running":
                job.state, changed = "running", True
            elif (
                job.kind == "train" and job.auto_resubmit
                and slurm_state in (_SLURM_OK | _SLURM_FAIL | _SLURM_CANCEL)
                and job.state not in ("done", "failed")
            ):
                # A train job left the cluster (walltime TIMEOUT, node fail, or a
                # clean exit). Decide done-vs-resume from the on-disk markers.
                # (User scancels never reach here: cancel() flips state out of the
                # on-cluster set before the next poll.)
                if self._handle_train_end(job, slurm_state):
                    changed = True
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

    # --------------------------------------------------------------- auto-resubmit

    def _train_markers(self, job: ClusterJob) -> tuple[int, bool]:
        """Read a train run's `.progress` (latest saved step) and `.complete`
        (training reached max_steps) markers in one SSH round-trip. Returns
        (step, complete); step is -1 if no checkpoint has landed yet. Raises
        ssh.SSHError on a transient connection failure so the caller can punt."""
        rundir = shlex.quote(submit.train_run_dir(job.run_name))
        # `|` separates the two fields; both halves may be empty.
        cmd = (
            f"cat {rundir}/.progress 2>/dev/null || true; printf '|'; "
            f"test -f {rundir}/.complete && printf DONE || true"
        )
        out = ssh.run(cmd, timeout=20, check=False).stdout
        prog, _, comp = out.partition("|")
        prog = prog.strip()
        return (int(prog) if prog.isdigit() else -1), ("DONE" in comp)

    def _handle_train_end(self, job: ClusterJob, slurm_state: str) -> bool:
        """A train job left the cluster. Mark it done if `.complete` exists, else
        resubmit it (resume from latest.pt under the same run_name) to ride out the
        24h walltime. Returns True if the job's state changed, False if we punted
        on a transient SSH error (re-decided on the next poll)."""
        try:
            step, complete = self._train_markers(job)
        except ssh.SSHError:
            return False  # VPN blip — leave state as-is, retry next poll

        if complete:
            job.state, job.error = "done", None
            return True

        # Loop guards: stop if we've hit the cap, or made no progress since the
        # last resubmit (a crash that never checkpoints would otherwise spin).
        if job.resubmit_count >= MAX_RESUBMITS:
            job.state = "failed"
            job.error = f"slurm: {slurm_state}; resubmit cap {MAX_RESUBMITS} hit at step {step}"
            return True
        if step <= job.last_resubmit_step:
            job.state = "failed"
            job.error = (
                f"slurm: {slurm_state}; not resubmitting — no checkpoint progress "
                f"since last submit (step {step} ≤ {job.last_resubmit_step})"
            )
            return True

        # Resume: re-sbatch the SAME script/env/flags → same run_name → train.py
        # auto-resumes from latest.pt.
        try:
            job.slurm_id = submit._submit(job.script, job.env, job.job_name, job.sbatch_flags)
            job.state, job.error = "pending", None
            job.resubmit_count += 1
            job.last_resubmit_step = step
        except Exception as e:  # noqa: BLE001
            job.state = "failed"
            job.error = f"resubmit after {slurm_state} failed: {e}"
        return True

    # ----------------------------------------------------------------- pull media

    def _pull(self, job: ClusterJob) -> None:
        """rsync the job's viz/ dir down and register pulled clips as outputs."""
        if not job.run_name:
            job.state, job.error = "failed", "no run_name to pull"
            return
        remote = f"{ssh.abs_remote(cfgmod.cluster_runs_dir())}/{cfgmod.cluster_model()}/viz/{job.run_name}/viz/"
        local = cache.media_dir() / "jobs" / job.id
        local.mkdir(parents=True, exist_ok=True)
        # The job can complete without writing a viz/ dir (e.g. all requested clip
        # ids missing from the dataset). rsync would then fail with an opaque
        # "change_dir … No such file or directory (code 23)"; probe first so we can
        # surface a useful reason instead.
        probe = ssh.run(f"test -d {shlex.quote(remote)}", timeout=15, check=False)
        if not probe.ok:
            job.state, job.error = "failed", (
                "job completed but wrote no viz/ output — likely no valid clips/prompts. "
                "Check the job log (the 'log' button)."
            )
            return
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
