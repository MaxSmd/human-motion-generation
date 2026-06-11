# RMG Interactive — Cluster Control Plane (Design Draft)

Extends the local app (`plan.md`, `server/`, `frontend/`) into a **remote control
plane** for the SLURM cluster: gate the UI on VPN reachability, browse/render
motion by submitting jobs on the cluster and pulling media back, monitor
`squeue`, launch viz/training/eval jobs, and watch their logs + metrics.

> **Status:** design only. No code in this doc — it's the blueprint to build from.

---

## 0. Ground truth (your setup)

| Thing | Value |
|---|---|
| SSH alias | `head` (= `slurm` = `cluster`) → `131.159.11.60`, key-based, **behind VPN** |
| Remote project | `~/riemann-motion-generation` (git) |
| Remote runs | `~/rmg-runs/<run>/{checkpoints,samples,viz,…}` |
| Remote eval | `~/riemann-motion-generation/runs/…` |
| Job runner | SLURM + `enroot` image `~/rmg.sqsh`; partition `24g`, `qos=students_normal`, `gpu:1` |
| Existing sbatch | `slurm/{visualize,train_rmg_base,train_rmg_large,train_ablation,evaluate,sanity_eval}.sbatch` — all parameterized by **env vars** (`MODE`, `CKPT`, `PROMPTS`, `RUNS_ROOT`, …) |
| Viz output | `~/rmg-runs/<RUN_NAME>/viz/*.gif` (+ `.npy` joints alongside) |

**Chosen data strategy:** *render on the cluster (real GPU/FK), then rsync the
media back.* Every clip/sample/prompt becomes (or joins) a SLURM job; the backend
tracks it, pulls `viz/*.gif` on completion, and serves it. Deterministic renders
are cached so repeats are instant.

---

## 1. Architecture

```
 Browser ──HTTP──▶ FastAPI backend ──ssh/rsync/sbatch──▶ head node ──SLURM──▶ GPU node
   │                   │   (control plane, holds job registry + media cache)      │
   │                   └──◀ rsync viz/*.gif ◀── ~/rmg-runs/<run>/viz ◀────────────┘
   └──◀ /media/<hash>.gif (served once pulled)
```

The backend stops being "warm model in memory" and becomes a **cluster broker**:
it shells out to `head` over the user's own SSH config, submits jobs, polls
their state, and syncs artifacts back. **No model and no heavy rendering ever run
locally.** The laptop only does *cheap* work on already-pulled tiny artifacts
(metric plots, `.npy` analysis plots, run-comparison tables — see §1.2).

### 1.1 Where it runs (LOCKED)

**Frontend + backend both run on the host laptop (M4). Backend = host process**
(`uvicorn` in the project venv), so SSH/rsync/VPN "just work" with the user's
existing `~/.ssh/config` — exactly like opening an SSH session by hand today.
The frontend is just HTTP to `localhost:8000` (Docker or `next` — doesn't matter).

- **No permanent process on the cluster.** We never hold a node / `salloc` /
  daemon. Connectivity is a normal **SSH ControlMaster** opened when the app
  starts and torn down when it stops (multiplexes many fast commands over one
  temporary connection — same footprint as the interactive SSH the user already
  keeps open).
- **macOS + Docker caveat (why backend is a host process, not a container):**
  Docker Desktop on Mac runs in a VM whose network does **not** transparently
  inherit the host VPN route, and SSH-agent/`~/.ssh` forwarding into the VM is
  fiddly. A host-process backend sidesteps all of it. *If* containerizing the
  backend is ever required, the escape hatch is to bind-mount a host-side SSH
  **ControlMaster socket** into the container and run all `ssh`/`rsync` through
  it (the VPN+auth stay on the host). Not the default.

`RMG_CLUSTER_MODE=1` switches the backend from "local files" to "cluster broker".
When off, today's local behavior is unchanged.

### 1.2 Compute split — what runs WHERE (LOCKED)

| Runs on **cluster** (heavy: needs model or full skeleton render) | Runs on **host** (cheap, fast, fine in a container) |
|---|---|
| Text→motion **generation / prompt sampling** (loads the model) | Pulling + caching media; serving `/media` |
| **Animated skeleton render** → GIF (clip / compare / samples) via `visualize.sbatch` | **Metric plots** from `metrics.csv` (matplotlib) |
| **Training** runs | **`.npy` analysis plots**: root trajectory, per-joint velocity, **jitter-over-time**, foot-contact — from the tiny `.npy` joints `visualize.py` dumps next to each GIF |
| **Evaluation** compute (FID / R-precision / jitter) | **Run-comparison tables**, incl. **LaTeX-rendered** paper-style tables (see §4.5) |

Rule of thumb: *if it loads the model or renders the animated skeleton, it goes
to the cluster.* Everything else is small enough for the M4 and stays local. The
local `server/render.py` animator is **bypassed** in cluster mode.

---

## 2. Connection gate (VPN reachability)

### Backend `GET /cluster/status`
Fast, cached (≈5 s TTL) probe, classifying the failure so the UI can be specific:

```
ssh -o BatchMode=yes -o ConnectTimeout=4 head 'hostname; sinfo -h -o %P | head'
```

| Result | `state` | UI message |
|---|---|---|
| timeout / no route to 131.159.11.60 | `vpn_down` | "Cluster unreachable — connect to the VPN." |
| connection refused / auth failure | `ssh_error` | "VPN up, SSH failing — check key / `~/.ssh/config`." |
| ok | `online` | unlock app; show node/partition summary |

Return `{ state, host, hostname?, partitions?, user?, latency_ms, checked_at }`.

### Frontend gate
A top-level `<ClusterGate>` polls `/cluster/status` every ~5 s. While `state !=
online`, render a **full-screen blocking overlay** (instrument "no-signal"
screen) with the reason + a "retry now" button; the tabs mount only when online.
A persistent header pill shows live link state + latency once connected.

---

## 3. Job model (the spine of everything)

One generic async abstraction backs viz, train, and eval.

```
ClusterJob {
  id            # our uuid
  kind          # "viz" | "train" | "eval"
  slurm_id      # set after sbatch
  state         # SUBMITTING → PENDING → RUNNING → PULLING → DONE | FAILED | CANCELLED
  params        # the request (mode, ckpt, prompts, …)
  run_name      # RUN_NAME passed to sbatch → where outputs land
  outputs       # [{ media_url, npy_url, caption }]  (filled after rsync)
  log_tail      # last N lines of slurm/logs/*-<slurm_id>.out
  submitted_at, started_at, finished_at, error?
}
```

**Lifecycle (backend job manager, one background poller):**
1. `sbatch` the relevant script with env vars → capture `Submitted batch job N`.
2. Poll `squeue --me` / `sacct -j N` → map SLURM state to ours (`PD→PENDING`,
   `R→RUNNING`, `CD→` pull, `F/CA/TO→FAILED/CANCELLED`).
3. On completion: `rsync -az head:~/rmg-runs/<run>/viz/ <media_dir>/<job>/` →
   register each `*.gif` (+ `.npy`) as `/media/...` outputs.
4. Stream `log_tail` on demand by `tail`-ing the job's `.out`/`.err`.

**Caching / batching (mitigate queue latency):**
- Cache key = hash(kind, params) → if media already pulled, return instantly (no
  job). Pre-warm GT + sample renders.
- `visualize.py` already renders *many* clips/prompts per invocation — submit a
  **whole subset / filmstrip in one job** rather than one job per clip. The UI
  shows a single job producing N tiles. This is the main latency lever (one queue
  wait, one enroot start, N renders).
- **No persistent cluster process** (per the locked decision): we do *not* hold a
  GPU with `salloc`/a render daemon. Each render is an independent `sbatch` whose
  enroot startup we accept; batching (above) amortizes it. SSH overhead is hidden
  by the session ControlMaster (§1.1).

---

## 4. Feature surfaces

### 4.1 squeue monitor (Cluster tab)
- `GET /cluster/squeue` → parse `squeue --me -o '%i|%j|%T|%M|%D|%R'` into rows
  (jobid, name, state, time, nodes, reason). Auto-refresh ~5 s.
- `POST /cluster/cancel/{slurm_id}` → `scancel`. Confirm in UI.
- Show our backend-launched jobs enriched with `kind` + output links; show
  foreign jobs read-only.

### 4.2 Viz submission (wires Generate / GT / Training tabs to the cluster)
Map each existing tab action to a `visualize.sbatch` env payload:

| Tab action | sbatch env |
|---|---|
| Generate (prompt) | `MODE=prompt CKPT=… MODEL_PRESET=… PRESET=… PROMPTS='…' NUM_FRAMES NUM_STEPS GUIDANCE USE_EMA` |
| GT clip(s) | `MODE=clip CLIPS=000021,…` |
| GT compare (vs prediction) | `MODE=compare CKPT=… CLIPS=… SUBSET_FRACTION SUBSET_SEED` |
| Training samples | `MODE=samples SAMPLES_FILE=~/rmg-runs/<run>/samples` |

- `POST /cluster/jobs/viz` `{mode, …}` → returns `job.id`; frontend polls
  `GET /cluster/jobs/{id}` until `DONE`, then renders the pulled tiles in the
  existing `MediaViewer`/grid. The spinner becomes a **job-state chip**
  (PENDING/RUNNING/PULLING) with elapsed time + a "view log" popover.
- `GET /cluster/runs` → `ssh head 'ls ~/rmg-runs'` + per-run probe (has
  checkpoints? has samples?) to populate run/checkpoint dropdowns from the
  cluster instead of local disk.

### 4.3 Training control (new tab section)
- Render a **config form** from `configs/` presets: model preset (`dit_base/large`),
  train preset (`rmg_base/large`), `subset_n`/`subset_fraction`/`subset_seed`,
  `max_steps`, `lr`, `guidance_scale`, `prior_sigma`, `precision`, `run_name`.
- **Dry-run preview**: show the exact `sbatch` command + resolved env before
  launching (no surprises). Guardrail: explicit "Launch training" confirm.
- `POST /cluster/jobs/train` → picks `train_rmg_base|large|ablation.sbatch`,
  passes overrides via env / Hydra-style args, returns the job. Track in squeue
  like any other job; surface `metrics.csv` + periodic sample dumps as they
  appear (ties into 4.2 `MODE=samples`).
- **Safety:** never auto-cancel; confirm launches/cancels; cap concurrent
  training jobs we start; tag all our jobs with a recognizable `--job-name`.

### 4.4 Evaluation (new tab) — *added per your note*
**Compute on cluster, present on host.** The FID/R-precision/jitter numbers need
the eval pipeline + model, so they run on the GPU node; the laptop only pulls the
tiny CSV/PNG artifacts and renders the views (§1.2).
- `POST /cluster/jobs/eval` → submit `evaluate.sbatch` (or `sanity_eval.sbatch`)
  for a chosen checkpoint/run. Track as a job.
- On completion, pull the small artifacts under `~/rmg-runs/<run>/`
  (`metrics.csv`, `*_metrics.png`, `jitter-*.png`, eval logs):
  - **Metric panels:** FID, R-precision (top-1/2/3), diversity, multimodality,
    jitter — parsed from `metrics.csv` (produced by `src/rmg/eval/*`).
  - **Plots:** show the rsync’d PNGs inline, *and/or* recompute cheap ones locally
    from `metrics.csv` for interactivity (zoom, overlay multiple runs).
- `GET /cluster/eval/{run}` → cached parsed metrics for instant re-display.

### 4.5 Local analysis & LaTeX tables (host, fast — *new*)
Everything here is cheap M4 compute on already-pulled tiny files; **no model, no
animation**.

- **`.npy` analysis plots.** `visualize.py` dumps `<clip>.npy` (T×22×3 joints)
  next to every GIF. Pull just the `.npy` (kilobytes) and compute static charts
  locally with matplotlib: root-trajectory (top-down path), per-joint speed,
  **jitter / 3rd-derivative over time**, foot-contact/skating, bone-length drift.
  `GET /analysis/npy?job=&idx=&kind=trajectory|jitter|speed|contacts`.
- **Training-curve plots.** Pull a run's `metrics.csv`/`train.log`, plot loss /
  metric curves locally; overlay several runs for comparison.
- **Run-comparison tables with LaTeX rendering** *(your ask).* A results table
  across runs/checkpoints (FID ↓, R@1/2/3 ↑, diversity, MM, jitter), best-per-
  column highlighted like a paper table. Two LaTeX touches:
  - inline math via **KaTeX** in headers/cells (`\text{FID}\downarrow`, `\pm`
    std, scientific notation);
  - a **"Copy LaTeX"** button that emits a ready `\begin{tabular}{lcccc} … \end{tabular}`
    (booktabs style, bold best) for pasting straight into the report.
  - `GET /analysis/table?runs=a,b,c` → normalized rows + a `latex` string the
    frontend can render (KaTeX) and copy.
- All of this works **offline** once artifacts are cached — only the *jobs*
  (render/train/eval) need the live VPN.

---

## 5. Endpoint summary (additions)

```
GET  /cluster/status                  reachability + classification (gate)
GET  /cluster/squeue                  live job table
POST /cluster/cancel/{slurm_id}       scancel (confirmed)
GET  /cluster/runs                    runs on the cluster (+ has ckpt/samples)
GET  /cluster/checkpoints?run=        checkpoints under a run

POST /cluster/jobs/viz                submit visualize.sbatch (clip|prompt|compare|samples)
POST /cluster/jobs/train              submit train_*.sbatch from a config form (dry-run first)
POST /cluster/jobs/eval               submit evaluate.sbatch
GET  /cluster/jobs                    our tracked jobs
GET  /cluster/jobs/{id}               one job: state, outputs, log_tail
GET  /cluster/jobs/{id}/log           streamed tail of .out/.err
GET  /cluster/eval/{run}              parsed eval metrics + plot URLs

# local, fast — compute on host from pulled artifacts (no VPN needed once cached)
GET  /analysis/npy                    {job,idx,kind} → static plot from pulled .npy
GET  /analysis/curves?run=            training-curve plot from metrics.csv/train.log
GET  /analysis/table?runs=a,b,c       run-comparison rows + a copy-ready LaTeX tabular

(reuses) GET /media/{file}            pulled gif/mp4/npy/png + locally-made plots
```

---

## 6. Backend modules (new)

- `server/cluster/ssh.py` — thin `ssh`/`rsync`/`scp` wrappers over the user's
  `~/.ssh/config` alias (`RMG_CLUSTER_HOST=head`); `BatchMode`, timeouts, no
  shell injection (argv lists, careful quoting of Hydra comma args — see
  `visualize.sbatch`’s escaping notes).
- `server/cluster/status.py` — reachability probe + classification + TTL cache.
- `server/cluster/jobs.py` — `ClusterJob` registry + background poller
  (`squeue`/`sacct`) + post-completion rsync; persists to a small on-disk JSON so
  jobs survive a backend restart.
- `server/cluster/submit.py` — env payload builders for viz/train/eval; dry-run
  command renderer.
- `server/cluster/eval.py` — pull + parse `metrics.csv` / plots.
- `server/analysis/plots.py` — **host-side** matplotlib from pulled `.npy`
  (trajectory/jitter/speed/contacts) + training curves. Cheap; reuses the math in
  `scripts/plot_jitter.py` / `plot_metrics.py`.
- `server/analysis/tables.py` — normalize metrics across runs + emit a booktabs
  LaTeX `tabular` (bold best per column) for the frontend's "Copy LaTeX".
- `server/config.py` — add `RMG_CLUSTER_MODE`, `RMG_CLUSTER_HOST` (`head`),
  `RMG_CLUSTER_RUNS` (`~/rmg-runs`), `RMG_CLUSTER_PROJECT`
  (`~/riemann-motion-generation`), `RMG_SSH_OPTS`, `RMG_SSH_CONTROL_PATH`
  (ControlMaster socket for the app session).

---

## 7. Frontend additions

- `<ClusterGate>` wrapper (full-screen no-signal overlay until `online`).
- Header link-state pill (state + latency + node count).
- New **Cluster** tab: `squeue` table (live, scancel), our-jobs list with
  state chips + log popovers.
- New **Train** + **Eval** sections (config form w/ dry-run preview; metric
  panels + plots).
- **Analysis** views: `.npy` static plots + training curves; a **run-comparison
  results table with LaTeX** rendered via **KaTeX** (add `katex` dep) and a
  "Copy LaTeX" button emitting a booktabs `tabular`.
- Generate/GT/Training tabs: swap the local spinner for the async **job-state
  chip** (PENDING→RUNNING→PULLING→tiles), driven by `GET /cluster/jobs/{id}`.

---

## 8. Cross-cutting concerns

- **Security / blast radius:** all actions run as the user on their own cluster
  via their keys. Still: confirm every launch/cancel; dry-run training; argv
  (never string-interpolate user text into shell); cap concurrent self-launched
  jobs; recognizable `--job-name` prefix (`rmgui-…`) so the user can audit.
- **Latency UX:** queue waits are minutes — lean on caching, batching, optimistic
  job chips, and "you can leave; it'll be here when it's done" (jobs persist).
- **Resilience:** poller tolerates transient VPN drops (job state unknown ≠
  failed); reconcile against `sacct` on reconnect; gate flips UI to "reconnect".
- **known_hosts / first connect:** document a one-time `ssh head true` so the
  host key is trusted before the backend probes (avoid interactive prompt).
- **No GPU locally:** in cluster mode the local renderer is unused; everything
  visual comes from pulled media.

---

## 9. Build order (when we implement)

1. `cluster/ssh.py` (host process + session **ControlMaster**) + `/cluster/status`
   + frontend **gate** (smallest end-to-end proof VPN/SSH works and the UI
   locks/unlocks).
2. `/cluster/squeue` + **Cluster tab** table + scancel.
3. Job manager (`jobs.py`) + `/cluster/jobs/viz` (`MODE=clip`) → submit→poll→rsync
   →serve the pulled GIF in the GT tab (the core loop).
4. Extend viz to prompt/compare/samples; wire Generate + Training tabs to jobs.
5. `/cluster/jobs/train` with config form + dry-run preview.
6. `/cluster/jobs/eval` + metric panels/plots.
7. **Local analysis**: `.npy` plots, training curves, and the **LaTeX
   run-comparison table** (KaTeX + Copy-LaTeX) — all host-side, works offline.
8. Polish: persistence, batching, caching/pre-warm, reconnect reconciliation.

---

## 10. Open decisions (resolve at build start)

- ~~Backend host vs container~~ → **LOCKED: host-process backend** (§1.1).
- ~~Generate path~~ → **LOCKED: always cluster** (loads the model ⇒ GPU node).
- ~~Data strategy~~ → **LOCKED: render on cluster + rsync media** (§0).

Still open:
1. **Job granularity:** confirm batched filmstrip per job (recommended) vs one
   job per render.
2. **sacct availability:** confirm `sacct` is enabled for post-exit state, else
   use "left `squeue` + output file present" as the done signal. *(verify on next
   cluster login)*
3. **known_hosts:** one-time `ssh head true` so the host key is trusted before
   the backend's ControlMaster opens (avoid an interactive prompt).
4. **Training guardrails:** max concurrent self-launched train jobs; fixed safe
   subset of editable Hydra overrides vs free-form.
5. **ControlMaster lifecycle:** auto-open on first cluster call and auto-close on
   backend shutdown, vs piggyback on a user-opened `ssh head` session.
```
