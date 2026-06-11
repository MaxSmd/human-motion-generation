# RMG Interactive — backend + frontend

A local web app to interact with the trained RMG model: prompt it, browse
ground-truth clips, and scrub through training-time sample dumps. See
[`../plan.md`](../plan.md) for the design.

- **Backend** (`server/`) — FastAPI. Keeps the model warm, renders motion to
  MP4/GIF on disk, serves it. A thin wrapper over `src/rmg`; nothing is ported.
- **Frontend** (`frontend/`) — Next.js + React + Tailwind. Three tabs
  (Generate / GT Browser / Training Viewer), plain `fetch` to the backend.

## Run with Docker Compose (the way to run it)

```bash
docker compose up --build
# frontend → http://localhost:3000   backend → http://localhost:8000
```

**Cluster mode is the default** (`RMG_CLUSTER_MODE=1`). The backend drives the
SLURM head node over SSH; compose mounts your `~/.ssh` read-only so it
authenticates exactly like you do by hand. Prerequisites:

1. **VPN up** and reachable from the container's network. The frontend shows a
   blocking "connect to VPN" gate until `ssh head` succeeds. *(macOS Docker
   Desktop runs in a VM — if the container can't reach the VPN, run the backend
   as a host process instead; see below.)*
2. **Host key trusted** once: `ssh head true` (writes `~/.ssh/known_hosts`,
   which is mounted in).
3. The three unified sbatch scripts (`slurm/rmg_{train,eval,viz}.sbatch`) present
   in the cluster's `~/riemann-motion-generation` checkout — i.e. **commit +
   push them, then `git pull` on the cluster**. The app never rsyncs scripts up;
   your git checkout is the single source of truth.

Jobs render on the GPU node; the backend rsyncs the media back into the
`rmg-media` volume and serves it. Nothing heavy runs locally.

GPU host: rebuild the backend with `--build-arg TORCH_INDEX_URL=…/cu121` and
uncomment the `deploy.resources` GPU block in `docker-compose.yml`.

### Local-files mode (no cluster)

Set `RMG_CLUSTER_MODE=0` in the compose `backend.environment` to use the older
local-render behavior (Generate / GT / Training from host-mounted artifacts).
Endpoints degrade gracefully (HTTP 503) when an artifact isn't present.

## Fallback: host-process backend (if the container can't reach the VPN)

On some macOS / VPN setups the Docker VM won't inherit the host VPN route. Then
run the **backend as a host process** (it natively uses your VPN + `~/.ssh`) and
keep the frontend in compose (or `npm run dev`):

```bash
# backend on the host
pip install -e . && pip install -r server/requirements.txt
RMG_CLUSTER_MODE=1 uvicorn server.app:app --reload --port 8000

# frontend
cd frontend && npm install && npm run dev   # http://localhost:3000
```

### Environment variables (backend)

| Var              | Meaning                                              | Default                                  |
|------------------|------------------------------------------------------|------------------------------------------|
| `RMG_RUNS_DIR`   | runs root (checkpoints + samples)                    | `<repo>/runs`                            |
| `RMG_DATA_ROOT`  | packed HumanML3D dir (zip, splits, offsets)          | repo `configs/data` default              |
| `RMG_OFFSETS`    | direct path to `target_offsets.pt` (overrides above) | `RMG_DATA_ROOT/target_offsets.pt`        |
| `RMG_CHECKPOINT` | explicit checkpoint for `/generate`                  | newest `*.pt` under `RMG_RUNS_DIR`       |
| `RMG_MEDIA_DIR`  | where rendered clips are written/served              | `<repo>/.media`                          |
| `RMG_CLUSTER_MODE` | `1` = SLURM control plane, `0` = local files       | `1`                                      |
| `RMG_CLUSTER_HOST` | SSH alias for the head node (`~/.ssh/config`)      | `head`                                   |
| `RMG_CLUSTER_RUNS` | remote runs dir                                    | `~/rmg-runs`                             |
| `RMG_CLUSTER_PROJECT` | remote git project (holds `slurm/*.sbatch`)     | `~/riemann-motion-generation`            |
| `RMG_SSH_CONTROL_PATH` | ControlMaster socket for the app session       | `~/.rmg-cm.sock`                         |

Frontend: `NEXT_PUBLIC_API_BASE` (baked at build time) — the URL the **browser**
uses to reach the backend, default `http://localhost:8000`.

## Endpoints

```
GET  /health                       backend status, device, ckpt, ffmpeg
GET  /checkpoints                  resume-able checkpoints under RMG_RUNS_DIR
POST /generate                     {text,guidance,num_steps,seed,num_frames,checkpoint?}
GET  /gt?subset_n=&subset_fraction=&subset_seed=   subset clip list (cid + caption)
GET  /gt/{cid}                     render a GT clip
GET  /runs                         runs that have sample dumps
GET  /runs/{run}/steps             dumped step numbers
GET  /runs/{run}/sample/{step}     render the fixed-prompt clips at that step
GET  /media/{file}                 rendered mp4/gif/npy
```

## Notes / placeholders

- Only the **T+R** (and T+R+P rotation) representations are wired for decode→FK.
  T+P (pre-shape) raises `501 Not Implemented` — see `render.decode_to_joints`.
- V1 is block-with-spinner (no job/poll or denoising-trajectory streaming).
- Constraint editor / in-browser 3D / eval panels are out of scope (V2+).
