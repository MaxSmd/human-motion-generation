# RMG Interactive Frontend — Plan

A locally-hosted web app to interact with the trained RMG model: prompt it,
browse ground-truth data, and step through training-time sample dumps.

**V1 decisions (locked):**
- **Rendering:** server-rendered video (reuse `scripts/visualize.py`'s renderer).
  No in-browser 3D — React just shows the rendered clip + a control panel.
- **V1 scope:** (1) Generate from prompt, (2) GT data browser, (3) Training
  sample viewer. **Constraint editor is explicitly out of scope for V1.**

**Core architectural fact:** the model, forward kinematics, and manifold math
are PyTorch. They live in a small **Python (FastAPI) backend** that keeps the
model warm in memory. **React is a thin client** — forms + an image/video
viewer — that calls the backend over HTTP and displays the rendered media it
returns. Nothing is ported to JS.

---

## 1. How this maps onto the existing codebase

The backend is mostly *thin wrappers* over code that already exists. Key
modules (`src/rmg/`):

| Module | Role | Used by |
|---|---|---|
| `representation/` (`registry`, `tplusr`, `skeleton`) | encode/decode manifold ↔ raw; **forward kinematics** (quats → 3D joints) | all 3 features (FK before render) |
| `manifolds/` (`sphere`, `euclidean`, `product`) | geodesic / exp / log / project_tangent | generation (inside sampler) |
| `flow/sampler.py` (`RiemannianEulerSampler.sample`) | **inference entry point** — text cond → motion | Generate |
| `flow/prior.py` (`WrappedGaussianPrior`) | start-noise for sampling | Generate |
| `models/text_encoder.py` (`Qwen3EmbeddingEncoder`) | prompt → `cond` embedding | Generate |
| `models/dit.py` (`RMGDiT`) | velocity network | Generate |
| `data/humanml3d.py` (`HumanML3DDataset`) | GT clips, subset selection | GT browser |
| `utils/checkpoint.py` (`find_latest_checkpoint`, `load_checkpoint`) | load weights | startup / ckpt list |
| `utils/ema.py` (`EMA`) | EMA weights for sampling | Generate |
| `eval/` (`guo_evaluator`, `metrics`) | FID / R-precision / jitter | (optional, later) |

Helpers in `scripts/visualize.py` to **reuse directly** (refactor to be
importable / return a media path instead of CLI-only):
- `_build_model(cfg, representation, use_ema)`, `_build_sampler(cfg, representation, skel)`
- `_subset_train_ids(data_root, splits_name, fraction, seed)` — GT browser listing
- `_render(...)` and the `.npy` joint dump — produces the rendered clip
- `forward_kinematics`, `tplusr_decode`, `_T2M_CHAINS` — decode → FK → skeleton

Training-sample dump format (from `scripts/train.py:345-362`):
`runs/<run>/samples/step-*.pt` → `{"texts": [...], "samples": (B,T,D) tensor}`.

> **Note:** `visualize.py` currently renders **MP4** (matplotlib animation +
> ffmpeg). Frontend uses `<video>` for MP4, or we add a GIF export option for
> `<img>`. Decide during build; MP4 is the smaller/cheaper default.

---

## 2. Backend — FastAPI

Single process. Loads checkpoint + sampler **once at startup** (slow; keep warm).
Each endpoint renders to a media file on disk and returns its URL. **Cache** by a
hash of request params so repeats are instant; GT and training-sample renders are
deterministic and worth pre-warming.

### Endpoints

```
GET  /health                         → { ckpt, device, representation, ambient_dim }
GET  /checkpoints                    → [ckpt paths]            (find_latest_checkpoint)

# Feature 1 — Generate from prompt
POST /generate {text, guidance, num_steps, seed}
     → text_encoder.encode → sampler.sample → FK → _render
     → { media_url, joints_npy_url }
     cache key = hash(text, guidance, num_steps, seed, ckpt)

# Feature 2 — GT data browser
GET  /gt?subset_n=&subset_seed=      → [{ cid, caption }]      (_subset_train_ids + dataset)
GET  /gt/{cid}                       → { media_url, caption }  (render GT clip; cache by cid)

# Feature 3 — Training sample viewer
GET  /runs                           → [run_id]                (scan RMG_RUNS_DIR)
GET  /runs/{run_id}/steps            → [step numbers]          (list step-*.pt)
GET  /runs/{run_id}/sample/{step}    → { gifs: [{ text, media_url }] }
                                        (load step-*.pt → decode → FK → render; cache)

# static media
GET  /media/{hash}.{mp4|gif}         → rendered files
```

### New backend code (the only real work)
- `server/app.py` — FastAPI app, CORS for `localhost:5173`, startup model load.
- `server/render.py` — in-process render: `(joints) → media path`, refactored
  out of `visualize.py`'s `_render`.
- `server/cache.py` — params-hash → media path dict (+ on-disk media dir).
- Light refactor of `visualize.py` so its helpers import cleanly (no Hydra-only
  / CLI-only assumptions in the reused functions).

---

## 3. Frontend — React (Vite)

Thin client. Three tabs; each is a form + media viewer + loading spinner
(generation takes a few seconds). No state library needed; `fetch` to
`http://localhost:8000`.

- **Generate** — text input + guidance / num_steps / seed controls → "Generate"
  → shows returned clip. Keep a session history strip of past generations.
- **GT browser** — subset selector (`subset_n`, `subset_seed`) → list of clips
  with captions → click → load clip.
- **Training viewer** — run dropdown → step slider/list → shows the fixed-prompt
  clips at that step (scrub to watch the model learn).

Stack: Vite + React, plain `fetch`, `<video controls>` (or `<img>` if GIF).

---

## 4. Open decisions (resolve before/at build start)

1. **Where does the backend run?** The model needs the checkpoint + ideally GPU.
   - (a) On the cluster (GPU), tunnel `localhost:8000` to the laptop, **or**
   - (b) scp a checkpoint down, run CPU inference locally (slow but demo-fine).
2. **Media format:** MP4 (`<video>`, cheaper) vs GIF (`<img>`, simpler). Default MP4.
3. **Generate latency UX:** block-with-spinner (simplest) vs job/poll vs
   websocket streaming the ODE trajectory (`sampler.sample(return_trajectory=True)`
   already supports a "watch it denoise" animation) — V1 = block-with-spinner.

---

## 5. Build order

1. `server/render.py` — extract reusable render-from-joints from `visualize.py`.
2. `server/app.py` — `/health`, `/generate` (prove the model→render→media loop).
3. Vite React shell + **Generate** tab end-to-end.
4. `/gt` + GT browser tab.
5. `/runs` + Training viewer tab.
6. Caching + pre-warm GT / training-sample renders.

**Out of scope (V2+):** constraint editor (joint pins / hinge limits / obstacle
placement → constrained sampler), interactive 3D (react-three-fiber), in-app
evaluation (FID / jitter panels).
