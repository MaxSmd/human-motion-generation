# RMG Showcase — Constraint Studio

A **separate, fully static** public site for the RMG project. No backend, no
cluster, no model at runtime: the server ships only static bytes and the
visitor's browser renders the motion in three.js. Designed to run on a Raspberry
Pi behind a Cloudflare Tunnel.

## Pages

| Route | What it shows |
|---|---|
| `/` | Overview — hero (live in-browser skeleton), the manifold idea, headline stats, model card |
| `/studio` | **Constraint Studio** — the fake-live generator (below) |
| `/results` | Interactive charts: FID gap, ω sweep, sampling-steps lever, generation forensics, R-precision + diversity, and the rendered-figure gallery |
| `/evaluation` | The "we found no bug" validation story — the stage-by-stage audit scorecard, systems table, the two FID traps, the forensics |
| `/training` | Training analysis — recipe, the geometry, the bf16 / antipodal-cut-locus stability story, cluster auto-resubmit |

All numbers live in `lib/results.js` (transcribed from `docs/figures/make_figs.py`
+ `docs/rmg_mid_validation.md`) — the single source of truth for every chart and
table. The charts are dependency-free SVG; math is styled Unicode (no KaTeX dep).

## The flagship — Constraint Studio

The **Constraint Studio** (`/studio`): pick a prompt + hard joint constraints, hit
*Generate*, watch a realistic RMG-sampler progress animation for a few seconds,
and the matching **pre-rendered** clip is revealed in an in-browser 3D player. It
looks live; it costs the server nothing.

## Run locally

```bash
cd showcase
npm install
npm run dev        # http://localhost:3100
```

## Build the static site

```bash
npm run build      # emits ./out  (pure HTML/CSS/JS + /data/*.npy)
npx serve out      # preview the static output
```

Deploy = copy `out/` to the Pi and serve it (Caddy/nginx) behind `cloudflared`
(`deploy/cloudflared/config.yml`).

## How the "generation" works

Nothing is generated at runtime. `lib/catalog.js` maps every reachable
selection — a prompt + a sorted set of constraint ids — to one pre-rendered clip
under `public/data/*.npy`. On *Generate*, `GenerationProgress` plays a staged
animation that mirrors the real sampler (encode text → 200 Riemannian-Euler ODE
steps → constraint projection → FK render), then `SkeletonPlayer` loads the
resolved `.npy` and animates it. The `.npy` is `(T, 22, 3)` float32 joint
positions, ~26 KB each.

## Extend the matrix (add prompts / constraints)

The clips ship as pre-rendered assets because generating them needs the GPU
model (cluster). To add more:

1. **Render on the cluster.** For each `(prompt, constraints)` pair you want,
   generate a clip with the existing pipeline (`src/rmg` sampler /
   `visualize.py mode=prompt`, threading `RMG_CONSTRAINTS` / `RMG_RANGES` exactly
   as the interactive app does) and dump the FK joint positions `(T, 22, 3)`.
2. **Normalize + drop in.** Run each through `scripts/normalize_npy.py` (ensures
   C-order float32 — the one format `lib/npy.js` requires) into
   `public/data/<name>.npy`.
3. **Register it.** Add a row to `CLIPS` in `lib/catalog.js`:
   ```js
   { prompt: "walk", constraints: ["elbow_lock", "knee_clamp"], file: "walk_both.npy", fps: 20 },
   ```
   and, if it's a new prompt/constraint, add it to `PROMPTS` / `CONSTRAINTS`.
   The UI derives which prompts and which constraint toggles to offer straight
   from the catalog — no component changes needed.

Keep every reachable selection resolvable: for a prompt, render at least the
unconstrained `[]` baseline plus each single constraint you expose (and any
combos you let the user reach).

## Seed data (current)

The four clips checked in are the constraint before/after demos you already had
(`L_ELBOW*` / `L_KNEE*`), renamed:

| prompt | constraints | file |
|---|---|---|
| wave | — | `wave_free.npy` |
| wave | left elbow locked | `wave_elbow.npy` |
| walk | — | `walk_free.npy` |
| walk | left knee clamped | `walk_knee.npy` |

Replace/expand them with a proper cluster render for the public launch.

## Files

```
app/layout.jsx            root layout + sticky nav
app/page.jsx              Overview (hero player, idea, stats, model card)
app/studio/page.jsx       Constraint Studio route
app/results/page.jsx      charts + figure gallery
app/evaluation/page.jsx   validation story + scorecard + tables
app/training/page.jsx     recipe + stability story
components/Nav.jsx         top navigation
components/ui.jsx          Page/Section/Panel/Stat/Eq/Table helpers
components/ConstraintStudio.jsx   selectors + orchestration + reveal
components/GenerationProgress.jsx the fake-sampler choreography (buildPlan + rAF animation)
components/SkeletonPlayer.jsx     in-browser 3D skeleton player (auto-grounds/centres)
components/LineChart.jsx  SVG line chart (ω + steps sweeps)
components/Lollipop.jsx   SVG log-scale FID gap
components/Bars.jsx       SVG grouped bars (forensics, functional)
lib/results.js            ALL measured numbers — single source of truth
lib/catalog.js            the constraint matrix → clip mapping
lib/npy.js                client-side .npy parser (copied from the app, unmodified)
public/data/*.npy         pre-rendered clips (seed; replace with cluster render)
public/figures/*.png      the six defense figures
scripts/normalize_npy.py  ensure dropped-in clips are C-order float32
```
