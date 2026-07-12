# Instructions — surface the RMG-mid validation analysis in the frontend Analysis tab

**Audience:** an engineer/agent extending the RMG interactive app's frontend.
**Goal:** fold the recently-completed *end-to-end validation of RMG-mid* into the
web UI's **Analysis** tab, so the story ("why is mid ~22× off the paper, and is it
a bug?") is browsable in-app instead of living only in `docs/rmg_mid_validation.md`
+ static PNGs in `docs/figures/`.

This is a **frontend + light backend** task. The heavy lifting (the experiments,
the measurements) is done — you are *presenting* known results, plus wiring two
already-computed analyses. Do **not** re-run cluster jobs.

---

## Background — what was found (context you need)

RMG-mid (112M params, 300k steps) scores **FID ≈ 0.96** full-split vs the paper's
0.043 (460M / 600k). We audited every pipeline stage and found **no correctness
bug in the model** — the gap is capacity + training budget. Along the way two
**evaluation-harness** bugs were found and fixed:

1. **Motion-embedding pairing bug** — `RealGuoEvaluator.encode_motion` mis-paired
   text↔motion on length ties, depressing R-precision/mm_dist (not FID). Fixed.
   Real R@1 went 0.34 → **0.513** (matches published 0.511).
2. **Missing length mask at generation** — sampling without a per-clip validity
   mask let short clips attend across the padded batch. Fixed (now default).
   **FID improved ~25%** (1.10 → 0.822 at mid ω6.5, 200 steps, 1024 clips).

The canonical write-up is **`docs/rmg_mid_validation.md`** (read it first — it has
the exact numbers and narrative) and six figures in **`docs/figures/`**
(`fig1_gap` … `fig6_functional`, plus `make_figs.py`).

**Final eval runs on the cluster** (use these — everything else was deleted):
`eval-mid` and `eval-base`, each the full 4096-clip test split × all 8 guidance
values, on the corrected harness. NOTE: a fresh **masked** re-sweep is in progress
and will replace them as `eval-mid-masked` / `eval-base-masked` — prefer the
`-masked` runs once present (they are the true, better numbers). Confirm the
current run names via the `/eval-runs` endpoint rather than hardcoding.

---

## What already exists (reuse it — do not rebuild)

`app/frontend/components/AnalysisTab.jsx` already renders four sections:

| # | Section | Component | Data source |
|---|---|---|---|
| 01 | Training | `TrainingCurves` (in AnalysisTab) | `api.runMetrics`, `api.runInfo` |
| 02 | Eval comparison | `EvalTab.jsx` (`EvalComparison`) | `/eval-runs`, `/eval/{run}`, `/analysis/table` |
| 03 | ODE step sweep | `OdeStepSweep.jsx` | eval runs |
| 04 | Clip comparison | `JointAnalysis` (in AnalysisTab) | `/metrics`, `/analysis/npy` |

Charts use **`LineChart.jsx`**. Section chrome is the `Section` helper in
AnalysisTab (numbered `01`…`04`). Backend eval plumbing is
`app/backend/analysis/eval_tables.py` (`list_eval_runs`, `fetch_results`,
`comparison()` → `{rows, best, sweeps, latex}`) exposed at
`app/backend/cluster/routes.py` (`/eval-runs`, `/eval/{run}`, `/analysis/table`,
`/metrics`, `/analysis/npy`).

The **guidance-sweep overlay** (fig2) and **ODE-step lever** (fig3) are therefore
*already in the app* (sections 02 and 03). Your job is to add what's missing and
tie it together as a validation narrative.

---

## Deliverables

### A. New Analysis section: **"05 · Validation"** (primary task)

Add a `Section n="05" title="Validation" sub="is the gap a bug? — end-to-end audit"`
to `AnalysisTab.jsx`, rendering a new `ValidationPanel` component
(`app/frontend/components/ValidationPanel.jsx`). It has four blocks:

1. **The gap (headline).** A small horizontal log-scale bar/lollipop replicating
   `fig1_gap`: GT-GT 0.0019 · paper 0.043 · **mid (live, best-FID from
   `eval-mid[-masked]`)** · base (live). Pull mid/base FID from the
   `/analysis/table` comparison response (`best` row per run) so it stays live;
   GT-GT and paper are constants. Annotate the mid→paper ratio ("≈ Nx"), computed.

2. **Pipeline scorecard.** Replicate `fig4_scorecard`: seven stages (Data
   processing, Container, Data loading, Representation, Training, Sampling,
   Evaluation) each with a ✓ and a one-line note. This is **static content** —
   hardcode the seven items from `docs/rmg_mid_validation.md` §2. A horizontal
   fl: row of rounded cards; keep it responsive (wrap to grid on narrow widths).

3. **Two harness bugs found + fixed.** A compact two-item list/callout:
   - *Pairing bug* — "R@1 0.34 → 0.513 (FID unchanged)"
   - *Length mask* — "FID 1.10 → 0.822 (~25%)"
   Each with a one-sentence mechanism. Static content from §"harness bugs".

4. **Generation forensics.** Replicate `fig5`/`fig6`: grouped bars of gen-vs-real
   for {angular-vel, translation-vel, |translation|} and {R@1, diversity}. See
   "wiring the forensics endpoint" below — ideally live, else static from the doc.

Match the existing dark aesthetic (`var(--signal)` cyan `#22d3ee`, amber
`#fbbf24`, white ink). Reuse `LineChart` where a chart fits; simple bars can be
divs. Follow the `EvalTab.jsx` data-fetch pattern (`useEffect` + `api.*`,
graceful empty states).

### B. Wire the generation-forensics endpoint (light backend)

The forensics numbers (quat-norm, angular/translation velocity, |translation| for
gen vs real) currently only exist in a throwaway script. Make them a first-class
endpoint so block 4 is live:

- The computation is ~30 lines (already prototyped): load a run's generated EMA
  sample (`runs/.../samples/step-*.pt`) and a sample of real packed clips, compute
  per-frame angular velocity `2·acos(|<q_t,q_{t+1}>|)`, translation velocity, and
  |translation|; return gen + real summary stats. Reference the exact formulas in
  `docs/rmg_mid_validation.md` appendix and the (git-history) analyze script.
- Add `GET /analysis/forensics?run=<train_run>` in `routes.py` → a new
  `app/backend/analysis/forensics.py`. It runs over SSH on the cluster (mirror how
  `eval_tables.fetch_results` shells out) OR reads a cached JSON the eval writes.
  Simplest robust option: have it read a `forensics.json` if present and 404
  otherwise; a follow-up can compute on demand.
- If wiring the live endpoint is out of scope for one pass, **hardcode block 4**
  from the doc's appendix and leave a `TODO(forensics-endpoint)` — the panel must
  render either way.

### C. Do NOT

- Do not regenerate the static `docs/figures/*.png` (the user explicitly does not
  want updated figures).
- Do not launch or modify cluster eval/training jobs.
- Do not touch the eval harness code (`src/shared/eval/*`, `src/rmg/scripts/
  evaluate.py`) — the fixes are done and a sweep is running against them.

---

## Exact numbers to hardcode (from `docs/rmg_mid_validation.md`; verify against it)

- **Gap:** GT-GT 0.0019 · paper (460M/600k) 0.043 · mid (112M/300k) ~0.96 (or
  ~0.7 once masked sweep lands — prefer live value) · base (25M) 8.19.
- **Bugs:** pairing → real R@1 0.34→0.513, mm_dist→3.10; length-mask → FID
  1.10→0.822, R@1 0.45→0.483.
- **Forensics (gen / real):** quat-norm 1.0000 / 1.0000; angular-vel 0.0875 /
  0.0645 rad/frame; translation-vel 0.0241 / 0.0171 m/frame; |translation| 0.514 /
  0.646 m. Prior σ=1.0 matches data std [0.62, 0.21, 0.99].
- **Functional:** mid gen R@1 ~0.45 (masked/fixed), GT ceiling 0.513, published
  0.511; diversity gen 8.61, GT 9.79, published 9.50.

---

## Acceptance

- Analysis tab shows a new **05 · Validation** section that renders without errors
  when eval runs exist, and degrades gracefully (skeleton/empty copy) when they
  don't.
- The gap headline and any per-run FID are **live** from `/analysis/table`; the
  scorecard and bug list are static; forensics is live if B is done, else static
  with a TODO.
- Visual style matches sections 01–04 (numbered header, dark theme, `LineChart`).
- `npm run build` (in `app/frontend/`) passes; no new eslint errors.

## Pointers

- Read first: `docs/rmg_mid_validation.md`, `app/frontend/components/AnalysisTab.jsx`,
  `app/frontend/components/EvalTab.jsx`, `app/backend/analysis/eval_tables.py`.
- Palette/mark reference (optional): the `dataviz` skill; figures in `docs/figures/`
  show the intended look for each block (`fig1`,`fig4`,`fig5`,`fig6`).
- API client: `app/frontend/lib/api.js` (`api.*`, `mediaUrl`).
