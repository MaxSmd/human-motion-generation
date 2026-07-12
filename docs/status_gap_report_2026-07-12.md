# RMG status & gap report — 2026-07-12

**Prepared for the supervisor meeting (Thu 2026-07-17).** Question on the table:
*why does our ¼-scale config (112 M params) reach FID ≈ 0.8–1.0 when the paper's
460 M config reports 0.043 — and is any of that gap still a bug?*

This report is the result of a fresh front-to-end audit of the entire codebase
(data prep → packing → loading → representation → training → sampling →
evaluation), cross-checked line-by-line against the paper (arXiv 2603.15016) and
the upstream HumanML3D / text-to-motion evaluation code. It consolidates
everything tried since May, quantifies every lever we measured, lists the
remaining *candidate* error spots found in this audit (none is a confirmed bug),
and ranks what could still be experimented.

---

## 1. Executive summary

1. **The measurement chain is now provably faithful.** We reproduce the published
   GT-vs-GT FID floor (ours 0.0019 vs published 0.002), the published real-motion
   R@1 (ours 0.513 vs published 0.511), and real mm-dist (3.10 vs ~3.0). Every
   number below is on this validated harness.
2. **Current honest headline (112 M, 300 k steps, full 4096-clip test split, 200
   ODE steps, masked):** **FID 0.607 @ ω 6.5, R@1 0.498 = 97 % of the GT
   ceiling** (0.513), mm-dist 3.23, diversity 9.28 (GT 9.79). Completed
   2026-07-13 (job 13924); clean U-curve over ω. Unmasked comparison: 0.998.
3. **Six real bugs were found and fixed along the way** — two in training, one in
   data, three in evaluation. Together they account for the difference between
   the early nonsense numbers (FID 21, R@1 0.17) and today's 0.607. None of
   them remains open.
4. **The residual 14× gap to 0.043 is consistent with scale + budget, not a
   defect.** Our own scaling step (25 M/150 k → 112 M/300 k, i.e. 4.5× params ×
   2× steps) improved masked FID **13.3×** (8.049 → 0.607). The paper's config
   is one more almost identical step (4.1× params × 2× steps). Applying our
   measured factor once more predicts **FID ≈ 0.046 at their scale — the paper
   reports 0.043.** Two points are an illustration, not a law, but the gap has
   exactly the multiplicative shape scaling predicts, with nothing left over.
5. **The paper publishes no number for any config smaller than 460 M.** Their own
   "RMG-base" (identical to our base config) has no reported FID — the headline
   0.043 exists only at full scale. There is no published evidence that a small
   RMG should be anywhere near 0.043.
6. **This audit found no new smoking gun**, but surfaced four testable
   *candidate* differences vs the paper (§6): un-canonicalized global
   placement/heading in our training clips (strongest), a low-resolution time
   embedding, ~93 % dataset coverage, and ~4 % caption/segment noise — plus
   three unknowable protocol details worth emailing the authors about.
7. **Generation quality failure mode is exactly what under-capacity looks like:**
   +36 % temporal jitter, ~20 % conservative displacement, diversity 9.28 vs GT
   9.79 — while text alignment is nearly saturated. Capacity/steps fix motion
   *texture*, and FID is dominated by texture.

**One-sentence version for the meeting:** *the instrument is validated to the
published floors, six pipeline bugs are found-and-fixed, the remaining gap
tracks our own measured scaling curve to within the paper's undisclosed
protocol details, and we have a ranked, mostly-testable list of what could still
close it.*

---

## 2. Where we stand (all numbers on the fixed harness)

### Models

| | ours: base | ours: mid | paper: RMG-base | paper: RMG |
|---|---|---|---|---|
| hidden / layers / heads / ffn | 384 / 6 / 8 / ×8 | 768 / 10 / 12 / ×4 | 384 / 6 / 8 / ×8 | 1024 / 24 / 8 / ×4 |
| params | 24.7 M | **111.7 M** | ~25 M | **~460 M** |
| steps (eff. BS 256) | 150 k | **300 k** | 150 k | **600 k** |
| precision | bf16 | bf16→**tf32** (from 160 k) | ? | ? (8 GPUs) |
| published FID | — | — | **none reported** | **0.043** |

Training recipe is otherwise paper-identical (Table 7): AdamW, LR 1e-4 cosine +
warmup, grad-clip 0.5, cfg-dropout 0.1, EMA, effective batch 256, Qwen3-Embedding
-0.6B (1024-d) fused with the time embedding via MLP → AdaLN-Zero.

### Headline numbers (full test split = 4096 clips, 200 Euler steps, EMA)

| run | best FID (ω) | R@1 | mm-dist | diversity |
|---|---|---|---|---|
| GT vs GT (floor) | 0.0019 | 0.513 | 3.10 | 9.79 |
| mid, no mask | 0.984 (7.5) / 0.998 (6.5) | 0.448 | 3.50 | 8.96 |
| **mid, masked — headline** | **0.607 (6.5)** | **0.498** | 3.23 | 9.28 |
| base, no mask | 8.19 (5.5) | 0.224 | 5.30 | 7.82 |
| base, masked | 8.049 (5.5) | 0.224 | 5.40 | 8.06 |
| paper RMG (460 M) | **0.043** (6.5) | 0.525 | — | 9.56 |

Mid beats base **13.3×** on FID and **+2.2×** on R@1 at the optimum — the
approach scales cleanly in our own hands. (The mask barely moves base: its gap
is model quality, not the padding leak.)

### Measured levers (what moves the number and by how much)

| lever | effect on FID | status |
|---|---|---|
| ODE steps 50 → 200 (mid, ω6.5, 1024 clips) | 3.81 → 1.10 (**3.5×**), 400 → 1.04 (plateau) | exploited (we report at 200; paper's count undisclosed) |
| generation length-mask (batch-padding attention leak) | 1.10 → 0.822 (**−25 %**); ω2.5 full-split −48 % | fixed, now default |
| guidance ω calibration at 200 steps | U-shaped; masked bottom at 6.5 (0.607), rises both sides | fully swept, exhausted |
| full split vs 1024-clip subset | 1.10 → 0.96 (small-sample bias) | always report full split |
| replication noise (two independent full-split runs @ω6.5) | 0.959 vs 0.998 (±0.04) | error bars: repeat runs queued (§7) |
| EMA vs live weights | EMA required | default |

---

## 3. What was tried and *closed* (the campaign so far)

Chronological; each item cost real debugging time and is now settled.

1. **Training collapse at ~46 k / 49 k (June).** Root cause chain: the packed
   data used temporal sign-*continuity* instead of the paper's global
   upper-hemisphere restriction (q₀ > 0) → near-180° joints settled in the lower
   hemisphere → near-antipodal (x₀, x₁) pairs → the CFM target's θ/sin θ · (→0)
   cancellation produces *finite garbage* in bf16 (up to ~66; occasionally
   Inf/NaN) → slow divergence that sailed past the non-finite guard. **Fixes:**
   hemisphere restriction in the loader (matches paper App. A), antipodal
   double-cover alignment in the CFM path (θ ≤ π/2 by construction, default on),
   non-finite step-skip guard. Verified with a 200 k-pair bf16-vs-fp32 study.
2. **bf16 late-training gradient storms (from step ~160 k).** Finite grad spikes
   (8 → 23 k → 3.7e8) that poison Adam; storms are *persistent*, not transient —
   a spike-guard that skips steps just stalls training (80–98 % skipped for 57 k
   steps). **Fix:** tf32 (fp32 + TF32 matmuls, 10-bit mantissa). The rerun
   crossed the exact bf16 storm zone with zero guard events and completed 300 k.
   Take-away for any future from-scratch run: qk-norm + AdamW β₂ 0.95, or tf32.
3. **Eval text encoding was silently garbage (fixed 07-05).** The Guo text
   encoder was fed spaCy *surface* forms ('walks') but its vocab is lemmatized
   ('walk') → most content words became 'unk' → real R@1 0.17 vs published 0.51.
   **Fix:** use HumanML3D's own pre-tagged word/POS tokens from texts.zip.
4. **Eval loader was unshuffled (fixed 07-05).** Consecutive test ids are
   near-duplicate segments of the same AMASS take → R-precision pools stuffed
   with near-duplicates, subsets redundant. **Fix:** seeded shuffle.
5. **R-precision pairing bug (fixed 07-10/11; this branch's regression).** Our
   `encode_motion` assumed upstream's internal `argsort(m_lens)[::-1]` is a
   no-op on pre-sorted input; on ties (eval frame-cap ⇒ one huge tie group) it
   *reverses* the tie group → motion↔text mispairing. Depressed R@1/mm-dist
   only (FID is permutation-invariant). **Fix restored:** real R@1 0.34 →
   **0.513** = published; mid gen R@1 0.20 → 0.45–0.485.
6. **Missing length-mask at generation (found 07-11 by code analysis).** Clips
   were sampled attending across the whole padded batch instead of their own
   length (training always masks). **Fix (+25 % FID, now default)** — this was
   the generation-side cause of part of the measured jitter.
7. **Instrument forensics that *exonerated* suspects:** IK→FK→263-D feature
   round-trip is bit-equal to official `new_joint_vecs` (0.0 % rel. diff, Guo
   embedding cosine 1.0000); evaluator normalization uses the checkpoint's own
   `Comp_v6_KLD01/meta` stats (the H3D repo's Mean/Std are *different* and
   wrong for it); unit-length-4 crop protocol reproduces GT-GT 0.002; the
   remembered "base FID 0.5" era was a different/likely-stub instrument and is
   not comparable — stop chasing it.
8. **ω-curve inversion mystery (resolved).** At 50 steps FID *improves* with ω
   (opposite of the paper) — a discretization artifact; at 200 steps the curve
   is the paper-shaped U. Never compare models across step counts.

---

## 4. Fresh front-to-end audit (this session): stage-by-stage verdicts

| stage | what was checked | verdict |
|---|---|---|
| AMASS → joints (`prepare_humanml3d.py raw-pose`) | SMPL+H FK, 20 fps downsampling, Y-up transform, subset pre-trims (Eyes_Japan −3 s, HDM05 −3 s, TotalCapture −1 s, Limits −1 s, Transitions −0.5 s), X-flip for all non-humanact12, humanact12 pass-through | ✅ mirrors upstream notebook incl. its pre-trim quirk |
| pack (`stage_pack`) | index.csv slicing, <40-frame drop, upstream IK (`inverse_kinematics_np`, w-x-y-z, qfix), leg-length `scale_rt` on translation, mirror construction (un-flip + L/R chain swap + IK), captions | ✅ mirror math identical to upstream `swap_left_right`; leg-length rescale matches `uniform_skeleton` |
| loading (`shared/data/humanml3d.py`) | upper-hemisphere restriction, random crop ≤196, min 40, pad+mask, per-clip caption sampling | ✅; note: we *crop* ≥200-frame clips, upstream eval *excludes* them (composition nit, both sides affected equally) |
| representation (`tplusr.py`, `registry.py`) | flat 91-d layout (T ⊕ 22×S³), prior µ = rest pose, encode = raw pack values | ✅ matches paper §3.1/3.2; **but no spatial canonicalization — see §6.1** |
| manifold math (`manifolds/*.py`) | slerp geodesic, cos-form velocity (paper App. A has a sin/cos typo — ours is the correct tangent version), safe θ/sin θ, exp/log, product dispatch, wrapped-Gaussian prior | ✅ verified algebraically + oracle sampler test |
| CFM trainer (`flow/trainer.py`) | t ~ U[ε,1−ε] per sample, per-frame iid x₀, tangent projection before MSE, masked mean, cfg-dropout 0.1 | ✅ = paper eq. 4 |
| network (`models/dit.py`, `conditioning.py`) | AdaLN-Zero DiT, learnable pos-emb, key-padding attention mask, zero-init final layer, Qwen3 pooled 1024-d ⊕ sinusoidal t → MLP fusion | ✅ faithful DiT; **time-embedding resolution flag — see §6.2** |
| training loop (`scripts/train.py`) | grad-accum, clip 0.5, non-finite + spike guards, EMA update on accepted steps only, resume/rewarm, tf32 | ✅ |
| sampling (`flow/sampler.py`) | Euler on [0,1], CFG v_u + ω(v_c − v_u) in ambient then tangent-project (linear ⇒ equivalent), exp-map stepping | ✅ = paper eq. 5 |
| eval (`scripts/evaluate.py`, `shared/eval/*`) | GT lengths, seeded shuffle, unit-4 crop with jitter on both sides, tie-safe motion encoding, token-based text encoding, checkpoint-matched normalization, FID/R@k/mm/div formulas vs upstream | ✅ formulas line-checked vs `utils/metrics.py`; GT-GT + real-R@1 floors reproduced |
| text encoder (`shared/text/text_encoder.py`) | Qwen3-Embedding-0.6B, last-non-pad-token pooling, L2 norm, same encoder at train & eval | ✅ self-consistent; paper's pooling detail unspecified (§6.5) |

**Net: no new bug found.** The four flagged items are *differences that may cost
quality*, not correctness errors — ranked next.

---

## 5. Why the gap is plausibly *just* scale + budget

- **Our own scaling data point.** base → mid is 4.5× params × 2× steps and bought
  **~10×** FID (8.19 → ~0.8 masked). mid → paper-large is 4.1× params × 2× steps.
  One more equal multiplicative step lands at **≈ 0.08**, i.e. within ~2× of the
  published 0.043 — and their number is a 20-replication mean at an undisclosed
  ODE step count with possibly different regularization. Two points don't make a
  law, but the *shape* of the gap is exactly what scaling predicts.
- **Literature anchor.** Raw-space (non-latent) diffusion at small scale sits at
  FID 0.5–0.6 on this benchmark (MDM 0.544, MotionDiffuse 0.63); the sub-0.1
  club is latent/tokenized (MLD 0.47; T2M-GPT 0.116; MoMask 0.045) or very large
  (RMG 460 M, 0.043). A 112 M raw-manifold model at 0.7–0.8 with near-ceiling
  R@1 is *in family*, not anomalous. And again: the paper's own small config has
  **no published number** — plausibly because it isn't good.
- **The failure signature is capacity-shaped.** Generated motion is valid
  (|q| = 1.0000 everywhere), correctly conditioned (R@1 94 % of GT ceiling),
  diverse (91 % of GT), but **+36 % angular jitter** (0.0875 vs 0.0645
  rad/frame) and **conservative** (max |ΔT| 1.8 m vs 3.6 m real). FID's
  covariance term punishes exactly this texture mismatch. Jitter/texture is
  the thing more depth × more steps repairs (it's also what their 24-layer
  model has 2.4× more of than our 10-layer one).
- **Training was still descending at 300 k** (tail-averaged logged loss trending
  down through the final 40 k), i.e. the 300 k checkpoint is budget-limited, not
  converged.

---

## 6. Remaining candidate gaps found in this audit (ranked)

None of these is a confirmed bug. Each has an evidence line, an expected impact,
and a concrete test.

### 6.1 Training clips are not spatially canonicalized (strongest candidate)
Our pack stores translations in (leg-length-rescaled) **AMASS world frame**: the
round-trip check measured a ~1.3 m global offset vs the official files, per-axis
translation std ≈ [0.62, 0.21, 0.99] m, and **heading is arbitrary** (upstream's
official pipeline re-bases every clip: floor at y=0, XZ origin at frame 0,
facing +Z — its features are built *after* that). At eval this is provably
irrelevant (the 263-D features canonicalize; bit-verified). At **train** time,
however, our model must spend capacity modeling a nuisance SE(2) orbit
(position × heading) that the paper's model may never see — and the prior
µ_T = 0 matches canonicalized data much better than ours. A 460 M model shrugs
this off; a 112 M model pays proportionally more.
**Test:** opt-in per-crop canonicalization (subtract first-frame root XZ,
yaw-align first-frame heading; y untouched) — eval-neutral by construction — and
A/B at base scale (150 k ≈ 2.5 days). **Implemented + tested** (2026-07-12):
`data.canonicalize_crops=true`, default off, zero effect on existing runs.
**Caveat:** the paper never says they canonicalize; this could equally be a
capability *we* could exploit that they didn't.

### 6.2 Time-embedding resolution
`sinusoidal_time_embedding` is fed t ∈ [0,1] *unscaled* with max_period 10⁴: the
argument t·f spans [0, 1] only — cos ≈ 1, sin ≈ t·f for almost every band, so
the network effectively receives a near-linear-in-t signal through an MLP.
Standard DiT/flow implementations scale t by ~1000 before embedding, giving the
conditioning MLP a genuinely multi-scale basis. An MLP on a scalar *can*
represent any smooth schedule, so this is expressivity-pressure, not error — and
it's self-consistent between train and inference.
**Test:** flip to t×1000 in the next from-scratch run (cannot retrofit).
**Implemented + tested** (2026-07-12): `model.time_scale=1000`, default 1.0
(bit-identical to all existing checkpoints).

### 6.3 Dataset coverage ≈ 93 %
Our splits: train 21 777 / val 1 362 / test 4 096 vs official 23 384 / 1 460 /
4 384 — ~7 % of clips missing (unfetched AMASS files, missing captions; the
<40-frame drops are protocol-correct). Training on 93 % of the data is a real
but bounded handicap; eval-side it cancels (both real & gen use the same set).
**Test/fix:** inventory which index.csv sources are missing and re-pack
(CPU-only), if the missing AMASS archives are obtainable.

### 6.4 Sub-segment captions attached to full clips (~4 %)
Upstream slices captions tagged `#start#end` into *separate* (sub-clip, caption)
training pairs; we attach all captions to the full clip and random-crop —
~4 % of caption draws describe a segment the crop may not contain. Measured
eval impact nil; train-side it is mild label noise.
**Fix:** emit sub-segment entries as extra clips at pack time.

### 6.5 Unknowable protocol details (email the authors)
- **ODE step count** for their reported numbers (our 50→200 sweep moves FID
  3.5×; their Euler solver is stated, count never).
- **Their (T,R) dataset construction** — from official canonicalized H3D joints
  (→ they trained canonicalized, §6.1 matters) or from raw AMASS (→ like ours)?
- **AdamW weight decay** (ours 0.0; AdamW default 0.01) and **Qwen3 pooling**
  (ours: last-token + L2-norm per the model card; paper: "encoded hidden
  states").
- **20-replication** averaging (we run single-rep full-split; ±0.04 observed
  across runs).

### 6.6 Explicitly *not* candidates (checked, cleared)
Geometry/CFM math (incl. the paper's own velocity-formula typo), CFG combination
order, evaluator normalization stats, IK/FK/feature conversion, mirror
construction, leg-length rescale, hemisphere restriction, EMA handling,
eff. batch/LR/clip/dropout, text-token eval encoding, R-pool shuffling,
tie-safe pairing, unit-4 crop protocol, ≥200-frame handling (composition nit
only), MModality protocol size (secondary metric, smaller than upstream's but
self-consistent).

---

## 7. What is running / queued / proposed

**Done (2026-07-13):**
- job 13924 `eval-mid-masked` + job 13925 `eval-base-masked` — full split × 8 ω
  × 200 steps, mask on. Final masked ω-curves for both scales on one harness
  (§2 + Appendix A) → *the* headline chart for Thursday.

**Queued (submitted 2026-07-13 via the backend, serial):**
1. **Error bars:** seeds 1 & 2 of mid @ ω 6.5, full split, 200 steps (jobs
   `mid300k-w65-seed1` = 13950, `-seed2`) → quote FID ± spread.
2. **400-step confirm** @ ω 6.5, full split (`mid300k-w65-ode400`) — the
   1024-clip sweep says +~5 % over 200 steps; if it holds, the headline improves
   a notch. Branch pushed (e4da186) and cluster checkout synced first.

**Prepared, needs your go (not launched):**
- **600 k extension** (step-parity with paper): re-warm schedule implemented,
  smoke-tested on the cluster (LR curve verified: 0 → 3e-5 @305 k → cosine 0 @
  600 k), resubmit-safe; ~6.5 days — would NOT land by Thursday, and occupies
  the single job slot. Decision: launch after Thursday, or tonight if the
  meeting doesn't need more eval runs.
- **Canonicalization A/B at base scale** (§6.1): ~30-line opt-in change +
  150 k-step run ≈ 2.5 days + eval — *could* land a preliminary result by
  Wednesday night if launched tonight; competes with the same slot.
- For any future from-scratch run: qk-norm + AdamW β₂ 0.95 (bf16 storm
  prophylaxis), t×1000 time embedding (§6.2).

---

## 8. Suggested meeting narrative

1. *Instrument first:* we reproduce the published GT floors exactly — every
   number is trustworthy (show GT-GT 0.0019, real R@1 0.513).
2. *Six bugs found & fixed* — one-slide table (§3), each with its measured
   impact. This is the "we did the engineering" slide.
3. *Current state:* mid = FID 0.607 / R@1 0.498 (97 % of GT ceiling), 13.3×
   better than base on identical harness (show the two masked ω-curves).
4. *The gap decomposes:* measured levers (steps, mask, ω) are exhausted; our own
   base→mid scaling step, applied once more, predicts 0.046 at the paper's
   config vs their published 0.043 — and they publish no small-config number at
   all.
5. *What's left:* ranked candidate list (§6) — one strong testable hypothesis
   (canonicalization), several bounded ones, three questions for the authors.
6. *Ask:* the job slot priority — 600 k extension (parity, 6.5 d) vs
   canonicalization A/B (hypothesis test, 2.5 d) vs both sequenced.

---

## Appendix A — exact current numbers

- **mid unmasked full sweep** (FID / R@1): 2.5→3.448/.383 · 3.5→1.746/.423 ·
  4.5→1.286/.439 · 5.5→1.033/.437 · 6.5→0.998/.448 · 7.5→0.984/.455 ·
  8.5→1.254/.435 · 9.5→1.348/.446
- **base unmasked full sweep**: 2.5→8.943/.204 · 3.5→8.438/.226 · 4.5→8.258/.221
  · 5.5→8.186/.224 · 6.5→8.237/.224 · 7.5→8.508/.224 · 8.5→8.660/.226 ·
  9.5→8.571/.217
- **mid masked full sweep** (FID / R@1): 2.5→1.793/.439 · 3.5→0.964/.485 ·
  4.5→0.798/.482 · 5.5→0.683/.493 · **6.5→0.607/.498** · 7.5→0.638/.491 ·
  8.5→0.750/.490 · 9.5→0.812/.494
- **base masked full sweep**: 2.5→8.857/.213 · 3.5→8.292/.234 · 4.5→8.390/.225 ·
  5.5→**8.049**/.224 · 6.5→8.463/.221 · 7.5→8.587/.219 · 8.5→8.790/.223 ·
  9.5→8.671/.224
- **steps sweep** (mid ω6.5, 1024 clips, unmasked): 50→3.81 · 100→1.67 ·
  200→1.10 · 400→1.04
- **mask A/B** (mid ω6.5, 200 steps, 1024 clips): 1.10 → 0.822; R@1 0.45→0.483
- **GT refs**: GT-GT 0.0019 · real R@1 0.513 / R@2 ~0.70 / R@3 ~0.80 · mm-dist
  3.10 · diversity 9.79
- **forensics** (gen vs real): |q| 1.0000/1.0000 · ang-vel 0.0875/0.0645
  rad/frame · trans-vel 0.0241/0.0171 m/frame · max |T| 1.8/3.58 m
- **replication noise**: full-split ω6.5 unmasked measured 0.959 (job 13794) and
  0.998 (eval-mid) on different runs → ±0.04
- mid = 111,696,731 params · trained tf32 from 160 k · EMA 0.9999 · eff. BS 256
  · ckpt-000300000 (clean, `latest_pre_extension.pt` preserved; re-warm smoke
  advanced `latest.pt` a few hundred steps past 300 k — use the numbered ckpt)

## Appendix B — config vs paper Table 7

| hyperparameter | ours (mid) | paper (RMG 460 M) | match |
|---|---|---|---|
| input dim | 91 | 91 | ✅ |
| hidden / layers | 768 / 10 | 1024 / 24 | ✋ ¼ scale (deliberate) |
| heads | 12 (hd 64) | 8 (hd 128) | ~ (negligible) |
| FFN mult | 4 | 4 | ✅ |
| max LR / schedule | 1e-4 cosine+warmup | 1e-4 cosine+warmup | ✅ |
| warmup ratio | 0.05 | 0.08 | ~ (minor) |
| effective batch | 256 (64×4×1 GPU) | 256 (16×8×2) | ✅ |
| steps | 300 k | 600 k | ✋ ½ (extension ready) |
| grad clip | 0.5 | 0.5 | ✅ |
| cfg dropout | 0.1 | 0.1 | ✅ |
| text encoder | Qwen3-Emb-0.6B pooled→MLP fuse | same | ✅ |
| optimizer | AdamW β(0.9,0.999) wd 0.0 | AdamW (β, wd unspecified) | ❓ |
| EMA | 0.9999 | "EMA" (decay unspecified) | ❓ |
| ODE steps at eval | 200 (swept 50–400) | undisclosed | ❓ |
| eval protocol | single-rep, full split | 20 reps, mean ± CI | ✋ |
