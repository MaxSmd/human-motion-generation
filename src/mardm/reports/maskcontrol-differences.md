# MaskControl → MARDM: what we translated exactly, and where we differ

Phase-1 spatial control (branch `mardm/control`) ports MaskControl's
(ControlMM, Pinyoanuntapong et al.) inference-time control to our continuous
masked-autoregressive MARDM. This note pins down which pieces are faithful
translations, which are structural adaptations, and which are our own
additions — so the paper text doesn't accidentally claim the wrong thing.

## Faithful translations

| MaskControl | Ours | Notes |
|---|---|---|
| Logits Optimization: gradient steps on the per-token logits against a differentiable joint-space loss | z-optimization: Adam on the transformer's per-token condition `z` (`mardm.control.guidance`) | `z` plays exactly the role logits play — it parameterizes the per-token generative distribution the head samples from. Optimizing the condition rather than the output keeps sampling inside the learned denoiser. |
| Differentiable Expectation Sampling (DES) to fake differentiability through categorical sampling + codebook lookup | **not needed** | Our whole path — z → DiffMLP euler ODE → AE decode → denormalize → FK — is exactly differentiable. This is the core pitch: continuous masked AR makes MaskControl's mechanisms exact. |
| Motion-consistency loss on controlled joints (their Eq. 3, `R(D(·))` to world frame) | `control_loss` in `mardm.control.losses`: decode → denormalize → `recover_joints_from_ric` → masked Euclidean error | Same loss, same world-frame recovery (cumsum root + rotate), no Gumbel machinery. |
| Speed/precision knob = iteration count (Fast/Medium/Accurate) | `inner_iters` (× `ode_steps_guidance`) | Same lever. |
| Zero-shot objective control (any differentiable joint loss) | inherited for free | Swap `control_loss` for any differentiable function of joints. |

## Structural adaptations (different by necessity, same intent)

**Remasking loop.** MaskControl's generator remasks *low-confidence* tokens
each iteration, so tokens perturbed by optimization keep getting re-predicted
by the transformer — the prior is implicitly re-applied every step ("implicit
proximal operator"). MARDM's `generate()` uses a *nested* mask schedule: a
fixed random ranking with a shrinking threshold, so a token that leaves the
mask set is committed forever and is never revisited. Consequence: heavily
perturbed tokens keep their perturbed values, and nothing pulls them back to
the manifold.

Our answer is **re-prediction repair** (`repair_rounds` in `GuidanceConfig`):
after the AR loop, re-predict every valid token with the *unguided* prior,
remask the `repair_frac` whose committed values the prior most disagrees with,
and re-predict those under light guidance (`repair_iters`). This is the
explicit version of the restore MaskControl gets implicitly from
confidence-based remasking. It is an adaptation, not in their paper in this
form.

**Where the prior's "warm start" comes from.** In MaskControl, inference-time
optimization starts from logits produced by the *trained* Logits Regularizer
(ControlNet copy), so only small perturbations are needed and realism barely
degrades (their ablation: removing the regularizer degrades FID 0.061 → 0.142
while control stays fine). Phase 1 has **no trained regularizer** — the
optimizer starts from the raw model's z and has to move it far. Phase 2 (the
trained condition regularizer) is the faithful translation of that component
and is expected to carry most of the realism recovery.

## Our own additions (not in the paper)

* **Repair-token selection by prior disagreement** (‖unguided re-prediction −
  committed latent‖ per token). MaskControl selects by confidence; our
  continuous tokens have no confidence score, so disagreement with the prior
  is the analogue we chose.
* **Direct latent post-optimization** (`post_iters`, MaskControl Eq. 8
  *analogue* — they optimize embeddings before decode; we optimize AE latents).
  Off by default; bypasses the diffusion prior.
* **Runtime trust-region levers** (implemented 2026-07-22 after the pivot to
  inference-only control): error-tolerance early stop for the inner loop
  (`tolerance` — stop at ~5 cm instead of grinding to mm), proximal anchor
  `λ‖z−z₀‖²` (`prox_weight`), and guidance scheduling over AR steps
  (`guidance_start_frac` / `guidance_ramp` — early steps have an empty
  context, so perturbing them does the most structural damage). Standard
  guided-diffusion trust-region machinery, **not** from MaskControl — must be
  labeled as our own ablation when reported.
* **Interleaved repair** (`repair_every`): repair rounds *inside* the AR loop
  over the committed prefix only, so the prior restores realism while the
  remaining context is still open — targeting the frozen-gait pathology that
  post-hoc repair reinforced (it re-predicted against fully frozen context).
  Also our own; MaskControl gets this effect implicitly from per-iteration
  confidence remasking.

## Evaluation differences

* Traj./Loc./Avg. error (0.5 m threshold, waypoints sampled from GT joints)
  follow the OmniControl protocol and are directly comparable to MaskControl's
  tables.
* **FID uses the STANDARD Guo 263-D evaluator** (original Comp_v6_KLD01
  checkpoint + its shipped normalization) — generated essential features are
  bridged back to 263-D via `essential_to_h3d` before embedding, and the real
  side feeds canonical new_joint_vecs. (Upstream MARDM retrains essential-dim
  evaluators; this repo deliberately does not.) Cross-paper FID comparison is
  therefore *closer* than first noted, but still not exact: our data is
  AMASS-reprocessed, gen features go through a positions -> process_file
  re-extraction, and control evals run on 512 clips (small-sample FID bias:
  0.268 vs 0.154 full-split for the same model). Within-run same-seed deltas
  remain the only fully clean comparison.

## Phase-1 numbers (2026-07-15, MARDM-M @300k canonical, CFG w=3.0)

512 test clips, 5 pelvis keyframes. Unguided baseline shares seeds with each
guided run (FID 0.268 is the 512-sample small-sample value, higher than the
full-test 0.154; only within-column deltas are meaningful).

| | Avg. err | Loc. err | Traj. err | FID | R@1 |
|---|---|---|---|---|---|
| unguided | 0.648 m | 0.363 | 0.541 | 0.268 | 0.504 |
| guided, optimization-only | 0.015 m | 0.003 | 0.012 | 1.447 | 0.385 |
| guided + repair (2 rounds, frac 0.5) | 0.076 m | 0.017 | 0.057 | 0.905 | 0.385 |

Optimization-only control matches MaskControl's accurate mode with zero
training; its 5.4× FID degradation is the measured cost of having neither their
regularizer (phase 2) nor a restore mechanism.

**Re-prediction repair recovers ~40% of the FID gap** (1.447 → 0.905, vs 0.268
floor) at a 5× control cost (0.015 → 0.076 m, still well inside the 0.5 m
threshold — loc err only 0.017). But it does NOT fix the qualitative failure it
was meant to: on the hardest conflict clip (004822, "walk in place" fighting the
prior's forward-walk habit) foot speed goes 0.57 → 0.10 → 0.06 m/s — repair
made the freeze slightly *worse*, because re-prediction restores consistency
with the committed context, and when >half the sequence already encodes frozen
legs the prior predicts more of them. Aggregate motion smoothness does improve
(mean jerk 121 → 77 m/s³ vs 57 unguided; mean foot speed 0.75 → 0.58 vs 0.56),
i.e. repair removes the high-frequency artifacts optimization injects but cannot
re-inject gait that optimization suppressed.

**Conclusion: the zero-training restore cannot substitute for the trained
regularizer.** MaskControl's implicit restore works because its
regularizer-warm-started perturbations are small, so the surrounding context
stays on-manifold and re-prediction pulls toward it. Ours are large, so
re-prediction pulls toward the pathology. This sharpens the phase-2 motivation:
the warm start is not an optimization nicety, it is what makes any restore
mechanism work.

## Phase-2 attempt 1 (2026-07-15): GT-anchored L_s trains a no-op — diagnosed

First regularizer training (90k steps, α=0.1) used the one-step clean estimate
`x̂₁ = x_t + (1−t)·v` for L_s, with `x_t = t·x₁ + (1−t)·x₀` built from the TRUE
token. **This leaks the ground truth into the loss anchor**: with GT-derived
control targets, L_s is minimized by accurate denoising alone — at large t,
x̂₁ ≈ x₁ regardless of the control signal; at small t the estimate is noise-
dominated. No gradient pressure to *use* S ever arises. The evidence was in
the training curve (l_s flat: 0.042 @step 1 → 0.072 val @90k) and confirmed by
eval (512 clips, same protocol as above):

| | Avg. err | Loc. err | Traj. err | FID | R@1 |
|---|---|---|---|---|---|
| reg_only (attempt 1) | 0.643 m | 0.374 | 0.564 | 0.334 | 0.523 |
| reg + opt (attempt 1) | 0.015 m | 0.004 | 0.016 | 1.383 | 0.348 |

reg_only ≡ unguided (0.643 vs 0.648 m); reg+opt ≡ phase-1 optimization-only.
The architecture is fine (zero-init identity verified); the loss was vacuous.

**Why MaskControl doesn't have this bug:** their DES-based L_s is computed on
the model's own *from-noise generation*, so it measures generation-time
control error. The one-step estimate was our shortcut, and it broke the loss's
information structure.

**Fix (attempt 2):** L_s samples the masked tokens from pure noise through
`ls_ode_steps` differentiable euler steps conditioned on z (the same machinery
phase-1 guidance uses), scattered into true-latent context. Exactly
differentiable, still no DES — the paper claim is unchanged; only the anchor
moved from GT to noise.

## Runtime levers (2026-07-23, job 15185): how far zero-training gets

The supervisor pivot to inference-only ruled out phase 2, so the question
became how much of the FID cost the *runtime* knobs can recover. An 8-arm
128-clip sweep (job 15149) found a three-point Pareto front, confirmed here at
the full 512-clip protocol. Same checkpoint, same seeds, same protocol as the
phase-1 table above, so the columns stack directly — the unguided row
reproduces (0.268 FID / R@1 0.516 vs 0.504).

| | Avg. err | Loc. err | Traj. err | FID | R@1 | gen s |
|---|---|---|---|---|---|---|
| unguided | 0.648 m | 0.363 | 0.541 | 0.268 | 0.516 | 179 |
| opt30 (phase-1) | 0.015 m | 0.003 | 0.012 | 1.447 | 0.385 | — |
| tol5_ramp | 0.052 m | 0.012 | 0.049 | 0.826 | 0.439 | 985 |
| tol5_inline | 0.049 m | 0.005 | 0.023 | 0.673 | 0.434 | 3456 |
| combo_all | 0.106 m | 0.036 | 0.098 | 0.392 | 0.471 | 1176 |

`tol5_inline` (tolerance stop + interleaved repair every 3 AR steps) closes
**66% of the FID excess over opt30** (1.179 → 0.405 above the floor) while
*improving* control over phase-1 post-hoc repair on both axes (0.905 / 0.076 m
→ 0.673 / 0.049 m). Interleaving is what does it: repairing the committed
prefix while the rest of the context is still open lets the correction spread,
instead of re-predicting into an already-frozen sequence — the exact failure
diagnosed in phase 1. R-precision recovers 0.385 → 0.434–0.471.

The 128-clip sweep ranked the arms in the same order as the 512-clip run, so
the cheap scale is order-predictive and is used for probing. FID magnitudes are
NOT comparable across scales (unguided floor 0.755 @128 vs 0.268 @512 vs 0.154
full split).

**What is still wrong.** All four levers act on the optimizer; none touch the
objective. `control_loss` is purely positional, so a static pose is a global
minimum of it whenever the prior's habitual motion conflicts with the
waypoints — on clip 004822 foot lift stays ≈30% of GT. FID does not see this
(a damped clip is still a plausible clip), which is why the evaluation now also
reports `foot_skate`, `motion_mag` and `jerk` against a GT reference row.

## Phase 3 (2026-07-24): change the objective, not the optimizer

Four inference-only attacks, all sweepable as `GuidanceConfig` fields:

1. **Objective augmentation.** `L = L_ctrl + λ_dyn·‖Δ_t J_g − Δ_t J_u‖² +
   λ_skate·L_skate`. The dynamics anchor takes J_u from an unguided sample of
   the same clip, generated internally with the RNG stream forked so arms stay
   noise-matched, and is applied to joints that are never constrained. It is
   **root-relative** by default: world-frame joint velocities carry the root
   translation, so anchoring them would fight the waypoints — root-relative
   anchors articulation (gait) while leaving the trajectory free. The foot term
   is geometric (horizontal foot velocity gated on foot height) because the 4
   binary contact channels sit in the 196 dims dropped from the 67-D essential
   group. The two terms are complements: skate alone cannot prevent freezing (a
   static pose has zero foot velocity), and the anchor alone does not enforce
   ground contact. This regularizes in motion space; `prox_weight` regularizes
   in z space, which is the wrong metric.
2. **Root-channel reparameterization** (`control/root_edit.py`), for
   `joints=[0]` only. Dims 0:4 are (yaw velocity, local xz velocity, height)
   and the pelvis world trajectory is exactly their cumulative integral, while
   the 63 local-position dims are invariant under edits to them. A waypoint is
   therefore a prefix-sum constraint, and the minimum-norm correction is
   piecewise constant between keyframes — closed form, no gradients, no
   diffusion calls. Exact control at unguided cost with gait preserved by
   construction; the cost it *does* pay is foot skate proportional to how far
   the pelvis had to move, which the optional polish over those four channels
   trades back against waypoint error.
3. **U-turn resampling.** RePaint's renoise-and-redenoise, applied to the
   transport path rather than to tokens: perturb the committed latents back to
   `uturn_t` along the linear interpolant and re-integrate the SiT head to t=1
   under light guidance. Unlike `_repair_round` it never has to guess which
   tokens guidance damaged — the whole sequence is re-solved jointly.
4. **Hard trust region + CFG.** Project z onto ‖z − z₀‖₂ ≤ r·‖z₀‖₂ after every
   Adam step, instead of a soft penalty whose λ trades off against control
   error; plus a per-arm CFG override, since every arm so far ran w=3.0 and
   stronger text conditioning opposes freezing directly.
