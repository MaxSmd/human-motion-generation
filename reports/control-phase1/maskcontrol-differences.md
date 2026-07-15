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
* *Considered but not implemented:* proximal anchor `λ‖z−z₀‖²` and an
  error-tolerance early stop for the inner loop. Standard guided-diffusion
  trust-region machinery, **not** from MaskControl — if we ever report them,
  they must be labeled as our own ablation.

## Evaluation differences

* Traj./Loc./Avg. error (0.5 m threshold, waypoints sampled from GT joints)
  follow the OmniControl protocol and are directly comparable to MaskControl's
  tables.
* **FID is not comparable to their published numbers**: ours is computed under
  the repo's Guo evaluator on our canonical-features stack; theirs under the
  standard redundant-263-D evaluator. Only within-run guided-vs-unguided
  deltas (same seeds) are meaningful.

## Phase-1 numbers (2026-07-15, MARDM-M @300k canonical, CFG w=3.0)

Optimization-only (no regularizer, no repair), 5 pelvis keyframes:

| | Avg. err | Loc. err | Traj. err | FID | R@1 |
|---|---|---|---|---|---|
| unguided (512 clips) | 0.648 m | 0.363 | 0.541 | 0.268 | 0.504 |
| guided (512 clips) | 0.015 m | 0.003 | 0.012 | 1.447 | 0.385 |

Control matches MaskControl's accurate mode with zero training; the 5.4× FID
degradation is the measured cost of having neither their regularizer (phase 2)
nor a restore mechanism — the motivating row for both. Qualitative signature:
on locomotion clips the optimizer satisfies pelvis waypoints by damping gait
(clip 004822: foot speed 0.57 → 0.10 m/s) rather than by cancelling steps.
