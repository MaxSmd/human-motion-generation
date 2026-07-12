# RMG Midterm — PowerPoint Build Spec

> **Purpose of this document.** A complete, self-contained brief for generating a
> 10–15 minute midterm presentation deck. It contains the design system, the exact
> on-slide text (kept minimal), full speaker notes, and a precise figure
> specification for every slide. Anything labelled *FIGURE* is a drawing brief —
> render it as a graphic, not as body text. Low-level math lives in the BACKUP
> section and should only surface if asked.
>
> **Talk length:** 10–15 min. **Audience:** technical but not in-the-weeds.
> **Golden rule from the guidelines:** keep it intuitive, don't dive deep on the
> main path, guide the audience linearly, no random ideas dropped at random points.

---

## 0. The through-line (memorize this; every slide serves it)

> **Motion generation matters → the *representation* is the bottleneck → we put
> motion on the right geometry (a manifold) → that geometry makes *control*
> (constraints) almost free → here is what works: three models and a live UI.**

Five acts, mapped to slides:
1. **Hook** (S1–S2): why text-to-motion is worth doing.
2. **Problem** (S3): every representation of a pose has a catch.
3. **Idea** (S4–S5): RMG = generation *on* a product manifold.
4. **Payoff** (S6–S9): the geometry makes constraints fall out — hard + soft, with a live frontend.
5. **Proof & path** (S10): visual results across three models, and what's next.

Each main slide should be speakable in ~60–90 seconds. 10 slides ≈ 12 minutes + buffer.

---

## 1. Design system (apply to every slide)

**Aesthetic:** "kinematics instrument" — engineering-blueprint feel, calm, precise,
not flashy. Dark theme. This mirrors the project's own web app.

**Palette**
| Role | Hex | Use |
|---|---|---|
| Background | `#0B1220` | deep navy, all slides |
| Panel / card | `#111A2E` | figure boxes, tables, callouts |
| Primary accent (signal) | `#22D3EE` | cyan — titles' underline, key terms, the manifold/velocity |
| Hard-constraint accent | `#DC2626` | red — exact projection |
| Soft-constraint accent | `#059669` | green — energy guidance |
| Targets / clamp band | `#D97706` | amber |
| Body text | `#E5E7EB` | near-white |
| Muted text | `#94A3B8` | captions, slide numbers, secondary |
| Hairline / grid | `#1E2A44` | subtle blueprint grid lines, dividers |

**Typography**
- Headings: a geometric/grotesque sans (e.g. *Bricolage Grotesque*, *Space Grotesk*, or fallback *Montperat/Arial Bold*). Large, tight tracking.
- Body: clean sans (e.g. *Inter*).
- Code / math tokens (e.g. `S³`, `Exp_x`, `R³ × (S³)²²`): monospace (*JetBrains Mono*), cyan.

**Layout grid**
- 16:9. Generous margins (~0.6in).
- Faint blueprint grid in the background at very low opacity (`#1E2A44`, ~6% alpha). Do not let it compete with content.
- Title at top-left, with a short cyan underline rule beneath it.
- Most content slides = **two zones**: left ~45% text, right ~55% figure. (Some slides are figure-dominant — noted per slide.)
- Slide number bottom-right in muted text. A thin footer label bottom-left: "RMG · Midterm".

**Bullets**
- Max ~5 bullets per slide, max ~8 words each. Prefer phrases over sentences.
- Use the cyan accent to highlight the *one* key term per bullet.
- No paragraphs on slides — paragraphs go to speaker notes.

**Figure placeholders**
- Where a figure isn't final yet, draw a panel-colored card with a dashed cyan
  border and the figure's one-line description centered in muted text, plus a small
  "FIGURE" tag top-left. The author will replace these with LaTeX/TikZ renders.
- Several figures already exist as TikZ in `rmg_constraints.tex` — those are marked
  **[exists: rmg_constraints.tex]** and just need compiling to PDF/PNG.

---

## 2. MAIN SLIDES

---

### Slide 1 — Title

**On-slide text**
- Title: **Riemannian Motion Generation**
- Subtitle: Text-conditioned human motion on a product manifold
- Footer line: *Author · Supervisor · Midterm · {date}*

**FIGURE (full-bleed hero, behind/with the title)**
- A single generated human skeleton captured mid-stride, drawn as cyan joints +
  thin bone lines on the dark blueprint grid. Slight motion-trail ghosting of 2–3
  previous frames in lower opacity to imply movement. No labels. Elegant, quiet.

**Speaker notes**
> Title slide. One line: "I'm building a text-to-motion generator that treats
> rotations as what they actually are — points on a sphere — and I'll show that
> this single choice makes the motion easy to *control*." Set the expectation that
> the talk goes problem → idea → payoff → results.

---

### Slide 2 — Why generate human motion?

**On-slide text**
- Heading: **Text → 3D human motion**
- Bullets:
  - "a person walks forward and waves" → animated skeleton
  - Hard: high-dimensional, temporal, must look **plausible**
  - Used in: games · film · **AR/VR** avatars · robotics · biomechanics · synthetic data

**FIGURE (right zone)**
- Left-to-right pipeline strip: a text prompt chip → arrow → three skeleton
  thumbnails (a walk cycle) → arrow → a ring of 5 small application icons
  (game controller, VR headset, robot arm, film clapperboard, DNA/biomech).
- Keep icons monochrome cyan line-art on panel cards.

**Speaker notes**
> Motivate fast. The dream is natural-language direction of a virtual human.
> Applications are broad — animation studios, game NPCs, VR telepresence, robot
> motion priors, and cheap synthetic training data. The challenge: the output is
> high-dimensional and temporal, and humans are *very* good at spotting motion
> that looks wrong. Don't dwell — 45 seconds, then pivot to "but how do you even
> represent a pose?"

---

### Slide 3 — How do you represent a pose? (the crux)

**On-slide text**
- Heading: **The representation is the bottleneck**
- One line under heading: A pose = root position + many joint **rotations**. How you encode rotation decides everything.

**FIGURE (dominant, center/right) — comparison table card**
Three rows, two columns, each row with a small red ✗ motif:

| Representation | The catch |
|---|---|
| **263-D HumanML3D vector** (positions + velocities + 6D rot, redundant) | over-parameterised & internally inconsistent — the model must *learn* to stay valid |
| **Euler / axis-angle** | gimbal lock · discontinuities · non-unique |
| **Quaternions in flat R⁴** | model fights to keep ‖q‖ = 1; renormalising drifts off the sphere |

Visual motifs beside each row: (1) a tangled redundant vector, (2) a spinning frame locking up (gimbal), (3) a point drifting off a circle back toward it with a corrective arrow.

**Speaker notes**
> This is the pivot slide — make the audience feel the problem. Every pose is a
> root translation plus a stack of joint rotations. The *quality of the whole
> system* hinges on how you encode rotation. The popular HumanML3D 263-D vector is
> redundant and internally inconsistent, so the network burns capacity just
> learning to produce *valid* poses. Euler angles gimbal-lock and are
> discontinuous. Flat quaternions look nice but live in R⁴, so diffusing them
> means constantly fighting to stay on the unit sphere — and renormalising after
> the fact quietly drifts. Land the line: "what if we never left the sphere in the
> first place?"

---

### Slide 4 — The RMG representation (the one idea)

**On-slide text**
- Heading: **Take the geometry of rotation seriously**
- Key equation (large, monospace, cyan): `M = R³ × (S³)²²`
- Bullets:
  - one **Euclidean** factor → root translation
  - 22 **independent spheres** → one unit quaternion per joint
  - a quaternion *stays* unit — **by construction**, not by renormalising

**FIGURE (right zone) — the product manifold**
- Left: a flat box labelled `R³` (translation), drawn as a small Cartesian frame.
- A large `×` (product) connector.
- Right: a cluster of 22 small spheres; on a couple of them draw a point with a
  short tangent arrow (foreshadows the velocity field). Label "one S³ per joint".
- Caption (muted): "Translation and rotation are different *kinds* of quantity —
  stop forcing them into one flat vector."

**Speaker notes**
> The core idea, said simply: a pose is a *point on a curved manifold*, not a flat
> vector. Specifically a **product manifold** — one ordinary Euclidean factor for
> the pelvis translation, times 22 independent 3-spheres, one unit quaternion per
> joint. Because we define the model *on* this manifold, a unit quaternion stays a
> unit quaternion at every single step — not by re-normalizing afterward, but
> because every operation is intrinsic to the sphere. Plant the seed for later:
> "the fact that those 22 spheres are *independent* is exactly what lets us pin
> one joint without touching the rest of the body." 60–75 seconds.

---

### Slide 5 — Generation *on* the manifold (go one level deeper)

**On-slide text**
- Heading: **Riemannian flow matching**
- Bullets:
  - learn a **velocity field**; flow noise (t=0) → data (t=1)
  - prior, interpolation, ODE step — all defined *on* the sphere
  - one Riemannian Euler step (boxed):
- Boxed equation (large, monospace): `x_{t+h} = Exp_x( h · Π v_θ(x, t, text) )`
  - small labels: `Π` = project to tangent space · `Exp` = walk the geodesic

**FIGURE (right zone) [partially exists: rmg_constraints.tex sampling loop]**
- A single sphere with: the current point `x`, a tangent velocity arrow `Πv`, and
  a short geodesic hop along the surface to `x_{t+h}`. Beside it, a tiny strip
  showing noise-cloud → clean walk, labelled "t: 0 → 1".

**Speaker notes**
> One level deeper, but stay intuitive. Generation is a *flow*: start from random
> noise on the manifold and follow a learned velocity field until you arrive at a
> real motion. The model predicts a velocity; we **project** it onto the tangent
> plane of the sphere (velocities must be tangent) and take an **exponential-map**
> step — i.e. we walk along the curved geodesic instead of stepping off into flat
> space. That's the only equation I'll show on the main track; the CFM objective
> and the exp/log maps are in backup. Key takeaway: "we never leave the manifold."

---

### Slide 6 — Where the project stands

**On-slide text**
- Heading: **Project state**
- Status lanes (each a pill with ✅ / ◐):
  - ✅ **RMG** — Riemannian flow matching, training (main result)
  - ◐ **MoMask** & **MARDM** — reproductions in the same repo
  - ✅ **Constraints on RMG** — hard + soft, sampling-time, no retraining
  - ✅ **Web app** — generate · visualise · *control* live + cluster control plane

**FIGURE (lower band) — shared-infrastructure strip**
- A horizontal "rail" diagram: one shared **HumanML3D pipeline** → one shared
  **Guo evaluator** → one shared **container**, feeding three model boxes (RMG /
  MoMask / MARDM). Emphasize "build once, reuse across models".

**Speaker notes**
> Quick status board, honest about maturity. RMG is implemented and training — the
> main result. MoMask and MARDM are reproductions sharing the same data pipeline,
> evaluator, and container (set their pill to match reality on the day — MARDM is
> further along than MoMask). Constraints on RMG are done and verified live.
> There's a web app that not only generates but lets you *control* motion
> interactively, plus a cluster control plane. The point of the shared-infra rail:
> adding a model is cheap because everything around it is shared. ~60 sec. Then:
> "now the payoff — why this representation was worth the trouble."

---

### Slide 7 — The payoff: constraints are *easy* here

**On-slide text**
- Heading: **The geometry makes control almost free**
- Bullets:
  - two hooks, both at **sampling time** — trained weights untouched
  - **hard** joint-angle limits → exact projection (closed form, one sphere)
  - **soft** world-space rules → energy gradient in the tangent space
  - elsewhere (flat / 263-D / discrete tokens): no clean handle on "this joint"

**FIGURE (dominant) [exists: rmg_constraints.tex — Fig. "loop"]**
- The sampling-loop diagram: `evaluate v_θ (+CFG)` → `Euler–Exp step` → loop back,
  with a **red** branch "hard projection — clamps x *after* the step" and a
  **green** branch "soft guidance — modifies v *before* the step" feeding the step.
- This figure is the spine of the next two slides — keep it identical there.

**Speaker notes**
> This is the thesis of the talk. Because the model lives on the right geometry,
> *control* drops out without any fine-tuning — we only touch the sampler. There
> are exactly two hooks. A **hard** hook that clamps the state right after each
> step (for joint-angle limits — exact, closed-form). A **soft** hook that nudges
> the velocity right before each step (for world-space goals — a differentiable
> energy). Contrast: in a flat 263-D model or a discrete-token model like
> MoMask, there's no single clean handle on "this one joint's rotation", so the
> same constraints are awkward or need retraining. Tell them the next two slides
> walk these two hooks. ~75 sec.

---

### Slide 8 — Hard constraint: fix / clamp a joint angle

**On-slide text**
- Heading: **Pin a joint — touch one sphere**
- Bullets:
  - "elbow = 90°", "knee ≤ 120°" — bend depends on **one** quaternion
  - independent spheres → **inpaint that factor**, leave the body alone
  - **exact** projection, not a penalty
  - knobs: strength (snap ↔ bias) · ease-in frames

**FIGURE (right zone) [exists: rmg_constraints.tex — Fig. "bend"]**
- Left half: bend geometry — incoming bone `u`, current outgoing direction
  `d = q·v`, the angle `α`, a dashed amber target `β`, and the in-plane correction
  arc.
- Right half: the clamp map (flat → diagonal → flat) with the amber feasible band
  `[α_min, α_max]`; note "fixed angle = zero-width band".
- **Plus a before/after pair** (small, bottom): the same elbow free vs pinned at
  90°. *(Render from the existing clips `L_ELBOWNONE.npy` vs `L_ELBOWX180.npy`,
  and/or `L_KNEENONE.npy` vs `L_KNEECLAMP.npy`.)*

**Speaker notes**
> First hook, concretely. The bend angle at a joint — how flexed the elbow is —
> turns out to depend on just *one* quaternion. And because our 22 spheres are
> independent, fixing that angle means overwriting (inpainting) that single
> spherical factor after each sampling step, leaving the rest of the body free to
> respond. It's an **exact projection** — a closed-form clamp on one quaternion —
> not a soft penalty you hope converges. Two knobs make it usable: a *strength*
> that lets you bias rather than hard-snap, and *ease-in frames* so a windowed
> constraint doesn't pop on and off. Show the before/after clip: "free elbow,
> versus the same prompt with the elbow pinned at 90° — the rest of the motion
> adapts naturally." ~75 sec.

---

### Slide 9 — Soft constraint: world-space rules + live frontend

**On-slide text**
- Heading: **Put the body in a room**
- Bullets:
  - "stay in the room" · "don't hit the table" · "plant the foot"
  - world position = **nonlinear FK** of all joints → no closed-form fix
  - differentiable **energy** through FK → nudges the velocity (scale-invariant)
  - exact spawn placement = rigid **SE(2)** drop at a marker

**FIGURE A (left-of-figure) [exists: rmg_constraints.tex — Fig. "room"]**
- Side-view room box with an obstacle; a stick body leaving the room; green `−∇E`
  arrows pushing the violating joints back inside; an amber "contact pull" arrow
  resting a hand on the obstacle top.

**FIGURE B (right-of-figure) — app screenshot**
- The web app's Room/Scene editor (3D room + draggable obstacle) **next to** the
  in-browser motion player showing the live violation HUD: violating joints drawn
  as enlarged **red** spheres, a violation timeline strip under the transport.

**Speaker notes**
> Second hook. Some requirements live in *world space* — stay inside the room,
> don't walk through the table, keep a foot planted. The catch: a joint's world
> position is a nonlinear forward-kinematics function of *all* the joint rotations
> at once, so there's no closed-form projection. Instead we define a
> differentiable **energy** — squared penetration into walls/obstacles via
> signed-distance fields, plus contact and anti-foot-skate terms — and
> backpropagate it through FK to nudge the velocity. It's *soft*: it steers, the
> model still drives. The one exact world-space op is spawn placement — a rigid
> SE(2) transform that drops the clip at a marker. Then demo the frontend: drag an
> obstacle, watch the body deflect, and the HUD light up red where it violates.
> Honesty note if asked: soft guidance *deflects* at solid obstacles but doesn't
> *climb* — stairs need a contact term (on the roadmap). ~90 sec.

---

### Slide 10 — Results & outlook

**On-slide text**
- Heading: **Results & next steps**
- Bullets:
  - qualitative: **RMG vs MoMask vs MARDM**, same prompt
  - RMG-base ≈ **24.7M** params, **FID ≈ 0.5**; scaling to **rmg-mid (~112M)** for lower FID & smoother motion
  - constraints verified **live** (fixed angles · hinge limits · room guidance)
  - next: tune soft-guidance weight · **contact/support** term (stairs, floor) · constraints for MoMask/MARDM · quantitative constraint metrics

**FIGURE (dominant) — 3-up comparison + tiny table**
- Three looping GIF panels side by side (one per model) on the *same* caption —
  the headline visual. Label each panel with the model name.
- Bottom-right: a small results table — `Model · #params · FID` — filled from your
  eval runs. (RMG-base row prefilled: 24.7M / ≈0.5; leave MoMask/MARDM blank until measured.)

**Speaker notes**
> Land it on something they can *see*. Three clips, same prompt, one per model —
> this is the moment that earns the talk. Quantitatively, RMG-base is ~24.7M
> params at FID about 0.5, and I'm scaling to a ~112M "mid" model to push FID
> lower and the motion smoother. The constraints are verified live on the cluster.
> What's next: tune the soft-guidance weight on the full model, add a
> contact/support term so feet can rest and the body can climb, extend constraints
> to MoMask and MARDM, and add quantitative constraint-satisfaction metrics. Close
> by restating the through-line in one sentence: "the right geometry gave us both
> valid motion *and* easy control." ~75 sec. Then take questions — backup slides
> below cover the math.

---
---

## 3. BACKUP SLIDES (appendix — only on demand)

> Visually quieter than the main deck. Same palette, but allow denser text and full
> equations. Title each "Backup · {topic}".

---

### B1 — The representation in detail (T+R + ablations)

**On-slide text / content**
- Layout: `flat[0:3] = τ` · `flat[3+4j : 3+4(j+1)] = q_j` · `D = 3 + 4·22 = 91`
- `q₀` = global root orientation; `q₁…q₂₁` = local joint rotations (SMPL tree)
- HumanML3D ships joint *positions* → **inverse kinematics** → per-joint quaternions
- then **sign-continuity** across time (`q ≡ −q` double cover — pick a smooth hemisphere)
- ablation reps: **T+P** (Kendall pre-shape of the joint cloud, rotations dropped) · **T+R+P** (three-factor)

**FIGURE:** the 22-joint SMPL kinematic tree with the flat 91-D vector layout annotated beneath it.

**Notes:** This is the "what's actually in the vector" slide. Mention that T+R is the main result; P is the interesting geometric alternative.

---

### B2 — Riemannian flow matching: the math

**On-slide text / content**
- conditional flow matching: regress `v_θ` onto the geodesic target velocity between prior and data
- need: S³ exponential & log maps, tangent projection `Π`, geodesic interpolation
- target velocity carries a `θ / sin θ` factor → blows up near the **antipodal cut locus** (θ → π)

**FIGURE:** a sphere with prior point, data point, the connecting geodesic, and the tangent target velocity; mark the antipodal point as a singularity (red).

**Notes:** Sets up B3. The `θ/sin θ` singularity is the seed of the training-stability story.

---

### B3 — Training stability (the war story)

**On-slide text / content**
- bf16 + antipodal cut locus → one bad batch → Inf/NaN; **NaN is absorbing** (poisons weights, EMA, Adam moments)
- ships ON: **non-finite step guard** — skip `opt.step()` + `ema.update()` when loss/grad non-finite
- **antipodal alignment** (flip data quat into prior's hemisphere, θ ≤ π/2) removes the singularity — *but only correct from scratch*; retrofitting shocks the target distribution and diverges
- subtlety: `grad_norm` can be `inf` because a finite-but-huge gradient's L2 norm overflows fp32 — not because any element is NaN

**FIGURE:** loss curve showing the NaN divergence and the recovery after quarantining bad checkpoints; inset: bf16 vs fp32 target-velocity magnitude as θ → π.

**Notes:** Only if a stability/precision question comes up. Good signal that the project is real and hard-won.

---

### B4 — Hard projection: the formula

**On-slide text / content**
- bend `α = arccos(u·d)`, with `d = q·v`
- clamp `β = clamp(α, α_min, α_max)`; in-plane axis `n = (u×d)/‖u×d‖`
- correction `Δ = ( cos(s(β−α)/2), sin(s(β−α)/2)·n )`; update `q ← Δ ⊗ q`
- touches only bend **magnitude** — twist about the bone and bend *direction* survive
- fixed angle = degenerate band (`α_min = α_max`) → one projector serves fixed & clamped
- hinge limits = swing–twist decomposition: clamp signed twist, shrink swing to identity

**FIGURE:** the clamp map (flat–diagonal–flat) beside a swing/twist decomposition sketch of a quaternion.

---

### B5 — Soft guidance: the formula

**On-slide text / content**
- clean-sample estimate `x̂₁ = Exp_x((1−t)v)`; world joints `X = FK(x̂₁)`; place into room
- energy `E(X) = Σ ReLU(sdf + m)²  +  Σ ReLU(‖X_j − tgt‖ − tol)²  +  E_skate`
  - terms: room/obstacle SDFs · contacts · anti-foot-skate
- guidance `g = Π ∇_x E`;  `v ← v − w · ‖v‖ · (g/‖g‖)` — velocity-relative, scale-invariant, composes with the hard hook
- SDFs for box / sphere / cylinder; padding `m` inflates keep-out so it **brakes before contact**
- soft *deflects* at solids, does **not** climb — stairs/floor need a contact/support term (TODO)

**FIGURE:** an SDF heatmap of a room with one obstacle, overlaid with the `−∇E` gradient field arrows.

---

### B6 — System & engineering

**On-slide text / content**
- one repo, multiple models: shared HumanML3D pack · shared **Guo evaluator** (FID / R@k / Diversity / MM-Dist) · shared enroot container
- web app: **FastAPI** backend (in-process RMG inference) + **Next.js** frontend
- cluster control plane: single SSH ControlMaster · one job at a time + local queue · **24h-walltime auto-resubmit** that resumes from `latest.pt`

**FIGURE:** architecture box diagram — frontend ↔ backend ↔ SSH/SLURM ↔ GPU, with the shared data mount.

---

### B7 — Evaluation protocol

**On-slide text / content**
- Guo et al. text-to-motion evaluator on HumanML3D
- metrics: **FID** (realism) · **R-precision@k** (text alignment) · **Diversity** · **Multimodal-Dist**
- classifier-free-guidance scale `ω` sweep → pick best-FID `ω`
- mirror-augment is **train-only** (be explicit for eval splits)

**FIGURE:** an FID-vs-ω sweep curve, marking the chosen operating point.

---

## 4. Pre-flight checklist (fill before presenting)

- [ ] MoMask / MARDM status pills on Slide 6 set to true maturity (MARDM has real transport/training code; MoMask is thinner).
- [ ] Slide 10 results table: real FID numbers for MoMask & MARDM (or mark "in progress").
- [ ] Slide 10: export the three same-prompt GIFs (RMG / MoMask / MARDM).
- [ ] Slide 8: render the elbow/knee before-after from `L_ELBOW*.npy` / `L_KNEE*.npy`.
- [ ] Compile the three TikZ figures from `rmg_constraints.tex` (loop / bend / room) to PNG/PDF.
- [ ] Confirm RMG-base param count & FID against the latest eval run before quoting.
- [ ] Author / supervisor / date on the title slide.
