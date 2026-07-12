# RMG — Midterm Presentation (content draft)

Target: 10–15 min · 10 main slides + backup · keep it intuitive, low-level lives in backup.

Guiding narrative (one sentence per act so the talk doesn't wander):
**Motion generation matters → the representation is the bottleneck → we put motion on the right geometry (a manifold) → that geometry makes *control* (constraints) almost free → here's what works, in three models and a live UI.**

---

## Slide 1 — Title

**Text**
- Riemannian Motion Generation (RMG)
- Text-conditioned human motion on a product manifold
- Your name · supervisor · date · "Midterm"

**Figure**
- One hero still: a generated skeleton mid-stride, clean blueprint/grid background (matches the app aesthetic). No words on it.

---

## Slide 2 — Human Motion Generation (motivation)

**Text**
- Goal: *text → 3D human motion* ("a person walks forward and waves")
- Why it's hard: high-dimensional, temporal, must look physically plausible
- Applications: games & film, AR/VR avatars, robotics, biomechanics, synthetic data

**Figure**
- Left→right strip: a text prompt → arrow → a 3-frame motion thumbnail.
- A small ring of application icons (controller, headset, robot arm, clapperboard) around the output. Intuitive, no detail.

---

## Slide 3 — How do you represent a pose? (and why it's the crux)

**Text**
- A pose = root position + many joint rotations
- The whole pipeline's quality hinges on *how you encode rotation*
- Three common choices, each with a catch ↓

**Figure**
- 3-column comparison card, one row of "the catch" under each:
  | Representation | Issue |
  |---|---|
  | **263-D HumanML3D feature vector** (positions + velocities + rot-6D, redundant) | over-parameterised, internally inconsistent, model must *learn* the constraints |
  | **Euler angles / axis-angle** | gimbal lock, discontinuities, non-unique |
  | **Quaternions in flat R⁴** | model must fight to keep ‖q‖=1; renormalise-after-the-fact drifts off the sphere |
- Visual cue: each option shown with a small red "✗" annotation (gimbal-lock spin, a quaternion drifting off a circle, a tangled feature vector).

---

## Slide 4 — The RMG representation (the one idea)

**Text**
- Take the geometry of rotations *seriously*
- A pose is a point on a **product manifold**: `M = R³ × (S³)²²`
  - one Euclidean factor → root translation
  - 22 independent spheres → one unit quaternion per joint
- A quaternion *stays* a unit quaternion at every step — by construction, not by renormalising

**Figure**
- Split diagram: on the left a flat box labelled `R³` (translation), on the right 22 small spheres each with a point + tangent arrow on it. Bracket them together as "× (product)".
- Caption hook: "Translation and rotation are *different kinds of quantity* — stop forcing them into one flat vector."

---

## Slide 5 — Project idea: generation *on* the manifold (go a bit deeper)

**Text**
- Generative model = **Riemannian flow matching**: learn a velocity field, integrate from noise (t=0) to data (t=1)
- Every operation is intrinsic to the manifold:
  - prior, interpolation, and ODE step all defined *on* the sphere
  - one **Riemannian Euler step**: `x_{t+h} = Exp_x( h · Πv_θ(x,t,text) )` — project velocity to tangent space, walk the geodesic
- Result: never leaves the manifold; geometry is respected end-to-end

**Figure**
- Reuse/adapt the sampling-loop TikZ from `rmg_constraints.tex` (the model → Euler–Exp step → loop). Show one curved sphere with a tangent velocity arrow and a short geodesic hop.
- Keep math to that single boxed equation; details go to backup.

---

## Slide 6 — Project state (where we are)

**Text**
- ✅ **RMG** implemented & training (Riemannian flow matching, the main result)
- ✅ **MoMask** & **MARDM** reproductions in the same repo (shared data pipeline / evaluator / container)
- ✅ **Constraints on RMG** — hard joint-angle + soft Euclidean (sampling-time, no retraining)
- ✅ **Web app** — generate, visualise, and *control* motion live; cluster control plane

**Figure**
- A simple status board / roadmap bar with 4 lanes (RMG · MoMask · MARDM · Constraints+App), each with a green check or "in progress" pill.
- Small note: "one HumanML3D pipeline, one Guo evaluator, one container shared across all models."

---

## Slide 7 — The payoff: constraints are *easy* on this representation

**Text**
- Claim: the right geometry makes *control* fall out for free
- Two hooks, both at **sampling time**, trained weights untouched:
  - **Hard** joint-angle limits → exact projection (closed form on one sphere)
  - **Soft** world-space rules → energy gradient in the tangent space
- Why it's hard elsewhere: flat / 263-D / discrete-token models have no single clean handle on "this joint's rotation"

**Figure**
- The two-hook loop figure from `rmg_constraints.tex` (Fig. "loop"): model → Euler–Exp step, with a red "hard projection (after step)" branch and a green "soft guidance (before step)" branch feeding back. This is the spine of the next two slides.

---

## Slide 8 — Hard constraint: fix / clamp a joint angle

**Text**
- "Elbow = 90°", "knee ≤ 120°" — a bend angle depends on **one** quaternion
- Because RMG factors the body into independent spheres → **inpaint that one factor**, leave the rest of the body alone
- Exact (closed-form projection), not a penalty; knobs: strength + ease-in frames

**Figure**
- The bend-geometry figure from `rmg_constraints.tex` (Fig. "bend"): incoming bone u, outgoing direction d=q·v, the angle α, and the clamp map (flat–diagonal–flat) on the right.
- Pair it with a before/after skeleton thumbnail: free elbow vs pinned-90° elbow (you already have `L_ELBOW*.npy` / `L_KNEE*.npy` clips for exactly this).

---

## Slide 9 — Soft constraint: world-space rules (+ live frontend)

**Text**
- "Stay in the room", "don't walk through the table", "plant the foot"
- World position is a *nonlinear* FK of all joints → no closed-form fix
- → differentiable **energy** backpropped through FK, nudges the velocity (scale-invariant, still model-driven)
- Showcase: drag a room/obstacle in the browser → motion adapts; also Euclidean spawn placement (exact SE(2))

**Figure**
- The room-guidance figure from `rmg_constraints.tex` (Fig. "room"): side-view room box, an obstacle, a stick body leaving the room with green −∇E arrows pushing it back.
- Screenshot of the app's Room/Scene editor + the in-browser 3D motion player with the live violation HUD (red joints when penetrating).

---

## Slide 10 — Results & outlook

**Text**
- Visual comparison: **RMG vs MoMask vs MARDM** on the same prompt (qualitative)
- RMG-base: ~24.7M params, FID ≈ 0.5; scaling up (rmg-mid ~112M) to push lower & smoother
- Constraints verified live on the cluster (fixed angles, hinge limits, room guidance)
- **Next:** tune soft-guidance weight on the full model · contact/support term (stairs, floor) · constraints for MoMask/MARDM · quantitative constraint-satisfaction metrics

**Figure**
- 3-up grid of looping GIFs (one per model) on the *same* caption — the headline visual moment of the talk.
- Small results table corner: model · #params · FID (fill from your eval runs).

---
---

# BACKUP SLIDES (low-level, pull up only if asked)

## B1 — The representation in detail (T+R, and ablations)
- Layout: `flat[0:3]=τ`, `flat[3+4j:3+4(j+1)]=q_j`, D = 3 + 4·22 = 91
- q₀ = global root orientation; q₁…q₂₁ = local joint rotations (SMPL tree)
- HumanML3D ships *positions* → inverse kinematics → per-joint quaternions, then **sign-continuity** across time (q ≡ −q double cover)
- Ablation reps: **T+P** (Kendall pre-shape of the joint cloud, rotations dropped) and **T+R+P** (three-factor)
- *Figure:* the kinematic tree with the flat-vector layout annotated beneath.

## B2 — Riemannian flow matching, the math
- CFM objective: regress vᵢ onto the geodesic target velocity between prior and data sample
- S³ exponential/log maps; tangent projection Π; geodesic interpolation
- Target velocity has a `θ/sin θ` factor → blows up near the **antipodal cut locus** (θ→π)
- *Figure:* a sphere with prior point, data point, geodesic, and the tangent velocity; mark the antipodal singularity.

## B3 — Training stability (the war story, if asked)
- bf16 + antipodal cut locus → one bad batch produces Inf/NaN; NaN is *absorbing* (poisons weights, EMA, Adam moments)
- Fix that ships ON: **non-finite step guard** (skip opt.step + ema.update when loss/grad non-finite)
- **Antipodal alignment** (flip data quat into prior's hemisphere, θ ≤ π/2) removes the singularity — *but only correct from scratch*; retrofitting shocks the target distribution and diverges
- grad_norm can be `inf` because a finite-but-huge gradient's L2 norm overflows fp32, not because any element is NaN
- *Figure:* loss curve with the NaN divergence + recovery; bf16 vs fp32 target-velocity magnitude near θ→π.

## B4 — Hard projection, the formula
- bend α = arccos(u·d), d = q·v; β = clamp(α, α_min, α_max); axis n = (u×d)/‖u×d‖
- Δ = (cos(s(β−α)/2), sin(s(β−α)/2)·n); q ← Δ⊗q
- Touches only bend *magnitude* — twist about the bone and bend *direction* survive
- Fixed angle = degenerate band α_min=α_max → one projector serves both fixed & clamped
- Hinge limits = swing-twist decomposition, clamp signed twist, shrink swing to identity
- *Figure:* the clamp-map plot + swing/twist decomposition sketch.

## B5 — Soft guidance, the formula
- clean-sample estimate x̂₁ = Exp_x((1−t)v); X = FK(x̂₁); place into room
- E(X) = Σ ReLU(sdf+m)² (room/obstacle SDFs) + Σ ReLU(‖Xⱼ−tgt‖−tol)² (contacts) + E_skate
- g = Π∇ₓE; v ← v − w·‖v‖·(g/‖g‖)  → velocity-relative (scale-invariant), composes with the hard hook
- box/sphere/cylinder SDFs; ReLU(violation)²; padding m inflates keep-out so it brakes before contact
- Soft *blocks/deflects* at solid obstacles, does **not** climb — true stair-climbing needs a contact/support term (TODO)
- *Figure:* SDF heatmap of a room with an obstacle + gradient field arrows.

## B6 — System / engineering (if asked)
- One repo, multiple models: shared HumanML3D pack, shared Guo evaluator (FID / R@k / Diversity / MM-Dist), shared enroot container
- Web app: FastAPI backend + Next.js frontend; in-process RMG inference + SLURM cluster control plane
- Cluster: single SSH ControlMaster, one job at a time + local backend queue, 24h-walltime **auto-resubmit** that resumes from `latest.pt`
- *Figure:* architecture box diagram (frontend ↔ backend ↔ SSH/SLURM ↔ GPU; data mount).

## B7 — Evaluation protocol (if asked)
- Guo et al. text-to-motion evaluator on HumanML3D
- Metrics: FID (realism), R-precision@k (text alignment), Diversity, Multimodal-Dist
- Guidance-scale (CFG ω) sweep → pick best-FID ω; mirror-augment train-only
- *Figure:* FID-vs-ω sweep curve.
