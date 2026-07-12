# RMG — the mathematics, front to end

*A methodical walk through what this project actually computes. The focus is the
geometry and the generative model — not buttons, jobs, or UI. Read this as the
story of a single number: a human motion, born as noise, flowed into a clip,
then bent and placed under hard and soft constraints.*

Notation: quaternions are `[w, x, y, z]` (scalar-first). The world frame is
HumanML3D's: **X right, Y up, Z forward, floor at y = 0**. The skeleton is the
22-joint SMPL body subset. `J = 22`.

---

## 0. The one idea

Almost every motion model represents a pose as a flat vector in `R^n` and
diffuses in that flat space. RMG instead takes the **geometry of rotations
seriously**: a pose is a point on a curved manifold, and generation is a flow
that lives *on* that manifold and never leaves it. A unit quaternion stays a
unit quaternion at every integration step — not because we re-normalize after
the fact, but because every operation (the prior, the interpolation, the ODE
step) is defined intrinsically on the sphere.

That single commitment is what makes the constraint story clean later: "hold
this joint angle" becomes "inpaint one spherical factor," and "stay in the room"
becomes "follow a gradient in the tangent space." Both fall out of the geometry
rather than being bolted on.

---

## 1. A motion as a point on a manifold

### 1.1 The representation (T + R)

A clip of `T` frames is encoded frame-by-frame. Each frame is

- a **root translation** `τ ∈ R³` (pelvis position), and
- one **unit quaternion per joint** `q_j ∈ S³`, `j = 0 … 21`, where `q_0` is the
  global root orientation and `q_1 … q_21` are local joint rotations relative to
  the parent in the kinematic tree (SMPL convention).

These are packed into a flat vector of dimension

```
D = 3 + 4·J = 3 + 4·22 = 91.
```

Layout (`representation/tplusr.py`):

```
flat[0:3]                  = τ            (translation)
flat[3+4j : 3+4(j+1)]      = q_j          (joint j's quaternion)
```

The HumanML3D clips ship as joint *positions*; the data pipeline runs **inverse
kinematics** to recover per-joint quaternions, then enforces sign **continuity**
across time (`q` and `−q` are the same rotation, so the raw IK output can flip;
we choose the hemisphere that makes the quaternion track continuously, which
matters because the network sees a smooth trajectory).

### 1.2 Why this is the right object

The point is that translation and rotation are *different kinds of quantity*.
Translation lives in flat Euclidean space; rotation lives on a sphere. Forcing
them into one flat vector and adding Gaussian noise treats a quaternion like a
displacement — and then you spend the whole model fighting to keep `‖q‖ = 1`.
RMG keeps them on the manifold where they belong:

```
M_RMG = R³ × (S³)^J         (paper eq. 3)
```

a **product manifold**: a Euclidean factor for the root translation and `J`
independent spheres, one per joint. The independence of the factors is not a
detail — it is exactly what lets us later pin a single joint without touching
the rest of the body (§6.4).

The codebase also implements two ablation variants (`representation/registry.py`):

- **T + P**: translation + a **Kendall pre-shape** (§2.4) of the joint cloud —
  rotations dropped entirely.
- **T + R + P**: both, on a three-factor product manifold.

T + R is the main result; P is the interesting geometric alternative.

---

## 2. The manifolds and their calculus

Every manifold implements the same small interface (`manifolds/base.py`): a
tangent projection, the exponential and logarithm maps, a constant-speed
geodesic and its velocity, and a wrapped-Gaussian sampler. Riemannian flow
matching needs nothing else. The metric on every factor is the one **induced
from the ambient Euclidean inner product**, so the tangent inner product
`⟨v, v⟩_g` is just the squared L2 norm of the tangent-projected vector — which
is why the training loss can be a plain MSE (§3.4).

### 2.1 Euclidean `R^d`

The trivial case, and a sanity baseline: every map collapses to the obvious
thing.

```
proj_tangent(x, u) = u
Exp_x(v)           = x + v
Log_x(y)           = y − x
geodesic(x0,x1,t)  = (1−t)·x0 + t·x1
```

On `R^d` Riemannian flow matching reduces *exactly* to standard Euclidean
conditional flow matching, with CFM target `x1 − x0`. The root translation rides
on this factor.

### 2.2 The sphere `S^d ⊂ R^{d+1}`

This is where the geometry earns its keep. A unit quaternion is a point on `S³`.

**Tangent projection** removes the radial component (the tangent plane at `x` is
orthogonal to `x`):

```
proj_tangent(x, u) = u − ⟨x, u⟩ x
```

**Exponential map** — walk along the sphere from `x` in tangent direction `v`,
for arc length `‖v‖`:

```
Exp_x(v) = cos(‖v‖)·x + sin(‖v‖)·v/‖v‖
```

**Logarithm map** — the inverse: the tangent vector at `x` pointing toward `y`,
with length equal to the geodesic (angular) distance `θ = arccos⟨x, y⟩`:

```
Log_x(y) = θ · (y − ⟨x,y⟩x) / sin θ
```

**Geodesic** between two points is **slerp**:

```
γ(t) = sin((1−t)θ)/sin θ · x0 + sin(tθ)/sin θ · x1
```

**A correctness note worth flagging** (`manifolds/sphere.py`): the paper's
appendix writes the geodesic velocity `γ̇(t)` with a `sin` where it should be a
`cos`. The code uses the correct derivative

```
γ̇(t) = θ/sin θ · ( −cos((1−t)θ)·x0 + cos(tθ)·x1 )
```

which is the only form that is actually tangent at `t = 0` and that equals
`Log_{γ(t)}(x1)/(1−t)` (verified algebraically). This matters because that
quantity *is* the training target (§3.3).

All of these have `θ → 0` and antipodal guards (Taylor-expanding `·/sin θ` near
zero), because slerp is numerically nasty exactly where the two rotations are
close — which is most of the time during late sampling.

### 2.3 The product manifold

`manifolds/product.py` is pure dispatch: a point is a flat `(…, 91)` tensor, it
gets `_split` into per-factor slices, each factor's operation runs on its slice,
and the results are `_join`ed back. Because the metric is a product metric, the
geodesic on `M_RMG` is just the per-factor geodesics run in parallel, and the
squared tangent norm is the sum of the per-factor squared norms. The network and
the constraints only ever see the flat 91-vector; the product structure is
invisible plumbing that guarantees every factor stays on its own sphere.

### 2.4 The Kendall pre-shape sphere (the elegant ablation)

`manifolds/preshape.py` is the most mathematically interesting alternative. A
**pre-shape** of `J` landmarks in `R^m` is the joint cloud after removing
translation and scale:

1. **center** the `J × m` matrix (column means zero), and
2. **normalize** to unit Frobenius norm.

What remains is a point on a hypersphere living in the centered subspace, of
intrinsic dimension `J·m − m − 1` (m centering constraints + 1 scale
constraint). So "the shape of the pose, modulo where it is and how big it is" is
literally a point on a sphere, and its geodesics are again slerp on the centered
subspace. The tangent projection composes two projections: kill the
centroid-shifting direction, then kill the radial direction. This is Kendall
shape space, and it lets the model generate *poses as shapes* without ever
parameterizing rotations — a clean geometric counterpoint to the quaternion
representation.

---

## 3. Generation as a flow on the manifold

The generative model is **Riemannian conditional flow matching** (RFM). The
mental model: define, for each ground-truth motion `x1`, a path that drags a
noise sample `x0` to `x1` along a geodesic; train a network to predict the
velocity field of that path; then at inference, integrate the field from fresh
noise.

### 3.1 The conditional probability path

Pick a noise point `x0` from a prior on the manifold and a data point `x1`. The
conditional path is the **geodesic** between them, traversed at constant speed:

```
x_t = γ(t) = geodesic(x0, x1, t),     t ∈ [0, 1].
```

On `R³` this is a straight line; on each `S³` it is slerp. At `t = 0` we are at
noise, at `t = 1` at data.

### 3.2 The wrapped-Gaussian prior

`x0` is **not** drawn from a flat Gaussian — that would put mass off the
manifold. Instead the prior is a **wrapped Gaussian** centered at a reference
point `μ` (`flow/prior.py`): draw tangent noise `ε ~ N(0, σ²I)` at `μ`, project
it into `T_μ M`, and push it onto the manifold with the exponential map:

```
x0 = Exp_μ( proj_tangent(μ, σ·ε) ).
```

For RMG the reference `μ` is the **rest pose**: `τ = 0` and `q_j = [1,0,0,0]`
(identity rotation) for every joint. So generation always starts from "a fuzzy
T-pose at the origin" and flows outward. Per-factor `σ` is allowed.

### 3.3 The conditional flow-matching target

The velocity that transports `x_t` toward `x1` along the geodesic is

```
v(x_t | x1) = (1 / (1−t)) · Log_{x_t}(x1)               (paper eq. 4)
```

i.e. point from the current state toward the data, with a magnitude that keeps
the trip on schedule. For the constant-speed geodesics here this reduces exactly
to the geodesic velocity `γ̇(t)` (§2.2), and the code uses that closed form
directly — both to avoid a `geodesic → log` round-trip and because it is
numerically symmetric at both endpoints. On `R³` it is simply `x1 − x0`, the
familiar linear-CFM target.

This target is the **regression label**: it depends only on `(x0, x1, t)`, never
on the network. That is the whole trick of flow matching — a simulation-free,
per-sample target that, in expectation over `x0`, trains the marginal velocity
field of the data distribution.

### 3.4 The training loss

One optimization step (`flow/trainer.py`):

1. sample `t ~ U[ε, 1−ε]` (bounded off the endpoints so `1/(1−t)` and slerp stay
   clean);
2. sample `x0` from the wrapped-Gaussian prior;
3. build `x_t = γ(x0, x1, t)` and the target `v_⋆ = γ̇(t)`;
4. predict `v_θ = model(x_t, t, cond)`, then **tangent-project** it:
   `v_θ ← proj_tangent(x_t, v_θ)`;
5. loss = masked mean of `‖v_θ − v_⋆‖²` over valid frames.

Because the metric is induced, the tangent inner product is the ambient L2 norm,
so the geometrically-correct loss is an ordinary MSE — no metric tensor to carry
around. The network may output anything in ambient `R^91`; projecting onto
`T_{x_t}M` before the loss means it is only ever scored on the component that can
actually move the point along the manifold.

**Classifier-free guidance** is trained by randomly replacing the text condition
with a null embedding with probability `p ≈ 0.1`, so the same weights model both
the conditional and unconditional velocity fields.

### 3.5 The network (briefly)

The backbone is a 1-D **DiT** (`models/dit.py`): each frame's 91-vector is a
token, with learned positional embeddings and **AdaLN-Zero** conditioning. A
fused (text, time) MLP produces shift/scale/gate triplets that modulate every
block; the gates and the final layer are **zero-initialized**, so at init the
network outputs exactly zero velocity (the identity flow) and learns to deviate.
Text comes from a frozen Qwen3 embedding; time enters through a sinusoidal
frequency embedding. The output is the **ambient velocity** `v_θ`, same shape as
the input, and the trainer/sampler are responsible for the tangent projection.
Nothing in the network knows about the manifold — all the geometry is in the
loss and the sampler. That separation is deliberate and is what keeps the model
architecture boring and the geometry honest.

---

## 4. Sampling: a Riemannian Euler integrator

Inference integrates the learned ODE from `t = 0` to `t = 1`
(`flow/sampler.py`). The update is the manifold analogue of an Euler step — take
the velocity, project it to the tangent space, and move along the manifold with
the exponential map:

```
x_{t+h} = Exp_{x_t}( h · proj_tangent(x_t, v_θ(x_t, t, cond)) )      (paper eq. 5)
```

Start from `x_0 ~ prior`, march over a uniform `t`-grid, end at a clip on the
manifold, then decode the 91-vector back into `(τ, q)` and run forward
kinematics to get joint positions for rendering.

**Classifier-free guidance** runs the model twice per step (conditional and
unconditional) and combines the **ambient** velocities *before* projection:

```
v = v_uncond + ω·(v_cond − v_uncond),     then project, then Exp.
```

Projection is linear, so combining before or after projecting is identical — the
code combines before to halve nothing but to keep the guidance scale acting on
the raw field.

---

## 5. Constrained generation — the interesting part

The headline question: **how do you impose user constraints on a trained model
without retraining it?** RMG answers with a clean dichotomy that falls straight
out of the geometry:

> **If a constraint has a closed-form projection onto a single manifold factor,
> enforce it exactly by projecting after every ODE step. If it does not (because
> it couples many joints through forward kinematics), enforce it softly with a
> differentiable energy whose gradient nudges the velocity.**

Joint-angle limits are the first kind. Room/obstacle/contact constraints are the
second.

### 5.1 Why the dichotomy is forced

A joint's **bend angle** is a function of *one* quaternion (§5.3). The set of
quaternions achieving a given bend is a submanifold of a single `S³` factor, and
we can write the nearest feasible quaternion in closed form. So we can *project*
— an exact hard constraint.

A joint **position** in the world is `FK(all 22 quaternions, root)` — a
deeply nonlinear function of the entire pose. "This foot is inside the room and
outside every obstacle" has no closed-form projection onto the 91-vector. So we
fall back to **guidance**: define a smooth penalty on positions, differentiate
it through FK, and steer the sampler down the gradient. A soft constraint.

The sampler exposes exactly these two hooks, applied every step after the Exp
update:

- `project_fn(x) → x` — snap constrained factors onto their feasible set;
- `energy_fn(x̂) → scalar` + `guidance_weight` — a penalty whose gradient bends
  the velocity.

### 5.2 The forward-kinematics off-by-one (an anatomy subtlety)

HumanML3D's FK (`shared/geometry/skeleton.py`) uses a **per-chain** convention:
walking down a kinematic chain, the offset of joint `chain[i]` is rotated by the
running rotation *after* multiplying in `quats[chain[i]]`. The consequence is
that `quats[j]` orients the bone *leading into* joint `j` (the parent→j
segment), **not** the bend *at* `j`.

So the anatomical bend at joint `j` — the angle between its incoming and outgoing
bones — is governed by the quaternion of the **next** joint along the chain (its
child):

```
elbow bend   ↔ quats[wrist]
knee bend    ↔ quats[ankle]
shoulder     ↔ quats[elbow]
neck         ↔ quats[head]
```

`bend_controller_index()` performs this remap. Without it, "pin the elbow" would
re-aim the upper arm and leave the elbow free — the exact bug an early demo
exposed. End-effectors (wrists, ankles, feet, head) have no outgoing bone, so
they have no representable bend; the root quaternion is global orientation, not a
bend. Both raise rather than silently doing nothing.

### 5.3 Bend-angle limits by projection

Let `u` be the (unit) incoming rest bone `parent(j) → j` and `v` the outgoing
rest bone `j → child(j)`, both read from the skeleton offsets. Let
`q := quats[child(j)]` be the controller quaternion. The current outgoing
direction is `d = q · v` (quaternion rotation), and the **bend** is

```
α = angle(u, d) = arccos⟨u, d⟩,      α = 0 ⇒ straight.
```

To clamp `α` into `[α_min, α_max]`, rotate `d` *within the bend plane* (the plane
spanned by `u` and `d`, whose normal is `n = u × d`) by the deficit `β − α`,
where `β = clamp(α, α_min, α_max)`:

```
Δ = quat(axis = n/‖n‖, angle = β − α)
q ← Δ ⊗ q
```

(`flow/constraints.py::bend_clamp`). Because `Δ` acts only on the outgoing
direction in the bend plane, it changes the bend **magnitude** while leaving the
joint's **twist about its own bone** and the **bend direction** untouched — the
model keeps everything it wanted except the one number we constrained. A fixed
angle is just `α_min = α_max`, so "pin to θ" and "limit to [lo, hi]" are the same
projector. When the outgoing bone is (anti)parallel to `u` the bend direction is
undefined (`n ≈ 0`) and `q` is left as-is.

Two refinements make the hold feel natural rather than a teleport:

- **strength** `s ∈ [0,1]`: apply only the fraction `s·(β − α)` of the
  correction each step. `s = 1` is a hard clamp; `s < 1` *biases* the joint
  toward the target and lets the prompt fight back. Because the projection runs
  every one of the ~50 ODE steps, even a small `s` accumulates into a firm hold —
  but a soft one.
- **ease_frames**: ramp `s` from 0 up to its value over a few frames at the edges
  of a partial frame window (and back down), so a windowed constraint does not
  snap on and off and spike the jerk.

### 5.4 Pinning a whole joint by inpainting (the geometric freebie)

Because the factors of `M_RMG` are **independent**, fixing an entire joint to a
target quaternion is *inpainting*: overwrite that joint's 4 coordinates with the
target after every step and let the network condition on the pinned value going
forward. The target is a unit quaternion, so the state stays on the manifold by
construction — no projection needed, no renormalization, no off-manifold drift.
This is the cleanest possible demonstration of why the product-manifold
representation pays off: a constraint that would be a painful equality
constraint in a flat model is a one-line tensor assignment here. (The current UI
drives the bend projector instead, which is the anatomically meaningful knob; the
inpainting hook remains in the sampler.)

### 5.5 The Euclidean room — soft guidance through FK

Room containment, obstacle avoidance, contacts, and foot anti-skate are all the
*second* kind of constraint. They share one machinery (`flow/scene.py`).

**Signed distance fields.** Each primitive has an SDF (negative inside, zero on
the surface, positive outside): an (optionally yaw-rotated) box, a sphere, a
Y-axis capped cylinder. The room itself is the *inside* of a box, so leaving it
is penalized by the box SDF with the sign flipped.

**The penalty.** For a set of placed world-space joints `p` and standoff margin
`m` (padding — brake this far *before* contact):

```
E_room(p)     = Σ_joints ReLU( sdf_room(p) + m )²
E_obstacle(p) = Σ_joints ReLU( m − sdf_obj(p) )²      (per obstacle)
```

aggregated as **sum over joints, mean over frames/batch**. This aggregation is a
deliberate choice: averaging over all 22 joints would let ~20 clear joints dilute
two penetrating feet to near-zero gradient (the reason high guidance weights once
felt inert). Summing keeps the penalty proportional to total penetration so the
gradient actually pushes.

**Guidance.** The penalty is evaluated not on the noisy state but on the
**clean-sample estimate** — the model's current guess of where it will end up:

```
x̂₁ = Exp_{x_t}( (1−t)·v_t ).
```

We backprop the energy through `decode → FK → place-in-room → SDF` to get a
gradient `g = ∂E/∂x`, tangent-project it, and subtract it from the velocity. The
subtlety is the **scaling** (`flow/sampler.py`):

```
v_t ← v_t − w · ‖v_t‖ · ĝ,        ĝ = g/‖g‖.
```

The guidance velocity is a *fraction* `w` of the model's own velocity magnitude.
This is scale-invariant: a fixed-length avoidance step would be swamped by a fast
"walk forward" field, while a relative step has the same authority regardless of
how fast the model is moving, and there is no divergence cliff. `w ≈ 1` means "as
strong as the motion itself."

### 5.6 Exact spawn placement, and the "do we ignore collisions early?" question

Spawn placement (put the clip at a chosen `(x, z, facing)` in the room) **is**
closed-form: a rigid `SE(2)` transform — a yaw about the up axis plus an `xz`
translation of the root. `place_motion` applies it on the representation
(translation + root quaternion); `place_joints` applies the algebraically
identical transform on FK output. The facing yaw is computed so that spawn
rotation is **absolute**: it aligns the clip's net walk direction to the
requested heading regardless of the model's canonical facing, falling back to the
raw heading for in-place motions (where net displacement is ~0 and heading is
undefined).

A natural worry: if placement is applied only to the *final* sample, are
collisions ignored during the front-to-end sampling, since the "real" position
only exists at the end? **No** — and this is the neat part. The avoidance energy
applies the *same* `place_joints` transform to `x̂₁` **inside every ODE step**,
so the gradient is always computed in the correct room frame. The final
`place_motion` is just the exact equivalent applied once for display. The two are
verified identical. (The only caveat: this is active when `guidance_weight > 0`;
at zero, "placement only" genuinely does no collision handling, by design.)

One more numerical point: the placement yaw is **detached** from the gradient.
Backprop through `atan2` of a tiny, noisy early-step trajectory otherwise
produces enormous gradients and diverges the sampler. Orientation is fixed by the
spawn; only avoidance should drive the gradient.

### 5.7 Contacts — attraction, not avoidance

Avoidance pushes joints *out* of regions. **Contact** pulls a joint *onto* a
target — which is what makes "sit on the box," "step on the stair," "hand on the
wall" read as real rather than as hovering-near. A contact specifies a joint, a
target, a tolerance `tol`, a weight, and a frame window; the energy is

```
E_contact = weight · mean_window ReLU( ‖p_joint − target‖ − tol )²,
```

zero once the joint is within `tol` of the target. The interesting bit is the
**differentiable target geometry**, computed per frame so the gradient can drag
the joint there:

- **obstacle top** (sit / step on): the target is the point on the object's *top
  face* nearest the joint. For a box this is the joint's `xz` **clamped to the
  footprint** (in the box's yaw-local frame, then rotated back) at top height;
  for a cylinder, the `xz` clamped to the disk; for a sphere, the apex.
- **floor**: target `y = 0`, `xz` free (it tracks the joint horizontally) — plant
  a foot/hand on the ground.
- **obstacle surface** (hand on a wall): step off the SDF toward the surface
  (`sdf ≈ 0`).
- **point**: a fixed `xyz`.

These compose additively with room/obstacle penalties into one `energy_fn`, so a
single guidance gradient simultaneously keeps the body in the room, out of
obstacles, and *on* the surfaces it should touch.

### 5.8 Foot anti-skate — contact as a velocity penalty

Foot-skate (planted feet sliding) is the classic giveaway of synthetic motion.
Because the energy sees all `T` frames at once, we can penalize it directly. For
the foot joints, define a soft **plant indicator** from foot height (a sigmoid
that is ~1 when a foot is near the floor) and penalize horizontal foot velocity
weighted by it:

```
plant_f,t = σ( (band − height_f,t) / scale )
E_skate   = weight · Σ_feet  mean_t  plant · ‖horizontal velocity‖²
```

with velocities from frame differences `(p_{t+1} − p_t)·fps`. A foot is free to
move while lifted (plant ≈ 0) and penalized for sliding while grounded
(plant ≈ 1). It is differentiable through FK like every other energy term, so it
just adds to the guidance gradient.

---

## 6. Evaluation (where the geometry meets the benchmark)

To score against the standard Guo et al. evaluator, a generated `(τ, q)` clip is
converted to the **263-D HumanML3D feature** (`representation/registry.py` →
`to_h3d_features`). For exact, bit-comparable agreement with the evaluator's
training distribution, the conversion is routed through upstream's own
`process_file` rather than a hand reimplementation — upstream re-runs IK with
`smooth_forward=True` *inside* the feature extractor, and skipping that diverges
the cont6D channel by ~100%. The pre-shape variant recovers positions by
rescaling the unit-Frobenius shape back to the canonical body's Frobenius norm
("recovered by P"); the rotation variant recovers via FK ("recovered by R").

---

## 7. What is mathematically interesting, in one paragraph

The elegance is in the *factoring of concerns*. The architecture is a generic
DiT that knows nothing about rotations. All the geometry lives in three small,
composable places: the **manifold interface** (exp/log/geodesic per factor), the
**flow-matching target** (a closed-form geodesic velocity that doubles as the
regression label), and the **sampler's two hooks**. Because rotations live on
their own spheres, the prior is a wrapped Gaussian about the rest pose, the loss
is an honest MSE in the induced metric, and — the payoff — constraints split
cleanly into *exact projections* on single factors (bend angles, joint pins) and
*soft gradients* through forward kinematics (room, obstacles, contacts, skate).
The same `Exp`/`Log` that define training define sampling, and the same product
structure that defines the model defines what can be constrained for free. The
curved geometry is not an obstacle the model fights — it is the thing that makes
everything downstream simple.
