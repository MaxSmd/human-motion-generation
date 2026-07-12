# RMG → 2D/3D Human Pose Estimation — Implementation Plan

*Turning the RMG motion prior into a two-stage monocular pose estimator:
2D keypoints (Stage 1) → Riemannian flow-matching lift onto `R³ × (S³)²²`
(Stage 2). Grounded in the current codebase; every hook names a real module.*

---

## 0. The core idea in one paragraph

RMG is **already** a conditional generative model on the right manifold. Text-
to-motion conditions the DiT on a `(B, text_dim=1024)` vector
(`models/conditioning.py::ConditioningFusion`). Pose estimation is the **same
model with the conditioning swapped**: replace the text vector with a 2D-keypoint
embedding, generate a single frame (`T=1`) or a short window instead of a clip,
and sample K hypotheses. Nothing in the manifold stack, the Euler–Exp sampler,
or CFG changes. Two things we get for free that the Euclidean-FM lifters
(FMPose, D3DP) do **not** have: (a) every hypothesis has correct bone lengths by
construction because we generate rotations over a fixed skeleton, and (b) the
existing constraint hooks (`flow/constraints.py`, `flow/scene.py`) become
estimation machinery — reprojection guidance and anatomical joint limits at
sampling time, training-free.

---

## 1. On YOLO26-x — does it make sense?

**As a black-box 2D detector: yes, fine.** `yolo26x-pose.pt` is the strongest
pose variant (71.6 % mAP-pose on COCO, RLE keypoint head), single `pip install
ultralytics`, one call returns `(17, 2)` keypoints + confidences per person. It
is a sensible Stage-1 choice and nothing downstream depends on which detector we
use.

**But two caveats that matter more than the YOLO version:**

1. **Convention mismatch (the real work).** YOLO26 emits **COCO-17** keypoints;
   RMG lives on the **SMPL-22** skeleton (`shared/geometry/skeleton.py`,
   `T2M_KINEMATIC_CHAINS`). COCO has no pelvis/spine/collar joints and puts
   shoulders/hips where SMPL does not. We never regress SMPL joints *from* COCO
   joints directly — instead COCO-17 is the **conditioning signal** and the model
   generates SMPL-22 rotations. So the mismatch is absorbed by the encoder, not
   by a lossy remap. The one place it bites: the **reprojection energy** (§5)
   must compare like-with-like, so we project FK'd SMPL joints and select only
   the ~12 joints with a clean COCO correspondence (shoulders, elbows, wrists,
   hips, knees, ankles + head proxy). Build this index map once.

2. **You do not need YOLO for the first two milestones.** The Tier-0 and Tier-1
   sanity runs (§6) use **projected ground-truth 2D** from HumanML3D joints under
   a synthetic camera — no detector noise, clean 2D↔3D pairing, isolates whether
   the manifold lift works at all. YOLO only enters at M4 (real-image inference),
   where detector noise and the COCO→SMPL gap are the new variables. Introducing
   it earlier just couples two unknowns.

**Verdict:** keep YOLO26-x, but treat it as a plug-in for M4. Build and validate
the whole pipeline on synthetic 2D first.

---

## 2. Where each piece hooks into the existing code

| Concern | Existing module | Change |
|---|---|---|
| Conditioning vector | `models/conditioning.py::ConditioningFusion` | add a keypoint encoder producing a `(B, cond_dim)` vector; reuse the `text_mlp`/fuse path or add a parallel `kp_mlp` |
| DiT contract | `models/dit.py::RMGDiT.forward(x, t, *, cond, drop_cond_mask, mask)` | **unchanged** — `cond` is already an opaque `(B, C)` tensor |
| Sampler | `flow/sampler.py::RiemannianEulerSampler.sample` | **unchanged** — already accepts `cond`, `energy_fn`, `guidance_weight`, CFG |
| Reprojection guidance | `flow/scene.py` (energy pattern), `flow/sampler.py` energy hook | new `energy_fn` = ‖Π(FK(x̂₁)) − y₂d‖²; sampler already back-props it through `Exp` |
| Joint-angle validity | `flow/constraints.py::build_bend_projector` | reuse as-is via `project_fn` |
| FK for reprojection | `shared/geometry/skeleton.py::forward_kinematics` | reuse; `tplusr_to_joints` for the packed layout |
| Representation | `representation/tplusr.py` (91-D), `representation/registry.py` | reuse T+R; also gives the Euclidean baseline for free (§7) |
| Trainer loss | `flow/trainer.py::compute_loss(model, batch, cond=...)` | **unchanged** — pass keypoint cond instead of text cond |
| Data | `data/humanml3d.py` | add a per-frame variant emitting `(quats+trans, 2D-keypoints)` pairs |

The conditioning being a single opaque `(B, cond_dim)` vector is the whole reason
this is cheap: the encoder is the only genuinely new network.

---

## 3. Stage 1 — 2D acquisition & normalization ("scaling")

Produces the conditioning signal `y₂d` and absorbs the camera/scale gauge.

1. **Synthetic path (M0–M3):** take HumanML3D GT joints (22×3), pick a camera
   (fixed intrinsics + sampled azimuth/elevation/distance), project to 2D. This
   is the training and sanity signal.
2. **Real path (M4):** `yolo26x-pose.pt` → COCO-17 keypoints + confidences.
3. **Normalization (shared by both):** root-center on pelvis (COCO: hip
   midpoint), scale by a robust bone-length proxy (e.g. shoulder-width or torso
   height) so the encoder sees **scale-invariant** 2D. This is the "scaling"
   half of the two-stage design and is what lets monocular scale ambiguity live
   **only** in the `R³` root-translation factor, never in the `(S³)²²` rotations.
4. Encode confidences as a 3rd channel per keypoint (0 for the synthetic path).

Encoder options, cheapest first:
- **MLP over flattened normalized keypoints** → `cond_dim`. Start here.
- **Small joint-graph GCN** (COCO skeleton edges) → pooled `cond_dim`. This is
  what FMPose/FMPose3D use; adopt if the MLP underfits.

Output dim = whatever the DiT `text_dim` is set to (default 1024); keep it equal
so `ConditioningFusion` needs no shape change.

---

## 4. Stage 2 — the manifold lift (training)

- **Target:** T+R representation, single frame `T=1` (per-frame estimation) or a
  short window `T=k` (temporal estimation — the more distinctive version, since
  RMG is natively a motion model; do per-frame first).
- **Model:** `RMGDiT` with `dit_small`/`dit_mid` config, `input_dim=91`,
  `max_seq_len` reduced. Reuse `flow/trainer.py::compute_loss` verbatim — it
  already takes `cond`, `drop_cond_mask` (for CFG), and `mask`.
- **Data pairs:** `data/humanml3d.py` variant yielding
  `(x₁ = packed T+R frame, cond = normalized 2D keypoints)`. IK-derived
  quaternions already exist in the pipeline (`humanml3d_io.py`), so pairs are one
  projection function away — **no new data infrastructure**.
- **CFG:** keep the existing drop mechanism (drop keypoint cond → null embedding)
  so we can trade fidelity vs. diversity at sample time exactly as text-to-motion
  does.
- **Compute:** FMPose reaches SOTA at **4.5 M params, 100 epochs, one A40** — a
  `dit_small`/`dit_mid` run on the existing SLURM path (`slurm/rmg/train`) is well
  within budget. Expect < 1 day on one GPU for the per-frame model.

---

## 5. Sampling & aggregation (inference)

Per image / per frame:

1. Encode `y₂d` → cond.
2. Draw **K hypotheses** (K≈200, matching FMPose) by sampling K prior draws and
   integrating with `RiemannianEulerSampler.sample`, `num_steps≈25`, CFG on.
3. **Reprojection guidance (optional, strong):** pass
   `energy_fn(x̂₁) = Σ_j ‖Π_cam(FK(x̂₁)_j) − y₂d_j‖²` over the COCO-corresponding
   joint subset, with `guidance_weight`. The sampler already forms the clean-
   sample estimate `x̂₁ = Exp_x((1−t)v)`, runs the energy, tangent-projects the
   gradient, and steers `v` — this is the exact code path used for room/obstacle
   guidance in `flow/scene.py`, repurposed to observation guidance (ScoreHMR /
   ZeDO style, but intrinsic to the manifold).
4. **Anatomical validity (optional):** `build_bend_projector` as `project_fn`
   guarantees every hypothesis respects joint-angle limits — a differentiator no
   FM/diffusion lifter has.
5. **Aggregation → point estimate:**
   - *Reprojection-min* (D3DP-style): FK each hypothesis, pick the one with
     lowest 2D reprojection error. Deterministic, strong.
   - *Karcher/geodesic mean:* per-joint quaternion Karcher mean of the K
     hypotheses (uses `manifolds/sphere.py` Log/Exp), root averaged in `R³`.
     Correct manifold mean; keeps the estimate ON the manifold.
   - Report both; keep the full K as the calibrated distribution.

---

## 6. Verification milestones (fastest first)

**M0 — Tier-0 zero-shot, ~1 day, no training.** Use the **existing trained RMG
checkpoint** (`runs/rmg/train/...`). Null the text cond, feed reprojection
`energy_fn` from projected-GT 2D of a held-out clip, sample. Success = samples
measurably pulled toward the observation (reprojection error drops vs.
unguided). This exercises the exact machinery in `rmg_constraints.tex` and
answers "does manifold guidance pull onto an observation" before writing any new
network. Deliverable: a script under `src/rmg/scripts/` + a metrics printout.

**M1 — data pairs + encoder, ~1–2 days.** Add the per-frame paired loader and the
keypoint encoder; unit-test shapes and that `compute_loss` runs end-to-end on a
tiny batch (mirror `tests/rmg` style). No accuracy claim yet.

**M2 — train the conditional per-frame model, ~2–3 days.** `dit_small`/`dit_mid`,
projected-GT 2D condition. Metrics after FK: **MPJPE / P-MPJPE**, plus the two
manifold-native wins — **bone-length consistency** (≈0 by construction) and
**joint-limit violation rate**. Compare against the Euclidean-FM baseline (§7).

**M3 — Human3.6M protocol, +1–2 weeks (optional, for citable numbers).** 17-joint
skeleton + IK to SMPL-22, standard train/test split, MPJPE under Protocol #1/#2.
This is the number reviewers want; everything before is internal validation.

**M4 — real images via YOLO26-x.** Swap projected-GT 2D for `yolo26x-pose`
output, add the COCO-17→SMPL-22 conditioning path and the joint-subset
reprojection map. New variables: detector noise + convention gap. Evaluate on
3DPW / in-the-wild clips.

---

## 7. The ablation that IS the paper

Because `representation/registry.py` includes a Euclidean `R^d` manifold, the
**same architecture, same trainer, same K-hypothesis inference** can lift in
position space (`R^(22×3)`) — i.e. **FMPose reimplemented for free**. Run three
heads at matched params:

- **(a) RMG manifold lift** on `R³ × (S³)²²` ← ours
- **(b) Euclidean FM** on joint positions ← FMPose-equivalent baseline
- **(c) direct-regression MLP** ← lower bound

Headline comparison: (a) vs (b) at equal budget on {MPJPE, P-MPJPE, bone-length
consistency, joint-limit violations, multi-hypothesis calibration}. The manifold
version should win decisively on the last three even if raw MPJPE is close —
that's the story: **estimation as constrained sampling from a Riemannian pose
prior**, producing animation-ready quaternions with no post-hoc IK.

---

## 8. Honest risks

- **Rotation-space MPJPE lags position-space** initially: FK chains accumulate
  error while the metric measures positions. Frame the contribution as validity
  + consistency + calibrated distributions + native quaternion output, not raw mm.
- **Novelty is combination-level** (RFM × HPE). The Euclidean-FM people own the
  task; the SO(3)-normalizing-flow people (HuProSO3, HuManiFlow) own the manifold
  with *discrete* flows. RFM on the articulated product manifold conditioned on
  2D appears to be the open cell — and the training-free constraint hooks are
  unique to us. The §7 ablation is what defends it.
- **Temporal version** (2D keypoint *sequence* → motion window) is where RMG is
  most differentiated — "manifold D3DP", untouched by the per-frame FM papers.
  Higher upside, do it after the per-frame model is validated.

---

## 9. Concrete first-week checklist

1. [ ] `scripts/pose_zeroshot.py` — M0: reprojection-guided sampling on the
       existing checkpoint; report guided vs unguided reprojection error.
2. [ ] `data/humanml3d.py` — per-frame paired variant `(T+R frame, proj-GT 2D)`
       + synthetic camera projector.
3. [ ] `models/conditioning.py` — `KeypointEncoder` (MLP first) → `cond_dim`.
4. [ ] `configs/` — `representation` (T=1 per-frame) + a `pose_estimation`
       train config; wire the encoder cond into the trainer call.
5. [ ] COCO-17↔SMPL-22 index map + reprojection joint-subset (needed for M0's
       energy and reused at M4).
6. [ ] Train `dit_small` per-frame; log MPJPE / P-MPJPE / bone-consistency /
       joint-limit rate.
7. [ ] Stand up the Euclidean-FM baseline (config-only via the `R^d` manifold)
       for the §7 table.
