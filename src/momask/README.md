# momask

Reproduction of *MoMask: Generative Masked Modeling of 3D Human Motions*
(Guo et al., CVPR 2024).

## Status

Implemented end to end — residual VQ-VAE tokenizer plus the masked and residual
token transformers — and the 263-D data path is verified by `tests/momask/`. There
is **no reproduction-quality evaluation yet**: the RVQ has no EMA codebook update
and no dead-code revival, so codebook collapse is the failure mode to expect and
the one the diagnostic runs kept hitting. The loader, Guo evaluator and 263-D
feature conversion all come from `shared`.

Load training data straight from `shared` — MoMask imports no other model package:

```python
from shared.data import H3D263Dataset

ds = H3D263Dataset(root, split="train")
```

This returns `sample.x1` as `(T-1, 263)` HumanML3D features, ready for
`momask.models.MotionRVQVAE`. To train on the *official* feature distribution
the Guo evaluator was fitted on, use
`shared.data.CanonicalHumanML3DDataset(root, canonical_dir=...)` instead, which
reads `new_joint_vecs/*.npy` directly.

## Layout

```text
src/momask/
    models/        MotionRVQVAE, MaskedMotionTransformer, ResidualTransformer,
                   CodebookResidualTransformer
    training/      Stage 1/2/3 single-step helpers
    tasks/         generation.py - text embedding -> 263-D motion sampling
    scripts/       train_momask · evaluate_momask{,_constraints} ·
                   visualize_momask_{checkpoint,constraints} ·
                   resample_momask_checkpoint · summarize_momask_sweep
configs/momask/    TODO: Hydra configs
tests/test_momask.py
```

Entry points run as `python -m momask.scripts.<name>`; the matching sbatch
wrappers are `slurm/*_momask*.sbatch`.

## What to reuse

- `shared.data.H3D263Dataset` / `shared.data.CanonicalHumanML3DDataset` + `shared.data.collate`
- `shared.eval.RealGuoEvaluator` + `shared.eval.{fid,r_precision,...}`
- `shared.text.{CLIPTextEncoder, Qwen3EmbeddingEncoder, RandomTextEncoder}`
- `shared.geometry.{H3D_FEATURE_DIM, PARENTS, recover_joints_from_ric, ...}`
- `shared.utils.{EMA, Logger, save_checkpoint, load_checkpoint, set_seed, ...}`

## Independent transformer training

The paper-style pipeline trains the M-Transformer and R-Transformer with
independent objectives. Both consume canonical motions cropped to at most 196
frames and tokens from the same frozen RVQ checkpoint. The R-Transformer uses
ground-truth lower RVQ levels during training, so the two jobs can run in
parallel.

Paper-style runs read the original HumanML3D `train.txt`, `val.txt`, and
`test.txt` files, including the `M*` mirrored motions. They do not use the
smaller packed `splits.json` population. Before launching a full run, submit the
data audit and confirm that the official split has roughly twice as many IDs as
the packed split, includes mirrors, and reports no missing motion/text assets:

```bash
sbatch slurm/momask/audit_momask_transformer_data.sbatch
```

The report is written to
`runs/momask_diagnostics/paper_transformer_data_audit.json`.

The stage jobs use the 12 GB partition with its compute-capability filter; the
current PyTorch image cannot run on the older Titan X nodes in that partition.
First submit short stage-specific probes and inspect the reported
CUDA peak memory. These are compute jobs; the launcher itself performs no
training on the head node:

```bash
bash slurm/momask/submit_momask_transformer_probes.sh
```

Once both probes show acceptable cluster utilization, submit the two full jobs:

```bash
bash slurm/momask/submit_momask_transformers_paperstyle.sh
```

The default assembled, evaluation-ready checkpoint is written to
`runs/momask-canonical-tokens-officialsplits196-clip500e/momask_smoke_latest.pt`.
Each component is selected using a fixed validation cache and repeatable
corruptions: masked-token CE for the M-Transformer and the same sampled-level
residual-token CE used to train the R-Transformer. The assembly job consumes
each run's `checkpoints/tokens_best_val.pt`, not its final
training checkpoint. FID still needs to be measured on the assembled model.
The R job depends on the M job and assembles both best checkpoints after its own
training completes, keeping the workflow within the two-job submission quota.
The launcher only submits SLURM jobs; training and checkpoint assembly do not
run on the head node. To resume a cancelled component independently, pass its
stage checkpoint as `TOKEN_CKPT` when submitting the corresponding masked or
residual wrapper.

The official implementation selects transformer checkpoints using generated
motion FID, while the training jobs above use validation token CE so they can
remain lightweight. After both 500-epoch stages finish, run the canonical
196-frame FID selector:

```bash
sbatch slurm/momask/select_momask_transformer_checkpoints.sbatch
```

This is a compute job. It follows the official components' independent FID
selection protocols over the complete validation split. M-Transformer candidates
are decoded with base tokens only, using 18 generation iterations and guidance
4. R-Transformer candidates receive ground-truth VQ base tokens and predict only
the residual layers, using guidance 2. Both use maximum length 196, temperature
1, top-k 0.9, sampling, and the official no-remasking behavior. Periodic,
CE-best, latest-periodic, and true final stage checkpoints are included.

Candidate assembly reuses one temporary file. Completed evaluations are reused
only when both checkpoint identities and the complete evaluation protocol match.
After selecting both components, the job assembles `momask_best_val_fid.pt` and
reports its normal full-generation validation metrics in
`selected_full_validation.json`. Results are written under
`runs/momask-canonical-tokens-officialsplits196-valfidselection-full/`. The test
split remains untouched. Set `CHECKPOINT_STRIDE=2` or higher for a quicker coarse
sweep, then use stride 1 for final selection.

## Inference-time joint constraints

`momask.constraints` supports sparse joint-position targets and exact/ranged
bend-angle targets without retraining. The constraint path keeps token sampling
unchanged, converts the generated RVQ tokens to their summed continuous latent,
and optimizes that latent through the frozen decoder and differentiable
HumanML3D joint recovery.

```python
from momask import JointPositionConstraint, LatentRefinementConfig
from momask.tasks import generate_h3d263_constrained

result = generate_h3d263_constrained(
    vqvae=vqvae,
    masked_transformer=masked,
    residual_transformer=residual,
    cond=text_embedding,
    seq_len=token_length,
    target_len=frame_length,
    mean=checkpoint["normalizer"]["mean"],
    std=checkpoint["normalizer"]["std"],
    position_constraint=JointPositionConstraint(targets, mask),
    refinement=LatentRefinementConfig(steps=50),
)

print(result.metrics)  # initial/final error and latent drift
```

Targets use clip-canonical HumanML3D coordinates, not an absolute scene frame.
The refined latent is close to but not guaranteed to remain on the exact RVQ
codebook manifold, so constraint error and motion-quality metrics must both be
reported. Integer token ids themselves are not differentiable. Set
`LatentRefinementConfig(root_weight=0.0)` when the constraint is intentionally
supposed to change the root trajectory.

### Room geometry constraints

`momask.scene_constraints` provides inference-time room containment and
nonpenetration for box, sphere, and upright-cylinder obstacles. The original
generated clip is rigidly aligned to a scene spawn marker once; that transform
is then frozen while the RVQ latent is refined. Collision loss checks every
joint and configurable interior samples along all 21 kinematic edges, rather
than only checking joint endpoints. It also checks interpolated body samples
between adjacent frames so a fast motion cannot tunnel through a thin obstacle.

A separate worst-penetration loss prevents the optimizer from improving the
average merely by crossing the obstacle in fewer frames. No avoidance route is
prescribed: the frozen model prior and collision gradients determine how the
motion changes. This is inference-time latent optimization; neither MoMask
transformer is retrained.

This stage enforces geometry only. It can push a motion out of an obstacle, but
it does not by itself teach the model to climb stairs or choose a jumping action.
Run a short compute-node probe against the validated legacy checkpoint with:

```bash
MAX_CLIPS=8 \
LATENT_VARIANTS=scene \
REFINEMENT_STEPS=100 \
sbatch slurm/momask/evaluate_momask_constraints.sbatch
```

The default scene is a 6x8x3 metre room. The actor spawns at `(0, -2)` facing
`+Z`, with a one-metre box one metre ahead at `z=-1`. `SCENE_ROOM_*`,
`SCENE_SPAWN_*`, `SCENE_OBSTACLE_*`, `SCENE_BODY_RADIUS`, `SCENE_WEIGHT`,
`SCENE_PEAK_WEIGHT`, and `SCENE_SWEPT_SAMPLES` can be overridden at submission.
The JSON compares `unconstrained` and
`scene_latent` quality and reports maximum/mean clearance violation, violating
body-point fraction, colliding-frame fraction, and swept collision metrics for
the intervals between frames. Source-specific
`scene_obstacle_*`, `scene_floor_*`, `scene_wall_*`, and `scene_ceiling_*`
metrics identify which geometry still fails. Render the same strong diagnostic
settings for three samples with:

```bash
sbatch slurm/momask/visualize_momask_scene_constraints.sbatch
```

Each GIF compares unconstrained and scene-refined motion in the configured
room, highlights penetrating body samples in red, shows the actual pelvis path
from above, and plots penetration by source over time.

The constraint evaluator compares the unconstrained sample, the existing root
trajectory baselines, wrist-position latent refinement, and knee/elbow-angle
latent refinement from the same generated tokens. Its SLURM wrapper defaults to
the validated legacy checkpoint at step 145k:

```bash
MAX_CLIPS=256 \
LATENT_VARIANTS=joint,angle \
sbatch slurm/momask/evaluate_momask_constraints.sbatch
```

The JSON reports FID, R@1/R@2/R@3, MM-Dist, and diversity for every variant. It
also reports wrist error and success within 5/10 cm, bend-angle error and success
within the configured tolerance, and root-trajectory drift. `JOINT_TARGET_SPACE`
defaults to `root-relative`, which aligns GT offsets to the generated pelvis and
heading; set it to `global` to test absolute clip-canonical joint targets. Bend
angles are unsigned magnitudes: they can constrain how much
a knee or elbow bends, but not the side of the bending plane by themselves.

### Fixed limbs relative to the body

`ParentRelativeJointConstraint` represents commands such as "keep the right arm
fixed relative to the body." At a generated reference frame, it records each
selected parent-to-child vector in a moving torso frame. For the right arm the
controlled edges are collar-to-shoulder, shoulder-to-elbow, and elbow-to-wrist;
for joint 5 the controlled edge is right-hip-to-right-knee. During refinement,
each selected edge must retain that reference vector while its parent may still
move with the body.

The torso frame uses Spine3 as its origin, the left-to-right collar direction
as its lateral axis, and the Spine3-to-Neck direction as its vertical axis.
World translation, turning, and leaning therefore do not count as limb motion.
This is less restrictive and more anatomically meaningful than independently
pinning every selected joint to the chest. The legacy
`TorsoRelativeJointConstraint` primitive remains available for experiments that
need the old absolute torso-offset behavior.

The default reference is generated frame 0, so this constraint needs no paired
ground-truth motion at inference time. It is dense, not sampled every 20
frames.

```bash
# Small metric smoke test: unconstrained versus body_fixed_latent.
MAX_CLIPS=8 \
LATENT_VARIANTS=body-fixed \
BODY_FIXED_JOINT_IDS=17,19,21 \
REFINEMENT_STEPS=100 \
PARENT_RELATIVE_WEIGHT=5 \
OUTPUT=runs/momask-canonical-tokens-clip200k/constraints/body_fixed_right_arm_smoke8.json \
sbatch slurm/momask/evaluate_momask_constraints.sbatch

# Render ground truth, generated motion, and the parent-relative result.
SAMPLES="0 500" \
sbatch slurm/momask/visualize_momask_body_fixed_constraints.sbatch

# Keep the right knee (joint 5) fixed relative to its moving right hip.
BODY_FIXED_JOINT_IDS=5 \
CONSTRAINT_TEXT="Right knee must stay fixed relative to the body." \
BODY_OUTPUT_PREFIX=body_fixed_right_knee \
SAMPLES="0 500" \
sbatch slurm/momask/visualize_momask_body_fixed_constraints.sbatch
```

The evaluator reports `parent_relative_l2_m`, success within 5/10 cm, FID,
R-Precision, MM-Dist, diversity, and root-trajectory drift. The GIF title names
the controlled joints and explains that their anatomical parent-to-child
vectors remain fixed. Ground truth is shown only for visual comparison.

Render the same wrist-position and knee/elbow-angle refinements as synchronized
GIFs. Each GIF shows the ground-truth reference, unconstrained generation,
joint-refined motion, and angle-refined motion. Green targets appear on active
anchor frames; the lower plots show wrist error in centimetres and bend-angle
error in degrees.

```bash
# One quick sample before the full render.
SAMPLES="0" SEEDS="0" \
sbatch slurm/momask/visualize_momask_joint_angle_constraints.sbatch

# Defaults to samples 0, 500, 1000, 1500, and 2000 with the validated 145k checkpoint.
sbatch slurm/momask/visualize_momask_joint_angle_constraints.sbatch
```

Outputs are written under
`runs/momask-canonical-tokens-clip200k/constraints/viz_joint_angle_step145k/`.

## Do not reuse

- Anything under `rmg.*`. MoMask operates on flat 263-D HumanML3D features and
  must not depend on rmg's manifold representation (`rmg.flow`, `rmg.manifolds`,
  `rmg.representation.tplusr`) or on rmg's package layout.

## Next Steps

1. Verify RVQ reconstruction quality:
   real motion -> encode -> decode -> 263-D feature should be close to the
   input before training the masked/residual transformers.
2. Scale the current smoke-tested components into full Hydra
   training/checkpoint scripts.
3. Add validation diagnostics for masked-token and residual-token prediction.
