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
`runs/momask-canonical-tokens-paperfaithful196-clip500e/momask_smoke_latest.pt`.
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

`TorsoRelativeJointConstraint` represents commands such as “keep the right arm
fixed relative to the torso.” It defines a moving chest frame from Spine3
(origin), the left-to-right collar direction (lateral axis), and the
Spine3-to-Neck direction (vertical axis). The forward axis is their cross
product. At a reference frame, selected joint positions are converted into this
local frame; the same local offsets are required at every valid generated
frame. World translation, turning, and leaning therefore do not count as arm
motion.

For the right arm, the constrained joints are R_Shoulder (17), R_Elbow (19),
and R_Wrist (21). The default reference is generated frame 0, so this
constraint needs no paired ground-truth motion at inference time. It is dense,
not sampled every 20 frames.

```bash
# Small metric smoke test: unconstrained versus body_fixed_latent.
MAX_CLIPS=8 \
LATENT_VARIANTS=body-fixed \
BODY_FIXED_JOINT_IDS=17,19,21 \
REFINEMENT_STEPS=100 \
TORSO_RELATIVE_WEIGHT=5 \
OUTPUT=runs/momask-canonical-tokens-clip200k/constraints/body_fixed_right_arm_smoke8.json \
sbatch slurm/momask/evaluate_momask_constraints.sbatch

# Render ground truth, generated motion, and the torso-relative result.
SAMPLES="0 500" \
sbatch slurm/momask/visualize_momask_body_fixed_constraints.sbatch

# Keep the right knee (joint 5) fixed relative to the moving torso.
BODY_FIXED_JOINT_IDS=5 \
CONSTRAINT_TEXT="Right knee must stay fixed relative to the torso." \
BODY_OUTPUT_PREFIX=body_fixed_right_knee \
SAMPLES="0 500" \
sbatch slurm/momask/visualize_momask_body_fixed_constraints.sbatch
```

The evaluator reports `torso_relative_l2_m`, success within 5/10 cm, FID,
R-Precision, MM-Dist, diversity, and root-trajectory drift. The GIF title names
the controlled shoulder, elbow, and wrist and states that they remain fixed
relative to the torso. Ground truth is shown only for visual comparison.

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
