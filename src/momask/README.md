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
