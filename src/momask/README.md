# momask

Reproduction of *MoMask: Generative Masked Modeling of 3D Human Motions*
(Guo et al., CVPR 2024).

## Status

Initial implementation is present. The packed HumanML3D loader, Guo et al.
evaluator, and 263-D HumanML3D feature conversion are shared with the rest of
the repository. See the top-level `README.md` for data setup and evaluator
status before launching full training runs.

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
    scripts/       train_momask_smoke · evaluate_momask{,_constraints} ·
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
