# momask

Reproduction of *MoMask: Generative Masked Modeling of 3D Human Motions*
(Guo et al., CVPR 2024).

## Status

Initial implementation is present. The packed HumanML3D loader, Guo et al.
evaluator, and 263-D HumanML3D feature conversion are shared with the rest of
the repository. See the top-level `README.md` for data setup and evaluator
status before launching full training runs.

Use the shared loader through the rmg compatibility wrapper in MoMask mode:

```python
from rmg.data import HumanML3DDataset

ds = HumanML3DDataset(root, split="train", output_mode="h3d_263")
```

This returns `sample.x1` as `(T-1, 263)` HumanML3D features, ready for
`momask.models.MotionRVQVAE`.

## Layout

```text
src/momask/
    models/        MotionRVQVAE, MaskedMotionTransformer, ResidualTransformer
    training/      Stage 1/2/3 single-step helpers
    tasks/         generation.py - text embedding -> 263-D motion sampling
configs/momask/    TODO: Hydra configs
scripts/           TODO: train/evaluate entry points
tests/test_momask.py
```

## What to reuse

- `rmg.data.HumanML3DDataset(..., output_mode="h3d_263")` + `rmg.data.collate`
- `rmg.eval.RealGuoEvaluator` + `rmg.eval.{fid,r_precision,...}`
- `rmg.models.text_encoder.Qwen3EmbeddingEncoder`
- `rmg.utils.{EMA, Logger, save_checkpoint, load_checkpoint, set_seed, ...}`

## Do not reuse

- `rmg.flow.*`, `rmg.manifolds.*`, or `rmg.representation.tplusr` inside the
  MoMask model. MoMask operates on flat 263-D HumanML3D features.

## Next Steps

1. Verify RVQ reconstruction quality:
   real motion -> encode -> decode -> 263-D feature should be close to the
   input before training the masked/residual transformers.
2. Scale the current smoke-tested components into full Hydra
   training/checkpoint scripts.
3. Add validation diagnostics for masked-token and residual-token prediction.
