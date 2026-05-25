# MoMask reproduction

Reproduction of *MoMask: Generative Masked Modeling of 3D Human Motions*
(Guo et al., CVPR 2024).

## Status

Placeholder package — no code yet. The data pipeline (HumanML3D packed
dataset, Guo et al. evaluator, 263-D feature conversion) lives in `rmg/` and
is shared across all reproductions. See the top-level `README.md` for **data
status** (the packed dataset is still being debugged — do not train against
it until `sanity_eval` passes).

## Suggested layout

```
src/momask/
    models/        ResidualVQ, MaskedTransformer, ResidualTransformer
    training/      Stage 1 (VQ), Stage 2 (masked), Stage 3 (residual) loops
    tasks/         generation.py — text → motion sampling
configs/momask/    Hydra configs (mirror configs/{model,train,data})
scripts/
    train_momask_vq.py
    train_momask_gen.py
    train_momask_res.py
    evaluate_momask.py
tests/test_momask*.py
```

## What to reuse from `rmg`

- `rmg.data.HumanML3DDataset` + `rmg.data.collate` — already returns 263-D-
  ready clips on the same manifold layout. For MoMask you'll want a variant
  that returns the **263-D HumanML3D feature directly** (not the (T+R, 91-D)
  encoding); add it as a flag on `HumanML3DDataset` rather than forking.
- `rmg.eval.RealGuoEvaluator` + `rmg.eval.{fid,r_precision,...}` — the
  evaluator is the single source of truth for metrics across all three
  reproductions.
- `rmg.models.text_encoder.Qwen3EmbeddingEncoder` — same text encoder so
  cross-method comparisons are apples-to-apples.
- `rmg.utils.{EMA, Logger, save_checkpoint, load_checkpoint, set_seed, ...}` —
  shared training-loop primitives.

## Don't reuse

- `rmg.flow.*`, `rmg.manifolds.*`, `rmg.representation.tplusr` — RMG-specific.
  MoMask operates on flat 263-D features, not on Riemannian manifolds.

## First steps

1. Wait for the green light on `sanity_eval` (top-level README "Data status").
2. Add a `mode="h3d_263"` option to `HumanML3DDataset.__init__` that returns
   the 263-D feature directly (call `representation.to_h3d_features` on the
   stored T+R, cache the result to disk on first read).
3. Implement `ResidualVQ` (Stage 1). Verify reconstruction quality first:
   real motion → encode → decode → 263-D feature should be ≈ original.
