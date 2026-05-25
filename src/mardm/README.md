# MARDM reproduction

Placeholder package — no code yet. The data pipeline (HumanML3D packed
dataset, Guo et al. evaluator, 263-D feature conversion) lives in `rmg/` and
is shared across all reproductions. See the top-level `README.md` for **data
status** (the packed dataset is still being debugged — do not train against
it until `sanity_eval` passes).

## Suggested layout

```
src/mardm/
    models/        backbone(s)
    training/      training loop
    tasks/         generation.py — text → motion sampling
configs/mardm/     Hydra configs
scripts/
    train_mardm.py
    evaluate_mardm.py
tests/test_mardm*.py
```

## What to reuse from `rmg`

- `rmg.data.HumanML3DDataset` + `rmg.data.collate` — already returns
  263-D-ready clips on the same manifold layout.
- `rmg.eval.RealGuoEvaluator` + `rmg.eval.{fid, r_precision, ...}` — single
  source of truth for metrics across all reproductions.
- `rmg.models.text_encoder.Qwen3EmbeddingEncoder` — same text encoder so
  cross-method comparisons are apples-to-apples.
- `rmg.utils.{EMA, Logger, save_checkpoint, set_seed, ...}` — shared
  training-loop primitives.

## Don't reuse

- `rmg.flow.*`, `rmg.manifolds.*`, `rmg.representation.tplusr` — RMG-specific.

## First steps

1. Wait for the green light on `sanity_eval` (top-level README "Data status").
2. Decide on input representation (263-D feature, or one of the T+R / T+P /
   T+R+P encodings via `rmg.representation.build_representation`).
3. Stand up a smoke training run (200 steps, tiny model) following
   `slurm/sanity_train.sbatch` as a template.
