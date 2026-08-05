# MARDM reproduction

Reproduction of *Rethinking Diffusion for Text-Driven Human Motion Generation:
Redundant Representations, Evaluation, and Masked Autoregression* (Meng et al.,
CVPR 2025) — a.k.a. **MARDM**. Upstream: https://github.com/neu-vi/MARDM.

This is a **scaled-down** reproduction (`mardm_mini`) built to recreate the
HumanML3D-format row of the RMG paper's Table 2/4 alongside RMG and MoMask
under matched compute. It is not a 1:1 port — see *Key design decisions* below.

## Status

Trained and evaluated. The paper-M configuration, retrained on canonical
features and scored with the standard 263-D Guo evaluator on the full HumanML3D
test split, reaches **FID 0.097 / R@1 0.499** at CFG *w* = 4.0 — at or slightly
better than the published 0.114 / 0.500. Tables and the runs behind them are in
`reports/tables/` (`eval_genm_canonical.tex` is the current one).

A spatial-control layer sits on top of the base model — see `control/` and
`reports/maskcontrol-differences.md`, which records what was translated from
MaskControl exactly, where this implementation departs, and the measured cost of
control at each phase.

The smaller `mardm_mini` scale (AE ≈ 18.8M, generation branch ≈ 14.7M) is still
in the configs and remains useful for cheap end-to-end checks.

## Method (two stages)

1. **Motion AutoEncoder (`models/autoencoder.py`)** — a 1D-ResNet encoder/
   decoder that compresses the *essential* 67-D feature sequence ×4 in time
   into 512-D latent tokens, trained with an L1 reconstruction loss.
2. **Masked-autoregressive generation branch (`models/mardm.py` +
   `models/diffmlps.py`)** — an AdaLN transformer over the frozen AE's latent
   tokens produces a per-token condition `z`; a per-token **SiT (velocity)
   diffusion MLP** denoises masked latents from `z`, with a cosine masking
   schedule and classifier-free guidance. Sampling is iterative masked-AR
   decoding (`tasks/generation.py`).

## Key design decisions

- **Essential representation = first 67 dims of the 263-D HumanML3D feature**
  (root angular/linear velocity + height + 21 local joint positions). We reuse
  `rmg`'s 263-D machinery and slice `[:, :67]` rather than re-deriving it
  (`representation/essential.py`). The redundant 196 dims (6D rotations, joint
  velocities, foot contacts) are dropped, per the paper.
- **Eval bridge back to 263-D via inverse kinematics.** The Guo evaluator needs
  full 263-D, so generated essential features are mapped back:
  de-normalize → `recover_joints_from_ric` → upstream HumanML3D IK → 263-D
  (`representation.essential_to_h3d`). The IK is the *same* one
  `scripts/prepare_humanml3d.py` uses, so it stays consistent with the packed
  data. (This is the "convert to joints and back" path of the paper's App. D.3
  and loses one frame — expected.)
- **AE latent dim kept at 512** (paper value). The cost delta vs. a smaller
  latent is negligible (~2–3% of step time), so we keep the paper's
  reconstruction headroom. The generation branch (transformer + diffusion MLP)
  is the scaling axis — all dims are Hydra config fields, so growing to
  paper-M/L/XL later is a config edit + retrain (reuse the same AE if its latent
  dim is unchanged).
- **SiT velocity, not DDPM.** We keep only the SiT (linear-path, velocity-
  prediction, ODE-sampling) head — the headline variant — and drop the DDPM
  variant + its gaussian-diffusion code. This adds a `torchdiffeq` dependency
  (used by the ODE sampler; already in `pyproject.toml` /
  `containers/requirements.txt`).
- **Text encoder swapped CLIP → `rmg`'s Qwen3-Embedding** (or `random` for
  smoke). MARDM uses CLIP only as a generic per-caption feature extractor, so
  the swap is architecturally neutral and — importantly — holds the text
  encoder constant across RMG/MARDM/MoMask, removing a confound from the
  comparison. `timm`'s MLP is likewise replaced by a plain GELU MLP.
- **z-normalization, not manifold normalization.** MARDM standardizes the
  essential dims with a train-split mean/std (`mardm.scripts.compute_stats`),
  unlike RMG's manifold structure.
- **Unit-variance AE latents for the SiT head — intentional deviation from
  upstream.** Upstream MARDM feeds the *raw* AE latents straight into the SiT
  diffusion head (no scaling anywhere in `AE.py` / `MARDM.py` / `DiffMLPs.py` /
  `train_MARDM.py`). Our AE's latents come out at std ≈ 0.13, and against the
  SiT N(0,1) prior that low SNR makes the trivial "predict the noise / collapse
  to the latent mean" solution a strong attractor — the velocity loss parks at
  `var(latent) ≈ 0.13² ≈ 0.02` and sampling returns noise (motions just jitter
  in place). Upstream escapes this with full-dataset, long-schedule training; at
  our scaled-down ≤1-day budget it does not. So we add a **per-channel
  `latent_scale` buffer to the AE** (`encode` ×scale, `decode` ÷scale;
  `forward`/reconstruction is bypassed and therefore unchanged), computed
  post-training as `1/std` of the raw encoder output and stored in the AE
  checkpoint (and EMA shadow). This makes the head see ~unit-variance latents —
  standard SiT/LDM practice — and lets the generator learn the signal within the
  compute budget. A deliberate, documented departure from a 1:1 port;
  `latent_scale=1` (the default for pre-existing checkpoints) recovers upstream
  behavior.
- **`tasks/` layering.** Sampling/orchestration lives in `tasks/generation.py`;
  the scripts are thin Hydra wrappers (models / training / tasks split).
- **Dropped from upstream** (irrelevant for this task): the DDPM head, the
  length estimator, zero-shot `edit()`, the action-conditioned mode, and the
  upstream data/eval utilities (we use `shared`'s).

## Layout

```
src/mardm/
    generation.py  text → sampled motion → 263-D
    masking.py     cosine schedule + BERT-style sub-masking
    models/        autoencoder.py · mardm.py · diffmlps.py
    transport/     vendored SiT (linear-path velocity flow matching)
    representation/essential.py — 67-D encode, stats, essential→263 bridge
    data/          dataset.py — EssentialDataset (wraps shared's loader)
    control/       guidance.py · losses.py · root_edit.py — inference-time joint control
    configs/       ae.yaml · gen.yaml · gen_m.yaml · control_sweep*.yaml
    scripts/       compute_stats.py · train_ae.py · train.py · evaluate.py ·
                   evaluate_control.py · generate_control.py ·
                   visualize.py · verify_features.py · eval_table.py
    reports/       results/ · tables/ · gifs/
slurm/mardm/       train_mardm_ae.sbatch · train_mardm.sbatch ·
                   evaluate_mardm.sbatch · overfit_mardm.sbatch
tests/mardm/       test_mardm_{representation,data,models,control}.py
```

## How to run

### Local smoke (CPU, synthetic data — validates the pipeline, not quality)

```bash
# 1. Tiny synthetic packed dataset (rmg helper, still at the repo root)
python scripts/build_synthetic_dataset.py --output-dir /tmp/synth \
    --num-train 16 --num-val 4 --num-test 4 --seq-len 80
# 2. Essential mean/std
python -m mardm.scripts.compute_stats --data-root /tmp/synth --out /tmp/synth/stats.pt
# 3. AE (stage 1)
python -m mardm.scripts.train_ae data.root=/tmp/synth stats_path=/tmp/synth/stats.pt \
    ae.width=32 ae.output_emb_width=16 ae.depth=2 \
    train.max_steps=20 train.micro_batch_size=4 train.grad_accum=1 train.precision=fp32 \
    data.num_workers=0 logging.use_wandb=false logging.use_tensorboard=false \
    run_name=ae-smoke output_dir=/tmp/mardm_ae_smoke
# 4. Generation branch (stage 2) — random text encoder, frozen AE from step 3
python -m mardm.scripts.train data.root=/tmp/synth stats_path=/tmp/synth/stats.pt \
    ae_checkpoint=/tmp/mardm_ae_smoke/checkpoints/latest.pt \
    ae.width=32 ae.output_emb_width=16 ae.depth=2 \
    text_encoder.type=random text_encoder.text_dim=64 \
    model.latent_dim=64 model.num_heads=4 model.ff_size=128 \
    model.diffmlps_width=64 model.diffmlps_depth=2 model.diffmlps_batch_mul=2 \
    train.max_steps=20 train.micro_batch_size=4 train.grad_accum=1 train.precision=fp32 \
    data.num_workers=0 logging.use_wandb=false logging.use_tensorboard=false \
    run_name=gen-smoke output_dir=/tmp/mardm_gen_smoke
```

### Cluster (enroot + slurm; full `mardm_mini`)

```bash
sbatch slurm/mardm/train_mardm_ae.sbatch            # stage 1 (also computes stats)

AE_CKPT=$HOME/rmg-runs/mardm-ae-XXXX/checkpoints/latest.pt \
    sbatch slurm/mardm/train_mardm.sbatch           # stage 2

AE_CKPT=$HOME/rmg-runs/mardm-ae-XXXX/checkpoints/latest.pt \
GEN_CKPT=$HOME/rmg-runs/mardm-gen-YYYY/checkpoints/latest.pt \
    sbatch slurm/mardm/evaluate_mardm.sbatch        # HumanML3D-format metrics
```

Env vars honored by the sbatch (see top-level `README.md`): `RMG_DATA_ROOT`,
`RMG_RUNS_DIR`, `HF_HOME`, `IMAGE`, `WANDB_MODE`, and `MARDM_STATS` (writable
path for the shared mean/std; defaults under `RMG_RUNS_DIR`).

### Overfit smoke (pipeline sanity check on real data)

Before committing the 1-day budget, verify the pipeline can drive losses down on
a tiny deterministic subset. If it *can't* memorize 0.5% of the data, something's
wrong (data, loss, optimization) — fix it before a full run.

```bash
sbatch slurm/mardm/overfit_mardm.sbatch
# Override the subset / step counts:
SUBSET_FRAC=0.01 AE_STEPS=5000 GEN_STEPS=10000 sbatch slurm/mardm/overfit_mardm.sbatch
```

The sbatch chains AE → gen on the subset with overfit-friendly hyperparameters
(`model.cond_drop_prob=0`, `model.dropout=0`).

**What to look for**:
- `ae/l1` drops to **~0** on the subset — strongest signal; the AE memorizes
  ~60 clips trivially. If this doesn't happen, fix the AE before touching gen.
- `gen/loss` drops sharply and plateaus low (won't hit 0 — the SiT velocity
  loss has irreducible variance across `t` and the noise/data pair).
- Optionally, sample the training captions from the gen checkpoint and check
  the generated motions reproduce the training motions.

Equivalent for a single stage via CLI:

```bash
# AE only, 0.5% subset:
python -m mardm.scripts.train_ae subset_frac=0.005 ...
# Gen only, same subset (needs an AE checkpoint):
python -m mardm.scripts.train subset_frac=0.005 \
    model.cond_drop_prob=0 model.dropout=0 ae_checkpoint=...
```

### Setting `max_steps` (throughput probe)

`train.max_steps` in the `mardm_mini` configs are placeholders. Run a short probe
(`train.max_steps=500`), read `steps_per_s` from the log, then set
`max_steps ≈ steps_per_s × budget_seconds`. Budget the AE (cheap, a few hours)
and the generation branch so the **sum ≤ ~23h** — the 1-day-per-model rule.

## What's reused from `shared` (not forked)

- `shared.data.HumanML3DDataset` + `collate` — wrapped by `EssentialDataset`.
- `shared.geometry.{tplusr_to_h3d_features_with_quats, recover_joints_from_ric}`
  — the 263-D encode/decode primitives behind the essential representation.
- `shared.eval.{RealGuoEvaluator, fid, r_precision, mm_distance, diversity,
  multimodality}` — the metrics, shared across all reproductions.
- `shared.text.{Qwen3EmbeddingEncoder, RandomTextEncoder}` — text conditioning.
- `shared.utils.{EMA, Logger, build_scheduler, save_checkpoint, find_latest_checkpoint,
  set_seed, ...}` — training-loop primitives.

## Caveats

- **IK eval bridge needs `external/HumanML3D`**: the `essential→263` conversion
  imports the upstream skeleton for IK, so initialize the submodule (the eval
  sbatch assumes it). The bridge test is guarded without it.
- **The preload cache duplicates the shared loader's preprocessing by hand.**
  `EssentialDataset._build_preload_cache` must mirror
  `HumanML3DDataset.__getitem__` exactly, or `preload=True` and `preload=False`
  yield different features. Cluster runs need `preload=True`.
- **Off-by-one in eval**: generated motion is one frame shorter than the
  ground truth after the joints→263 round-trip; the evaluator handles per-sample
  lengths, and this matches the paper's reported conversion.
