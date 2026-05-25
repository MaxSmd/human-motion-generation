# Motion-generation reproductions on HumanML3D

Single repo, three reproductions, one shared data + eval pipeline:

| Package        | Paper                                                                 | Owner    | Status |
|----------------|------------------------------------------------------------------------|----------|--------|
| `src/rmg/`     | Riemannian Motion Generation (Miao, Huang, Li 2026)                    | Julian   | code complete, retrain pending data fixes |
| `src/momask/`  | MoMask: Generative Masked Modeling of 3D Human Motions (Guo+ 2024)     | *open*   | placeholder |
| `src/mardm/`   | MARDM                                                                  | *open*   | placeholder |

Common infrastructure (data loading, evaluator wrapper, metrics, training-loop
utilities, container, slurm) is shared via the `rmg` package and the
top-level `scripts/`, `configs/`, `slurm/`, `containers/`.

> [!IMPORTANT]
> **The packed dataset is currently broken — do not train anything yet.**
> See [Data status](#data-status) below.

---

## Repo layout

```
src/
  rmg/                  Riemannian flow matching on (R³ × S³^22) — paper main result
    data/               HumanML3D packed dataset (shared with momask/mardm)
    eval/               Guo et al. evaluator wrapper + FID/R@k/Diversity/MM-Dist
    flow/               flow matching trainer, sampler, prior, geodesic interp
    manifolds/          R^d, S^d, pre-shape, ProductManifold
    models/             DiT, conditioning, text encoders (random + Qwen3)
    representation/     T+R / T+P / T+R+P encodings + 263-D conversion
    tasks/              generation, ssl, recognition stubs
    utils/              EMA, logger, seeding, checkpointing, scheduler
  momask/               placeholder — see src/momask/README.md
  mardm/                placeholder — see src/mardm/README.md

configs/                Hydra configs (rmg-specific today; add momask/, mardm/ as you go)
  data/                 local_submodule.yaml | cluster_mounted.yaml
  model/                dit_base.yaml | dit_large.yaml
  representation/       t_plus_r.yaml | t_plus_p.yaml | t_plus_r_plus_p.yaml | (dt/dr stubs)
  train/                rmg_base.yaml | rmg_large.yaml
  train.yaml            top-level composition

scripts/                Python entry points (Hydra-driven)
  train.py              RMG training
  evaluate.py           RMG eval with CFG sweep
  sanity_eval.py        decode + eval pipeline check on REAL data (no model needed)
  prepare_humanml3d.py  AMASS → packed dataset (two stages)
  diagnose_h3d_conversion.py
                        per-block diff of our 263-D vs upstream's process_file

slurm/                  sbatch templates (12g + 24g partitions; QoS: students_normal)
  smoke.sbatch          5-min container/import check
  sanity_train.sbatch   2-min tiny-DiT end-to-end pipeline check
  sanity_eval.sbatch    decode/eval check on real data (NO training)
  prep_data.sbatch      full prep: AMASS → joints → packed zip
  train_rmg_base.sbatch RMG-base (6L/384h, 150k steps, ~3.5d)
  train_rmg_large.sbatch
                        RMG-large (24L/1024h, 600k steps)
  train_ablation.sbatch parameterized by REPRESENTATION env var
  evaluate.sbatch       CKPT=... sbatch evaluate.sbatch
  init_eval_assets.sbatch
                        one-time text-to-motion submodule + Guo checkpoint setup

containers/             enroot build instructions + Dockerfile + requirements.txt
external/               git submodules: HumanML3D, text-to-motion
tests/                  pytest suite — manifolds, representation, flow, eval, models
```

Each package is independently importable: `import rmg`, `import momask`,
`import mardm`. The package-finder in `pyproject.toml` picks all three up
automatically.

---

## Quickstart

### Local (CPU, tests + smoke runs only)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                          # manifold + representation + flow tests
python scripts/train.py model=dit_base train=rmg_base \
    train.max_steps=20 train.micro_batch_size=4 train.grad_accum=2 \
    text_encoder.type=random run_name=local-smoke
```

### Cluster (TUM head node, enroot + slurm)

Once-per-account:

```bash
# 1. Build the container (see containers/BUILD.md — 5 min on a 24g interactive job)
# 2. Init eval assets (text-to-motion submodule + Guo checkpoint)
sbatch slurm/init_eval_assets.sbatch
# 3. Smoke-test
sbatch slurm/smoke.sbatch
```

Then for a training run:

```bash
sbatch slurm/train_rmg_base.sbatch
# logs land in slurm/logs/rmg-base-<jobid>.{out,err}
# checkpoints land in $HOME/rmg-runs/<run_name>/checkpoints/
```

---

## Container build

See `containers/BUILD.md` for the full enroot build recipe. Short version:

1. Open an interactive 24g job.
2. `enroot import` upstream `pytorch/pytorch:2.9.0-cuda13.0-cudnn9-devel`.
3. `enroot create --name rmg` from it.
4. `enroot start --root --rw --mount /mnt:mnt ... rmg`, then `pip install -r containers/requirements.txt`.
5. `enroot export --output ~/rmg.sqsh rmg`.

Every sbatch starts a fresh container from `~/<USER>/rmg.sqsh`. If you change
`requirements.txt`, rebuild the image; do **not** `pip install` inside an
sbatch (it bloats every job and races with file locks).

The image is ~6 GB built. **One person should build it and put it on the
shared mount once** — see below for how to share.

---

## Cluster mounts: per-user vs shared

The cluster gives each of us a per-user `/mnt/home/<user>/` mount, and there's
a shared mount `/mnt/projects/drl4cvb/human-motion/`. Default rule: **anything that's expensive to produce and identical
across users goes on the shared mount, read-only; everything else stays
per-user.**

| Artifact                                  | Where           | Why |
|-------------------------------------------|-----------------|-----|
| AMASS raw `.npz` files                    | shared, RO      | ~40 GB, never changes after download |
| SMPL+H + DMPL body models (`*.npz`)        | shared, RO      | licensed, identical for everyone |
| `external/HumanML3D` submodule clone       | shared, RO      | static; saves 14k+ inode quota per user |
| `external/text-to-motion` + Guo checkpoint | shared, RO      | static; ~600 MB checkpoint |
| `joints_cache/` (Stage 1 output, ~30min GPU) | shared, RW    | one person runs it, everyone reads |
| **`humanml3d_packed/` (Stage 2 output, the zip)** | shared, RW | one person re-packs, everyone reads — see Data status |
| `~/rmg.sqsh` container image               | shared, RO copy | symlink your `$HOME/rmg.sqsh` to a shared copy |
| HF cache (`~/.cache/huggingface`)          | shared, RW      | Qwen3 weights, sentencepiece tokenizers; safe to share |
| `~/rmg-runs/` checkpoints + samples        | **per-user**    | write-heavy, large, per-experiment |
| `slurm/logs/`                              | **per-user** (committed empty) | per-user job output |
| wandb / tensorboard outputs                | **per-user**    | one wandb run per training job |

### Coordination rule

The shared mount has **one writer per artifact at a time**. Concretely:

- **AMASS / body models / submodules / Guo checkpoint**: write *once*, then
  `chmod -R a-w` so nobody can accidentally clobber.
- **`joints_cache/`**: only re-run Stage 1 if you've changed `stage_raw_pose`
  (we won't for a while). Coordinate over Slack first.
- **`humanml3d_packed/`**: this *will* change while we fix decode bugs.
  Convention: name the dir with a date/version suffix
  (`humanml3d_packed_v3_xflip/`) and update `configs/data/cluster_mounted.yaml`
  in a single commit when a new version goes live. Don't overwrite the
  current dir in-place — concurrent eval/training jobs will read garbage.
- **Container image**: same versioning. `rmg-2026-05-25.sqsh` not `rmg.sqsh`.

### Env vars

Set these in your `~/.bashrc` on the cluster:

```bash
export RMG_DATA_ROOT=/mnt/shared/motion/humanml3d_packed_v3_xflip
export RMG_RUNS_DIR=$HOME/rmg-runs
export HF_HOME=/mnt/shared/motion/hf-cache
export IMAGE=/mnt/shared/motion/rmg-2026-05-25.sqsh    # or ~/rmg.sqsh if you built your own
```

The sbatch scripts honor these; if a user wants to point at a private dataset
build for a specific experiment, they override `RMG_DATA_ROOT` for that job
only.

### Bootstrap for a new teammate

```bash
ssh head
# 1. Use the shared container instead of building your own
ln -s /mnt/shared/motion/rmg-2026-05-25.sqsh ~/rmg.sqsh
# 2. Clone the repo to your home
git clone <repo-url> ~/motion-reproductions && cd ~/motion-reproductions
# 3. Symlink eval assets so init_eval_assets.sbatch isn't needed
ln -s /mnt/shared/motion/external/HumanML3D       external/HumanML3D
ln -s /mnt/shared/motion/external/text-to-motion  external/text-to-motion
# 4. Point at the shared dataset
mkdir -p external/data
ln -s /mnt/shared/motion/humanml3d_packed_v3_xflip external/data/humanml3d_packed
# 5. Smoke test
sbatch slurm/smoke.sbatch
sbatch slurm/sanity_train.sbatch   # 2 min end-to-end
```

---

## <a id="data-status"></a>Data status — read before training

The packed dataset has had three known bugs. **Until `sanity_eval` reports
PASS on real data, no method should train against this data**: every metric
you'd compute would be against a broken eval baseline.

### What's been fixed

1. **FK convention** (in `src/rmg/representation/skeleton.py::forward_kinematics`).
   We used SMPL-standard "rotation OF the outgoing frame"; HumanML3D's upstream
   uses "rotation that takes the canonical bone direction to the observed one"
   (j's own quaternion participates in placing j itself). Fixed by rotating
   the offset by `gq = global_parent * local_j` instead of `global_parent`.
   *Improvement on sanity_eval: diversity_real 4.6 → 7.1.*

2. **Translation rescaling at prep** (in `scripts/prepare_humanml3d.py::stage_pack`).
   Upstream's `uniform_skeleton` rescales root translation by
   `scale_rt = tgt_leg_len / src_leg_len` so XZ velocity is in canonical-body
   scale. We weren't applying this. *Improvement on sanity_eval: none
   measured (so the body-size variance in HumanML3D may be smaller than I
   thought) but the fix is correct in principle and we keep it.*

3. **X-flip + subset pre-trim at prep** (also in `stage_pack`).
   Upstream's `raw_pose_processing.ipynb` cell 11 applies `data[..., 0] *= -1`
   to *every* non-humanact12 clip, and pre-trims a few seconds from
   `Eyes_Japan_Dataset`, `MPI_HDM05`, `TotalCapture`, `MPI_Limits`, and
   `Transitions_mocap`. Without the flip our IK assigns L/R joint rotations
   to the wrong side and R-precision collapses (captions say "left", motions
   look right). *Expected to close the remaining gap to paper.*

### What needs verification

Run `sanity_eval` against a freshly re-packed dataset. Expected real-data
numbers (HumanML3D ground-truth row, Guo et al. 2022):

| Metric           | Expected   | Last seen (pre-X-flip fix) | Verdict |
|------------------|------------|----------------------------|---------|
| `diversity_real` | ≈ 9.503    | 7.118                       | not yet |
| `R@1 / R@2 / R@3`| 0.51 / 0.70 / 0.80 | 0.11 / 0.21 / 0.27 | not yet |
| `MM-Dist`        | ≈ 2.974    | 5.86                        | not yet |
| `FID(real, real)`| ≈ 0.002    | 0.76                        | not yet |

To re-verify:

```bash
sbatch slurm/prep_data.sbatch                 # ~10 min CPU
MAX_CLIPS=512 sbatch slurm/sanity_eval.sbatch   # ~3 min
cat slurm/logs/rmg-sanity-eval-*.out | grep -A 10 verdict
```

### If sanity_eval still fails

Two open questions if the next round doesn't pass:

- **Our `forward_kinematics` may still differ from upstream's somewhere
  subtle.** Easy test: `scripts/diagnose_h3d_conversion.py` does an element-
  wise diff per feature block; if `cont6d` is still far from zero, our FK is
  not bit-comparable. *Add a regression test that pins this.*
- **`spaCy` POS tagging may not match** what the Guo evaluator was trained
  against (different model versions ⇒ different POS tags ⇒ different word/POS
  embeddings ⇒ different text features). The text encoder is a BiGRU over
  GloVe + POS one-hot; if R-precision is still bad but `diversity_real` is
  fine, this is the suspect.

### When sanity passes

- Existing `rmg-base` checkpoint is trained on L/R-confused data → **toss it,
  retrain from scratch** (3.5 days). The model learned anti-handedness.
- Anyone starting MoMask / MARDM gets a clean dataset to train against.

---

## Adding a new method

Workflow we'll all follow:

1. **Open a feature branch** named `<method>/<short-desc>`, e.g.
   `momask/vq-stage1`. Don't push to `main`.
2. **Code lives under your package**: `src/<method>/`. Keep it self-contained.
   Pull anything you need from `rmg`; don't reach into the other method's
   internals.
3. **Configs under `configs/<method>/`**, hydra-composable. Mirror the
   `configs/train.yaml` top-level pattern.
4. **Scripts under top-level `scripts/`**, named `train_<method>.py`,
   `evaluate_<method>.py`. Each one is hydra-decorated, single entry point.
5. **Sbatch under `slurm/`**, named `train_<method>_*.sbatch` and
   `evaluate_<method>.sbatch`. Copy the existing `train_rmg_base.sbatch`
   as a template — same partition + QoS + image conventions.
6. **Tests under `tests/test_<method>_*.py`**. Use `RandomGuoEvaluator` (in
   `rmg.eval`) for CI-friendly tests that don't need the upstream checkpoint.
7. **PR into `main`** with a one-paragraph description; another teammate
   reviews. Until we have a CI runner, the reviewer runs `pytest` locally
   before merging.

### Shared changes

If your method needs a new dataloader mode, a new metric, a new text encoder,
etc. — **add it to `rmg/` and import from there**, don't fork the file into
your package. Coordinate the change (one PR, all three of us reading it).

---

## Common pitfalls

- **`weights_only=False`** is required when loading our packed `.pt` blobs
  (they're a `dict[str, Tensor | list[str]]`, which torch ≥2.6 rejects under
  strict mode). Safe because we produce these files ourselves.
- **`nn.MultiheadAttention` doesn't always dispatch to flash/SDPA.**
  For speed-critical paths use `F.scaled_dot_product_attention` directly.
- **`wandb_mode=offline` by default**; cluster egress isn't always available.
  Override with `WANDB_MODE=online sbatch ...` when you've confirmed.
- **Don't `pip install` inside an sbatch.** Rebuild the container instead;
  see `containers/BUILD.md`.
- **`mirror_augment=True` is train-only.** Always pass `False` for eval splits
  — our dataset class respects this when `split != "train"` but be explicit.
- **HumanML3D L/R variable naming is swapped** (`l_hip, r_hip, sdr_r, sdr_l =
  [2, 1, 17, 16]` — `l_hip` holds index 2 which is the R_Hip). If you write
  any new code that touches these, copy upstream's variable names verbatim
  and keep them confined.

---

## Contact / process

- **Pipeline / data / shared infra changes**: ping the team channel before
  merging. We don't want to push two prep changes that both touch the packed
  dataset on the same day.
- **Method-internal changes** (model architecture, hyperparams, etc.): merge
  freely on your branch + PR.
- **Cluster outages / quota issues**: post to the channel; whoever's free
  triages.
