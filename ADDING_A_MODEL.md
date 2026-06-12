# Adding a model to this repo

This repo hosts several motion-generation reproductions side by side (`rmg`
today; `momask`, `mardm` next), sharing one HumanML3D data pipeline, one Guo
evaluator, one container, and one web + cluster control plane. Here's how to
slot your model in without forking shared code.

## 1. Branch

Work on `<model>/<short-desc>` (e.g. `momask/vq-stage1`). PR into `main`; don't
push to `main` directly.

## 2. Your package — `src/<model>/`

Self-contained, mirroring rmg's shape:

```
src/<model>/
  configs/        Hydra configs (data/ model/ representation/ train/ + train.yaml)
  scripts/        entry points: train.py, evaluate.py, visualize.py (hydra-decorated)
  models/ training/ tasks/ ...
```

- Run entrypoints as modules: `python -m <model>.scripts.train`.
- Keep Hydra `config_path="../configs"` (scripts and configs live together under
  your package, so it resolves to `src/<model>/configs`).
- `pyproject.toml`'s `packages.find` already includes `<model>*` — it's picked up
  automatically once `src/<model>/__init__.py` exists.

## 3. Reuse shared code — don't fork it

`src/shared/` holds everything model-agnostic — **import it, don't copy:**
- `shared.data` — the HumanML3D pack pipeline (`prepare_humanml3d`) and dataset
  machinery (`mirror_motion`, `select_clip_ids`, `read_clip`, `random_crop`,
  `pad_batch`).
- `shared.geometry` — SMPL skeleton + `forward_kinematics` + the 263-D HumanML3D
  feature conversion (`tplusr_to_h3d_features_upstream`). Compose `shared.data`
  with your own *encode*: `{translation, quats}` → `forward_kinematics` → 263-D
  (rmg instead encodes to its manifold).
- `shared.eval` — Guo evaluator + FID/R@k/Diversity/MM-Dist.
- `shared.utils` — EMA, checkpointing, logging, seeding, scheduler.

New shared metric / dataloader util? Add it to `src/shared/` in one PR — don't fork.

Model-specific deps go in a `[project.optional-dependencies]` group named after
your model; install with `pip install -e '.[<model>]'`.

## 4. Cluster jobs — `slurm/<model>/`

Copy `slurm/rmg/{train,eval,viz}.sbatch` into `slurm/<model>/` and swap the
`python -m rmg.scripts.*` calls for yours. The shared jobs at the `slurm/` top
level — `prep_data.sbatch` (builds the shared packed dataset), `build_image.sbatch`,
`ensure_eval_assets.sh` — are reused as-is; don't duplicate them.

## 5. Runs — `runs/<model>/{train,eval,viz}/`

Set `RUNS_ROOT=${REPO}/runs/<model>/<kind>` in your sbatch (mirror rmg's
defaults). The backend lists and serves runs by model automatically; set
`RMG_CLUSTER_MODEL=<model>` so the control plane targets yours.

## 6. Tests — `tests/<model>/`

Put your tests in `tests/<model>/`. Use `RandomGuoEvaluator` (in `rmg.eval`) for
CI-friendly tests that don't need the upstream Guo checkpoint. `pytest` discovers
and runs everything.

## 7. Web app (optional)

`app/backend` drives SLURM generically; `app/frontend` distinguishes runs by the
`model` field on the run listing. Wire your model into the UI selector when ready.
