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

The Guo evaluator + metrics and training utilities are shared, in `src/shared/`
(`shared.eval`, `shared.utils`) — **import them, don't copy.** The HumanML3D
loader (`rmg.data`) bakes in rmg's representation, so it's not generic yet: build
your own dataset/encoding, reusing `shared` where you can. New shared metric or
utility? Add it to `src/shared/` in one PR — don't fork.

Model-specific deps go in a `[project.optional-dependencies]` group named after
your model; install with `pip install -e '.[<model>]'`.

## 4. Cluster jobs — `slurm/<model>/`

Copy `slurm/rmg/{train,eval,viz}.sbatch` into `slurm/<model>/` and swap the
`python -m rmg.scripts.*` calls for yours. Keep the same partition / QoS / image
conventions; everything stays env + `OVERRIDES` configurable. Shared jobs
(`build_image.sbatch`, `prep_data.sbatch`, `ensure_eval_assets.sh`) stay at the
`slurm/` top level — reuse them, don't duplicate.

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
