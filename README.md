# Motion-generation reproductions on HumanML3D

Several motion-generation reproductions in one repo — `rmg` today, `momask` and
`mardm` next — sharing one HumanML3D data pipeline, one Guo evaluator, one
container, and one web + cluster control plane. Adding a model:
[`ADDING_A_MODEL.md`](ADDING_A_MODEL.md).

---

## Repo layout

```
src/
  rmg/                  Riemannian flow matching on (R³ × S³^22)
    configs/            Hydra configs (data / model / representation / train + train.yaml)
    scripts/            entry points: train · evaluate · visualize
    data/ eval/ flow/ manifolds/ models/ representation/ tasks/ utils/
  momask/  mardm/       other models (placeholders — see ADDING_A_MODEL.md)

app/
  backend/              FastAPI: SLURM control plane + in-process rmg inference
  frontend/             Next.js UI

scripts/                shared data tooling — prepare_humanml3d.py (AMASS → packed)
slurm/
  rmg/                  per-model jobs: train.sbatch · eval.sbatch · viz.sbatch
  build_image.sbatch    prep_data.sbatch    ensure_eval_assets.sh    (shared)
containers/             enroot image: Dockerfile + requirements.txt
external/               git submodules: HumanML3D, text-to-motion
tests/rmg/              pytest suite (per model)
runs/<model>/{train,eval,viz}/   run outputs (gitignored)
```

Each model package is independently importable (`import rmg`); `pyproject.toml`
picks them up automatically.

---

## Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
# tiny CPU smoke run
python -m rmg.scripts.train model=dit_base train=rmg_base \
    train.max_steps=20 train.micro_batch_size=4 train.grad_accum=2 \
    text_encoder.type=random run_name=local-smoke
```

The web app (cluster control plane + inference):

```bash
RMG_CLUSTER_MODE=1 uvicorn backend.app:app --app-dir app --reload --port 8000
# or full stack (backend + frontend):
docker compose up --build
```

---

## Cluster

### Build the environment / container

The training+eval environment is a single enroot image (PyTorch + our deps).
Build it once on an interactive 24g job — or just `sbatch slurm/build_image.sbatch`:

```bash
enroot import -o /tmp/base.sqsh 'docker://pytorch/pytorch:2.9.0-cuda13.0-cudnn9-devel'
enroot create --name rmg /tmp/base.sqsh
enroot start --root --rw --mount /mnt:mnt rmg   # then: pip install -r containers/requirements.txt
enroot export -o ~/rmg.sqsh rmg
```

Every sbatch starts a fresh container from `~/rmg.sqsh`. Change `containers/requirements.txt`
→ rebuild the image; never `pip install` inside an sbatch.

### Run jobs

```bash
sbatch slurm/rmg/train.sbatch                       # default RMG-base
MODEL_PRESET=dit_base PRESET=rmg_base \
  OVERRIDES='data.subset_n=16 train.max_steps=15000' \
  sbatch slurm/rmg/train.sbatch                     # everything is env + OVERRIDES
sbatch slurm/rmg/eval.sbatch                        # Guo eval assets bootstrap on first run
# logs → slurm/logs/ ; outputs → <project>/runs/rmg/{train,eval,viz}/<run>/
```

### Storage

The only **shared** cluster storage is the data mount under
`/mnt/projects/drl4cvb/data/` — `humanml3d/`, `text-to-motion/`, and the packed
`data/`. Everything else is **per-user**: your repo clone, your `~/rmg.sqsh`
image, and all run outputs under `<project>/runs/`.

```bash
export RMG_DATA_ROOT=/mnt/projects/drl4cvb/data/humanml3d_packed   # shared dataset
export IMAGE=$HOME/rmg.sqsh                                        # your image
```

---

## Common pitfalls

- **`weights_only=False`** is required when loading our packed `.pt` blobs
  (`dict[str, Tensor | list[str]]`, which torch ≥2.6 rejects under strict mode).
  Safe because we produce these files ourselves.
- **`nn.MultiheadAttention` doesn't always dispatch to flash/SDPA.** For
  speed-critical paths use `F.scaled_dot_product_attention` directly.
- **`WANDB_MODE=offline` by default**; cluster egress isn't always available.
  Override with `WANDB_MODE=online sbatch ...` once confirmed.
- **Don't `pip install` inside an sbatch** — rebuild the container instead.
- **`mirror_augment=True` is train-only.** Pass `False` for eval splits (the
  dataset respects this when `split != "train"`, but be explicit).
- **HumanML3D L/R variable naming is swapped** (`l_hip, r_hip, sdr_r, sdr_l =
  [2, 1, 17, 16]` — `l_hip` holds index 2, the R_Hip). If you touch these, copy
  upstream's variable names verbatim and keep them confined.
