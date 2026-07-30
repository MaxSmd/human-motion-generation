# Human motion representation

Text-to-motion reproductions on HumanML3D. Three models share one data pipeline,
one Guo evaluator, one container, and one web + cluster control plane:

- **RMG** — the primary contribution. Riemannian flow matching on the pose
  manifold **R³ × (S³)²²**: the root translates in Euclidean space while all 22
  joint rotations live on the quaternion sphere, so the model integrates an ODE
  along geodesics instead of denoising a flat vector. Adds sampling-time
  **constraints** (joint-angle limits, pelvis waypoints, scene collision) that
  project exactly onto the manifold rather than being penalised into place.
- **MARDM** — masked autoregressive diffusion, plus a spatial-control layer
  (guidance, a trained condition regularizer, and a closed-form root edit).
- **MoMask** — residual VQ-VAE tokenizer with masked and residual token
  transformers.

A public write-up of the RMG results is at
[riemannian-motion.space](https://riemannian-motion.space).

## Results

**RMG on the full HumanML3D test split.** Each row is at its own best guidance
scale ω and ODE step count:

| | params | FID ↓ | R@1 ↑ | R@3 ↑ | MM-Dist ↓ | Diversity → |
|---|---|---|---|---|---|---|
| Real motion (reference) | — | 0.0019 | 0.513 | 0.797 | 3.10 | 9.79 |
| **RMG-mid** (ours, ω 6.5 / 800) | 111.7 M | **0.429** | 0.507 | 0.790 | 3.13 | 9.06 |
| RMG-base (ours, ω 5.5 / 200) | 24.7 M | 8.049 | 0.224 | 0.479 | 5.40 | 8.06 |
| RMG paper (600k steps) | ≈460 M | 0.043 | 0.525 | — | — | 9.56 |

Retrieval quality essentially matches the paper: R@1 0.507 against their 0.525,
with our value just under the 0.513 real-motion reference and theirs marginally
above it. FID does not match — we land an order of magnitude higher. Scaling
accounts for the bulk of that: RMG-base → RMG-mid is 4.5× the parameters for a
**13× FID improvement** (8.049 → 0.607, both at 200 ODE steps, each at its own
best ω), and we stop at roughly a quarter of the paper's compute (111.7 M / 300k
vs ≈460 M / 600k).

Two caveats on that FID column: ours is a **single pass** over the test split
while the published figure is a 20-replication mean, and our own replication
spread is ±0.024. FID is also not comparable across papers unless the guidance
scale and feature normalisation match.

**MARDM** reproduces its published result. Ours, retrained on canonical features
and evaluated with the standard 263-D Guo evaluator at *w* = 4.0: **FID 0.097**,
R@1 0.499 — against the paper's 0.114 / 0.500.

**MARDM spatial control.** The pelvis trajectory is the exact cumulative sum of
four root channels, so a waypoint is a prefix-sum constraint with a closed-form
minimum-norm correction. That gives **exact waypoint satisfaction (0.000 m) at
unguided cost** — 45 s per clip, versus 1216 s for optimizer-based guidance that
still leaves 0.016 m of error. Where guidance is unavoidable, a hard trust region
holds FID at 0.786 against a 0.755 unguided floor (128-clip probe).

**MoMask** is implemented end to end and its data path is verified, but it has no
reproduction-quality evaluation yet.

Numbers above come from `showcase/lib/results.js` (RMG),
`src/mardm/reports/tables/` (MARDM), and
`src/mardm/reports/maskcontrol-differences.md` (control), each of which records
the run and eval job it came from.

## Setup

```bash
git clone --recurse-submodules https://github.com/julsmzr/human-motion-representation.git
cd human-motion-representation
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                    # 256 tests, no GPU or data needed
```

Training and evaluation need the packed HumanML3D dataset. Build it once from
AMASS + the HumanML3D submodule, then point `MGEN_DATA_ROOT` at it:

```bash
python -m shared.data.prepare_humanml3d       # writes humanml3d.zip + splits.json
export MGEN_DATA_ROOT=/path/to/humanml3d_packed
```

The web app (cluster control plane + in-process inference) runs from images:

```bash
docker compose up --build      # frontend :3000 · backend :8000
```

Code changes need a rebuild — the images bake `src/` rather than mounting it.

## Development

```
src/shared/     data · geometry · eval · text · utils   (model-agnostic; the only shared dependency)
src/rmg/        flow · manifolds · models · representation · configs · scripts
src/momask/     models · tasks · training · scripts
src/mardm/      models · control · representation · tasks · configs · scripts
app/            FastAPI backend (SLURM control plane) + Next.js frontend
showcase/       standalone static site for the public write-up
slurm/          rmg/ · momask/ · mardm/ job wrappers; shared prep at the root
tests/          rmg · shared · momask · mardm · app
```

Each model package is importable on its own (`import rmg`) and depends only on
`shared` — never on another model package. `pyproject.toml` picks them up
automatically.

A tiny CPU run that exercises the whole rmg training path:

```bash
python -m rmg.scripts.train model=dit_base train=rmg_base \
    train.max_steps=20 train.micro_batch_size=4 train.grad_accum=2 \
    text_encoder.type=random run_name=local-smoke
```

On the cluster, every job is a fresh enroot container plus env overrides:

```bash
sbatch slurm/rmg/train.sbatch                     # also eval.sbatch, viz.sbatch
OVERRIDES='data.subset_n=16 train.max_steps=15000' sbatch slurm/rmg/train.sbatch
# logs → slurm/logs/ ; outputs → runs/<model>/{train,eval,viz}/<run>/
```

Build the image once with `sbatch slurm/build_image.sbatch`; change
`containers/requirements.txt` and rebuild rather than installing inside a job.
