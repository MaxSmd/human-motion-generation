# RMG — Riemannian Motion Generation

Reimplementation of *Riemannian Motion Generation: A Unified Framework for Human Motion
Representation and Generation via Riemannian Flow Matching* (Miao, Huang, Li 2026).

Designed to extend later to *Hierarchical Self-Supervised Representation Learning for
Human Motion on Product Manifolds*, so the manifold ops, representation, and encoder
backbone are factored to be reused across tasks (generation / SSL / recognition).

## Layout

```
src/rmg/
  manifolds/      Geometry primitives: R^d, S^d, pre-shape, ProductManifold
  representation/ Skeleton, T+R encode/decode, HumanML3D / MotionStreamer IO
  flow/           Riemannian flow matching: prior, geodesic interp, trainer, sampler
  models/         DiT backbone, Qwen3 text encoder, conditioning
  data/           HumanML3D dataset + collate
  eval/           Guo et al. evaluator wrapper, FID / R@k / Diversity / MModality
  tasks/          generation.py (this paper) + SSL/recognition stubs
  utils/          EMA, seeding, logging
configs/          Hydra configs: data, model, train
containers/       Dockerfile + enroot BUILD.md
slurm/            sbatch templates
scripts/          train.py, sample.py, evaluate.py, prepare_humanml3d.py
tests/            pytest suite
external/         git submodules: HumanML3D, text-to-motion, packed dataset
```

## Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                       # manifold + representation tests
```

## On the cluster

See `containers/BUILD.md` for the one-time enroot image build, and `slurm/` for
sbatch templates. Data lives behind a single `data.root` config knob — switch
between the local submodule and a cluster-mounted path with no code changes.
