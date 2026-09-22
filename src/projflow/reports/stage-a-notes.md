# ProjFlow Stage A — faithful upstream run of Table 1

Stage A runs the pinned upstream code (`external/ProjFlow` @ `9550501`) with its
own evaluator (Meng et al. 67-D, `cr8br0ze/evaluators_humanml3d`) and the
released `ACMDM_Raw_Flow_S_PatchSize22` prior. Setup:
`slurm/projflow/setup_upstream.sbatch`; grid: `slurm/projflow/eval_upstream.sbatch`;
table: `python -m projflow.scripts.collect_upstream <log dir> --out results/stageA-table1`.

## Protocol

- OmniControl grid: joints {pelvis 0, left foot 10, right foot 11, head 15,
  left wrist 20, right wrist 21} x densities {1, 2, 5, 25 %, 100 %}. Upstream
  reads `--intensity` 1/2/5 as a keyframe count and anything else as a
  percentage of the clip, so 25/100 are the paper's 49/196 keyframes.
- A joint's row is the mean over its five densities; "Average" is the mean
  over the six joints (paper Tables 1 and 7).
- CFG 3, 100 ODE steps, batch 32, seed 3407, one replication per cell — all
  upstream defaults.

## Deviations from the paper's setup

1. **Test set: 4,213 of 4,384 clips.** The 171 missing clips are all mirrored
   HumanAct12 clips (`M`-prefix). Our shared HumanML3D preparation
   (`src/shared/data/prepare_humanml3d.py`) writes the left/right mirror only
   for non-HumanAct12 clips, so they have no canonical features; mirroring the
   original clip's joints is not a substitute (median 1.8 cm, max 8.8 cm
   error on 200 clip pairs). Upstream's loader skips clips it cannot load, so
   every cell evaluates on the remaining 4,213. Decision (2026-09-21): report,
   do not regenerate.
2. **`new_joints` rebuilt.** The shared copy kept a single original file, so all
   28,037 clips were rebuilt as `recover_from_ric(new_joint_vecs)` from our
   canonical features with upstream's own function
   (`projflow.scripts.build_new_joints`); the surviving original matches to
   2.9e-6 m.
3. **Environment.** Our torch 2.9 / numpy 2 container instead of upstream's
   torch 2.2 / numpy 1.21. `slurm/projflow/compat/sitecustomize.py` restores
   `np.float` and `torch.load(weights_only=False)`; `timm` is pinned to
   upstream's 1.0.9. The upstream code itself is unmodified.
4. **Packed execution.** Cluster limits (one running job, one GPU per job,
   low-memory jobs cancelled) mean cells run as concurrent processes on one
   GPU (`slurm/projflow/run_cells.sh`). Each cell is still an independent
   upstream process with its own seed.

## Result (job 25775, all 30 cells)

`results/stageA-table1/table1.md` (per-cell: `cells.json`). Average over the
six joints, ours / paper: FID 0.108 / 0.097, R@3 0.766 / 0.779, Diversity
10.89 / 10.65, foot skate 0.0635 / 0.0603, Traj/Loc/Avg err 0.0000 / 0.0000 in
every cell. Exact constraint satisfaction reproduces; FID is within the
single-replication noise below and stays under the zero-shot DNO baseline
(0.147).

R@3 is lower and Diversity higher for all six joints. The evaluator's own
ground-truth row on our 4,213-clip data (identical in every cell;
`results/stageA-table1/ground_truth.json`) is shifted the same way: R@3 0.779
vs the paper's GT 0.795, Diversity 10.665 vs 10.455. Relative to its own
reference, ProjFlow lands where the paper's does (R@3 -0.013 vs -0.016,
Diversity +0.22 vs +0.20), so these gaps come from the evaluation data
(deviations 1-2), not from the sampler.

## Run-to-run noise

The pelvis/1-keyframe cell ran twice with identical code and seed: standalone
on a 2080 Ti (probe, job 25767) and pooled on an RTX 3090 (job 25775). FID
0.125 vs 0.109, R@3 0.765 vs 0.761. Sampling is not bit-reproducible across
GPUs (TF32 matmuls on Ampere, non-deterministic kernels, amplified over 100
ODE steps), so treat ~±0.02 FID per single-replication cell as noise when
comparing with the paper.
