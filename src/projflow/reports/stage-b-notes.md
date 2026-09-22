# ProjFlow Stage B — our port, our 263-D evaluator

Stage A ran the pinned upstream code with its own 67-D evaluator
([stage-a-notes.md](stage-a-notes.md)). Stage B re-implements ProjFlow inside
`src/projflow/` on our data and evaluation stack, so its numbers are directly
comparable with our MARDM control results and with the paper's *legacy*
(263-D Guo) supplementary Table 8.

Code: `projflow/{models/acmdm,text,sampler/*,control,data}.py`; evaluation
`projflow.scripts.evaluate` (`slurm/projflow/eval_port.sbatch`); validation
`projflow.scripts.check_port` (`slurm/projflow/check_port.sbatch`) and
`tests/projflow/`.

## Is the port the same method?

| check | result |
|---|---|
| metric, pseudo-observations, trust/variance vs upstream helpers (CPU, float64) | exact |
| batched projection vs upstream's per-sample dense solve | 1e-9 |
| full 20-step loop with a toy velocity field, ProjFlow on and off | 1e-5 |
| **guided velocity**, same checkpoint/inputs, vs upstream `forward_with_CFG` (job 25796) | rel. 7.6e-7 |
| **full 100-step sampler** on GPU, same noise/keyframes/mixing ε | rel. 4.2e-7 |
| CLIP text features (transformers fp16) vs OpenAI `clip` fp16 | cosine 0.9999993 |
| our `recover_joints_from_ric` vs a surviving original `new_joints` file | 2.9e-6 m |

Differences from upstream that do not change the math: the CFG batch carries
only the conditional half (upstream carries both; the model reads only the
first), the projection solves all (sample, frame) blocks in one batched
Cholesky, and `use_projflow` is split into the three Table 3 switches.

## Result (job 25804, 30 cells, 8 h 38 min, batch 1024)

`results/stageB-table8/table.md`, per cell `results/stageB-table8/cells/`.
Embeddings stay on the cluster (`~/rmg-runs/projflow-stageB-table8/embeddings`,
984 MB) so cells can be re-scored without resampling.

| | FID | R@3 | Diversity | Foot skate | Traj/Loc/Avg err |
|---|---|---|---|---|---|
| **Ours, average over 6 joints** | **0.098** | **0.781** | 9.57 | 0.0633 | 0.0000 |
| Paper, supp. Table 8 average | 0.074 | 0.752 | 9.065 | 0.0624 | 0.00 |
| Ours, pelvis | 0.098 | 0.787 | 9.58 | 0.0663 | 0.0000 |
| Paper, supp. Table 8 pelvis | 0.083 | 0.755 | 9.096 | 0.0651 | 0.00 |

* **Exact control reproduces**: the largest keyframe error over all 30 cells is
  4.8e-7 m, per-cell mean 1e-8 m.
* **Realism/text match reproduce**: R@3 is 0.03 *above* the paper's and sits at
  the real-motion reference (0.783); foot skating matches (0.0633 vs 0.0624);
  FID is 0.098 vs 0.074, i.e. 0.024 higher, with per-cell spread 0.055-0.151.
* Ours is a single replication per cell; Stage A put that noise at ~±0.02 FID.

## Which FID to read

Generated motion reaches the 263-D evaluator through HumanML3D's `process_file`
(the only route from joint positions). The real side can be either the canonical
`new_joint_vecs` or the GT joints through that same conversion, and the choice
moves FID more than the method does, so every cell reports all three:

| column | meaning | average |
|---|---|---|
| `fid_processed` | real joints through the same conversion — symmetric, as upstream scores it; **the paper-comparable number** | 0.098 |
| `fid` | against canonical features: also measures the conversion | 0.187 |
| `fid_floor` | canonical vs processed real: the conversion's own FID | 0.064 |

The processed real reference also matches the paper's legacy GT row more closely
(R@3 0.783 vs 0.797, Diversity 9.48 vs 9.503) than the canonical one
(Diversity 9.27).

## Deviations

1. **Test set**: 4,525 entries from 4,213 of 4,384 test clips — the 171 mirrored
   HumanAct12 clips have no canonical features (deviation 1 of Stage A;
   upstream's own loader built 4,512 entries from the same protocol).
2. **Sampling batch 1024**, not upstream's 32 (which forced Stage A's packed
   jobs). Independent per sample; it changes throughput, not the method.
3. **Evaluation protocol** follows upstream (sub-segment captions, length
   rounded to 4 frames, random crop, CFG 3, 100 ODE steps, seed per cell).
4. **Foot skating** is reported twice: upstream's `calculate_skating_ratio`
   (over all 196 padded frames, as upstream computes it — the column above) and
   `mardm.control.losses.motion_metrics` for comparison with our MARDM numbers
   (0.386; a different definition, not a different motion).
5. Upstream's Loc./Avg. err divide by `mask.sum()` over (T, 22, 3), i.e. three
   times the keyframe count, so its published values are a third of the true
   distance. Irrelevant at 0, but it affects the baselines it is compared with;
   we report the per-keyframe value and `*_upstream` alongside.
