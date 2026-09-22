"""Stage B: evaluate the ported ProjFlow sampler with our 263-D Guo evaluator.

Runs cells of the OmniControl grid (controlled joint × keyframe density), each
over the whole HumanML3D test protocol (projflow.data). Per cell:

  * realism / text match with the Guo evaluator: generated joints -> 263-D via
    HumanML3D's own `process_file` (positions_to_h3d_features_upstream); real
    side scored two ways: (a) `real = canonical`, the same crop of canonical
    new_joint_vecs, and (b) `real = processed`, the GT joint crop through the same
    process_file path as the generated motion (symmetric, as upstream feeds real
    and generated joints through one back_process). FID, R@1-3 (pools of 32),
    MM-Dist, Diversity, plus both real reference rows. All embeddings are saved
    (<out_dir>/embeddings/*.npz) so cells can be re-scored without resampling.
  * control error, OmniControl definitions as upstream: Traj. err (any keyframe
    > 0.5 m), Loc. err (keyframes > 0.5 m), Avg. err (mean distance, m).
  * foot skating twice: upstream's `calculate_skating_ratio` (over all 196
    padded frames, as upstream computes it) and ours (mardm.control.losses
    .motion_metrics over valid frames, as in our MARDM control evaluation).

Results: <out_dir>/cells/joint{j}_density{d}.json, <out_dir>/embeddings/, <out_dir>/table.md.

Usage (cluster, see slurm/projflow/eval_port.sbatch):
    python -m projflow.scripts.evaluate --canonical-dir .../new_joint_vecs \\
        --checkpoint .../ACMDM_Raw_Flow_S_PatchSize22/model/latest.tar \\
        --offsets .../humanml3d_packed/target_offsets.pt --out-dir ~/rmg-runs/projflow-stageB
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

from mardm.control.losses import motion_metrics
from projflow.control import generate_control
from projflow.data import ControlTestSet, JointNormalizer, batch_draws
from projflow.models import ACMDM
from projflow.sampler import ProjFlowConfig
from projflow.text import ClipTextEncoder
from shared.eval.guo_evaluator import RealGuoEvaluator
from shared.eval.metrics import diversity, fid, mm_distance, r_precision
from shared.geometry.humanml3d_upstream import positions_to_h3d_features_upstream
from shared.geometry.skeleton import Skeleton

JOINTS = {0: "Pelvis", 10: "Left foot", 11: "Right foot", 15: "Head", 20: "Left wrist", 21: "Right wrist"}
DENSITIES = (1, 2, 5, 25, 100)
# ACMDM-S-PS22+ProjFlow under the legacy (263-D Guo) protocol, supp. Table 8.
PAPER_TABLE8 = {
    "Pelvis": {"fid": 0.083, "r3": 0.755, "diversity": 9.096, "skate_upstream": 0.0651},
    "Average": {"fid": 0.074, "r3": 0.752, "diversity": 9.065, "skate_upstream": 0.0624},
}


def cell_list(spec: str) -> list[tuple[int, int]]:
    """'all' or comma-separated 'joint:density' pairs."""
    if spec == "all":
        return [(j, d) for j in JOINTS for d in DENSITIES]
    return [tuple(int(v) for v in item.split(":")) for item in spec.split(",")]


def skating_ratio_upstream(joints: np.ndarray) -> np.ndarray:
    """Upstream `calculate_skating_ratio` on (B, T, 22, 3) world joints -> (B,)."""
    feet = joints[:, :, [10, 11], :].transpose(0, 2, 3, 1)                 # (B, 2, 3, T)
    plane_vel = np.linalg.norm(feet[:, :, [0, 2], 1:] - feet[:, :, [0, 2], :-1], axis=2) * 20.0
    vel_avg = uniform_filter1d(plane_vel, axis=-1, size=5, mode="constant", origin=0)
    height = feet[:, :, 1, :]
    contact = (height[:, :, :-1] < 0.05) & (height[:, :, 1:] < 0.05)
    skating = contact & (plane_vel > 0.5) & (vel_avg > 0.5)
    skating = skating[:, 0] | skating[:, 1]
    return skating.sum(axis=1) / skating.shape[1]


def control_errors(gen: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """OmniControl per-sample errors, upstream `calculate_trajectory_error` (strict),
    normalised per keyframe cell.

    gen/gt (T, 22, 3) world joints, mask (T, 22) bool keyframe cells.
    Returns [traj_fail@0.2, traj_fail@0.5, loc_fail@0.2, loc_fail@0.5, mean_dist].
    """
    dist = np.linalg.norm(gen - gt, axis=-1)[mask]
    n = max(mask.sum(), 1)
    return np.array([
        float(not (dist <= 0.2).all()), float(not (dist <= 0.5).all()),
        (dist > 0.2).sum() / n, (dist > 0.5).sum() / n, dist.sum() / n,
    ])


_SKELETON: Skeleton | None = None


def _init_feature_worker(offsets: torch.Tensor) -> None:
    global _SKELETON
    torch.set_num_threads(1)
    _SKELETON = Skeleton(offsets=offsets)


def _h3d_features(joints: np.ndarray) -> np.ndarray:
    """World joints (T, 22, 3) -> (T-1, 263) via HumanML3D's process_file (runs in a worker)."""
    return positions_to_h3d_features_upstream(torch.from_numpy(joints), _SKELETON).numpy()


def pad_stack(feats: list[torch.Tensor]) -> torch.Tensor:
    out = torch.zeros(len(feats), max(f.shape[0] for f in feats), feats[0].shape[1])
    for i, f in enumerate(feats):
        out[i, : f.shape[0]] = f
    return out


def evaluate_cell(joint: int, density: int, *, args, testset, normalizer, model, text_encoder,
                  evaluator, pool, pf: ProjFlowConfig, device) -> dict:
    seed = args.seed + 1000 * joint + density
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n = len(testset) if args.max_entries <= 0 else min(args.max_entries, len(testset))
    order = rng.permutation(len(testset))[:n]

    text_emb, real_emb, proc_emb, gen_emb = [], [], [], []
    errs, skate_up, ours = [], [], []
    t0 = time.time()
    t_sample = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for start in range(0, n, args.batch_size):
        draws = [testset.draw(int(i), rng) for i in order[start:start + args.batch_size]]
        control, lengths = batch_draws(draws, normalizer)
        ts = time.time()
        samples, mask = generate_control(
            model, text_encoder, [d.caption.text for d in draws], lengths.to(device),
            control.to(device), [joint], density, args.cfg, pf)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_sample += time.time() - ts

        gen = normalizer.denormalize(samples.permute(0, 2, 3, 1)).cpu()            # (B, L, 22, 3)
        padded = gen.clone()
        valid = torch.arange(gen.shape[1])[None] < lengths[:, None]
        padded[~valid] = normalizer.mean                                           # upstream: zeros -> mean pose
        skate_up.append(skating_ratio_upstream(padded.numpy()))
        ours.append({k: v * len(draws) for k, v in motion_metrics(gen, lengths).items()})

        key = mask[:, 0].bool().cpu().numpy()                                      # (B, L, 22)
        gen_np = gen.numpy()
        for i, d in enumerate(draws):
            errs.append(control_errors(gen_np[i, : d.length], d.joints, key[i, : d.length]))

        gen_feats = [torch.from_numpy(f) for f in
                     pool.map(_h3d_features, [gen_np[i, : d.length] for i, d in enumerate(draws)], chunksize=16)]
        gen_emb.append(evaluator.encode_motion(pad_stack(gen_feats), lengths - 1).cpu().numpy())
        real_emb.append(evaluator.encode_motion(
            pad_stack([torch.from_numpy(d.vecs) for d in draws]), lengths).cpu().numpy())
        proc_feats = [torch.from_numpy(f) for f in pool.map(_h3d_features, [d.joints for d in draws], chunksize=16)]
        proc_emb.append(evaluator.encode_motion(pad_stack(proc_feats), lengths - 1).cpu().numpy())
        text_emb.append(evaluator.encode_text_from_tokens([list(d.caption.tokens) for d in draws]).cpu().numpy())
        print(f"[projflow-eval] joint {joint} density {density}: {start + len(draws)}/{n} "
              f"({time.time() - t0:.0f}s)", flush=True)

    text_emb, real_emb, proc_emb, gen_emb = (np.concatenate(x) for x in (text_emb, real_emb, proc_emb, gen_emb))
    emb_dir = Path(args.out_dir).expanduser() / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(emb_dir / f"joint{joint}_density{density}.npz",
                        text=text_emb, real_canonical=real_emb, real_processed=proc_emb, gen=gen_emb)
    errs = np.stack(errs)
    metric_rng = np.random.default_rng(seed)
    ours_mean = {k: sum(o[k] for o in ours) / n for k in ours[0]}
    return {
        "joint": joint, "joint_name": JOINTS.get(joint, str(joint)), "density": density, "n": int(n),
        "seed": seed, "wall_seconds": round(time.time() - t0, 1), "sampling_seconds": round(t_sample, 1),
        "peak_gpu_gb": round(torch.cuda.max_memory_reserved(device) / 2**30, 2) if device.type == "cuda" else 0.0,
        "fid": fid(real_emb, gen_emb),
        "fid_processed": fid(proc_emb, gen_emb),
        "fid_floor": fid(real_emb, proc_emb),   # canonical vs processed real: the conversion's own FID
        "r_precision": r_precision(text_emb, gen_emb, top_k=3, rng=metric_rng).tolist(),
        "mm_dist": mm_distance(text_emb, gen_emb),
        "diversity": diversity(gen_emb, diversity_times=300, rng=metric_rng),
        "r_precision_real": r_precision(text_emb, real_emb, top_k=3, rng=np.random.default_rng(seed)).tolist(),
        "mm_dist_real": mm_distance(text_emb, real_emb),
        "diversity_real": diversity(real_emb, diversity_times=300, rng=np.random.default_rng(seed)),
        "r_precision_real_processed": r_precision(text_emb, proc_emb, top_k=3,
                                                  rng=np.random.default_rng(seed)).tolist(),
        "diversity_real_processed": diversity(proc_emb, diversity_times=300, rng=np.random.default_rng(seed)),
        "traj_err": float(errs[:, 1].mean()), "loc_err": float(errs[:, 3].mean()),
        "avg_err": float(errs[:, 4].mean()), "max_keyframe_err": float(errs[:, 4].max()),
        # Upstream divides Loc./Avg. err by mask.sum() over (T, 22, 3), i.e. 3x the
        # keyframe count; its reported values are these / 3 (irrelevant at 0).
        "loc_err_upstream": float(errs[:, 3].mean() / 3), "avg_err_upstream": float(errs[:, 4].mean() / 3),
        "skate_upstream": float(np.concatenate(skate_up).mean()),
        "skate_ours": ours_mean["foot_skate"], "motion_mag": ours_mean["motion_mag"], "jerk": ours_mean["jerk"],
    }


def write_table(out_dir: Path) -> str:
    cells = [json.loads(p.read_text()) for p in sorted((out_dir / "cells").glob("*.json"))]
    cols = ("fid", "fid_processed", "fid_floor", "r3", "diversity", "mm_dist", "skate_upstream", "skate_ours",
            "traj_err", "loc_err", "avg_err")

    def row(cs: list[dict]) -> dict:
        get = {"r3": lambda c: c["r_precision"][2]}
        return {k: float(np.mean([get.get(k, lambda c: c[k])(c) for c in cs])) for k in cols}

    rows = {}
    for jid, name in JOINTS.items():
        cs = [c for c in cells if c["joint"] == jid]
        if cs:
            rows[name] = (row(cs), len(cs))
    if rows:
        rows["Average"] = ({k: float(np.mean([r[k] for r, _ in rows.values()])) for k in cols},
                           sum(n for _, n in rows.values()))
    lines = ["| Joint | cells | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for name, (r, n_cells) in rows.items():
        lines.append(f"| **{name}** | {n_cells} | " + " | ".join(f"{r[k]:.4f}" for k in cols) + " |")
        if name in PAPER_TABLE8:
            p = PAPER_TABLE8[name]
            lines.append(f"| {name} (paper T8) | | " + " | ".join(
                f"{p[k]:.4f}" if k in p else "" for k in cols) + " |")
    if cells:
        def mean(f):
            return float(np.mean([f(c) for c in cells]))
        lines.append(
            f"\nReal-motion reference (mean over cells) — canonical: R@3 "
            f"{mean(lambda c: c['r_precision_real'][2]):.4f}, Diversity {mean(lambda c: c['diversity_real']):.4f}; "
            f"processed: R@3 {mean(lambda c: c['r_precision_real_processed'][2]):.4f}, Diversity "
            f"{mean(lambda c: c['diversity_real_processed']):.4f}; paper legacy GT: R@3 0.797, Diversity 9.503. "
            f"`fid` scores against canonical real features, `fid_processed` against real joints "
            f"through the same process_file path, `fid_floor` is canonical vs processed real.")
    table = "\n".join(lines) + "\n"
    (out_dir / "table.md").write_text(table)
    return table


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--canonical-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--offsets", required=True, help="humanml3d_packed/target_offsets.pt (process_file skeleton)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--humanml3d-repo", default="external/HumanML3D")
    p.add_argument("--text-to-motion-repo", default="external/text-to-motion")
    p.add_argument("--stats-dir", default="external/ProjFlow/utils/22x3_mean_std/t2m")
    p.add_argument("--cells", default="all", help="'all' or 'joint:density,...'")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--cfg", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--max-entries", type=int, default=0, help="smoke runs: evaluate only this many entries")
    p.add_argument("--feature-workers", type=int, default=4, help="processes for joints -> 263-D")
    p.add_argument("--clip-dtype", choices=("float16", "float32"), default="float16")
    p.add_argument("--no-kinematic-metric", action="store_true")
    p.add_argument("--no-pseudo-obs", action="store_true")
    p.add_argument("--no-noise-mixing", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir).expanduser()
    (out_dir / "cells").mkdir(parents=True, exist_ok=True)
    pf = replace(ProjFlowConfig(), use_kinematic_metric=not args.no_kinematic_metric,
                 use_pseudo_obs=not args.no_pseudo_obs, use_noise_mixing=not args.no_noise_mixing)
    (out_dir / "config.json").write_text(json.dumps({**vars(args), "projflow": asdict(pf)}, indent=2, default=str))

    h3d = Path(args.humanml3d_repo) / "HumanML3D"
    testset = ControlTestSet(args.canonical_dir, h3d / "test.txt", h3d / "texts.zip")
    print(f"[projflow-eval] {len(testset)} entries; {len(testset.missing)} test clips without "
          f"canonical features/texts", flush=True)
    normalizer = JointNormalizer(args.stats_dir)
    model = ACMDM.from_upstream_checkpoint(args.checkpoint).to(device)
    text_encoder = ClipTextEncoder(device, getattr(torch, args.clip_dtype))
    evaluator = RealGuoEvaluator(args.text_to_motion_repo, args.humanml3d_repo, device=device)
    offsets = torch.load(args.offsets, weights_only=True)
    # process_file is single-threaded numpy; fork workers before any CUDA work in them.
    pool = mp.get_context("fork").Pool(args.feature_workers, _init_feature_worker, (offsets,))

    for joint, density in cell_list(args.cells):
        path = out_dir / "cells" / f"joint{joint}_density{density}.json"
        if path.exists() and not args.overwrite:
            print(f"[projflow-eval] skip {path.name} (exists)", flush=True)
            continue
        res = evaluate_cell(joint, density, args=args, testset=testset, normalizer=normalizer, model=model,
                            text_encoder=text_encoder, evaluator=evaluator, pool=pool, pf=pf,
                            device=device)
        path.write_text(json.dumps(res, indent=2))
        print(f"[projflow-eval] {path.name}: FID {res['fid']:.4f} R@3 {res['r_precision'][2]:.4f} "
              f"div {res['diversity']:.3f} avg_err {res['avg_err']:.2e} ({res['wall_seconds']:.0f}s, "
              f"sampling {res['sampling_seconds']:.0f}s, peak GPU {res['peak_gpu_gb']} GB)", flush=True)
        print(write_table(out_dir), flush=True)


if __name__ == "__main__":
    main()
