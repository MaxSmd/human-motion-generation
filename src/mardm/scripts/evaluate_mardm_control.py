"""Evaluate MARDM under spatial-control guidance: FID/R-precision + control error.

Layer-2 control evaluation (phase 1, zero training): samples the test split with
`mardm.control.generate_guided` — waypoints drawn from each clip's GT joints
(OmniControl protocol) — and reports, per run:

  * realism vs the real test distribution: FID, R@1/2/3, MM-Dist, Diversity
    (same Guo-evaluator harness as `evaluate_mardm`);
  * control accuracy: Traj. err / Loc. err (@0.5 m) / Avg. err over the
    constrained cells.

With `ctrl.baseline=true` (default) the same clips are ALSO sampled unguided
with identical seeds, so the guided-vs-unguided delta isolates what guidance
costs in realism and buys in control — MaskControl's "w/o Logits Regularizer"
row, measured on our stack.

Real-side features come from canonical new_joint_vecs files (`ctrl.real_h3d_dir`),
falling back to the clip's essential feature -> 263-D bridge. GT joints for
waypoints come from the dataset's x1 (set `data.canonical_dir` so they match
the training distribution).

Run (cluster):
    python -m mardm.scripts.evaluate_mardm_control \\
        ae_checkpoint=.../mardm-ae-canon/checkpoints/latest.pt \\
        +ctrl.checkpoint=.../mardm-gen-m-canon/checkpoints/latest.pt \\
        text_encoder.type=qwen3 +ctrl.evaluator=real \\
        data.canonical_dir=.../h3d-canonical/new_joint_vecs \\
        +ctrl.real_h3d_dir=.../h3d-canonical/new_joint_vecs +ctrl.max_clips=512
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from mardm.control import ControlSignal, GuidanceConfig, generate_guided
from mardm.data import EssentialDataset
from mardm.models import AE, MARDM, AEConfig, MARDMConfig
from mardm.representation import denormalize, essential_to_h3d
from mardm.scripts.evaluate_mardm import (
    _build_text_encoder,
    _load_caption_tokens,
    _load_frozen_ae,
    _load_stats,
    _pad_stack,
)
from shared.eval import (
    RandomGuoEvaluator,
    RealGuoEvaluator,
    diversity,
    fid,
    mm_distance,
    r_precision,
)
from shared.geometry import Skeleton, recover_joints_from_ric
from shared.utils import EMA, load_checkpoint, set_seed


def _load_mardm(cfg: DictConfig, ckpt: str | Path, use_ema: bool, ae: AE,
                device: torch.device) -> MARDM:
    mardm = MARDM(MARDMConfig(
        ae_dim=ae.output_emb_width, text_dim=cfg.text_encoder.text_dim,
        **OmegaConf.to_container(cfg.model, resolve=True),
    )).to(device)
    state = load_checkpoint(Path(ckpt), map_location=device)
    mardm.load_state_dict(state.model)
    if use_ema and state.ema is not None:
        ema = EMA(mardm, decay=0.0)
        ema.load_state_dict(state.ema)
        ema.copy_to(mardm)
        print(f"[ctrl-eval] loaded MARDM EMA weights from step {state.step}", flush=True)
    else:
        print(f"[ctrl-eval] loaded MARDM live weights from step {state.step}", flush=True)
    mardm.eval()
    return mardm


def main_impl(cfg: DictConfig) -> None:
    set_seed(int(cfg.ctrl.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir) / "eval_control"
    out_dir.mkdir(parents=True, exist_ok=True)

    mean, std = _load_stats(cfg.stats_path)
    mean_d, std_d = mean.to(device), std.to(device)
    text_encoder = _build_text_encoder(cfg)
    ae = _load_frozen_ae(cfg, device)
    mardm = _load_mardm(cfg, cfg.ctrl.checkpoint, bool(cfg.ctrl.use_ema), ae, device)
    ds_rate = ae.downsample_rate
    skeleton = Skeleton(offsets=torch.load(Path(cfg.data.root) / cfg.data.offsets_name,
                                           weights_only=True))
    evaluator = (
        RealGuoEvaluator(text_to_motion_repo=cfg.ctrl.text_to_motion_repo,
                         humanml3d_repo=cfg.ctrl.humanml3d_repo, device=device)
        if cfg.ctrl.evaluator == "real" else RandomGuoEvaluator()
    )
    caption_tokens = (_load_caption_tokens(cfg.ctrl.humanml3d_repo)
                      if cfg.ctrl.vip_tokens and cfg.ctrl.evaluator == "real" else None)

    dataset = EssentialDataset(
        root=cfg.data.root, split=str(cfg.ctrl.split),
        window_size=None, min_seq_len=cfg.data.min_seq_len, max_seq_len=cfg.data.max_seq_len,
        canonical_dir=cfg.data.get("canonical_dir"),
    )
    n_total = len(dataset)
    if int(cfg.ctrl.max_clips) > 0:
        n_total = min(n_total, int(cfg.ctrl.max_clips))
    bs = int(cfg.ctrl.batch_size)
    real_dir = Path(cfg.ctrl.real_h3d_dir) if cfg.ctrl.real_h3d_dir else None

    gcfg = GuidanceConfig(
        inner_iters=int(cfg.ctrl.inner_iters), lr=float(cfg.ctrl.lr),
        ode_steps_guidance=int(cfg.ctrl.ode_steps_guidance),
        ode_steps_final=int(cfg.ctrl.ode_steps_final),
        post_iters=int(cfg.ctrl.post_iters), post_lr=float(cfg.ctrl.post_lr),
        repair_rounds=int(cfg.ctrl.repair_rounds), repair_frac=float(cfg.ctrl.repair_frac),
        repair_iters=int(cfg.ctrl.repair_iters),
    )
    run_cfgs: dict[str, GuidanceConfig] = {"guided": gcfg}
    if bool(cfg.ctrl.baseline):
        run_cfgs["unguided"] = GuidanceConfig(inner_iters=0, post_iters=0,
                                              ode_steps_final=gcfg.ode_steps_final)
    joint_ids = [int(j) for j in cfg.ctrl.joints]
    thresh = float(cfg.ctrl.threshold)
    use_upstream = bool(cfg.ctrl.use_upstream_features)
    print(f"[ctrl-eval] {n_total} clips (bs={bs}); joints={joint_ids} "
          f"keyframes={int(cfg.ctrl.num_keyframes)}; inner_iters={gcfg.inner_iters} "
          f"lr={gcfg.lr} cfg_w={float(cfg.ctrl.guidance)}", flush=True)

    real_emb, text_emb = [], []
    gen_emb: dict[str, list] = {name: [] for name in run_cfgs}
    seq_fail: dict[str, list] = {name: [] for name in run_cfgs}
    cell_dists: dict[str, list] = {name: [] for name in run_cfgs}
    n_real_fallback = 0
    n_text_fallback = 0

    for start in tqdm(range(0, n_total, bs), desc="ctrl-eval"):
        idxs = range(start, min(start + bs, n_total))
        samples = [dataset[i] for i in idxs]
        b = len(samples)
        texts = [s.text for s in samples]
        cids = [s.clip_id for s in samples]
        latent_lens = torch.tensor([max(1, s.x1.shape[0] // ds_rate) for s in samples])
        dec_lens = latent_lens * ds_rate
        t_max = int(dec_lens.max())

        # Waypoints from GT joints (deterministic per clip index).
        targets = torch.zeros(b, t_max, 22, 3)
        mask = torch.zeros(b, t_max, 22, dtype=torch.bool)
        for i, (gi, s) in enumerate(zip(idxs, samples)):
            gt_joints = recover_joints_from_ric(s.x1)        # (L, 22, 3) world frame
            t_i = int(dec_lens[i])
            gen = torch.Generator().manual_seed(int(cfg.ctrl.seed) * 100003 + gi)
            k = min(int(cfg.ctrl.num_keyframes), t_i)
            frames = torch.randperm(t_i, generator=gen)[:k]
            targets[i, :t_i] = gt_joints[:t_i]
            for j in joint_ids:
                mask[i, frames, j] = True
        control = ControlSignal(targets, mask)

        # Real + text side (once per batch).
        real_feats = []
        for i, s in enumerate(samples):
            f = None
            if real_dir is not None:
                p = real_dir / f"{cids[i]}.npy"
                if p.exists():
                    f = torch.from_numpy(np.load(p)[: int(cfg.data.max_seq_len) - 1]
                                         .astype(np.float32))
            if f is None:
                n_real_fallback += 1
                f = essential_to_h3d(s.x1, skeleton, humanml3d_repo=cfg.ctrl.humanml3d_repo,
                                     use_upstream=use_upstream)
            real_feats.append(f)
        real_lens = torch.tensor([f.shape[0] for f in real_feats], dtype=torch.long)
        real_emb.append(evaluator.encode_motion(_pad_stack(real_feats), real_lens).cpu().numpy())
        if caption_tokens is not None:
            toks = [caption_tokens.get(cid, {}).get(cap) for cid, cap in zip(cids, texts)]
            if all(t is not None for t in toks):
                te = evaluator.encode_text_from_tokens(toks)
            else:
                n_text_fallback += sum(t is None for t in toks)
                te = evaluator.encode_text_from_strings(texts)
        else:
            te = evaluator.encode_text_from_strings(texts)
        text_emb.append(te.cpu().numpy())

        cond = text_encoder.encode(texts, device=device)
        m_lens = latent_lens.to(device)
        for name, run_cfg in run_cfgs.items():
            set_seed(int(cfg.ctrl.seed) * 7919 + start)      # identical noise across runs
            latents = generate_guided(
                mardm, ae, cond, m_lens, control, mean, std,
                timesteps=int(cfg.ctrl.timesteps), cond_scale=float(cfg.ctrl.guidance),
                guidance=run_cfg,
            )
            with torch.no_grad():
                essential = ae.decode(latents)               # (B, t_max, 67) normalized
                joints = recover_joints_from_ric(
                    denormalize(essential, mean_d, std_d)).cpu()

            dist = torch.linalg.vector_norm(joints - targets, dim=-1)
            dist = torch.where(mask, dist, torch.zeros_like(dist))
            seq_fail[name].extend(((dist > thresh).flatten(1).any(dim=1)).tolist())
            cell_dists[name].append(dist[mask])

            feats = []
            for i in range(b):
                ess_i = essential[i, : int(dec_lens[i])]
                feats.append(essential_to_h3d(ess_i, skeleton, mean=mean, std=std,
                                              humanml3d_repo=cfg.ctrl.humanml3d_repo,
                                              use_upstream=use_upstream))
            lens = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
            gen_emb[name].append(evaluator.encode_motion(_pad_stack(feats), lens).cpu().numpy())

    real_np = np.concatenate(real_emb, 0)
    text_np = np.concatenate(text_emb, 0)
    if caption_tokens is not None:
        print(f"[ctrl-eval] VIP tokens: {n_text_fallback} caption(s) fell back to spaCy", flush=True)
    if real_dir is not None and n_real_fallback:
        print(f"[ctrl-eval] {n_real_fallback} real clip(s) fell back to essential->263 bridge",
              flush=True)
    rng = np.random.default_rng(int(cfg.ctrl.seed))
    results: dict = {
        "n_clips": int(real_np.shape[0]),
        "joints": joint_ids,
        "num_keyframes": int(cfg.ctrl.num_keyframes),
        "guidance_config": gcfg.__dict__,
        "cfg_w": float(cfg.ctrl.guidance),
        "r_precision_real": r_precision(text_np, real_np, top_k=3, rng=rng).tolist(),
        "diversity_real": diversity(real_np, diversity_times=int(cfg.ctrl.diversity_times), rng=rng),
        "mm_dist_real": mm_distance(text_np, real_np),
    }
    for name in run_cfgs:
        gen_np = np.concatenate(gen_emb[name], 0)
        dists = torch.cat(cell_dists[name])
        results[name] = {
            "fid": fid(real_np, gen_np),
            "r_precision": r_precision(text_np, gen_np, top_k=3, rng=rng).tolist(),
            "mm_dist": mm_distance(text_np, gen_np),
            "diversity": diversity(gen_np, diversity_times=int(cfg.ctrl.diversity_times), rng=rng),
            "traj_err": float(np.mean(seq_fail[name])),
            "loc_err": float((dists > thresh).float().mean()),
            "avg_err": float(dists.mean()),
        }
        print(f"[ctrl-eval] {name}: fid={results[name]['fid']:.3f} "
              f"r1={results[name]['r_precision'][0]:.3f} "
              f"traj={results[name]['traj_err']:.3f} loc={results[name]['loc_err']:.3f} "
              f"avg={results[name]['avg_err']:.3f}m", flush=True)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"[ctrl-eval] done — results under {out_dir}", flush=True)


@hydra.main(config_path="../configs", config_name="gen", version_base=None)
def main(cfg: DictConfig) -> None:
    ctrl_cfg = OmegaConf.create({
        "checkpoint": "???",
        "split": "test",
        "use_ema": True,
        "evaluator": "random",                 # real | random (harness smoke)
        "text_to_motion_repo": "external/text-to-motion",
        "humanml3d_repo": "external/HumanML3D",
        "use_upstream_features": True,
        "vip_tokens": True,
        "real_h3d_dir": "",                    # canonical new_joint_vecs for the REAL side
        "guidance": 3.0,                       # CFG scale (canonical sweep optimum)
        "timesteps": 18,
        "batch_size": 16,
        "max_clips": -1,
        "seed": 0,
        "diversity_times": 300,
        "threshold": 0.5,                      # OmniControl 50 cm
        "joints": [0],
        "num_keyframes": 5,
        "inner_iters": 30,
        "lr": 0.05,
        "ode_steps_guidance": 8,
        "ode_steps_final": 25,
        "post_iters": 0,
        "post_lr": 0.01,
        "repair_rounds": 0,                    # re-prediction repair passes
        "repair_frac": 0.5,
        "repair_iters": 10,
        "baseline": True,                      # also run unguided with same seeds
    })
    cfg.ctrl = OmegaConf.merge(ctrl_cfg, cfg.get("ctrl", OmegaConf.create({})))
    main_impl(cfg)


if __name__ == "__main__":
    main()
