"""Zero-training spatial control check: guided vs unguided MARDM sampling.

Phase-1 validation of MaskControl-style inference-time guidance on an existing
MARDM checkpoint (no retraining, no model changes). For each clip from the
chosen split it:

  1. recovers GT joints and samples sparse waypoints from them (OmniControl
     protocol: `guid.joints` controlled at `guid.num_keyframes` random frames);
  2. samples the model on the clip's caption twice with the same seed —
     without guidance and with z-optimization guidance (the ONLY difference);
  3. reports Traj./Loc./Avg. control error for both, plus a metrics.json
     aggregate under `${output_dir}/guided/`.

Joint trajectories land next to the metrics as `.npy` (T, 22, 3) plus a
`control-<cid>.npz` (targets + mask) for local re-rendering; `+guid.render=true`
also writes GIFs.

Run (cluster):
    python -m mardm.scripts.guided_generate_mardm +data=cluster_mounted \\
        ae_checkpoint=runs/mardm-ae-XXXX/checkpoints/latest.pt \\
        +guid.checkpoint=runs/mardm-gen-YYYY/checkpoints/latest.pt \\
        text_encoder.type=qwen3 +guid.num_clips=16 +guid.verbose=true
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from mardm.control import (
    ControlSignal,
    GuidanceConfig,
    control_metrics,
    generate_guided,
    latents_to_joints,
)
from mardm.data import EssentialDataset
from mardm.models import AE, MARDM, AEConfig, MARDMConfig
from shared.geometry import recover_joints_from_ric
from shared.text import Qwen3EmbeddingEncoder, RandomTextEncoder
from shared.utils import EMA, load_checkpoint, set_seed


def _load_stats(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    blob = torch.load(Path(path), weights_only=True)
    return blob["mean"], blob["std"]


def _build_text_encoder(cfg: DictConfig):
    t = cfg.text_encoder.type
    if t == "random":
        return RandomTextEncoder(text_dim=cfg.text_encoder.text_dim)
    if t == "qwen3":
        return Qwen3EmbeddingEncoder(
            model_name=cfg.text_encoder.model_name,
            cache_dir=cfg.text_encoder.cache_dir,
            max_length=cfg.text_encoder.max_length,
        )
    raise ValueError(f"Unknown text_encoder.type {t!r}")


def _load_frozen_ae(cfg: DictConfig, device: torch.device) -> AE:
    ae = AE(AEConfig(**OmegaConf.to_container(cfg.ae, resolve=True))).to(device)
    state = load_checkpoint(Path(cfg.ae_checkpoint), map_location=device)
    ae.load_state_dict(state.model, strict=False)  # tolerate pre-latent_scale checkpoints
    if cfg.ae_use_ema and state.ema is not None:
        ema = EMA(ae, decay=0.0)
        ema.load_state_dict(state.ema)
        ema.copy_to(ae)
    ae.eval()
    return ae


def _load_mardm(cfg: DictConfig, ckpt: str | Path, ae: AE, device: torch.device) -> MARDM:
    mardm = MARDM(MARDMConfig(
        ae_dim=ae.output_emb_width, text_dim=cfg.text_encoder.text_dim,
        **OmegaConf.to_container(cfg.model, resolve=True),
    )).to(device)
    state = load_checkpoint(Path(ckpt), map_location=device)
    mardm.load_state_dict(state.model)
    if cfg.guid.use_ema and state.ema is not None:
        ema = EMA(mardm, decay=0.0)
        ema.load_state_dict(state.ema)
        ema.copy_to(mardm)
        print(f"[guided] loaded MARDM EMA weights from step {state.step}", flush=True)
    else:
        print(f"[guided] loaded MARDM live weights from step {state.step}", flush=True)
    mardm.eval()
    return mardm


def main_impl(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir) / "guided"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[guided] device={device}  out_dir={out_dir}", flush=True)

    mean, std = _load_stats(cfg.stats_path)
    text_encoder = _build_text_encoder(cfg)
    ae = _load_frozen_ae(cfg, device)
    mardm = _load_mardm(cfg, cfg.guid.checkpoint, ae, device)
    ds_rate = ae.downsample_rate

    dataset = EssentialDataset(
        root=cfg.data.root, split=str(cfg.guid.split),
        subset_frac=cfg.get("subset_frac"), limit_clips=cfg.get("limit_clips"),
        window_size=None, min_seq_len=cfg.data.min_seq_len, max_seq_len=cfg.data.max_seq_len,
        canonical_dir=cfg.data.get("canonical_dir"),
    )
    zf = zipfile.ZipFile(Path(cfg.data.root) / cfg.data.zip_name)
    n_clips = min(len(dataset), int(cfg.guid.num_clips))

    gcfg = GuidanceConfig(
        inner_iters=int(cfg.guid.inner_iters), lr=float(cfg.guid.lr),
        ode_steps_guidance=int(cfg.guid.ode_steps_guidance),
        ode_steps_final=int(cfg.guid.ode_steps_final),
        post_iters=int(cfg.guid.post_iters), post_lr=float(cfg.guid.post_lr),
        repair_rounds=int(cfg.guid.repair_rounds), repair_frac=float(cfg.guid.repair_frac),
        repair_iters=int(cfg.guid.repair_iters),
        verbose=bool(cfg.guid.verbose),
    )
    baseline_cfg = GuidanceConfig(
        inner_iters=0, post_iters=0, ode_steps_final=gcfg.ode_steps_final)
    joint_ids = [int(j) for j in cfg.guid.joints]
    print(f"[guided] controlling joints {joint_ids} at {int(cfg.guid.num_keyframes)} keyframes; "
          f"inner_iters={gcfg.inner_iters} lr={gcfg.lr} post_iters={gcfg.post_iters}", flush=True)

    per_clip: dict[str, dict] = {}
    for idx in range(n_clips):
        sample = dataset[idx]
        cid = sample.clip_id
        gt_ess = sample.x1                                   # (L, 67) raw essential
        L = gt_ess.shape[0]
        latent_len = max(1, L // ds_rate)
        caption = torch.load(io.BytesIO(zf.read(f"{cid}.pt")),
                             weights_only=False)["texts"][0]

        gt_joints = recover_joints_from_ric(gt_ess)          # (L, 22, 3)
        keyframe_gen = torch.Generator().manual_seed(int(cfg.guid.seed) + idx)
        control = ControlSignal.from_gt_joints(
            gt_joints, joint_ids=joint_ids, num_keyframes=int(cfg.guid.num_keyframes),
            length=latent_len * ds_rate, generator=keyframe_gen,
        )

        cond = text_encoder.encode([caption], device=device)
        m_lens = torch.tensor([latent_len], device=device)
        runs: dict[str, dict] = {}
        for name, run_cfg in (("unguided", baseline_cfg), ("guided", gcfg)):
            set_seed(int(cfg.guid.seed) + idx)               # identical noise draws
            latents = generate_guided(
                mardm, ae, cond, m_lens, control, mean, std,
                timesteps=int(cfg.guid.timesteps), cond_scale=float(cfg.guid.guidance),
                guidance=run_cfg,
            )
            with torch.no_grad():
                joints = latents_to_joints(latents.permute(0, 2, 1), ae,
                                           mean.to(device), std.to(device)).cpu()
            m = control_metrics(joints, control)
            runs[name] = m
            np.save(out_dir / f"{name}-{cid}.npy", joints[0].numpy().astype(np.float32))
            if bool(cfg.guid.render):
                from mardm.scripts.visualize_mardm import _render
                _render(joints[0].numpy(), out_dir / f"{name}-{cid}.gif",
                        title=f"{name.upper()} [{cid}] avg={m['avg_err']:.3f}m",
                        fps=int(cfg.guid.fps))

        np.savez(out_dir / f"control-{cid}.npz",
                 targets=control.targets[0].numpy().astype(np.float32),
                 mask=control.mask[0].numpy())
        per_clip[cid] = {"caption": caption, "length": int(L), **{
            f"{name}_{k}": v for name, m in runs.items() for k, v in m.items()}}
        print(f"[guided] {cid}  L={L:3d}  avg_err {runs['unguided']['avg_err']:.3f}m -> "
              f"{runs['guided']['avg_err']:.3f}m  loc_err {runs['unguided']['loc_err']:.2f} -> "
              f"{runs['guided']['loc_err']:.2f}  cap={caption[:44]!r}", flush=True)

    summary = {"per_clip": per_clip, "n_clips": len(per_clip),
               "joints": joint_ids, "num_keyframes": int(cfg.guid.num_keyframes),
               "guidance_config": gcfg.__dict__}
    for name in ("unguided", "guided"):
        for k in ("traj_err", "loc_err", "avg_err"):
            summary[f"mean_{name}_{k}"] = float(
                np.mean([v[f"{name}_{k}"] for v in per_clip.values()]))
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[guided] mean avg_err: unguided={summary['mean_unguided_avg_err']:.3f}m  "
          f"guided={summary['mean_guided_avg_err']:.3f}m", flush=True)
    print(f"[guided] mean loc_err(0.5m): unguided={summary['mean_unguided_loc_err']:.2f}  "
          f"guided={summary['mean_guided_loc_err']:.2f}", flush=True)
    print(f"[guided] done — trajectories + metrics under {out_dir}", flush=True)


@hydra.main(config_path="../configs", config_name="gen", version_base=None)
def main(cfg: DictConfig) -> None:
    guid_cfg = OmegaConf.create({
        "checkpoint": "???",       # required: gen-branch checkpoint
        "split": "test",
        "num_clips": 8,
        "guidance": 2.0,           # CFG scale (see evaluate_mardm defaults)
        "timesteps": 18,           # masked-AR sampling iterations
        "use_ema": True,
        "seed": 0,
        "joints": [0],             # controlled joint ids (0 = pelvis)
        "num_keyframes": 5,        # sparse waypoints sampled from GT
        "inner_iters": 30,         # z gradient steps per AR step
        "lr": 0.05,
        "ode_steps_guidance": 8,   # euler steps inside the guidance graph
        "ode_steps_final": 25,     # euler steps for committed samples
        "post_iters": 0,           # direct latent optimization after AR loop
        "post_lr": 0.01,
        "repair_rounds": 0,        # re-prediction repair passes after the AR loop
        "repair_frac": 0.5,        # fraction of tokens remasked per repair round
        "repair_iters": 10,        # light-guidance inner steps during repair
        "render": False,           # also write GIFs (needs matplotlib)
        "fps": 20,
        "verbose": False,          # print inner-loop loss trajectory
    })
    cfg.guid = OmegaConf.merge(guid_cfg, cfg.get("guid", OmegaConf.create({})))
    main_impl(cfg)


if __name__ == "__main__":
    main()
