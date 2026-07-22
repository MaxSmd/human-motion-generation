"""Visualize + score a MARDM checkpoint on the clips it trained on.

This is the *overfit / memorization* check, not the distributional eval
(`evaluate_mardm.py`). For each clip in the deterministic train subset the model
saw, it:

  1. renders the ground-truth motion  -> `real-<cid>.gif`
  2. samples the model on that clip's own caption (matched length) and renders
     the prediction                    -> `gen-<cid>.gif`
  3. reports per-clip **MPJPE** (mean per-joint position error, GT vs pred) and
     writes a `metrics.json` + `manifest.json` sidecar.

Both GT and prediction go through the *same* `recover_joints_from_ric` path, so
the two skeletons share a frame/convention and are directly comparable (no IK,
no `external/HumanML3D` needed). A near-zero MPJPE + visually matching GIFs means
the model memorized the subset — the thing an overfit smoke is supposed to show.

Run (cluster):
    python -m mardm.scripts.visualize_mardm \\
        ae_checkpoint=runs/mardm-overfit-ae-XXXX/checkpoints/latest.pt \\
        +viz.checkpoint=runs/mardm-overfit-gen-XXXX/checkpoints/latest.pt \\
        text_encoder.type=qwen3 subset_frac=0.003 \\
        +viz.num_clips=8 +viz.guidance=1.5 +viz.timesteps=18

Outputs land under `${output_dir}/viz/`. `scp` the `*.gif` down to view; the
`*.npy` next to each GIF holds the raw (T, 22, 3) joints for local re-render.
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

# Non-interactive backend BEFORE any pyplot import.
import matplotlib
matplotlib.use("Agg")

from mardm.data import EssentialDataset
from mardm.models import AE, MARDM, AEConfig, MARDMConfig
from mardm.representation import denormalize
from shared.geometry import recover_joints_from_ric
from shared.text import Qwen3EmbeddingEncoder, RandomTextEncoder
from shared.utils import EMA, load_checkpoint, set_seed


# HumanML3D 22-joint kinematic chains (same as upstream's t2m_kinematic_chain).
_T2M_CHAINS: tuple[tuple[int, ...], ...] = (
    (0, 2, 5, 8, 11),       # right leg
    (0, 1, 4, 7, 10),       # left leg
    (0, 3, 6, 9, 12, 15),   # spine + head
    (9, 14, 17, 19, 21),    # right arm
    (9, 13, 16, 18, 20),    # left arm
)


def _render(joints: np.ndarray, save_path: Path, title: str, fps: int,
            markers: np.ndarray | None = None) -> Path:
    """Render (T, 22, 3) joint positions to a GIF (Pillow — no ffmpeg).

    `markers` (K, 3), optional: static world-space control waypoints, drawn as
    fixed red X's the skeleton should thread — makes a spatial-control result
    legible (you see the controlled joint pass through its target points).

    Also dumps the raw joints next to the GIF as `<stem>.npy` for local
    re-render to MP4 with a system ffmpeg.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    npy_path = save_path.with_suffix(".npy")
    np.save(npy_path, joints.astype(np.float32))

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    T = joints.shape[0]
    span = joints.reshape(-1, 3)
    if markers is not None and len(markers):
        span = np.concatenate([span, markers.reshape(-1, 3)], axis=0)
    lo, hi = span.min(axis=0), span.max(axis=0)
    center = (lo + hi) / 2
    radius = float(np.max(hi - lo)) / 2 * 1.1 + 1e-3

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    chain_lines = [ax.plot([], [], [], "-o", linewidth=2, markersize=3)[0]
                   for _ in _T2M_CHAINS]
    title_text = ax.set_title("")

    def _setup_axes():
        # HumanML3D is Y-up; matplotlib 3D treats Z as up, so swap Y<->Z.
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[2] - radius, center[2] + radius)
        ax.set_zlim(center[1] - radius, center[1] + radius)
        ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_zlabel("y")
        ax.view_init(elev=15, azim=-70)
        if markers is not None and len(markers):
            # Y<->Z swap to match the skeleton's axis convention above.
            ax.scatter(markers[:, 0], markers[:, 2], markers[:, 1],
                       c="red", marker="x", s=80, linewidths=2, depthshade=False)

    def update(t):
        _setup_axes()
        for line, chain in zip(chain_lines, _T2M_CHAINS):
            line.set_data(joints[t, list(chain), 0], joints[t, list(chain), 2])
            line.set_3d_properties(joints[t, list(chain), 1])
        title_text.set_text(f"{title}\nframe {t + 1}/{T}")
        return chain_lines + [title_text]

    ani = FuncAnimation(fig, update, frames=T, interval=1000 // fps, blit=False)
    gif_path = save_path.with_suffix(".gif")
    ani.save(str(gif_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"[viz] wrote {gif_path.name}  (+ joints {npy_path.name})", flush=True)
    return gif_path


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
    if cfg.viz.use_ema and state.ema is not None:
        ema = EMA(mardm, decay=0.0)
        ema.load_state_dict(state.ema)
        ema.copy_to(mardm)
        print(f"[viz] loaded MARDM EMA weights from step {state.step}", flush=True)
    else:
        print(f"[viz] loaded MARDM live weights from step {state.step}", flush=True)
    mardm.eval()
    return mardm


@torch.no_grad()
def _recon_sweep(mardm, ae, text_encoder, dataset, zf, mean, std, device,
                 n_clips: int, ratios: list[float]) -> None:
    """Teacher-forced latent reconstruction at controlled mask ratios.

    For each clip: encode the GT to true latents, mask a fraction r of the
    tokens (rest kept as true context), run ONE prediction step, and measure
    RMSE between predicted and true latents on the masked tokens. This isolates
    whether the diffusion head works given context (low r) vs the all-masked
    generation regime (r=1.0):
      * low+flat across r  -> head/sampling fine; collapse is elsewhere.
      * low at small r, blows up toward r=1 -> the model can denoise WITH
        context but not from caption-only; the all-masked first generation
        step is out-of-distribution. That's the bug, and it's training-side.
    """
    print("\n[sweep] teacher-forced latent reconstruction (1 step); "
          "rmse = pred vs true latent on masked tokens:", flush=True)
    agg: dict[float, list[float]] = {r: [] for r in ratios}
    lat_std, lat_absmean = [], []
    for idx in range(n_clips):
        sample = dataset[idx]
        cid = sample.clip_id
        caption = torch.load(io.BytesIO(zf.read(f"{cid}.pt")),
                             weights_only=False)["texts"][0]
        gt_norm = ((sample.x1 - mean) / std).unsqueeze(0).to(device)
        z_true = ae.encode(gt_norm).permute(0, 2, 1)          # (1, L, ae_dim)
        # AE-latent scale check: the SiT head's prior is N(0,1), so well-
        # conditioned flow matching wants these ~unit std. Far from 1 ⇒ P2.
        lat_std.append(float(z_true.std()))
        lat_absmean.append(float(z_true.abs().mean()))
        b, L, _ = z_true.shape
        cond = text_encoder.encode([caption], device=device)
        padding_mask = torch.zeros(b, L, dtype=torch.bool, device=device)
        row = []
        for r in ratios:
            num = max(1, int(round(r * L)))
            torch.manual_seed(0)
            perm = torch.randperm(L, device=device)
            is_mask = torch.zeros(b, L, dtype=torch.bool, device=device)
            is_mask[0, perm[:num]] = True
            latents_in = torch.where(is_mask.unsqueeze(-1),
                                     mardm.mask_latent.to(device), z_true)
            pred = mardm.forward_with_CFG(latents_in, cond, padding_mask, cfg=1.0, mask=is_mask)
            err = float((pred[is_mask] - z_true[is_mask]).pow(2).mean().sqrt())
            row.append(err)
            agg[r].append(err)
        print(f"[sweep]   {cid}: " + "  ".join(f"r={r:.2f}:{e:.3f}" for r, e in zip(ratios, row)),
              flush=True)
    print("[sweep]   MEAN: " + "  ".join(f"r={r:.2f}:{float(np.mean(agg[r])):.3f}" for r in ratios),
          flush=True)
    print(f"[sweep]   AE-latent stats: std={float(np.mean(lat_std)):.3f} "
          f"abs_mean={float(np.mean(lat_absmean)):.3f}  (SiT head wants std≈1; "
          f"far from 1 ⇒ latent-scale bug, P2)", flush=True)


@torch.no_grad()
def _caption_cosine(text_encoder, dataset, zf, device, n_clips: int) -> None:
    """Pairwise cosine of the caption embeddings. If ~0.99 the text encoder
    isn't separating the prompts, so the diffusion stack can't either."""
    caps = [torch.load(io.BytesIO(zf.read(f"{dataset[i].clip_id}.pt")),
                       weights_only=False)["texts"][0] for i in range(n_clips)]
    E = text_encoder.encode(caps, device=device)                 # (N, text_dim)
    E = E / E.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cos = (E @ E.T).cpu()
    off = cos[~torch.eye(cos.shape[0], dtype=torch.bool)]
    print(f"\n[cond] caption embedding pairwise cosine: min={off.min():.3f} "
          f"mean={off.mean():.3f} max={off.max():.3f}", flush=True)
    print("[cond]   (≈0.99 ⇒ Qwen3 pooling is non-discriminative — text encoder "
          "is the bottleneck, not the diffusion stack)", flush=True)
    for c in caps:
        print(f"[cond]   - {c[:72]}", flush=True)


@torch.no_grad()
def _ae_reconstruct(ae, dataset, zf, mean, std, device, n_clips: int,
                    out_dir: Path, fps: int) -> None:
    """Render GT vs decode(encode(GT)) for a trained AE — a pure reconstruction
    sanity check, no generation branch. Prints per-clip MPJPE (global + pose-only
    local) and writes real-<cid>.gif / aerecon-<cid>.gif."""
    locs, globs = [], []
    for idx in range(n_clips):
        sample = dataset[idx]
        cid = sample.clip_id
        gt_ess = sample.x1
        gt_joints = recover_joints_from_ric(gt_ess).numpy()
        gt_norm = ((gt_ess - mean) / std).unsqueeze(0).to(device)
        recon_ess = denormalize(ae.decode(ae.encode(gt_norm))[0].cpu(), mean, std)
        recon_joints = recover_joints_from_ric(recon_ess).numpy()
        m = min(gt_joints.shape[0], recon_joints.shape[0])
        gt_m, rc_m = gt_joints[:m], recon_joints[:m]
        g = float(np.linalg.norm(gt_m - rc_m, axis=-1).mean())
        loc = float(np.linalg.norm((gt_m - gt_m[:, 0:1]) - (rc_m - rc_m[:, 0:1]), axis=-1).mean())
        globs.append(g); locs.append(loc)
        print(f"[ae-recon] {cid}  L={gt_ess.shape[0]:3d}  MPJPE={g:.4f}  local={loc:.4f}", flush=True)
        _render(gt_joints, out_dir / f"real-{cid}.gif", title=f"GT [{cid}]", fps=fps)
        _render(recon_joints, out_dir / f"aerecon-{cid}.gif",
                title=f"AE-RECON [{cid}] local={loc:.3f}", fps=fps)
    if locs:
        print(f"[ae-recon] mean MPJPE={float(np.mean(globs)):.4f}  "
              f"mean local={float(np.mean(locs)):.4f}  (local≈0 ⇒ AE reproduces the pose; "
              f"global inflated by root drift on locomotion)", flush=True)


@torch.no_grad()
def main_impl(cfg: DictConfig) -> None:
    set_seed(int(cfg.viz.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir) / "viz"
    print(f"[viz] device={device}  out_dir={out_dir}", flush=True)

    mean, std = _load_stats(cfg.stats_path)
    ae = _load_frozen_ae(cfg, device)

    dataset = EssentialDataset(
        root=cfg.data.root, split=str(cfg.viz.get("split", "train")),
        subset_frac=cfg.get("subset_frac"), limit_clips=cfg.get("limit_clips"),
        window_size=None, min_seq_len=cfg.data.min_seq_len, max_seq_len=cfg.data.max_seq_len,
        canonical_dir=cfg.data.get("canonical_dir"),
    )
    n_render = min(len(dataset), int(cfg.viz.num_clips))
    zf = zipfile.ZipFile(Path(cfg.data.root) / cfg.data.zip_name)

    # AE-only mode: render GT vs decode(encode(GT)); no generation branch needed.
    if bool(cfg.viz.get("ae_only", False)):
        print(f"[viz] AE-only reconstruction check on {n_render} clips "
              f"(split={cfg.viz.get('split', 'train')})", flush=True)
        _ae_reconstruct(ae, dataset, zf, mean, std, device, n_render, out_dir, int(cfg.viz.fps))
        return

    if cfg.viz.checkpoint in (None, "", "???"):
        raise ValueError("set +viz.checkpoint=<path/to/gen latest.pt> (or +viz.ae_only=true)")
    print(f"[viz] guidance={float(cfg.viz.guidance)}  timesteps={int(cfg.viz.timesteps)}  "
          f"(cond_drop_prob={cfg.model.cond_drop_prob}: use guidance=1.0 if it was 0)", flush=True)
    text_encoder = _build_text_encoder(cfg)
    mardm = _load_mardm(cfg, cfg.viz.checkpoint, ae, device)
    ds_rate = ae.downsample_rate
    print(f"[viz] {len(dataset)} subset clips; rendering first {n_render}", flush=True)

    if bool(cfg.viz.sweep):
        _recon_sweep(mardm, ae, text_encoder, dataset, zf, mean, std, device,
                     n_clips=min(n_render, int(cfg.viz.sweep_clips)),
                     ratios=[0.1, 0.25, 0.5, 0.75, 1.0])
        _caption_cosine(text_encoder, dataset, zf, device, n_render)

    manifest: list[dict] = []
    metrics: dict[str, dict] = {}
    gen_pools: list[torch.Tensor] = []   # per-clip time-mean of gen latents
    true_pools: list[torch.Tensor] = []  # ...and of the true latents
    for idx in range(n_render):
        sample = dataset[idx]
        cid = sample.clip_id
        gt_ess = sample.x1                                   # (L, 67) raw essential
        L = gt_ess.shape[0]
        caption = torch.load(io.BytesIO(zf.read(f"{cid}.pt")),
                             weights_only=False)["texts"][0]

        gt_joints = recover_joints_from_ric(gt_ess).numpy()  # (L, 22, 3)

        # Sample the model on this clip's caption at its (latent-rounded) length.
        latent_len = max(1, L // ds_rate)
        cond = text_encoder.encode([caption], device=device)
        latents = mardm.generate(
            cond, m_lens=torch.tensor([latent_len], device=device),
            timesteps=int(cfg.viz.timesteps), cond_scale=float(cfg.viz.guidance),
        )
        # --- latent-space collapse probe (isolates gen branch from the AE) ---
        # Compare the GENERATED latents against the AE's TRUE latents for this
        # clip and against the learned mask token. If gen latents sit on the
        # mask token (rmse_mask≈0) and have ~no temporal variance, the masked-AR
        # decode never moved off the mask → collapse, not an AE problem.
        gt_norm = ((gt_ess - mean) / std).unsqueeze(0).to(device)         # (1, L, 67)
        z_true = ae.encode(gt_norm)                                       # (1, ae_dim, Lz)
        Lz = min(latents.shape[-1], z_true.shape[-1])
        mask_tok = mardm.mask_latent.detach().reshape(1, -1, 1).to(device)  # (1, ae_dim, 1)
        rmse_true = float((latents[..., :Lz] - z_true[..., :Lz]).pow(2).mean().sqrt())
        rmse_mask = float((latents - mask_tok).pow(2).mean().sqrt())
        gen_tvar = float(latents.std(dim=-1).mean())   # temporal spread of gen latents
        true_tvar = float(z_true.std(dim=-1).mean())   # ...vs the real target's
        gen_pools.append(latents.mean(dim=-1).flatten().cpu())
        true_pools.append(z_true.mean(dim=-1).flatten().cpu())
        print(f"[probe] {cid}  rmse(gen,true)={rmse_true:.3f}  rmse(gen,mask)={rmse_mask:.3f}  "
              f"tvar gen/true={gen_tvar:.3f}/{true_tvar:.3f}", flush=True)

        pred_ess = denormalize(ae.decode(latents)[0].cpu(), mean, std)   # (latent_len*ds, 67)
        pred_joints = recover_joints_from_ric(pred_ess).numpy()
        # AE reconstruction (decode∘encode of GT, NO diffusion) = the decode floor.
        # gen ≈ true latents, so gen joints can't beat this. If aerecon wiggles
        # too, the AE/essential→joints path is the culprit, not the diffusion.
        recon_ess = denormalize(ae.decode(z_true)[0].cpu(), mean, std)
        recon_joints = recover_joints_from_ric(recon_ess).numpy()
        mr = min(gt_joints.shape[0], recon_joints.shape[0])
        recon_mpjpe = float(np.linalg.norm(gt_joints[:mr] - recon_joints[:mr], axis=-1).mean())

        m = min(gt_joints.shape[0], pred_joints.shape[0])
        gt_m, pred_m = gt_joints[:m], pred_joints[:m]
        mpjpe = float(np.linalg.norm(gt_m - pred_m, axis=-1).mean())
        # Root-relative (pose-only): subtract joint-0 each frame so global
        # trajectory drift (integrated root velocity) doesn't dominate. This
        # isolates "did it memorize the pose" from "did the root trajectory drift".
        gt_rel = gt_m - gt_m[:, 0:1, :]
        pred_rel = pred_m - pred_m[:, 0:1, :]
        mpjpe_local = float(np.linalg.norm(gt_rel - pred_rel, axis=-1).mean())
        metrics[cid] = {"mpjpe": mpjpe, "mpjpe_local": mpjpe_local}
        print(f"[viz] {cid}  L={L:3d}  MPJPE={mpjpe:.4f}  local={mpjpe_local:.4f}  "
              f"cap={caption[:46]!r}", flush=True)

        print(f"[aerecon] {cid}  mpjpe(aerecon,gt)={recon_mpjpe:.4f}  "
              f"(decode∘encode floor; gen can't beat this)", flush=True)

        gt_gif = _render(gt_joints, out_dir / f"real-{cid}.gif",
                         title=f"GT [{cid}] {caption[:50]}", fps=int(cfg.viz.fps))
        recon_gif = _render(recon_joints, out_dir / f"aerecon-{cid}.gif",
                            title=f"AE-RECON [{cid}] mpjpe={recon_mpjpe:.3f}", fps=int(cfg.viz.fps))
        pred_gif = _render(pred_joints, out_dir / f"gen-{cid}.gif",
                           title=f"PRED [{cid}] mpjpe={mpjpe:.3f}", fps=int(cfg.viz.fps))
        manifest.append({"file": gt_gif.name, "kind": "gt", "clip_id": cid, "caption": caption})
        manifest.append({"file": recon_gif.name, "kind": "aerecon", "clip_id": cid})
        manifest.append({"file": pred_gif.name, "kind": "pred", "clip_id": cid,
                         "caption": caption, "mpjpe": mpjpe})

    if metrics:
        glob = np.array([v["mpjpe"] for v in metrics.values()])
        loc = np.array([v["mpjpe_local"] for v in metrics.values()])
        summary = {"per_clip": metrics, "n_clips": len(metrics),
                   "mean_mpjpe": float(glob.mean()), "median_mpjpe": float(np.median(glob)),
                   "mean_mpjpe_local": float(loc.mean()), "median_mpjpe_local": float(np.median(loc))}
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"\n[viz] mean MPJPE        = {summary['mean_mpjpe']:.4f}  (global; "
              f"inflated by root drift on locomotion)", flush=True)
        print(f"[viz] mean MPJPE local  = {summary['mean_mpjpe_local']:.4f}  (pose-only; "
              f"the real memorization signal — lower = better)", flush=True)

    # Inter-clip diversity: if generation collapses to one output regardless of
    # caption, gen spread << true spread (every GIF looks the same).
    if len(gen_pools) > 1:
        def _mean_pdist(rows: list[torch.Tensor]) -> float:
            X = torch.stack(rows)
            d = torch.cdist(X, X)
            n = X.shape[0]
            return float(d[~torch.eye(n, dtype=torch.bool)].mean())
        g, t = _mean_pdist(gen_pools), _mean_pdist(true_pools)
        print(f"[cond] inter-clip latent spread (mean pairwise L2): gen={g:.3f} true={t:.3f}  "
              f"ratio={g / max(t, 1e-6):.2f}", flush=True)
        print("[cond]   (gen ≪ true ⇒ generation collapses to a caption-independent "
              "output — the 'every GIF looks the same' symptom)", flush=True)
    print(f"[viz] done — GIFs + metrics under {out_dir}", flush=True)


@hydra.main(config_path="../configs", config_name="gen", version_base=None)
def main(cfg: DictConfig) -> None:
    viz_cfg = OmegaConf.create({
        "checkpoint": "???",   # required: gen-branch checkpoint
        "num_clips": 8,        # how many subset clips to render/score
        "guidance": 1.0,       # CFG scale; 1.0 = pure conditional. A cond_drop_prob=0
                               # model has an UNTRAINED unconditional branch, so >1 mixes
                               # in garbage — keep 1.0 unless the model trained with CFG dropout.
        "timesteps": 18,       # masked-AR sampling iterations
        "use_ema": True,
        "fps": 20,
        "seed": 0,
        "sweep": True,         # run the teacher-forced mask-ratio reconstruction probe
        "sweep_clips": 4,      # how many clips to average the sweep over
        "ae_only": False,      # render only GT vs decode(encode(GT)) — no gen needed
        "split": "train",      # which split to pull clips from (ae_only / rendering)
    })
    cfg.viz = OmegaConf.merge(viz_cfg, cfg.get("viz", OmegaConf.create({})))
    main_impl(cfg)


if __name__ == "__main__":
    main()
