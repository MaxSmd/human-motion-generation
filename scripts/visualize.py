"""Visualize human motion — either real clips from the packed dataset or
text-conditioned samples from a trained checkpoint.

Modes:
  mode=clip   +viz.clips='000021,000019,000022'
              → renders each packed clip's stored (translation, quats) as MP4

  mode=prompt +viz.checkpoint=... +viz.prompts='a person walks forward|sits down'
              → loads checkpoint, samples via the Riemannian Euler ODE with CFG,
                renders each sampled motion as MP4

Outputs land under `${output_dir}/viz/`. Both paths go through
`rmg.representation.forward_kinematics` so the rendered skeleton uses the same
HumanML3D-convention FK the evaluator expects.

Run via `slurm/visualize.sbatch`; not designed for local CPU use (Qwen3 text
encoder is heavy, and MP4 export needs ffmpeg in the container).
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# Non-interactive backend BEFORE plot_3d_motion's pyplot import.
import matplotlib
matplotlib.use("Agg")

REPO = Path(__file__).resolve().parent.parent

from rmg.flow import (  # noqa: E402
    RiemannianEulerSampler,
    SamplerCfg,
    WrappedGaussianPrior,
)
from rmg.models import (  # noqa: E402
    DiTConfig,
    Qwen3EmbeddingEncoder,
    RandomTextEncoder,
    RMGDiT,
)
from rmg.representation import (  # noqa: E402
    Skeleton,
    build_representation,
    forward_kinematics,
)
from rmg.representation.tplusr import decode as tplusr_decode  # noqa: E402
from rmg.utils import EMA, load_checkpoint, set_seed  # noqa: E402


# HumanML3D 22-joint kinematic chains (same as upstream's t2m_kinematic_chain).
_T2M_CHAINS: tuple[tuple[int, ...], ...] = (
    (0, 2, 5, 8, 11),       # right leg
    (0, 1, 4, 7, 10),       # left leg
    (0, 3, 6, 9, 12, 15),   # spine + head
    (9, 14, 17, 19, 21),    # right arm
    (9, 13, 16, 18, 20),    # left arm
)


def _render(joints: np.ndarray, save_path: Path, title: str, fps: int) -> None:
    """Render (T, 22, 3) joint positions to GIF (via Pillow — no ffmpeg).

    Also dumps the raw joints next to the GIF as `<stem>.npy` so the same
    motion can be re-rendered to MP4 locally with a system ffmpeg if you want.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Always save joints — cheap insurance, useful for local re-render.
    npy_path = save_path.with_suffix(".npy")
    np.save(npy_path, joints.astype(np.float32))

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    T = joints.shape[0]
    # Shared axis limits so the camera doesn't jitter between frames.
    pts = joints.reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = (lo + hi) / 2
    radius = float(np.max(hi - lo)) / 2 * 1.1 + 1e-3

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    chain_lines = [ax.plot([], [], [], "-o", linewidth=2, markersize=3)[0]
                   for _ in _T2M_CHAINS]
    title_text = ax.set_title("")

    def _setup_axes():
        # HumanML3D is Y-up; matplotlib's 3D viewer treats Z as up by default,
        # so swap Y↔Z for display.
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[2] - radius, center[2] + radius)
        ax.set_zlim(center[1] - radius, center[1] + radius)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("y")
        ax.view_init(elev=15, azim=-70)

    def update(t):
        _setup_axes()
        for line, chain in zip(chain_lines, _T2M_CHAINS):
            xs = joints[t, list(chain), 0]
            ys = joints[t, list(chain), 2]   # Y/Z swap for display
            zs = joints[t, list(chain), 1]
            line.set_data(xs, ys)
            line.set_3d_properties(zs)
        title_text.set_text(f"{title}\nframe {t + 1}/{T}")
        return chain_lines + [title_text]

    ani = FuncAnimation(fig, update, frames=T, interval=1000 // fps, blit=False)
    # Force GIF extension regardless of caller (PillowWriter doesn't do MP4).
    gif_path = save_path.with_suffix(".gif")
    ani.save(str(gif_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"[visualize] wrote {gif_path}  (+ joints at {npy_path.name})", flush=True)


def _load_real_clip(data_root: Path, clip_id: str) -> tuple[torch.Tensor, torch.Tensor, str]:
    zip_path = data_root / "humanml3d.zip"
    with zipfile.ZipFile(zip_path) as zf:
        blob = torch.load(io.BytesIO(zf.read(f"{clip_id}.pt")), weights_only=False)
    return blob["translation"].float(), blob["quats"].float(), blob["texts"][0]


def _build_text_encoder(cfg: DictConfig):
    t = cfg.text_encoder.type
    if t == "qwen3":
        return Qwen3EmbeddingEncoder(
            model_name=cfg.text_encoder.model_name,
            cache_dir=cfg.text_encoder.cache_dir,
            max_length=cfg.text_encoder.max_length,
        )
    return RandomTextEncoder(text_dim=cfg.text_encoder.text_dim)


def _build_model(cfg: DictConfig, representation, device) -> RMGDiT:
    dit_cfg = DiTConfig(
        input_dim=representation.ambient_dim,
        hidden_dim=cfg.model.hidden_dim,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        ffn_mult=cfg.model.ffn_mult,
        text_dim=cfg.model.text_dim,
        time_freq_dim=cfg.model.time_freq_dim,
        max_seq_len=cfg.model.max_seq_len,
    )
    model = RMGDiT(dit_cfg).to(device)
    state = load_checkpoint(cfg.viz.checkpoint, map_location=device)
    model.load_state_dict(state.model)
    if state.ema is not None:
        ema = EMA(model, decay=0.0)
        ema.load_state_dict(state.ema)
        ema.copy_to(model)
        print(f"[visualize] loaded EMA at step {state.step}", flush=True)
    else:
        print(f"[visualize] loaded live weights at step {state.step}", flush=True)
    model.eval()
    return model


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    viz_cfg = OmegaConf.create({
        "mode": "clip",                                # clip | prompt
        "checkpoint": "???",                           # required for prompt
        "clips": "000021,000019,000022,000026",        # comma-separated
        "prompts": "a person walks forward in a circle"
                   "|a person sits down on the floor"
                   "|a person waves their left hand"
                   "|a person does jumping jacks",
        "num_frames": 100,                             # length of sampled motion
        "num_sample_steps": 50,
        "guidance_scale": 6.5,
        "fps": 20,
        "seed": 0,
    })
    cfg.viz = OmegaConf.merge(viz_cfg, cfg.get("viz", OmegaConf.create({})))
    set_seed(int(cfg.viz.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir) / "viz"
    print(f"[visualize] mode={cfg.viz.mode}  device={device}  out_dir={out_dir}", flush=True)

    rep_kwargs = {k: v for k, v in dict(cfg.representation).items() if k not in ("name",)}
    representation = build_representation(cfg.representation.name, **rep_kwargs)

    data_root = Path(cfg.data.root)
    target_offsets = torch.load(
        data_root / cfg.data.offsets_name, weights_only=True,
    ).float()
    skel = Skeleton(offsets=target_offsets)

    if cfg.viz.mode == "clip":
        # ---- Real packed clips → FK → render ----
        clip_ids = [c.strip() for c in str(cfg.viz.clips).split(",") if c.strip()]
        for cid in clip_ids:
            try:
                translation, quats, caption = _load_real_clip(data_root, cid)
            except KeyError:
                print(f"[visualize] clip {cid!r} not in packed zip — skipping", flush=True)
                continue
            joints = forward_kinematics(skel, quats, translation).numpy()
            _render(joints, out_dir / f"real-{cid}.mp4",
                    title=f"[{cid}] {caption[:60]}", fps=int(cfg.viz.fps))
        return

    if cfg.viz.mode == "prompt":
        # ---- Text-conditioned sampling from a checkpoint ----
        if cfg.viz.checkpoint in (None, "", "???"):
            raise ValueError("mode=prompt requires +viz.checkpoint=<path/to/latest.pt>")

        text_encoder = _build_text_encoder(cfg)
        model = _build_model(cfg, representation, device)

        M = representation.build_manifold()
        if hasattr(representation, "prior_mu_from_skeleton"):
            mu = representation.prior_mu_from_skeleton(skel)
        else:
            mu = representation.prior_mu()
        prior = WrappedGaussianPrior(M, mu, sigma=cfg.train.prior_sigma)
        sampler = RiemannianEulerSampler(
            manifold=M, prior=prior,
            cfg=SamplerCfg(
                num_steps=int(cfg.viz.num_sample_steps),
                guidance_scale=float(cfg.viz.guidance_scale),
            ),
        )

        prompts = [p.strip() for p in str(cfg.viz.prompts).split("|") if p.strip()]
        print(f"[visualize] sampling {len(prompts)} prompts "
              f"({int(cfg.viz.num_frames)} frames, "
              f"{int(cfg.viz.num_sample_steps)} ODE steps, "
              f"ω={float(cfg.viz.guidance_scale)})", flush=True)

        with torch.no_grad():
            cond = text_encoder.encode(prompts, device=device)
            samples = sampler.sample(
                model, shape=(len(prompts), int(cfg.viz.num_frames)), cond=cond,
            )                                                # (B, T, ambient_dim)

        for i, prompt in enumerate(prompts):
            tpr = tplusr_decode(samples[i])
            joints = forward_kinematics(
                skel, tpr.quaternions.float(), tpr.translation.float()
            ).cpu().numpy()
            safe = "".join(c if c.isalnum() else "_" for c in prompt)[:48]
            _render(joints, out_dir / f"gen-{i:02d}-{safe}.mp4",
                    title=prompt[:60], fps=int(cfg.viz.fps))
        return

    raise ValueError(f"unknown viz.mode {cfg.viz.mode!r}")


if __name__ == "__main__":
    main()
