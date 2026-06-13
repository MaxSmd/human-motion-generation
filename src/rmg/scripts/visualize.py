"""Visualize human motion — either real clips from the packed dataset or
text-conditioned samples from a trained checkpoint.

Modes:
  mode=clip   +viz.clips='000021,000019,000022'
              → renders each packed clip's stored (translation, quats) as MP4

  mode=prompt +viz.checkpoint=... +viz.prompts='a person walks forward|sits down'
              → loads checkpoint, samples via the Riemannian Euler ODE with CFG,
                renders each sampled motion as MP4

  mode=compare +viz.checkpoint=... +viz.clips='000021,000019'
              → for each clip, renders BOTH the GT motion and the model's
                prediction conditioned on that clip's own caption (matched to
                the GT length). Outputs a paired real-<cid> / gen-<cid> per clip
                so GT vs prediction can be shown side by side. This is the mode
                for "what did the model learn on the clips it actually saw."

  mode=samples +viz.samples_file='runs/<run>/samples'
              → renders the training-time sample dumps (the 3 fixed prompts the
                trainer generates every sample_every steps). No checkpoint /
                model / GPU needed — the motions are already in the .pt file;
                we just decode → FK → render. Point samples_file at a DIRECTORY
                to render every step-*.pt inside (step-sorted), or pass one/more
                .pt paths comma-separated to pick specific steps.

Outputs land under `${output_dir}/viz/`. Both paths go through
`rmg.representation.forward_kinematics` so the rendered skeleton uses the same
HumanML3D-convention FK the evaluator expects.

Run via `slurm/rmg/viz.sbatch`; not designed for local CPU use (Qwen3 text
encoder is heavy, and MP4 export needs ffmpeg in the container).
"""

from __future__ import annotations

import io
import json
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
    CONSTRAINABLE_REPRESENTATIONS,
    RiemannianEulerSampler,
    SamplerCfg,
    WrappedGaussianPrior,
    build_inpaint_targets,
    parse_constraints,
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
from shared.utils import EMA, load_checkpoint, set_seed  # noqa: E402


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
    return gif_path


def _write_manifest(out_dir: Path, entries: list[dict]) -> None:
    """Write a sidecar `manifest.json` describing every rendered file so the
    app can label each clip with its TRUE caption / clip id / kind instead of
    guessing from the filename. This is what keeps the viewer aligned: the
    backend reads this verbatim rather than parsing stems. Each entry is
    `{file, kind, clip_id, caption, step}` (clip_id/step optional)."""
    if not entries:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(entries, indent=2))
    print(f"[visualize] wrote manifest.json ({len(entries)} entries)", flush=True)


def _subset_train_ids(
    data_root: Path, splits_name: str, fraction: float, seed: int,
) -> list[str]:
    """Replay HumanML3DDataset's train-subset selection and return the sorted
    regular (non-mirror) clip IDs the model trained on. Identical logic to the
    dataset + mode=info so picks are consistent across the codebase."""
    import json
    import random as _random
    with open(data_root / splits_name) as f:
        splits = json.load(f)
    regular = [c for c in splits["train"] if not c.startswith("M")]
    n_keep = max(1, int(round(len(regular) * fraction)))
    rng = _random.Random(seed)
    keep_reg = set(rng.sample(regular, k=n_keep))
    return sorted(keep_reg)


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
    use_ema = bool(cfg.viz.get("use_ema", True))
    if use_ema and state.ema is not None:
        ema = EMA(model, decay=0.0)
        ema.load_state_dict(state.ema)
        ema.copy_to(model)
        print(f"[visualize] loaded EMA at step {state.step}", flush=True)
    elif not use_ema:
        print(f"[visualize] using LIVE (non-EMA) weights at step {state.step}", flush=True)
    else:
        print(f"[visualize] loaded live weights at step {state.step}", flush=True)
    model.eval()
    return model


def _build_sampler(cfg, representation, skel) -> RiemannianEulerSampler:
    """Riemannian Euler ODE sampler with CFG, sharing the training prior."""
    M = representation.build_manifold()
    if hasattr(representation, "prior_mu_from_skeleton"):
        mu = representation.prior_mu_from_skeleton(skel)
    else:
        mu = representation.prior_mu()
    prior = WrappedGaussianPrior(M, mu, sigma=cfg.train.prior_sigma)
    return RiemannianEulerSampler(
        manifold=M, prior=prior,
        cfg=SamplerCfg(
            num_steps=int(cfg.viz.num_sample_steps),
            guidance_scale=float(cfg.viz.guidance_scale),
        ),
    )


def _load_constraint_specs(cfg) -> list[dict]:
    """Sampling-time joint-angle constraints for mode=prompt. Read from a JSON
    env var `RMG_CONSTRAINTS` (set by the app's viz-job submitter — avoids
    quoting a structured list through the Hydra CLI), falling back to
    `cfg.viz.constraints` if present. Returns [] when none."""
    import json
    import os

    raw = os.environ.get("RMG_CONSTRAINTS", "").strip()
    if raw:
        try:
            specs = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"RMG_CONSTRAINTS is not valid JSON: {e}") from e
    else:
        specs = cfg.viz.get("constraints") if "constraints" in cfg.viz else None
        specs = OmegaConf.to_container(specs, resolve=True) if specs is not None else []
    return list(specs or [])


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    viz_cfg = OmegaConf.create({
        "mode": "clip",                                # clip | prompt | compare | samples | info
        "checkpoint": "???",                           # required for prompt/compare
        "samples_file": "",                            # required for mode=samples (one or more step-*.pt, comma-separated)
        "clips": "000021,000019,000022,000026",        # comma-separated
        "prompts": "a person walks forward in a circle"
                   "|a person sits down on the floor"
                   "|a person waves their left hand"
                   "|a person does jumping jacks",
        "num_frames": 100,                             # length of sampled motion
        "num_sample_steps": 50,
        "guidance_scale": 6.5,
        "use_ema": True,                               # prompt/compare: EMA vs live weights.
                                                       # At low step counts EMA still carries
                                                       # heavy random-init weight (decay 0.9999
                                                       # ⇒ ~7k-step half-life) — set False to
                                                       # see the actual trained weights.
        "fps": 20,
        "seed": 0,
        # For mode=info — replay the train subset selection logic so we can
        # tell whether the model saw a given clip during 1%-subset training.
        # MUST match the values used when launching training.
        "subset_fraction": 0.01,
        "subset_seed": 0,
        "list_subset_n": 30,                            # how many subset clips to dump
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

    if cfg.viz.mode == "info":
        # ---- Report split membership + subset hits for queried clip IDs ----
        import json
        import random as _random
        with open(data_root / cfg.data.splits_name) as f:
            splits = json.load(f)

        # Replay HumanML3DDataset's subset selection exactly.
        sub_frac = float(cfg.viz.subset_fraction)
        sub_seed = int(cfg.viz.subset_seed)
        train_ids = list(splits["train"])
        regular = [c for c in train_ids if not c.startswith("M")]
        n_keep = max(1, int(round(len(regular) * sub_frac)))
        rng = _random.Random(sub_seed)
        keep_reg = set(rng.sample(regular, k=n_keep))
        subset_train = set(keep_reg) | {f"M{c}" for c in keep_reg}

        print(
            f"\nsplit sizes: train={len(train_ids)} val={len(splits['val'])} "
            f"test={len(splits['test'])}", flush=True,
        )
        print(
            f"train subset (fraction={sub_frac}, seed={sub_seed}): "
            f"{len(subset_train)} clips ({len(keep_reg)} regular + their mirrors)",
            flush=True,
        )

        queries = [c.strip() for c in str(cfg.viz.clips).split(",") if c.strip()]
        if queries:
            print(f"\nquery results — did the model see these during training?")
            print(f"  {'clip_id':<12}  {'split':<6}  {'in_subset':<10}  verdict", flush=True)
            print(f"  {'-' * 60}", flush=True)
            for cid in queries:
                where = next((s for s in ("train", "val", "test")
                              if cid in splits[s]), "NOT_FOUND")
                in_sub = cid in subset_train
                saw = (where == "train") and in_sub
                tag = "MODEL SAW" if saw else (
                    "in train but not subset" if where == "train" else f"never (in {where})"
                )
                print(f"  {cid:<12}  {where:<6}  {str(in_sub):<10}  {tag}", flush=True)

        # Dump a few subset clips' captions so user can pick prompts/clips
        # the model definitely overfit on.
        n_list = int(cfg.viz.list_subset_n)
        if n_list > 0:
            sample_ids = sorted(c for c in subset_train if not c.startswith("M"))[:n_list]
            print(f"\nfirst {len(sample_ids)} regular clips in the train subset "
                  f"(model HAS trained on these):", flush=True)
            zip_path = data_root / "humanml3d.zip"
            with zipfile.ZipFile(zip_path) as zf:
                for cid in sample_ids:
                    try:
                        blob = torch.load(io.BytesIO(zf.read(f"{cid}.pt")),
                                          weights_only=False)
                        cap = blob["texts"][0][:80]
                        n_caps = len(blob["texts"])
                        T = blob["translation"].shape[0]
                        print(f"  {cid}  T={T:3d}  caps={n_caps}  cap[0]={cap!r}",
                              flush=True)
                    except KeyError:
                        print(f"  {cid}: (not present in zip)", flush=True)
        return

    if cfg.viz.mode == "clip":
        # ---- Real packed clips → FK → render ----
        clip_ids = [c.strip() for c in str(cfg.viz.clips).split(",") if c.strip()]
        manifest: list[dict] = []
        for cid in clip_ids:
            try:
                translation, quats, caption = _load_real_clip(data_root, cid)
            except KeyError:
                print(f"[visualize] clip {cid!r} not in packed zip — skipping", flush=True)
                continue
            joints = forward_kinematics(skel, quats, translation).numpy()
            gif = _render(joints, out_dir / f"real-{cid}.mp4",
                          title=f"[{cid}] {caption[:60]}", fps=int(cfg.viz.fps))
            manifest.append({"file": gif.name, "kind": "gt", "clip_id": cid, "caption": caption})
        _write_manifest(out_dir, manifest)
        return

    if cfg.viz.mode == "prompt":
        # ---- Text-conditioned sampling from a checkpoint ----
        if cfg.viz.checkpoint in (None, "", "???"):
            raise ValueError("mode=prompt requires +viz.checkpoint=<path/to/latest.pt>")

        text_encoder = _build_text_encoder(cfg)
        model = _build_model(cfg, representation, device)
        sampler = _build_sampler(cfg, representation, skel)

        prompts = [p.strip() for p in str(cfg.viz.prompts).split("|") if p.strip()]
        n_frames = int(cfg.viz.num_frames)
        print(f"[visualize] sampling {len(prompts)} prompts "
              f"({n_frames} frames, "
              f"{int(cfg.viz.num_sample_steps)} ODE steps, "
              f"ω={float(cfg.viz.guidance_scale)})", flush=True)

        # Optional sampling-time joint-angle pins (broadcast across the batch).
        fixed_values = fixed_mask = None
        specs = _load_constraint_specs(cfg)
        if specs:
            if cfg.representation.name not in CONSTRAINABLE_REPRESENTATIONS:
                raise ValueError(
                    f"joint-angle constraints need a quaternion representation "
                    f"({CONSTRAINABLE_REPRESENTATIONS}); got {cfg.representation.name!r}."
                )
            fixed_values, fixed_mask = build_inpaint_targets(
                parse_constraints(specs), num_frames=n_frames,
                num_joints=int(representation.num_joints), device=device,
            )
            print(f"[visualize] applying {len(specs)} joint-angle constraint(s): "
                  f"{specs}", flush=True)

        with torch.no_grad():
            cond = text_encoder.encode(prompts, device=device)
            samples = sampler.sample(
                model, shape=(len(prompts), n_frames), cond=cond,
                fixed_values=fixed_values, fixed_mask=fixed_mask,
            )                                                # (B, T, ambient_dim)

        manifest = []
        for i, prompt in enumerate(prompts):
            tpr = tplusr_decode(samples[i])
            joints = forward_kinematics(
                skel, tpr.quaternions.float(), tpr.translation.float()
            ).cpu().numpy()
            safe = "".join(c if c.isalnum() else "_" for c in prompt)[:48]
            gif = _render(joints, out_dir / f"gen-{i:02d}-{safe}.mp4",
                          title=prompt[:60], fps=int(cfg.viz.fps))
            manifest.append({"file": gif.name, "kind": "pred", "caption": prompt})
        _write_manifest(out_dir, manifest)
        return

    if cfg.viz.mode == "compare":
        # ---- GT clip vs model prediction on that clip's own caption ----
        # For each clip: render the stored GT motion, then condition the model
        # on the clip's caption and sample at the GT's frame count so the two
        # GIFs are directly comparable.
        if cfg.viz.checkpoint in (None, "", "???"):
            raise ValueError("mode=compare requires +viz.checkpoint=<path/to/latest.pt>")

        clips_raw = str(cfg.viz.clips).strip()
        if clips_raw.lower() in ("", "auto"):
            # Auto-pick clips the model definitely trained on — no need to run
            # mode=info first. Uses the same subset_fraction/subset_seed as training.
            n_pick = max(1, int(cfg.viz.get("list_subset_n", 4)))
            n_pick = min(n_pick, 6)   # keep the render job bounded
            clip_ids = _subset_train_ids(
                data_root, cfg.data.splits_name,
                float(cfg.viz.subset_fraction), int(cfg.viz.subset_seed),
            )[:n_pick]
            print(f"[visualize] compare auto-picked {len(clip_ids)} in-subset clips "
                  f"(fraction={cfg.viz.subset_fraction}, seed={cfg.viz.subset_seed}): "
                  f"{clip_ids}", flush=True)
        else:
            clip_ids = [c.strip() for c in clips_raw.split(",") if c.strip()]
        if not clip_ids:
            raise ValueError("mode=compare requires +viz.clips='<id>,...' or 'auto'")

        text_encoder = _build_text_encoder(cfg)
        model = _build_model(cfg, representation, device)
        sampler = _build_sampler(cfg, representation, skel)

        manifest = []
        for cid in clip_ids:
            try:
                translation, quats, caption = _load_real_clip(data_root, cid)
            except KeyError:
                print(f"[visualize] clip {cid!r} not in packed zip — skipping", flush=True)
                continue

            # Cap length at the model's max_seq_len — the DiT can't process
            # longer sequences, and training crops clips to <= max_seq_len
            # anyway. Crop the GT to the same length so GT vs PRED are aligned.
            max_T = int(cfg.model.max_seq_len)
            full_T = int(translation.shape[0])
            n_frames = min(full_T, max_T)
            if full_T > max_T:
                print(f"[visualize] compare {cid}: clip is {full_T} frames > "
                      f"max_seq_len {max_T} — cropping to first {max_T}", flush=True)
            translation = translation[:n_frames]
            quats = quats[:n_frames]

            # GT (cropped to n_frames).
            gt_joints = forward_kinematics(skel, quats, translation).numpy()
            gt_gif = _render(gt_joints, out_dir / f"real-{cid}.mp4",
                             title=f"GT [{cid}] {caption[:55]}", fps=int(cfg.viz.fps))
            manifest.append({"file": gt_gif.name, "kind": "gt", "clip_id": cid, "caption": caption})

            # Prediction: same caption, matched (capped) length.
            print(f"[visualize] compare {cid}: sampling {n_frames} frames for "
                  f"caption {caption[:60]!r}", flush=True)
            with torch.no_grad():
                cond = text_encoder.encode([caption], device=device)
                samples = sampler.sample(model, shape=(1, n_frames), cond=cond)
            tpr = tplusr_decode(samples[0])
            pred_joints = forward_kinematics(
                skel, tpr.quaternions.float(), tpr.translation.float()
            ).cpu().numpy()
            pred_gif = _render(pred_joints, out_dir / f"gen-{cid}.mp4",
                               title=f"PRED [{cid}] {caption[:55]}", fps=int(cfg.viz.fps))
            manifest.append({"file": pred_gif.name, "kind": "pred", "clip_id": cid, "caption": caption})
        _write_manifest(out_dir, manifest)
        return

    if cfg.viz.mode == "samples":
        # ---- Training-time sample dumps (no model needed) ----
        # Each step-*.pt is {"texts": [...], "samples": (B, T, ambient_dim)} of
        # EMA generations the trainer saved. Decode → FK → render per prompt.
        #
        # `samples_file` accepts any of:
        #   - a directory          → renders ALL step-*.pt inside, step-sorted
        #   - one or more .pt paths → comma-separated
        raw = [f.strip() for f in str(cfg.viz.samples_file).split(",") if f.strip()]
        if not raw:
            raise ValueError(
                "mode=samples requires +viz.samples_file=<dir | path/to/step-*.pt> "
                "(a directory renders every step-*.pt inside; comma-separate paths "
                "to pick specific steps)"
            )
        files: list[Path] = []
        for entry in raw:
            p = Path(entry).expanduser()
            if p.is_dir():
                found = sorted(p.glob("step-*.pt"))
                if not found:
                    print(f"[visualize] no step-*.pt files under {p} — skipping",
                          flush=True)
                files.extend(found)
            else:
                files.append(p)
        if not files:
            raise ValueError(
                f"mode=samples: no .pt files resolved from {cfg.viz.samples_file!r}"
            )
        print(f"[visualize] rendering {len(files)} sample dump(s)", flush=True)
        manifest = []
        for p in files:
            if not p.exists():
                print(f"[visualize] samples file {p} not found — skipping", flush=True)
                continue
            blob = torch.load(p, map_location="cpu", weights_only=False)
            texts = blob["texts"]
            samples = blob["samples"]                     # (B, T, ambient_dim)
            step_tag = p.stem                             # e.g. "step-000000500"
            try:
                step = int(step_tag.split("-")[1])
            except (IndexError, ValueError):
                step = None
            print(f"[visualize] {step_tag}: {samples.shape[0]} prompts, "
                  f"{samples.shape[1]} frames", flush=True)
            for i, text in enumerate(texts):
                tpr = tplusr_decode(samples[i].float())
                joints = forward_kinematics(
                    skel, tpr.quaternions.float(), tpr.translation.float()
                ).cpu().numpy()
                safe = "".join(c if c.isalnum() else "_" for c in text)[:40]
                gif = _render(joints, out_dir / f"{step_tag}-{i:02d}-{safe}.mp4",
                              title=f"[{step_tag}] {text[:55]}", fps=int(cfg.viz.fps))
                manifest.append({"file": gif.name, "kind": "sample", "caption": text, "step": step})
        _write_manifest(out_dir, manifest)
        return

    raise ValueError(f"unknown viz.mode {cfg.viz.mode!r}")


if __name__ == "__main__":
    main()
