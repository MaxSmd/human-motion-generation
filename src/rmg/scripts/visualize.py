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
import os
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
    build_bend_projector,
    build_room_energy_fn,
    parse_bends,
    parse_scene,
    place_motion,
    scene_energy_series,
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
from shared.utils import EMA, load_checkpoint, set_seed, write_progress  # noqa: E402
from shared.render import render_joints  # noqa: E402


def _render(joints: np.ndarray, save_path: Path, title: str, fps: int,
            scene: dict | None = None, constraints: list[dict] | None = None,
            energy: dict | None = None):
    """Render (T, 22, 3) joint positions via the shared multi-panel animation.

    Delegates the actual drawing to `shared.render.render_joints` (the same
    renderer the backend uses), which lays out three orthographic panels plus a
    3-D perspective, draws the optional room/obstacle `scene`, and dumps the raw
    joints next to the media as `<stem>.npy`. Produces MP4 when a system ffmpeg
    is present, else a GIF.

    Returns the written media `Path`, or `None` if the motion is entirely
    non-finite (nothing renderable). Partially-corrupt motions still render:
    matplotlib silently breaks line segments at NaN frames, so corruption shows
    up as gaps/missing limbs rather than crashing the whole job.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Guard against non-finite joints (e.g. a sample dump whose decoded motion
    # diverged to NaN/Inf). Compute finiteness up front so we can warn about
    # partial corruption and skip entirely-dead motions before rendering.
    pts = joints.reshape(-1, 3)
    finite_rows = np.isfinite(pts).all(axis=1)
    if not finite_rows.all():
        n_bad = int((~finite_rows).sum())
        print(f"[visualize] WARNING: {save_path.stem} has {n_bad}/{pts.shape[0]} "
              f"non-finite joint positions (NaN/Inf) — the decoded motion is "
              f"corrupt (likely a diverged sample).", flush=True)
    if not finite_rows.any():
        # Still dump the raw joints (NaN/Inf preserved) for inspection, then bail.
        npy_path = save_path.with_suffix(".npy")
        np.save(npy_path, joints.astype(np.float32))
        print(f"[visualize] SKIP {save_path.stem}: motion is entirely non-finite, "
              f"nothing to render (saved raw joints to {npy_path.name}).", flush=True)
        return None

    media_path, npy_path = render_joints(
        joints, save_path, title=title, fps=fps, fmt="auto", scene=scene,
        constraints=constraints, energy=energy)
    print(f"[visualize] wrote {media_path}  (+ joints at {npy_path.name})", flush=True)
    return media_path


def _constraint_viz(bend_specs, num_frames: int) -> list[dict]:
    """Resolve parsed `BendConstraint`s into the flat dicts the renderer's
    constraint panel wants: `{joint, name, min_deg, max_deg, frame_start,
    frame_end}` with the joint index/name resolved and the frame window
    clamped to `num_frames`."""
    from shared.geometry.skeleton import JOINT_NAMES

    out = []
    for c in bend_specs:
        j = c.joint if isinstance(c.joint, int) else JOINT_NAMES.index(c.joint)
        fs, fe = c.frame_window(num_frames)
        out.append({
            "joint": j,
            "name": JOINT_NAMES[j],
            "min_deg": float(c.min_deg),
            "max_deg": float(c.max_deg),
            "frame_start": fs,
            "frame_end": fe,
        })
    return out


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
        time_scale=float(cfg.model.get("time_scale", 1.0)),
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


def _load_specs(cfg, env_var: str, cfg_key: str) -> list[dict]:
    """Sampling-time constraint specs for mode=prompt. Read from a JSON env var
    (set by the app's viz-job submitter — avoids quoting a structured list
    through the Hydra CLI), falling back to `cfg.viz.<cfg_key>`. Returns []
    when none. Used for both fixed-angle constraints and hinge ranges."""
    raw = os.environ.get(env_var, "").strip()
    if raw:
        try:
            specs = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"{env_var} is not valid JSON: {e}") from e
    else:
        specs = cfg.viz.get(cfg_key) if cfg_key in cfg.viz else None
        specs = OmegaConf.to_container(specs, resolve=True) if specs is not None else []
    return list(specs or [])


def _room_guidance(cfg) -> float:
    return float(os.environ.get("RMG_ROOM_GUIDANCE") or cfg.viz.get("room_guidance", 0.0))


def _load_batch() -> list[dict] | None:
    """Fused mode=prompt batch, from JSON env var `RMG_BATCH`: one entry per render,
    each with its OWN prompt + constraints/ranges/scene/num_frames/seed. The app
    fuses every compatible queued viz job into one submission this way, so a whole
    study (e.g. the constraint analysis' 24 cells) renders in ONE job instead of
    taking 24 turns at the back of the SLURM queue.

    Returns None when unset — the per-job env vars (PROMPTS + RMG_CONSTRAINTS/…)
    then drive the run as before, one constraint set broadcast over every prompt."""
    raw = os.environ.get("RMG_BATCH", "").strip()
    if not raw:
        return None
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"RMG_BATCH is not valid JSON: {e}") from e
    if not items:
        raise ValueError("RMG_BATCH is set but empty")
    return list(items)


def _batch_group_key(item: dict) -> str:
    """Items sharing this key sample together in one batched ODE. Everything in it
    is baked into the sampling call itself — the projector and room energy are built
    for a specific constraint set and frame count, and the seed must be set before
    the batch runs — so items differing in any of it need their own pass."""
    return json.dumps([
        item.get("constraints") or [], item.get("ranges") or [], item.get("scene"),
        float(item.get("room_guidance") or 0.0),
        int(item.get("num_frames", 100)), int(item.get("seed", 0)),
    ], sort_keys=True, default=str)


def _build_constraints(cfg, representation, skel, device, item, n_frames):
    """Sampling-time constraints for one group: fixed angles (inpainting) + hinge
    ranges (swing-twist projection) + euclidean room/obstacle guidance & exact spawn
    placement. Returns (project_fn, energy_fn, guidance_weight, constraint_viz,
    scene_obj) — all inert when the item carries no constraints."""
    project_fn = energy_fn = None
    guidance_weight = 0.0
    constraint_viz: list[dict] = []
    c_specs = list(item.get("constraints") or [])
    r_specs = list(item.get("ranges") or [])
    scene_obj = parse_scene(item.get("scene"))
    if not (c_specs or r_specs or scene_obj):
        return project_fn, energy_fn, guidance_weight, constraint_viz, scene_obj
    if cfg.representation.name not in CONSTRAINABLE_REPRESENTATIONS:
        raise ValueError(
            f"constraints need a quaternion representation "
            f"({CONSTRAINABLE_REPRESENTATIONS}); got {cfg.representation.name!r}."
        )
    nj = int(representation.num_joints)
    bend_specs = [*parse_bends(c_specs), *parse_bends(r_specs)]
    if bend_specs:
        project_fn = build_bend_projector(
            bend_specs, skel, num_frames=n_frames, num_joints=nj, device=device)
        constraint_viz = _constraint_viz(bend_specs, n_frames)
        print(f"[visualize] applying {len(c_specs)} bend pin(s) + {len(r_specs)} bend range(s): "
              f"{c_specs} {r_specs}", flush=True)
    if scene_obj:
        guidance_weight = float(item.get("room_guidance") or 0.0)
        if guidance_weight:
            energy_fn = build_room_energy_fn(scene_obj, skel, num_joints=nj)
        print(f"[visualize] room scene: {scene_obj.room} m, {len(scene_obj.objects)} obstacle(s), "
              f"spawn={scene_obj.spawn}, guidance={guidance_weight}", flush=True)
    return project_fn, energy_fn, guidance_weight, constraint_viz, scene_obj


def _load_scene(cfg) -> dict | None:
    """Room/scene dict for mode=prompt — from JSON env var `RMG_SCENE` (set by
    the app) or `cfg.viz.scene`. Returns None when absent."""
    raw = os.environ.get("RMG_SCENE", "").strip()
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"RMG_SCENE is not valid JSON: {e}") from e
    if "scene" in cfg.viz:
        return OmegaConf.to_container(cfg.viz.scene, resolve=True)
    return None


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    viz_cfg = OmegaConf.create({
        "mode": "clip",                                # clip | prompt | compare | samples | info
        "checkpoint": "???",                           # required for prompt/compare
        "samples_file": "",                            # required for mode=samples (one or more step-*.pt, comma-separated)
        "clips": "000021,000019,000022,000026",        # comma-separated
        # mode=compare: clip ids whose GT render the caller already has (the app's
        # GT registry). GT is pure in its clip id, so re-rendering it is wasted GPU
        # time — render only the prediction for these and let the caller pair them.
        "skip_gt": "",                                 # comma-separated
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
        skipped: list[str] = []
        for idx, cid in enumerate(clip_ids):
            write_progress(out_dir, stage="render", inner={"i": idx, "n": len(clip_ids)})
            try:
                translation, quats, caption = _load_real_clip(data_root, cid)
            except KeyError:
                print(f"[visualize] clip {cid!r} not in packed zip — skipping", flush=True)
                skipped.append(cid)
                continue
            joints = forward_kinematics(skel, quats, translation).numpy()
            gif = _render(joints, out_dir / f"real-{cid}.mp4",
                          title=f"[{cid}] {caption[:60]}", fps=int(cfg.viz.fps))
            if gif is None:
                skipped.append(cid)
                continue
            manifest.append({"file": gif.name, "kind": "gt", "clip_id": cid, "caption": caption})
        # Fail loudly if nothing rendered (e.g. all clip ids invalid) rather than
        # exiting 0 with no output dir — that left the app pulling a non-existent
        # viz/ and surfacing a cryptic rsync error. A non-zero exit makes the job
        # show FAILED with this clear reason in the log.
        if not manifest:
            raise ValueError(
                f"no requested clips were found in the packed dataset "
                f"({data_root}): {skipped or clip_ids}. Pick ids from the GT browser."
            )
        write_progress(out_dir, stage="done", complete=True,
                       inner={"i": len(clip_ids), "n": len(clip_ids)})
        _write_manifest(out_dir, manifest)
        return

    if cfg.viz.mode == "prompt":
        # ---- Text-conditioned sampling from a checkpoint ----
        if cfg.viz.checkpoint in (None, "", "???"):
            raise ValueError("mode=prompt requires +viz.checkpoint=<path/to/latest.pt>")

        text_encoder = _build_text_encoder(cfg)
        model = _build_model(cfg, representation, device)
        sampler = _build_sampler(cfg, representation, skel)

        # Every render is a batch ITEM: its own prompt, constraints, scene, frame
        # count and seed. A plain job builds one item per prompt, all sharing the
        # job-level constraint env vars; a FUSED job (RMG_BATCH) carries several
        # jobs' worth of items, each with its own. One code path serves both.
        batch = _load_batch()
        if batch is None:
            shared = {
                "constraints": _load_specs(cfg, "RMG_CONSTRAINTS", "constraints"),
                "ranges": _load_specs(cfg, "RMG_RANGES", "ranges"),
                "scene": _load_scene(cfg),
                "room_guidance": _room_guidance(cfg),
                "num_frames": int(cfg.viz.num_frames),
                "seed": int(cfg.viz.seed),
            }
            batch = [
                {**shared, "prompt": p.strip()}
                for p in str(cfg.viz.prompts).split("|") if p.strip()
            ]
        if not batch:
            raise ValueError("mode=prompt has no prompts to render")

        # The projector/room energy are built for a specific constraint set and the
        # seed is set per pass, so only items agreeing on all of it can share an ODE
        # integration. Group them; each group samples once, batched.
        groups: dict[str, list[dict]] = {}
        for item in batch:
            groups.setdefault(_batch_group_key(item), []).append(item)
        # Cap the ODE batch so a big fused group can't OOM the (12g) card.
        chunk_size = int(os.environ.get("RMG_BATCH_CHUNK", "8"))
        chunks = [
            g[i:i + chunk_size] for g in groups.values() for i in range(0, len(g), chunk_size)
        ]
        print(f"[visualize] sampling {len(batch)} prompt(s) in {len(chunks)} batch(es) "
              f"over {len(groups)} constraint group(s) "
              f"({int(cfg.viz.num_sample_steps)} ODE steps, "
              f"ω={float(cfg.viz.guidance_scale)})", flush=True)

        manifest = []
        done = 0
        for chunk in chunks:
            head = chunk[0]                    # group-wide by construction
            n_frames = int(head.get("num_frames", 100))
            set_seed(int(head.get("seed", 0)))
            project_fn, energy_fn, guidance_weight, constraint_viz, scene_obj = (
                _build_constraints(cfg, representation, skel, device, head, n_frames))
            scene_dict = head.get("scene")
            prompts = [it["prompt"] for it in chunk]

            # A chunk is sampled in one batched ODE integration, so the bar sits at
            # "sampling" until it returns, then advances per rendered prompt.
            write_progress(out_dir, stage="sampling", inner={"i": done, "n": len(batch)})
            with torch.no_grad():
                cond = text_encoder.encode(prompts, device=device)
                samples = sampler.sample(
                    model, shape=(len(prompts), n_frames), cond=cond,
                    fixed_values=None, fixed_mask=None, project_fn=project_fn,
                    energy_fn=energy_fn, guidance_weight=guidance_weight,
                )                                            # (B, T, ambient_dim)

            for i, item in enumerate(chunk):
                write_progress(out_dir, stage="render", inner={"i": done, "n": len(batch)})
                prompt = item["prompt"]
                tpr = tplusr_decode(samples[i])
                if scene_obj is not None:
                    tpr.translation, tpr.quaternions = place_motion(
                        tpr.translation, tpr.quaternions, scene_obj.spawn)
                joints_t = forward_kinematics(
                    skel, tpr.quaternions.float(), tpr.translation.float()
                )                                            # (T,J,3), already placed
                joints = joints_t.cpu().numpy()
                # World-constraint penalty over time + per-joint glow — the "how is
                # the skeleton being punished" overlay. Computed on the SAME placed
                # joints via the SAME SDFs the guidance energy used, so it's faithful.
                energy = scene_energy_series(joints_t, scene_obj) if scene_obj is not None else None
                safe = "".join(c if c.isalnum() else "_" for c in prompt)[:48]
                # `done` (not the chunk index) keeps names unique across groups.
                gif = _render(joints, out_dir / f"gen-{done:02d}-{safe}.mp4",
                              title=prompt[:60], fps=int(cfg.viz.fps), scene=scene_dict,
                              constraints=constraint_viz or None, energy=energy)
                done += 1
                if gif is None:
                    continue
                # `job` routes the file back to the app job that asked for it when
                # this run is a fusion of several; absent on a plain job.
                entry = {"file": gif.name, "kind": "pred", "caption": prompt}
                if item.get("job"):
                    entry["job"] = item["job"]
                manifest.append(entry)
        write_progress(out_dir, stage="done", complete=True,
                       inner={"i": len(batch), "n": len(batch)})
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

        skip_gt = {c.strip() for c in str(cfg.viz.get("skip_gt", "")).split(",") if c.strip()}
        if skip_gt:
            print(f"[visualize] compare: caller already holds GT for "
                  f"{sorted(skip_gt & set(clip_ids))} — rendering prediction only",
                  flush=True)

        text_encoder = _build_text_encoder(cfg)
        model = _build_model(cfg, representation, device)
        sampler = _build_sampler(cfg, representation, skel)

        manifest = []
        for idx, cid in enumerate(clip_ids):
            write_progress(out_dir, stage="sample", inner={"i": idx, "n": len(clip_ids)})
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

            # GT (cropped to n_frames) — unless the caller already has this one.
            if cid not in skip_gt:
                gt_joints = forward_kinematics(skel, quats, translation).numpy()
                gt_gif = _render(gt_joints, out_dir / f"real-{cid}.mp4",
                                 title=f"GT [{cid}] {caption[:55]}", fps=int(cfg.viz.fps))
                if gt_gif is not None:
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
            if pred_gif is not None:
                manifest.append({"file": pred_gif.name, "kind": "pred", "clip_id": cid, "caption": caption})
        write_progress(out_dir, stage="done", complete=True,
                       inner={"i": len(clip_ids), "n": len(clip_ids)})
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
        for idx, p in enumerate(files):
            write_progress(out_dir, stage="render", inner={"i": idx, "n": len(files)})
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
                # Diagnose where any corruption originates (the dump itself) so
                # the user can tell a diverged sample apart from a viz bug.
                if not torch.isfinite(tpr.translation).all() or not torch.isfinite(tpr.quaternions).all():
                    n_t = int((~torch.isfinite(tpr.translation)).sum())
                    n_q = int((~torch.isfinite(tpr.quaternions)).sum())
                    print(f"[visualize] WARNING: {step_tag} prompt {i} sample is "
                          f"non-finite in the dump (translation={n_t}, quats={n_q}) "
                          f"— the model's saved generation diverged at this step.",
                          flush=True)
                joints = forward_kinematics(
                    skel, tpr.quaternions.float(), tpr.translation.float()
                ).cpu().numpy()
                safe = "".join(c if c.isalnum() else "_" for c in text)[:40]
                gif = _render(joints, out_dir / f"{step_tag}-{i:02d}-{safe}.mp4",
                              title=f"[{step_tag}] {text[:55]}", fps=int(cfg.viz.fps))
                if gif is None:
                    continue
                manifest.append({"file": gif.name, "kind": "sample", "caption": text, "step": step})
        if not manifest:
            raise ValueError(
                "mode=samples: every rendered motion was non-finite — the saved "
                "sample dumps are corrupt (model generation diverged to NaN/Inf). "
                "Inspect the .npy files written next to each (attempted) GIF."
            )
        write_progress(out_dir, stage="done", complete=True,
                       inner={"i": len(files), "n": len(files)})
        _write_manifest(out_dir, manifest)
        return

    raise ValueError(f"unknown viz.mode {cfg.viz.mode!r}")


if __name__ == "__main__":
    main()
