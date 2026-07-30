"""Evaluate an RMG checkpoint on HumanML3D using the Guo et al. evaluator.

Pipeline:
  1. Load the model checkpoint (uses EMA weights by default).
  2. Iterate the test split: for each clip's text, sample a motion via the
     Riemannian Euler ODE with classifier-free guidance.
  3. Convert RMG samples (T+R) → 263-D HumanML3D features (paper §D.3).
  4. Encode real and generated motions through the Guo evaluator.
  5. Compute FID, R@1/R@2/R@3, MM-Dist, Diversity, and MultiModality.
  6. Optionally sweep classifier-free guidance ω ∈ [2.5, 9.5] (paper Fig. 3).

Run example:
    python -m rmg.scripts.evaluate \\
        +model=dit_base \\
        +train=rmg_base \\
        eval.checkpoint=runs/rmg-base-foo/checkpoints/latest.pt \\
        eval.guidance_scales='[2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5]' \\
        eval.evaluator=real

Set `eval.evaluator=random` for a smoke run that doesn't need the Guo
checkpoint (yields meaningless numbers — used to validate the pipeline).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from rmg.data import HumanML3DDataset, collate
from shared.eval import (
    RandomGuoEvaluator,
    RealGuoEvaluator,
    diversity,
    fid,
    mm_distance,
    multimodality,
    r_precision,
)
from shared.eval.metrics import crop_to_unit_length
from shared.eval.motion_quality import motion_quality
from shared.eval.text_tokens import CaptionTokenLookup, encode_texts_prefer_tokens
from rmg.flow import (
    CONSTRAINABLE_REPRESENTATIONS,
    RiemannianEulerSampler,
    SamplerCfg,
    WrappedGaussianPrior,
    bend_angles_deg,
    build_bend_projector,
    apply_path_facing,
    build_trajectory_energy_fn,
    build_trajectory_projector,
    compose_projectors,
    describe_control,
    flat_to_joints,
    merge_metrics,
    parse_bends,
    sample_control_signal,
    stack_controls,
    trajectory_metrics,
)
from rmg.models import (
    DiTConfig,
    Qwen3EmbeddingEncoder,
    RandomTextEncoder,
    RMGDiT,
)
from rmg.representation import (
    NUM_JOINTS,
    Representation,
    Skeleton,
    build_representation,
    decode,
    tplusr_to_h3d_features_with_quats,
)
from shared.utils import EMA, load_checkpoint, set_seed, write_progress


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_dataset(cfg: DictConfig, split: str, representation: Representation) -> HumanML3DDataset:
    return HumanML3DDataset(
        root=cfg.data.root,
        split=split,
        max_seq_len=cfg.data.max_seq_len,
        min_seq_len=cfg.data.min_seq_len,
        zip_name=cfg.data.zip_name,
        splits_name=cfg.data.splits_name,
        offsets_name=cfg.data.offsets_name,
        representation=representation,
        # Honor the fast-iteration subset so a train-split eval restricts to the
        # exact clips the model saw (subset only applies to split == "train").
        # No-op for val/test, where HumanML3DDataset ignores these.
        subset_fraction=cfg.data.subset_fraction,
        subset_seed=cfg.data.subset_seed,
        subset_n=int(cfg.data.get("subset_n", 0)),
    )


def _build_model(
    cfg: DictConfig, ckpt_path: Path, use_ema: bool, device: torch.device,
    representation: Representation,
) -> RMGDiT:
    dit_cfg = DiTConfig(
        input_dim=representation.ambient_dim,
        hidden_dim=cfg.model.hidden_dim,
        depth=cfg.model.depth, num_heads=cfg.model.num_heads, ffn_mult=cfg.model.ffn_mult,
        text_dim=cfg.model.text_dim, time_freq_dim=cfg.model.time_freq_dim,
        time_scale=float(cfg.model.get("time_scale", 1.0)),
        max_seq_len=cfg.model.max_seq_len,
    )
    model = RMGDiT(dit_cfg).to(device)
    state = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(state.model)
    if use_ema and state.ema is not None:
        ema = EMA(model, decay=0.0)         # decay value irrelevant; just a vehicle
        ema.load_state_dict(state.ema)
        ema.copy_to(model)
        print(f"[evaluate] loaded EMA weights from step {state.step}")
    else:
        print(f"[evaluate] loaded live weights from step {state.step}")
    model.eval()
    return model


def _build_text_encoder(cfg: DictConfig, device: torch.device):
    t = cfg.text_encoder.type
    if t == "random":
        return RandomTextEncoder(text_dim=cfg.text_encoder.text_dim)
    if t == "qwen3":
        return Qwen3EmbeddingEncoder(
            model_name=cfg.text_encoder.model_name,
            cache_dir=cfg.text_encoder.cache_dir,
            max_length=cfg.text_encoder.max_length,
        )
    raise ValueError(t)


def _build_evaluator(cfg: DictConfig, device: torch.device):
    if cfg.eval.evaluator == "real":
        return RealGuoEvaluator(
            text_to_motion_repo=cfg.eval.text_to_motion_repo,
            humanml3d_repo=cfg.eval.humanml3d_repo,
            device=device,
        )
    return RandomGuoEvaluator()


def _load_target_offsets(cfg: DictConfig) -> Skeleton:
    p = Path(cfg.data.root) / cfg.data.offsets_name
    offs = torch.load(p, weights_only=True)
    return Skeleton(offsets=offs)


# ---------------------------------------------------------------------------
# Sample → 263-D features
# ---------------------------------------------------------------------------


@torch.no_grad()
def _sample_and_featurize(
    model: RMGDiT,
    sampler: RiemannianEulerSampler,
    text_encoder,
    skeleton: Skeleton,
    representation: Representation,
    texts: list[str],
    lengths: torch.Tensor,
    guidance_scale: float,
    device: torch.device,
    use_length_mask: bool = False,
    bends: list | None = None,
    bend_stats: dict | None = None,
    control=None,
    traj_stats: list | None = None,
    quality_stats: list | None = None,
    fps: float = 20.0,
) -> list[torch.Tensor]:
    """Generate motions for `texts` and return per-sample 263-D feature tensors.
    Decoding is delegated to the configured Representation (T+R uses §D.3
    directly; T+P uses pre-shape rescaling; T+R+P honors `decode_via`)."""
    B = len(texts)
    Tmax = int(lengths.max().item())
    cond = text_encoder.encode(texts, device=device)
    # Optional length mask: True = valid frame. Passing it makes each clip's
    # frames attend only within its own length during sampling (as in training),
    # instead of across the whole padded batch-Tmax. Frames beyond L are cropped
    # away regardless.
    mask = None
    if use_length_mask:
        mask = torch.arange(Tmax, device=device)[None, :] < lengths.to(device)[:, None]
    # Sampling-time joint constraints. The projector is rebuilt per batch because
    # it is compiled against Tmax, which varies. Off unless `bends` is given, so
    # an unconstrained eval is byte-for-byte the path it always was.
    project_fn = None
    if bends:
        project_fn = build_bend_projector(
            bends, skeleton, Tmax, device=device, dtype=torch.float32,
        )
    # Trajectory (spatial mask-control) signal, one per clip, stacked into a
    # per-row control. Composed AFTER the bend projector so the root absorbs
    # whatever the bend clamp just moved (see `compose_projectors`).
    energy_fn = guidance_weight = None
    if control is not None:
        project_fn = compose_projectors(
            project_fn,
            build_trajectory_projector(
                control, skeleton, Tmax, device=device, dtype=torch.float32,
            ),
        )
        energy_fn = build_trajectory_energy_fn(control, skeleton)
        guidance_weight = float(control.guidance_weight) if energy_fn is not None else 0.0
    samples = sampler.sample(
        model, shape=(B, Tmax), cond=cond, mask=mask, guidance_scale=guidance_scale,
        project_fn=project_fn,
        energy_fn=energy_fn, guidance_weight=(guidance_weight or 0.0),
    )                                                       # (B, Tmax, ambient_dim)

    if control is not None and control.face_path:
        samples = apply_path_facing(samples, control, skeleton, project_fn=project_fn)

    # Control fidelity + the physical cost of enforcing it, measured on the FK
    # positions of the sample the projector actually acted on — the same
    # quantity the GT targets were read from, so the two are commensurable.
    if quality_stats is not None or (control is not None and traj_stats is not None):
        gen_joints = flat_to_joints(samples, skeleton)
        if control is not None and traj_stats is not None:
            traj_stats.append(trajectory_metrics(gen_joints, control, lengths=lengths))
        # Measured whether or not a constraint is active: a constrained run's
        # foot-skate means little without the UNCONSTRAINED number from the same
        # model to read it against (the real-data floor alone can't separate
        # "RMG is rough" from "enforcing the constraint made it rough").
        if quality_stats is not None:
            quality_stats.append(motion_quality(gen_joints, lengths=lengths, fps=fps))

    # Constraint satisfaction, measured on the valid frames of every clip from
    # the same quantity the projector acted on.
    if bends and bend_stats is not None:
        for c in bends:
            ang = bend_angles_deg(samples, c.joint, skeleton)   # (B, Tmax)
            key = f"{c.joint}"
            st = bend_stats.setdefault(key, {"min_deg": c.min_deg, "max_deg": c.max_deg,
                                             "worst": 0.0, "worst_violation": 0.0, "n": 0})
            for i in range(B):
                L = int(lengths[i].item())
                a = ang[i, :L]
                lo_v = float((c.min_deg - a).clamp(min=0).max().item())
                hi_v = float((a - c.max_deg).clamp(min=0).max().item())
                st["worst"] = max(st["worst"], float(a.max().item()))
                st["worst_violation"] = max(st["worst_violation"], lo_v, hi_v)
                st["n"] += 1
    feats = []
    for i in range(B):
        L = int(lengths[i].item())
        feats.append(representation.to_h3d_features(samples[i, :L], skeleton))
    return feats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    # Eval-specific sub-tree (defaults injected here, overridable at CLI):
    eval_cfg = OmegaConf.create({
        "checkpoint": "???",
        "split": "test",
        "use_ema": True,
        "evaluator": "random",                  # real | random
        "text_to_motion_repo": "external/text-to-motion",
        "humanml3d_repo": "external/HumanML3D",
        "guidance_scales": [6.5],
        "num_sample_steps": 50,
        "batch_size": 32,
        "max_clips": -1,                        # -1 = all in split
        "mm_num_texts": 30,                     # MModality: how many texts to evaluate
        "mm_repeats": 10,                       # MModality: K samples per text
        "diversity_times": 300,
        "seed": 0,
        # Sampling-time joint constraints as "JOINT:MIN:MAX", comma-separated
        # for several, e.g. "L_Knee:0:10" or "L_Knee:10:10,R_Elbow:90:90".
        # A pin is MIN == MAX. Deliberately bracket- and quote-free so it
        # survives a hydra CLI override and the shell unmangled. Empty string =
        # unconstrained, which is the path every existing result was measured on.
        "constraints": "",
        # --- Trajectory (spatial mask-control) evaluation -------------------
        # Off unless `control_joints` is set, so an unconstrained eval is
        # byte-for-byte the path every existing result was measured on.
        #
        # The protocol: for each test clip, read the REFERENCE motion's joint
        # positions at `control_density` randomly chosen frames and require the
        # generated motion to reproduce them. Reported as Traj./Loc./Avg. error
        # next to FID and R-precision, plus foot-skate and jerk — because the
        # interesting question for RMG is not whether the targets are hit
        # (root absorption makes that exact) but what hitting them costs.
        #
        #   control_joints:  "" (off) | pelvis | left_foot | right_foot | head
        #                    | left_wrist | right_wrist | cross
        #   control_density: keyframes per clip — 1, 2, 5, 49, or "all"
        #   control_mode:    project (exact) | guide (gradient, the baselines'
        #                    mechanism) | hybrid (project, then guide the residual)
        #   control_blend:   interp | local | none — how the root correction is
        #                    spread between keyframes (the error/smoothness dial)
        "control_joints": "",
        "control_density": 5,
        "control_mode": "project",
        "control_blend": "interp",
        "control_blend_frames": 10,
        "control_guidance": 1.0,
        "control_axes": "xyz",
        # Also turn the body to face along the target path. Position control
        # alone drags the body sideways/backwards (measured 125° between facing
        # and travel on a forced path, vs 13° unconstrained); heading is exactly
        # projectable too, so this is a separate arm worth scoring — it cannot
        # change the position error, only FID / skate / R-precision.
        "control_face_path": False,
        "control_face_strength": 1.0,
        # Pass a per-clip validity mask to the sampler so generation attends
        # only within each clip's true length (matches training). ON by default:
        # measured ~25% FID improvement (1.10→0.822 @ mid ω6.5, 200 steps, 1024
        # clips) vs the legacy no-mask path. Set False to reproduce legacy runs.
        "use_length_mask": True,
    })
    cfg.eval = OmegaConf.merge(eval_cfg, cfg.get("eval", OmegaConf.create({})))
    set_seed(int(cfg.eval.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Representation ---
    rep_kwargs = {k: v for k, v in dict(cfg.representation).items() if k not in ("name",)}
    representation = build_representation(cfg.representation.name, **rep_kwargs)

    # --- Data ---
    ds = _build_dataset(cfg, split=cfg.eval.split, representation=representation)
    # Seeded shuffle, always. test.txt lists segments of the same AMASS source
    # sequence under consecutive ids, so an unshuffled loader (a) evaluates a
    # highly redundant subset when eval.max_clips > 0 and (b) fills each
    # R-precision pool with near-duplicate motions, making text→motion
    # discrimination artificially hard and depressing R@k.
    loader = DataLoader(
        ds, batch_size=cfg.eval.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(int(cfg.eval.seed)),
        collate_fn=collate, num_workers=0, drop_last=False,
    )

    # --- Model + EMA ---
    model = _build_model(cfg, Path(cfg.eval.checkpoint), cfg.eval.use_ema, device, representation)
    text_encoder = _build_text_encoder(cfg, device)

    M = representation.build_manifold()
    if hasattr(representation, "prior_mu_from_skeleton") and ds._skeleton is not None:
        mu = representation.prior_mu_from_skeleton(ds._skeleton)
    else:
        mu = representation.prior_mu()
    prior = WrappedGaussianPrior(M, mu, sigma=cfg.train.prior_sigma)
    sampler = RiemannianEulerSampler(
        manifold=M, prior=prior,
        cfg=SamplerCfg(num_steps=int(cfg.eval.num_sample_steps), guidance_scale=6.5),
    )

    skeleton = _load_target_offsets(cfg)

    # Sampling-time constraints, if any. Parsed once; the projector itself is
    # rebuilt per batch because it compiles against that batch's frame count.
    bends = []
    spec_str = str(cfg.eval.get("constraints", "") or "").strip()
    if spec_str:
        specs = []
        for part in spec_str.split(","):
            joint, lo, hi = part.strip().split(":")
            specs.append({"joint": joint, "bend_min": float(lo), "bend_max": float(hi)})
        bends = parse_bends(specs)
        print(f"[eval] CONSTRAINED sampling: {spec_str}")
        print("[eval] NOTE: constraining every clip moves the samples off the data "
              "distribution on purpose, so FID is expected to rise. Read R-precision "
              "and multimodal distance for text fidelity, and the bend readout for "
              "whether the limit held.")
    bend_stats: dict = {}

    # --- Trajectory (spatial mask-control) protocol ---
    control_joints = str(cfg.eval.get("control_joints", "") or "").strip()
    control_on = bool(control_joints)
    if control_on:
        if cfg.representation.name not in CONSTRAINABLE_REPRESENTATIONS:
            raise ValueError(
                f"trajectory control needs a quaternion representation "
                f"({CONSTRAINABLE_REPRESENTATIONS}); this run uses "
                f"{cfg.representation.name!r}"
            )
        raw_density = cfg.eval.control_density
        density = str(raw_density) if isinstance(raw_density, str) else int(raw_density)
        control_kwargs = dict(
            density=density,
            joint_set=control_joints,
            mode=str(cfg.eval.control_mode),
            blend=str(cfg.eval.control_blend),
            blend_frames=int(cfg.eval.control_blend_frames),
            guidance_weight=float(cfg.eval.control_guidance),
            axes=str(cfg.eval.control_axes),
            face_path=bool(cfg.eval.control_face_path),
            face_strength=float(cfg.eval.control_face_strength),
        )
        print(f"[eval] TRAJECTORY control: joints={control_joints} density={density} "
              f"mode={cfg.eval.control_mode} blend={cfg.eval.control_blend} "
              f"axes={cfg.eval.control_axes}")
        print("[eval] NOTE: targets are read from each clip's REFERENCE motion, so a "
              "perfect method reproduces the reference trajectory. Read Traj./Loc./Avg. "
              "error for control fidelity and foot-skate/jerk for what enforcing it cost.")

    fps = float(cfg.data.get("fps", 20.0))
    # Physical readouts (foot-skate / jerk / root speed) need FK off the flat
    # state, so they are available exactly for the quaternion representations.
    quality_on = cfg.representation.name in CONSTRAINABLE_REPRESENTATIONS

    def _batch_control(x1: torch.Tensor, lengths: torch.Tensor, rng):
        """One control signal per clip in the batch, stacked into a per-row one.

        GT targets come from FK on the dataset's encoded `x1` with the SAME
        skeleton used to score the samples, so target and measurement are the
        same operation and any offsets-mismatch cancels.
        """
        gt_joints = flat_to_joints(x1, skeleton)                    # (B, T, J, 3)
        per_clip = [
            sample_control_signal(
                gt_joints[i].cpu(), length=int(lengths[i].item()), rng=rng, **control_kwargs
            )
            for i in range(gt_joints.shape[0])
        ]
        return stack_controls(per_clip)

    evaluator = _build_evaluator(cfg, device)
    try:
        token_lookup = CaptionTokenLookup(cfg.eval.humanml3d_repo)
    except FileNotFoundError as e:
        print(f"[eval] WARN: {e} — falling back to spaCy text tagging (worse R-precision)")
        token_lookup = None

    # --- Iterate test set, gather real + generated features at all guidance scales ---
    all_results: dict[float, dict[str, float | list]] = {}
    # Live progress markers (read by the app's job_progress): outer = which ω of
    # the sweep, inner = batch within the current ω. Written into <run>/eval/.
    eval_out = Path(cfg.output_dir) / "eval"
    n_omega = len(cfg.eval.guidance_scales)
    n_batches = len(loader)

    for omega_idx, omega in enumerate(cfg.eval.guidance_scales):
        print(f"\n=== guidance ω = {omega} ===")
        real_motion_feats, gen_motion_feats, text_feats = [], [], []
        n_seen = 0
        n_text_fallback = 0
        crop_rng = np.random.default_rng(int(cfg.eval.seed) + int(float(omega) * 10))
        # Keyframe draw is seeded per ω so every guidance level sees the SAME
        # control signals — otherwise an ω sweep would confound the two.
        control_rng = np.random.default_rng(int(cfg.eval.seed))
        traj_stats: list = []
        quality_stats: list = []
        real_quality: list = []
        outer = {"i": omega_idx, "n": n_omega, "label": f"ω={omega}"}
        write_progress(eval_out, stage="sample", outer=outer, inner={"i": 0, "n": n_batches})

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"sample ω={omega}")):
            x1 = batch.x1.to(device)
            mask = batch.mask
            lengths = batch.lengths

            control = _batch_control(x1, lengths, control_rng) if control_on else None
            if control is not None and batch_idx == 0 and omega_idx == 0:
                print(f"[eval] control signal (clip 0): {describe_control(control)}")

            # Generated motions: sample, decode via the configured representation, build 263-D features.
            gen_feats = _sample_and_featurize(
                model, sampler, text_encoder, skeleton, representation,
                texts=batch.texts, lengths=lengths,
                guidance_scale=float(omega), device=device,
                use_length_mask=bool(cfg.eval.use_length_mask),
                bends=bends,
                bend_stats=bend_stats,
                control=control,
                traj_stats=traj_stats,
                quality_stats=(quality_stats if quality_on else None),
                fps=fps,
            )
            # Foot-skate/jerk of the REFERENCE motions — the floor those numbers
            # should be read against (real data is not skate-free either).
            if quality_on:
                real_quality.append(
                    motion_quality(flat_to_joints(x1, skeleton), lengths=lengths, fps=fps)
                )

            # Real motions: same code path, just on the dataset's encoded x1.
            real_feats = []
            for i in range(x1.shape[0]):
                L = int(lengths[i].item())
                real_feats.append(representation.to_h3d_features(x1[i, :L], skeleton))

            # Upstream eval protocol: crop clips to multiples of unit_length=4
            # (the movement encoder is a stride-4 conv; raw lengths misalign
            # the co-embedding and depress R-precision/mm_dist — real-motion
            # R@1 measured 0.344 vs the published 0.511 without this). Same
            # jittered crop on real and gen, mirroring dataset.py.
            real_feats = [crop_to_unit_length(f, crop_rng) for f in real_feats]
            gen_feats = [crop_to_unit_length(f, crop_rng) for f in gen_feats]

            real_lens = torch.tensor([f.shape[0] for f in real_feats])
            gen_lens = torch.tensor([f.shape[0] for f in gen_feats])
            real_padded = torch.zeros(len(real_feats), int(real_lens.max()), 263)
            gen_padded = torch.zeros(len(gen_feats), int(gen_lens.max()), 263)
            for i, (rf, gf) in enumerate(zip(real_feats, gen_feats)):
                real_padded[i, : rf.shape[0]] = rf
                gen_padded[i, : gf.shape[0]] = gf

            real_emb = evaluator.encode_motion(real_padded, real_lens)
            gen_emb = evaluator.encode_motion(gen_padded, gen_lens)
            # Prefer HumanML3D's pre-tagged word/POS tokens: the Guo text
            # encoder was trained on their custom *_VIP tags, which spaCy
            # tagging never produces — the from_strings path silently halves
            # R-precision (harness bug found 2026-07-05: real-motion R@1 was
            # 0.17 vs the published ~0.51).
            text_emb, n_fb = encode_texts_prefer_tokens(
                evaluator, token_lookup, batch.clip_ids, batch.texts,
            )
            n_text_fallback += n_fb

            real_motion_feats.append(real_emb.cpu().numpy())
            gen_motion_feats.append(gen_emb.cpu().numpy())
            text_feats.append(text_emb.cpu().numpy())

            n_seen += x1.shape[0]
            write_progress(eval_out, stage="sample", outer=outer,
                           inner={"i": batch_idx + 1, "n": n_batches})
            if cfg.eval.max_clips > 0 and n_seen >= cfg.eval.max_clips:
                break

        def _tick(label: str, t0: float) -> float:
            t1 = time.perf_counter()
            print(f"[ω={omega}] {label}: {t1 - t0:.2f}s", flush=True)
            return t1

        if n_text_fallback:
            print(f"[ω={omega}] WARN: {n_text_fallback} captions missing pre-tagged "
                  f"tokens — encoded via spaCy fallback", flush=True)

        t = time.perf_counter()
        real_motion_feats = np.concatenate(real_motion_feats, axis=0)
        gen_motion_feats = np.concatenate(gen_motion_feats, axis=0)
        text_feats = np.concatenate(text_feats, axis=0)
        t = _tick(f"concat (shapes real={real_motion_feats.shape} gen={gen_motion_feats.shape} text={text_feats.shape})", t)

        rng = np.random.default_rng(int(cfg.eval.seed))
        results = {}
        results["fid"] = fid(real_motion_feats, gen_motion_feats)
        t = _tick(f"fid={results['fid']:.4f}", t)
        results["r_precision"] = r_precision(text_feats, gen_motion_feats, top_k=3, rng=rng).tolist()
        t = _tick(f"r_precision={results['r_precision']}", t)
        results["mm_dist"] = mm_distance(text_feats, gen_motion_feats)
        t = _tick(f"mm_dist={results['mm_dist']:.4f}", t)
        results["diversity"] = diversity(gen_motion_feats, diversity_times=int(cfg.eval.diversity_times), rng=rng)
        t = _tick(f"diversity={results['diversity']:.4f}", t)
        results["diversity_real"] = diversity(real_motion_feats, diversity_times=int(cfg.eval.diversity_times), rng=rng)
        t = _tick(f"diversity_real={results['diversity_real']:.4f}", t)

        # ---- MultiModality (re-sampled per-text generations) ----
        mm_texts = []
        mm_lengths = []
        mm_controls = []
        seen_clips = set()
        mm_rng = np.random.default_rng(int(cfg.eval.seed) + 1)
        for batch in loader:
            mm_gt = flat_to_joints(batch.x1, skeleton) if control_on else None
            for i, cid in enumerate(batch.clip_ids):
                if cid in seen_clips:
                    continue
                mm_texts.append(batch.texts[i])
                mm_lengths.append(int(batch.lengths[i].item()))
                # Under trajectory control, MultiModality must be measured
                # under control too — K samples of ONE text sharing ONE control
                # signal. Sampling them unconstrained inside a constrained run
                # would report a number that doesn't describe the setting.
                if control_on:
                    mm_controls.append(sample_control_signal(
                        mm_gt[i].cpu(), length=int(batch.lengths[i].item()),
                        rng=mm_rng, **control_kwargs,
                    ))
                seen_clips.add(cid)
                if len(mm_texts) >= int(cfg.eval.mm_num_texts):
                    break
            if len(mm_texts) >= int(cfg.eval.mm_num_texts):
                break
        t = _tick(f"populated {len(mm_texts)} MM texts", t)

        K = int(cfg.eval.mm_repeats)
        mm_per_text = []
        n_mm = len(mm_texts)
        for mm_idx, (text, L) in enumerate(zip(mm_texts, mm_lengths)):
            write_progress(eval_out, stage="multimodality", outer=outer,
                           inner={"i": mm_idx, "n": n_mm})
            tmm = time.perf_counter()
            cond = text_encoder.encode([text] * K, device=device)
            t_cond = time.perf_counter()
            # The projector is compiled against the frame count, which is this
            # text's L, so it is rebuilt per text. Without this the multimodality
            # branch would sample unconstrained inside a constrained run and
            # report a number that does not describe the setting it is filed
            # under (it came back identical across constraint specs).
            mm_project_fn = None
            if bends:
                mm_project_fn = build_bend_projector(
                    bends, skeleton, int(L), device=device, dtype=torch.float32,
                )
            mm_energy_fn, mm_gw = None, 0.0
            if control_on:
                # The control signal has a leading batch axis of 1, so it
                # broadcasts across all K repeats of this text unchanged.
                mm_ctrl = mm_controls[mm_idx]
                mm_project_fn = compose_projectors(
                    mm_project_fn,
                    build_trajectory_projector(
                        mm_ctrl, skeleton, int(L), device=device, dtype=torch.float32,
                    ),
                )
                mm_energy_fn = build_trajectory_energy_fn(mm_ctrl, skeleton)
                mm_gw = float(mm_ctrl.guidance_weight) if mm_energy_fn is not None else 0.0
            samples = sampler.sample(
                model, shape=(K, L), cond=cond, guidance_scale=float(omega),
                project_fn=mm_project_fn, energy_fn=mm_energy_fn, guidance_weight=mm_gw,
            )
            t_sample = time.perf_counter()
            feats = []
            for k in range(K):
                f = representation.to_h3d_features(samples[k, :L], skeleton)
                feats.append(crop_to_unit_length(f, crop_rng))
            t_feat = time.perf_counter()
            mm_lens = torch.tensor([f.shape[0] for f in feats])
            padded = torch.zeros(K, int(mm_lens.max()), 263)
            for k, f in enumerate(feats):
                padded[k, : f.shape[0]] = f
            emb = evaluator.encode_motion(padded, mm_lens)
            mm_per_text.append(emb.cpu().numpy())
            t_emb = time.perf_counter()
            print(f"[ω={omega} MM {mm_idx + 1}/{len(mm_texts)}] L={L} "
                  f"cond={t_cond - tmm:.2f}s sample={t_sample - t_cond:.2f}s "
                  f"feat={t_feat - t_sample:.2f}s emb={t_emb - t_feat:.2f}s "
                  f"(total {t_emb - tmm:.2f}s)", flush=True)
        t = _tick(f"MM done ({len(mm_texts)} texts × K={K})", t)
        results["multimodality"] = multimodality(np.stack(mm_per_text, axis=0))
        t = _tick(f"multimodality={results['multimodality']:.4f}", t)

        # Constraint satisfaction rides along with the metrics it qualifies: a
        # constrained FID means nothing without evidence the limit actually held.
        if bend_stats:
            results["constraints"] = {
                "spec": spec_str,
                "joints": {k: dict(v) for k, v in bend_stats.items()},
            }

        # Same idea for spatial control: a constrained FID means nothing without
        # evidence of whether the targets were hit — and what hitting them cost.
        def _avg(rows: list, key: str) -> float:
            return float(np.mean([r[key] for r in rows])) if rows else 0.0

        _QK = ("foot_skate_ratio", "jerk", "root_speed")
        if quality_on:
            # Reported for every run, constrained or not — this is the column a
            # constrained result has to be read against.
            results["quality"] = {k: _avg(quality_stats, k) for k in _QK}
            results["quality_real"] = {k: _avg(real_quality, k) for k in _QK}

        if control_on:
            results["trajectory"] = {
                "joints": control_joints,
                "density": control_kwargs["density"],
                "mode": control_kwargs["mode"],
                "blend": control_kwargs["blend"],
                "blend_frames": control_kwargs["blend_frames"],
                "guidance_weight": control_kwargs["guidance_weight"],
                "axes": control_kwargs["axes"],
                "face_path": control_kwargs["face_path"],
                "face_strength": control_kwargs["face_strength"],
                "control": merge_metrics(traj_stats),
            }
            tm = results["trajectory"]["control"]
            q, qr = results.get("quality", {}), results.get("quality_real", {})
            print(f"[ω={omega}] traj: avg_err={tm.get('avg_err', 0):.4f}m "
                  f"loc_err@0.5={tm.get('loc_err_0.5', 0):.4f} "
                  f"traj_err@0.5={tm.get('traj_err_0.5', 0):.4f} "
                  f"| skate={q.get('foot_skate_ratio', 0):.4f} "
                  f"(real {qr.get('foot_skate_ratio', 0):.4f})",
                  flush=True)

        all_results[float(omega)] = results
        print(json.dumps(results, indent=2), flush=True)

        # Save incrementally after every ω so a wall-time hit doesn't lose
        # everything. Each write overwrites the file with the cumulative results.
        out_dir = Path(cfg.output_dir) / "eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "results.json", "w") as f:
            json.dump(
                {str(k): v for k, v in all_results.items()},
                f, indent=2, default=float,
            )
        print(f"[evaluate] wrote partial results ({len(all_results)}/{len(cfg.eval.guidance_scales)} ω) "
              f"to {out_dir / 'results.json'}", flush=True)
        # ω done → advance the outer bar so the app reflects the finished level
        # even during the gap before the next ω's first batch.
        write_progress(eval_out, stage="sample",
                       outer={"i": omega_idx + 1, "n": n_omega, "label": f"ω={omega}"},
                       inner={"i": n_batches, "n": n_batches})

    write_progress(eval_out, stage="done", complete=True,
                   outer={"i": n_omega, "n": n_omega})
    (eval_out / ".complete").write_text(str(len(all_results)))
    print(f"\n[evaluate] done — all {len(all_results)} guidance levels saved.", flush=True)


if __name__ == "__main__":
    main()
