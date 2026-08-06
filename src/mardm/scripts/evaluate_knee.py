"""Joint-angle control check: can the guidance stop a knee from bending?

RMG constrains a joint's *bend angle* in closed form because it generates unit
quaternions (flow.constraints.bend_clamp). MARDM has no rotation coordinate to
project — but the decoded motion is joint *positions*, and a bend angle is a
differentiable function of three of them (hip, knee, ankle). So we express the
same "knee can't bend" limit as a soft objective on decoded joints and pull it
in through the existing z-optimization guidance (mardm.control), no retraining.

For each leg-heavy prompt it samples the model twice with the same seed:

  1. `unconstrained` — the plain masked-AR sample (no guidance);
  2. `knee-locked`   — guidance with a bend limit on the chosen knee (bend
     <= `knee.max_deg`), plus the usual dynamics anchor + foot-skate terms so
     the rest of the gait stays alive while the leg straightens.

Reports, per arm, the joint-angle satisfaction (mean/max bend, % frames over the
limit) and realism diagnostics (foot-skate, motion magnitude, jerk), and writes
a side-by-side `unconstrained | knee-locked` GIF per prompt under
`${output_dir}/knee/`.

Run (cluster):
    python -u -m mardm.scripts.evaluate_knee +data=cluster_mounted \\
        ae_checkpoint=.../mardm-ae-canon/checkpoints/latest.pt \\
        text_encoder.type=qwen3 \\
        +knee.checkpoint=.../mardm-gen-m-canon/checkpoints/latest.pt \\
        +knee.side=left +knee.render=true +knee.verbose=true
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import hydra

from mardm.control import (
    ControlSignal,
    GuidanceConfig,
    bend_metrics,
    generate_guided,
    latents_to_joints,
    motion_metrics,
)
from mardm.models import AE, MARDM, MARDMConfig
from mardm.scripts.generate_control import (
    _build_text_encoder,
    _load_frozen_ae,
    _load_stats,
)
from shared.utils import EMA, load_checkpoint, set_seed


def _load_gen(cfg: DictConfig, ae: AE, device: torch.device) -> MARDM:
    """Load the generation branch (mirrors generate_control._load_mardm but reads
    `knee.use_ema` instead of the guid-config key)."""
    mardm = MARDM(MARDMConfig(
        ae_dim=ae.output_emb_width, text_dim=cfg.text_encoder.text_dim,
        **OmegaConf.to_container(cfg.model, resolve=True),
    )).to(device)
    state = load_checkpoint(Path(cfg.knee.checkpoint), map_location=device)
    mardm.load_state_dict(state.model)
    if bool(cfg.knee.use_ema) and state.ema is not None:
        ema = EMA(mardm, decay=0.0)
        ema.load_state_dict(state.ema)
        ema.copy_to(mardm)
        print(f"[knee] loaded MARDM EMA weights from step {state.step}", flush=True)
    else:
        print(f"[knee] loaded MARDM live weights from step {state.step}", flush=True)
    mardm.eval()
    return mardm

# (hip, knee, ankle) joint indices per leg — the bend triplet the limit acts on.
KNEE_TRIPLET = {"left": (1, 4, 7), "right": (2, 5, 8)}

# Prompts where a knee bend is central, so locking one leg is visible.
DEFAULT_PROMPTS: list[tuple[str, int]] = [
    ("a person walks forward at a steady pace", 160),
    ("a person sits down on a chair", 140),
    ("a person squats down and stands back up", 140),
    ("a person kicks with their leg", 120),
    ("a person walks up a flight of stairs", 160),
    ("a person jogs in place", 140),
]


def _bend_config(cfg: DictConfig) -> tuple[str, tuple[int, int, int]]:
    side = str(cfg.knee.side).lower()
    if side not in KNEE_TRIPLET:
        raise ValueError(f"knee.side must be 'left' or 'right', got {side!r}")
    return side, KNEE_TRIPLET[side]


def main_impl(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir) / "knee"
    out_dir.mkdir(parents=True, exist_ok=True)
    side, triplet = _bend_config(cfg)
    max_deg = float(cfg.knee.max_deg)
    print(f"[knee] device={device}  out_dir={out_dir}  side={side} triplet={triplet} "
          f"max_deg={max_deg}", flush=True)

    mean, std = _load_stats(cfg.stats_path)
    text_encoder = _build_text_encoder(cfg)
    ae = _load_frozen_ae(cfg, device)
    mardm = _load_gen(cfg, ae, device)
    ds_rate = ae.downsample_rate

    # Prompts: (text, frames). Config `knee.prompts` (list of strings) overrides
    # the curated set; `knee.frames` sets a uniform length for those.
    if cfg.knee.get("prompts"):
        default_len = int(cfg.knee.get("frames", 150))
        prompts = [(str(p), default_len) for p in cfg.knee.prompts]
    else:
        prompts = list(DEFAULT_PROMPTS)
    if cfg.knee.get("num_prompts"):
        prompts = prompts[: int(cfg.knee.num_prompts)]

    # knee-locked guidance stack; the bend term + dynamics anchor + skate.
    guided = GuidanceConfig(
        inner_iters=int(cfg.knee.inner_iters), lr=float(cfg.knee.lr),
        ode_steps_guidance=int(cfg.knee.ode_steps_guidance),
        ode_steps_final=int(cfg.knee.ode_steps_final),
        dyn_weight=float(cfg.knee.dyn_weight), skate_weight=float(cfg.knee.skate_weight),
        bend_triplet=triplet, bend_max_deg=max_deg, bend_weight=float(cfg.knee.weight),
        trust_radius=float(cfg.knee.trust_radius),
        verbose=bool(cfg.knee.verbose),
    )
    baseline = GuidanceConfig(inner_iters=0, ode_steps_final=guided.ode_steps_final)
    runs_spec = [("unconstrained", baseline), ("knee-locked", guided)]

    per_prompt: dict[str, dict] = {}
    for pi, (text, frames) in enumerate(prompts):
        latent_len = max(1, int(frames) // ds_rate)
        T = latent_len * ds_rate
        cond = text_encoder.encode([text], device=device)
        m_lens = torch.tensor([latent_len], device=device)
        lengths = torch.tensor([T])
        # No positional waypoints — the bend limit alone drives the guided arm.
        control = ControlSignal(torch.zeros(1, T, 22, 3), torch.zeros(1, T, 22, dtype=torch.bool))

        joints_by_arm: dict[str, np.ndarray] = {}
        arm_metrics: dict[str, dict] = {}
        for name, gcfg in runs_spec:
            set_seed(int(cfg.knee.seed) + pi)                # identical noise draws
            latents = generate_guided(
                mardm, ae, cond, m_lens, control, mean, std,
                timesteps=int(cfg.knee.timesteps), cond_scale=float(cfg.knee.guidance),
                guidance=gcfg,
            )
            with torch.no_grad():
                joints = latents_to_joints(latents.permute(0, 2, 1), ae,
                                           mean.to(device), std.to(device)).cpu()
            if not bool(torch.isfinite(joints).all()):
                # An infeasible constraint (e.g. a straight knee while sitting)
                # can still push the sample off-manifold. Record it as diverged
                # and move on rather than crashing the run / renderer.
                print(f"[knee] p{pi} {name:13s}  DIVERGED (non-finite joints) — "
                      f"skipping  {text[:40]!r}", flush=True)
                arm_metrics[name] = {"diverged": True}
                continue
            bm = bend_metrics(joints, triplet, lengths, max_deg)
            mm = motion_metrics(joints, lengths)
            arm_metrics[name] = {**bm, **mm}
            joints_by_arm[name] = joints[0].numpy().astype(np.float32)
            np.save(out_dir / f"{name}-p{pi}.npy", joints_by_arm[name])
            print(f"[knee] p{pi} {name:13s}  mean_bend={bm['mean_bend_deg']:5.1f} "
                  f"max_bend={bm['max_bend_deg']:5.1f}  viol={bm['violation_frac']:.2f} "
                  f"skate={mm['foot_skate']:.3f} jerk={mm['jerk']:.1f}  {text[:40]!r}",
                  flush=True)

        if bool(cfg.knee.render) and {"unconstrained", "knee-locked"} <= joints_by_arm.keys():
            from mardm.scripts.visualize import _render_compare
            panels = [
                (joints_by_arm["unconstrained"], "Unconstrained"),
                (joints_by_arm["knee-locked"],
                 f"{side.capitalize()} knee-locked (<={max_deg:.0f} deg, "
                 f"mean {arm_metrics['knee-locked']['mean_bend_deg']:.0f})"),
            ]
            _render_compare(panels, out_dir / f"compare-p{pi}.gif",
                            suptitle=text, fps=int(cfg.knee.fps))

        per_prompt[f"p{pi}"] = {"prompt": text, "frames": int(T), **{
            f"{name}_{k}": v for name, m in arm_metrics.items() for k, v in m.items()}}

    names = [n for n, _ in runs_spec]
    summary = {
        "per_prompt": per_prompt, "n_prompts": len(per_prompt), "runs": names,
        "side": side, "triplet": list(triplet), "max_deg": max_deg,
        "guidance_config": guided.__dict__,
    }
    for name in names:
        for k in ("mean_bend_deg", "max_bend_deg", "violation_frac",
                  "foot_skate", "motion_mag", "jerk"):
            vals = [v[f"{name}_{k}"] for v in per_prompt.values()
                    if f"{name}_{k}" in v]                 # skip diverged arms
            summary[f"mean_{name}_{k}"] = float(np.mean(vals)) if vals else float("nan")
    summary["n_diverged"] = sum(
        1 for v in per_prompt.values() if v.get("knee-locked_diverged"))
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    print("\n[knee] === summary (mean over prompts) ===", flush=True)
    print(f"[knee] {'arm':14s} {'mean_bend':>9s} {'max_bend':>9s} {'viol%':>7s} "
          f"{'skate':>7s} {'jerk':>7s}", flush=True)
    for name in names:
        print(f"[knee] {name:14s} {summary[f'mean_{name}_mean_bend_deg']:9.1f} "
              f"{summary[f'mean_{name}_max_bend_deg']:9.1f} "
              f"{100 * summary[f'mean_{name}_violation_frac']:7.1f} "
              f"{summary[f'mean_{name}_foot_skate']:7.3f} "
              f"{summary[f'mean_{name}_jerk']:7.1f}", flush=True)
    print(f"[knee] done — metrics + GIFs under {out_dir}", flush=True)


@hydra.main(config_path="../configs", config_name="gen", version_base=None)
def main(cfg: DictConfig) -> None:
    knee_cfg = OmegaConf.create({
        "checkpoint": "???",       # required: gen-branch checkpoint
        "side": "left",            # which knee to lock ('left' | 'right')
        "max_deg": 10.0,           # bend-angle limit (deg); 0 = perfectly straight
        "weight": 1.0,             # bend-loss weight
        "dyn_weight": 1.0,         # velocity anchor to the unguided gait
        "skate_weight": 1.0,       # foot-skate penalty
        "trust_radius": 0.5,       # hard trust region on z drift (keeps infeasible
                                   # constraints from pushing z off-manifold → NaN)
        "guidance": 3.0,           # CFG scale
        "timesteps": 18,           # masked-AR sampling iterations
        "inner_iters": 30,         # z gradient steps per AR step
        "lr": 0.05,
        "ode_steps_guidance": 8,
        "ode_steps_final": 25,
        "use_ema": True,
        "seed": 0,
        "prompts": [],             # override the curated leg-heavy prompt set
        "frames": 150,             # length for override prompts
        "num_prompts": 0,          # 0 = all
        "render": False,
        "fps": 20,
        "verbose": False,
    })
    cfg.knee = OmegaConf.merge(knee_cfg, cfg.get("knee", OmegaConf.create({})))
    main_impl(cfg)


if __name__ == "__main__":
    main()
