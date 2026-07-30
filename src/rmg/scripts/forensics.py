"""Generation forensics: generated motion against real motion, on quantities
that never pass through the Guo evaluator.

FID and R-precision both go through the evaluator's learned motion encoder, so
they say nothing about whether the raw state is well-formed. This script reads
the flat manifold state directly and reports four quantities for generated and
for real clips, on the same batches:

  * quaternion norm            — is every joint still a unit quaternion
  * angular velocity           — 2·acos(|<q_t, q_{t+1}>|), rad / frame
  * translation velocity       — ‖p_t − p_{t-1}‖, m / frame
  * root displacement          — ‖p_last − p_first‖, m, plus the max over clips

All are computed over each clip's valid frames only, so padding never enters.
The generated side uses the same sampler settings as `evaluate.py`, which is the
point: an earlier ad-hoc pass measured these on a different sampler path, and its
numbers are not comparable to the ones reported alongside them.

Run:
    python -m rmg.scripts.forensics \\
        model=dit_mid train=rmg_mid data=cluster_mounted \\
        +eval.checkpoint=.../ckpt-000300000.pt \\
        +eval.num_sample_steps=800 +eval.guidance_scales=[6.5] \\
        +eval.max_clips=512
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from rmg.data import collate
from rmg.flow import RiemannianEulerSampler, SamplerCfg, WrappedGaussianPrior
from rmg.representation import NUM_JOINTS, build_representation, forward_kinematics
from rmg.representation.tplusr import decode as tplusr_decode
from shared.utils import set_seed, write_progress

from .evaluate import (
    _build_dataset,
    _build_model,
    _build_text_encoder,
    _load_target_offsets,
)


def _split_state(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat (T, D) state → (translation (T, 3), quaternions (T, J, 4))."""
    p = x[..., :3]
    q = x[..., 3 : 3 + 4 * NUM_JOINTS].reshape(*x.shape[:-1], NUM_JOINTS, 4)
    return p, q


def _jerk(joints: torch.Tensor) -> float:
    """Mean ‖third difference‖ of joint positions, over joints and frames.

    The same quantity the showcase measures per clip: metres per frame³ at 20 fps.
    Computed from decoded joint positions so it is comparable to the numbers taken
    off the rendered arrays, and so real and generated go through one path.
    """
    if joints.shape[0] < 4:
        return float("nan")
    x = joints.double()
    j = x[3:] - 3 * x[2:-1] + 3 * x[1:-2] - x[:-3]
    return float(j.norm(dim=-1).mean())


def _clip_stats(x: torch.Tensor, skeleton=None) -> dict[str, float]:
    """The measured quantities for one clip's valid frames, as plain floats."""
    p, q = _split_state(x.double())
    # Unit-norm check on the raw state: no renormalisation before measuring, or
    # the quantity would be 1.0 by construction.
    qn = q.norm(dim=-1)
    # Angular velocity between consecutive frames, per joint. |<q_t, q_t+1>| is
    # sign-invariant, so an antipodal representative does not inflate the angle.
    dots = (q[:-1] * q[1:]).sum(-1).abs().clamp(max=1.0)
    ang = 2.0 * torch.acos(dots)
    tvel = (p[1:] - p[:-1]).norm(dim=-1)
    out = {
        "quat_norm": float(qn.mean()),
        "angular_vel": float(ang.mean()) if ang.numel() else 0.0,
        "translation_vel": float(tvel.mean()) if tvel.numel() else 0.0,
        "root_disp": float((p[-1] - p[0]).norm()),
    }
    if skeleton is not None:
        tpr = tplusr_decode(x.float())
        joints = forward_kinematics(skeleton, tpr.quaternions.float(), tpr.translation.float())
        out["jerk"] = _jerk(joints)
    return out


def _accumulate(acc: dict[str, list], stats: dict[str, float]) -> None:
    for k, v in stats.items():
        acc.setdefault(k, []).append(v)


def _summarise(acc: dict[str, list]) -> dict[str, float]:
    out = {k: float(sum(v) / len(v)) for k, v in acc.items() if v}
    if acc.get("root_disp"):
        out["root_disp_max"] = float(max(acc["root_disp"]))
    # Jerk is heavy-tailed across clips, so the median and the inter-quartile
    # range describe it better than the mean. The spread here is across clips,
    # which is the unit that varies; it is not a seed replicate.
    js = sorted(v for v in acc.get("jerk", []) if v == v)
    if js:
        def pct(p):
            if len(js) == 1:
                return js[0]
            i = p * (len(js) - 1)
            lo, hi = int(i), min(int(i) + 1, len(js) - 1)
            return js[lo] + (js[hi] - js[lo]) * (i - lo)
        out["jerk_median"] = pct(0.5)
        out["jerk_q1"] = pct(0.25)
        out["jerk_q3"] = pct(0.75)
        out["jerk_n"] = len(js)
    out["n"] = len(acc.get("quat_norm", []))
    return out


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    eval_cfg = OmegaConf.create({
        "checkpoint": "???",
        "split": "test",
        "use_ema": True,
        "guidance_scales": [6.5],
        "num_sample_steps": 200,
        "batch_size": 32,
        "max_clips": 512,
        "seed": 0,
        "use_length_mask": True,
    })
    cfg.eval = OmegaConf.merge(eval_cfg, cfg.get("eval", OmegaConf.create({})))
    set_seed(int(cfg.eval.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rep_kwargs = {k: v for k, v in dict(cfg.representation).items() if k not in ("name",)}
    representation = build_representation(cfg.representation.name, **rep_kwargs)

    ds = _build_dataset(cfg, split=cfg.eval.split, representation=representation)
    loader = DataLoader(
        ds, batch_size=cfg.eval.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(int(cfg.eval.seed)),
        collate_fn=collate, num_workers=0, drop_last=False,
    )

    model = _build_model(cfg, Path(cfg.eval.checkpoint), cfg.eval.use_ema, device, representation)
    text_encoder = _build_text_encoder(cfg, device)

    M = representation.build_manifold()
    if hasattr(representation, "prior_mu_from_skeleton") and ds._skeleton is not None:
        mu = representation.prior_mu_from_skeleton(ds._skeleton)
    else:
        mu = representation.prior_mu()
    prior = WrappedGaussianPrior(M, mu, sigma=cfg.train.prior_sigma)
    omega = float(cfg.eval.guidance_scales[0])
    sampler = RiemannianEulerSampler(
        manifold=M, prior=prior,
        cfg=SamplerCfg(num_steps=int(cfg.eval.num_sample_steps), guidance_scale=omega),
    )

    skeleton = _load_target_offsets(cfg)

    out_dir = Path(cfg.output_dir) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_acc: dict[str, list] = {}
    real_acc: dict[str, list] = {}
    n_seen = 0
    limit = int(cfg.eval.max_clips)
    n_batches = len(loader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="forensics")):
            if limit > 0 and n_seen >= limit:
                break
            write_progress(out_dir, stage="sample",
                           outer={"i": 0, "n": 1, "label": f"ω={omega}"},
                           inner={"i": batch_idx, "n": n_batches})
            lengths = batch.lengths
            Tmax = int(lengths.max().item())
            cond = text_encoder.encode(batch.texts, device=device)
            mask = None
            if bool(cfg.eval.use_length_mask):
                mask = torch.arange(Tmax, device=device)[None, :] < lengths.to(device)[:, None]
            samples = sampler.sample(
                model, shape=(len(batch.texts), Tmax), cond=cond, mask=mask,
                guidance_scale=omega,
            )
            x1 = batch.x1.to(device)
            for i in range(samples.shape[0]):
                L = int(lengths[i].item())
                if L < 4:
                    continue
                _accumulate(gen_acc, _clip_stats(samples[i, :L], skeleton))
                _accumulate(real_acc, _clip_stats(x1[i, :L], skeleton))
                n_seen += 1

    result = {
        "settings": {
            "checkpoint": str(cfg.eval.checkpoint),
            "guidance_scale": omega,
            "num_sample_steps": int(cfg.eval.num_sample_steps),
            "split": str(cfg.eval.split),
            "use_ema": bool(cfg.eval.use_ema),
            "use_length_mask": bool(cfg.eval.use_length_mask),
            "seed": int(cfg.eval.seed),
            "n_clips": n_seen,
        },
        "generated": _summarise(gen_acc),
        "real": _summarise(real_acc),
    }
    (out_dir / "forensics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
