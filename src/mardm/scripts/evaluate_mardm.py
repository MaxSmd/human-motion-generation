"""Evaluate a MARDM checkpoint on HumanML3D (HumanML3D format) via the Guo evaluator.

Pipeline (mirrors scripts/evaluate.py, swapping RMG's manifold sampler for the
MARDM generation task):
  1. Load the generation-branch checkpoint (EMA) + the frozen AE + text encoder.
  2. Iterate the test split. Real motions are featurized to true 263-D from the
     packed (translation, quaternions) via T+R; generated motions go text ->
     MARDM.generate -> AE.decode -> essential_to_h3d -> 263-D.
  3. Encode both through the Guo evaluator; compute FID, R@1/2/3, MM-Dist,
     Diversity, MultiModality. Optionally sweep classifier-free guidance.

Run (cluster):
    python scripts/evaluate_mardm.py +data=cluster_mounted \\
        ae_checkpoint=runs/mardm-ae-XXXX/checkpoints/latest.pt \\
        eval.checkpoint=runs/mardm-gen-YYYY/checkpoints/latest.pt \\
        eval.evaluator=real text_encoder.type=qwen3

`eval.evaluator=random` exercises the pipeline without the Guo checkpoint
(meaningless numbers). The 263-D bridge needs the external/HumanML3D submodule.
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from mardm.models import AE, MARDM, AEConfig, MARDMConfig
from mardm.tasks import generate_h3d_features
from shared.data import HumanML3DDataset, collate
from shared.eval import (
    RandomGuoEvaluator,
    RealGuoEvaluator,
    diversity,
    fid,
    mm_distance,
    multimodality,
    r_precision,
)
from shared.geometry import Skeleton, tplusr_to_h3d_features_with_quats
from shared.text import Qwen3EmbeddingEncoder, RandomTextEncoder
from shared.utils import EMA, load_checkpoint, set_seed

# rmg's T+R reference pipeline builds the ground-truth 263-D features the Guo
# evaluator compares against — an intentional cross-method dependency in eval.
from rmg.representation import TRRepresentation, decode


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
    ae.load_state_dict(state.model)
    if cfg.ae_use_ema and state.ema is not None:
        ema = EMA(ae, decay=0.0)
        ema.load_state_dict(state.ema)
        ema.copy_to(ae)
    ae.eval()
    return ae


def _load_mardm(cfg: DictConfig, ae: AE, device: torch.device) -> MARDM:
    mardm = MARDM(MARDMConfig(
        ae_dim=ae.output_emb_width, text_dim=cfg.text_encoder.text_dim,
        **OmegaConf.to_container(cfg.model, resolve=True),
    )).to(device)
    state = load_checkpoint(Path(cfg.eval.checkpoint), map_location=device)
    mardm.load_state_dict(state.model)
    if cfg.eval.use_ema and state.ema is not None:
        ema = EMA(mardm, decay=0.0)
        ema.load_state_dict(state.ema)
        ema.copy_to(mardm)
        print(f"[eval] loaded MARDM EMA weights from step {state.step}")
    else:
        print(f"[eval] loaded MARDM live weights from step {state.step}")
    mardm.eval()
    return mardm


def _real_h3d(x1: torch.Tensor, length: int, skeleton: Skeleton) -> torch.Tensor:
    """T+R sample -> true (length-1, 263) HumanML3D feature."""
    tpr = decode(x1[:length])
    return tplusr_to_h3d_features_with_quats(tpr.translation, tpr.quaternions, skeleton)


def _pad_stack(feats: list[torch.Tensor], dim: int = 263) -> torch.Tensor:
    tmax = max(f.shape[0] for f in feats)
    out = torch.zeros(len(feats), tmax, dim)
    for i, f in enumerate(feats):
        out[i, : f.shape[0]] = f
    return out


@hydra.main(config_path="../configs", config_name="gen", version_base=None)
def main(cfg: DictConfig) -> None:
    eval_cfg = OmegaConf.create({
        "checkpoint": "???",
        "split": "test",
        "use_ema": True,
        "evaluator": "random",                 # real | random
        "text_to_motion_repo": "external/text-to-motion",
        "humanml3d_repo": "external/HumanML3D",
        "guidance_scales": [4.5],              # paper's HumanML3D CFG scale
        "timesteps": 10,                       # masked-AR iterations
        "batch_size": 32,
        "max_clips": -1,
        "mm_num_texts": 30,
        "mm_repeats": 10,
        "diversity_times": 300,
        "seed": 0,
    })
    cfg.eval = OmegaConf.merge(eval_cfg, cfg.get("eval", OmegaConf.create({})))
    set_seed(int(cfg.eval.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean, std = _load_stats(cfg.stats_path)
    text_encoder = _build_text_encoder(cfg)
    ae = _load_frozen_ae(cfg, device)
    mardm = _load_mardm(cfg, ae, device)

    skeleton = Skeleton(offsets=torch.load(Path(cfg.data.root) / cfg.data.offsets_name, weights_only=True))
    evaluator = (
        RealGuoEvaluator(text_to_motion_repo=cfg.eval.text_to_motion_repo,
                         humanml3d_repo=cfg.eval.humanml3d_repo, device=device)
        if cfg.eval.evaluator == "real" else RandomGuoEvaluator()
    )

    # Real motions: T+R representation so we can recover (translation, quats).
    ds = HumanML3DDataset(
        root=cfg.data.root, split=cfg.eval.split, max_seq_len=cfg.data.max_seq_len,
        min_seq_len=cfg.data.min_seq_len, mirror_augment=False, zip_name=cfg.data.zip_name,
        splits_name=cfg.data.splits_name, offsets_name=cfg.data.offsets_name,
        representation=TRRepresentation(),
    )
    loader = DataLoader(ds, batch_size=cfg.eval.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0, drop_last=False)

    all_results: dict[float, dict] = {}
    for omega in cfg.eval.guidance_scales:
        print(f"\n=== guidance w = {omega} ===")
        real_emb, gen_emb, text_emb = [], [], []
        n_seen = 0
        for batch in tqdm(loader, desc=f"sample w={omega}"):
            x1 = batch.x1.to(device)
            lengths = batch.lengths

            real_feats = [_real_h3d(x1[i], int(lengths[i]), skeleton) for i in range(x1.shape[0])]
            gen_feats, gen_lens = generate_h3d_features(
                mardm, ae, text_encoder, batch.texts, lengths - 1,
                guidance=float(omega), timesteps=int(cfg.eval.timesteps),
                mean=mean, std=std, skeleton=skeleton, device=device,
                humanml3d_repo=cfg.eval.humanml3d_repo,
            )
            real_emb.append(evaluator.encode_motion(_pad_stack(real_feats), lengths - 1).cpu().numpy())
            gen_emb.append(evaluator.encode_motion(_pad_stack(gen_feats), gen_lens).cpu().numpy())
            text_emb.append(evaluator.encode_text_from_strings(batch.texts).cpu().numpy())

            n_seen += x1.shape[0]
            if cfg.eval.max_clips > 0 and n_seen >= cfg.eval.max_clips:
                break

        real_emb = np.concatenate(real_emb, 0)
        gen_emb = np.concatenate(gen_emb, 0)
        text_emb = np.concatenate(text_emb, 0)
        rng = np.random.default_rng(int(cfg.eval.seed))

        results = {
            "fid": fid(real_emb, gen_emb),
            "r_precision": r_precision(text_emb, gen_emb, top_k=3, rng=rng).tolist(),
            "mm_dist": mm_distance(text_emb, gen_emb),
            "diversity": diversity(gen_emb, diversity_times=int(cfg.eval.diversity_times), rng=rng),
            "diversity_real": diversity(real_emb, diversity_times=int(cfg.eval.diversity_times), rng=rng),
        }

        # MultiModality: K re-samples per text.
        mm_texts, mm_lengths, seen = [], [], set()
        for batch in loader:
            for i, cid in enumerate(batch.clip_ids):
                if cid in seen:
                    continue
                mm_texts.append(batch.texts[i])
                mm_lengths.append(int(batch.lengths[i]))
                seen.add(cid)
                if len(mm_texts) >= int(cfg.eval.mm_num_texts):
                    break
            if len(mm_texts) >= int(cfg.eval.mm_num_texts):
                break
        K = int(cfg.eval.mm_repeats)
        mm_per_text = []
        for text, L in zip(mm_texts, mm_lengths):
            feats, lens = generate_h3d_features(
                mardm, ae, text_encoder, [text] * K, torch.full((K,), L - 1, dtype=torch.long),
                guidance=float(omega), timesteps=int(cfg.eval.timesteps),
                mean=mean, std=std, skeleton=skeleton, device=device,
                humanml3d_repo=cfg.eval.humanml3d_repo,
            )
            mm_per_text.append(evaluator.encode_motion(_pad_stack(feats), lens).cpu().numpy())
        results["multimodality"] = multimodality(np.stack(mm_per_text, axis=0))

        all_results[float(omega)] = results
        print(json.dumps(results, indent=2), flush=True)

        out_dir = Path(cfg.output_dir) / "eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "results.json", "w") as f:
            json.dump({str(k): v for k, v in all_results.items()}, f, indent=2, default=float)

    print(f"\n[evaluate_mardm] done — {len(all_results)} guidance level(s) saved.", flush=True)


if __name__ == "__main__":
    main()
