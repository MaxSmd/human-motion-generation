"""Eval-harness sanity check: metrics on REAL motions only (no model).

Motivation: both rmg-base (historically FID 0.5) and rmg-mid suddenly score
FID 8.7 / 21 with near-floor R-precision on the same harness. This script
isolates the harness from the models: it featurizes REAL clips through the
exact evaluate.py code path (dataset → to_h3d_features → Guo evaluator) and
reports metrics whose healthy values are known a priori:

  * FID(real half A, real half B)   → ≈ 0 (sampling noise only)
  * R-precision(text, real motion)  → ≈ 0.51 / 0.70 / 0.80 (HumanML3D paper)
  * mm_dist(text, real motion)      → ≈ 2.9–3.5
  * diversity(real)                 → ≈ 9.5

If these are healthy the harness is fine and the models' scores are real.
If these are broken, every model FID measured on this harness is bogus.

Run (mirrors evaluate.py wiring):
    python -m rmg.scripts.eval_sanity data=cluster_mounted \\
        eval.evaluator=real eval.max_clips=512
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

from rmg.data import collate
from shared.eval import diversity, fid, mm_distance, r_precision
from shared.eval.metrics import crop_to_unit_length
from shared.eval.text_tokens import CaptionTokenLookup, encode_texts_prefer_tokens
from shared.utils import set_seed

from rmg.scripts.evaluate import (
    _build_dataset,
    _build_evaluator,
    _load_target_offsets,
)
from rmg.representation import build_representation


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    eval_cfg = OmegaConf.create({
        "split": "test",
        "evaluator": "real",
        "text_to_motion_repo": "external/text-to-motion",
        "humanml3d_repo": "external/HumanML3D",
        "batch_size": 32,
        "max_clips": 512,
        "diversity_times": 300,
        "seed": 0,
    })
    cfg.eval = OmegaConf.merge(eval_cfg, cfg.get("eval", OmegaConf.create({})))
    set_seed(int(cfg.eval.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rep_kwargs = {k: v for k, v in dict(cfg.representation).items() if k not in ("name",)}
    representation = build_representation(cfg.representation.name, **rep_kwargs)

    ds = _build_dataset(cfg, split=cfg.eval.split, representation=representation)
    # Seeded shuffle — consecutive test.txt ids are near-duplicate segments of
    # the same AMASS sequence; see the matching note in evaluate.py.
    loader = DataLoader(
        ds, batch_size=cfg.eval.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(int(cfg.eval.seed)),
        collate_fn=collate, num_workers=0, drop_last=False,
    )
    skeleton = _load_target_offsets(cfg)
    evaluator = _build_evaluator(cfg, device)
    try:
        token_lookup = CaptionTokenLookup(cfg.eval.humanml3d_repo)
    except FileNotFoundError as e:
        print(f"[sanity] WARN: {e} — spaCy fallback (expect halved R-precision)")
        token_lookup = None

    # Featurize once (expensive: FK + upstream IK per clip), embed in passes.
    all_feats: list[torch.Tensor] = []
    text_embs = []
    n_seen = 0
    n_text_fallback = 0
    for batch in tqdm(loader, desc="featurize real"):
        x1 = batch.x1.to(device)
        lengths = batch.lengths
        for i in range(x1.shape[0]):
            L = int(lengths[i].item())
            all_feats.append(representation.to_h3d_features(x1[i, :L], skeleton).cpu())
        temb, n_fb = encode_texts_prefer_tokens(
            evaluator, token_lookup, batch.clip_ids, batch.texts,
        )
        n_text_fallback += n_fb
        text_embs.append(temb.cpu().numpy())
        n_seen += x1.shape[0]
        if cfg.eval.max_clips > 0 and n_seen >= cfg.eval.max_clips:
            break

    text = np.concatenate(text_embs, axis=0)
    print(f"[sanity] {len(all_feats)} real clips featurized"
          f" ({n_text_fallback} captions via spaCy fallback)")

    def embed_pass(rng: np.random.Generator, jitter: bool, bs: int = 32) -> np.ndarray:
        """Crop every clip to unit-length (upstream protocol) and embed."""
        out = []
        for s in range(0, len(all_feats), bs):
            chunk = [crop_to_unit_length(f, rng, jitter=jitter) for f in all_feats[s:s + bs]]
            lens = torch.tensor([c.shape[0] for c in chunk])
            padded = torch.zeros(len(chunk), int(lens.max()), 263)
            for i, c in enumerate(chunk):
                padded[i, : c.shape[0]] = c
            out.append(evaluator.encode_motion(padded, lens).cpu().numpy())
        return np.concatenate(out, axis=0)

    seed = int(cfg.eval.seed)
    real_a = embed_pass(np.random.default_rng(seed), jitter=True)
    real_b = embed_pass(np.random.default_rng(seed + 1), jitter=True)

    rng = np.random.default_rng(seed)
    half = real_a.shape[0] // 2
    perm = rng.permutation(real_a.shape[0])
    results = {
        # Upstream's "GT" row protocol: same motions embedded twice with
        # different random crop draws — published ≈ 0.002.
        "fid_gt_gt_protocol": float(fid(real_a, real_b)),
        # Disjoint halves (small-sample-biased; kept for continuity).
        "fid_real_vs_real": float(fid(real_a[perm[:half]], real_a[perm[half:2 * half]])),
        "r_precision_real": r_precision(text, real_a, top_k=3, rng=rng).tolist(),
        "mm_dist_real": float(mm_distance(text, real_a)),
        "diversity_real": float(diversity(real_a, diversity_times=int(cfg.eval.diversity_times), rng=rng)),
    }
    print(json.dumps(results, indent=2))

    out = Path(cfg.output_dir) / "eval" / "sanity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"[sanity] wrote {out}")


if __name__ == "__main__":
    main()
