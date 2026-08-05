"""Diagnose the eval-time 263-D feature space.

Two questions, one run:

  (A) ISOLATION — does HumanML3D's *canonical* stored `new_joint_vecs` 263, fed
      through the same Guo evaluator, reproduce diversity ~9.5? If yes, the
      evaluator + normalization are fine and the shortfall (diversity_real≈6.6)
      is in our T+R->263 reconstruction. If it ALSO gives ~6.6, then 6.6 is the
      correct "real" value for this evaluator setup and diversity_real is NOT
      anomalous (refocus R@1 elsewhere).

  (B) PER-BLOCK DIFF — for a handful of clips, compare our reconstructed 263
      (the exact `_real_h3d` path the evaluator scores) against the canonical
      `new_joint_vecs` for the same clip, broken down by the HumanML3D feature
      blocks, so we can see which channels diverge.

Run (cluster, in-container, paths match the eval you ran):

    python -m mardm.scripts.verify_features --config-name=gen_m \
        ae_checkpoint=$AE_CKPT +eval.checkpoint=$GEN_CKPT \
        +verify.h3d_vecs=external/HumanML3D/HumanML3D/new_joint_vecs \
        +verify.n_clips=128

(ae/gen checkpoints are NOT used for any math here — they're only required
 because the config marks them mandatory; pass the ones you evaluated.)
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from shared.data import HumanML3DDataset, collate
from shared.eval import RealGuoEvaluator, diversity
from shared.geometry import (
    Skeleton,
    tplusr_to_h3d_features_with_quats,
    tplusr_to_h3d_features_upstream,
)
from rmg.representation import TRRepresentation, decode

# HumanML3D 263-D layout for 22 joints: 4 + (J-1)*3 + (J-1)*6 + J*3 + 4.
BLOCKS = [
    ("root_rot_vel ", 0, 1),
    ("root_lin_vxz ", 1, 3),
    ("root_height_y", 3, 4),
    ("ric_local_pos", 4, 67),
    ("rot_6d       ", 67, 193),
    ("local_vel    ", 193, 259),
    ("foot_contact ", 259, 263),
]


def _real_h3d(x1: torch.Tensor, length: int, skeleton: Skeleton) -> torch.Tensor:
    """Exactly the eval's real path: T+R sample -> (length-1, 263)."""
    tpr = decode(x1[:length])
    return tplusr_to_h3d_features_with_quats(tpr.translation, tpr.quaternions, skeleton)


def _pad_stack(feats: list[torch.Tensor], dim: int = 263) -> torch.Tensor:
    tmax = max(f.shape[0] for f in feats)
    out = torch.zeros(len(feats), tmax, dim)
    for i, f in enumerate(feats):
        out[i, : f.shape[0]] = f
    return out


def _best_offset_diff(recon: np.ndarray, canon: np.ndarray) -> tuple[int, np.ndarray]:
    """Try frame offsets in {-1,0,1}; pick the one minimizing whole-vector MAE.
    Returns (offset, per-frame-aligned abs-diff array of shape (L, 263))."""
    best = None
    for off in (-1, 0, 1):
        if off >= 0:
            a, b = recon[off:], canon
        else:
            a, b = recon, canon[-off:]
        L = min(a.shape[0], b.shape[0])
        if L < 2:
            continue
        d = np.abs(a[:L] - b[:L])
        mae = d.mean()
        if best is None or mae < best[0]:
            best = (mae, off, d)
    return best[1], best[2]


@hydra.main(config_path="../configs", config_name="gen", version_base=None)
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vcfg = cfg.get("verify", OmegaConf.create({}))
    n_clips = int(vcfg.get("n_clips", 128))
    vecs_dir = Path(vcfg.get("h3d_vecs", "external/HumanML3D/HumanML3D/new_joint_vecs"))
    max_len = 196

    eval_cfg = cfg.get("eval", OmegaConf.create({}))
    t2m = eval_cfg.get("text_to_motion_repo", "external/text-to-motion")
    h3d = eval_cfg.get("humanml3d_repo", "external/HumanML3D")

    print(f"[verify] new_joint_vecs dir: {vecs_dir}  (exists={vecs_dir.exists()})", flush=True)

    skeleton = Skeleton(offsets=torch.load(Path(cfg.data.root) / cfg.data.offsets_name,
                                           weights_only=True))
    ds = HumanML3DDataset(
        root=cfg.data.root, split=eval_cfg.get("split", "test"),
        max_seq_len=cfg.data.max_seq_len, min_seq_len=cfg.data.min_seq_len,
        mirror_augment=False, zip_name=cfg.data.zip_name,
        splits_name=cfg.data.splits_name, offsets_name=cfg.data.offsets_name,
        representation=TRRepresentation(),
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate,
                        num_workers=0, drop_last=False)
    evaluator = RealGuoEvaluator(text_to_motion_repo=t2m, humanml3d_repo=h3d, device=device)

    recon_feats: list[torch.Tensor] = []
    recon_lens: list[int] = []
    canon_feats: list[torch.Tensor] = []
    canon_lens: list[int] = []
    block_err = {name: [] for name, _, _ in BLOCKS}
    offsets_used: list[int] = []
    n_missing = 0

    upstream_feats: list[torch.Tensor] = []
    upstream_lens: list[int] = []
    n_upstream_fail = 0

    avail = sorted(p.name for p in vecs_dir.glob("*.npy")) if vecs_dir.exists() else []
    print(f"[verify] vecs_dir has {len(avail)} .npy files; sample: {avail[:5]}", flush=True)
    dbg_ids: list[str] = []

    for batch in loader:
        x1 = batch.x1.to(device)
        for i in range(x1.shape[0]):
            if len(recon_feats) >= n_clips:
                break
            L = int(batch.lengths[i])
            cid = batch.clip_ids[i]
            if len(dbg_ids) < 10:
                dbg_ids.append(repr(cid))
                if len(dbg_ids) == 10:
                    print(f"[verify] sample clip_ids: {dbg_ids}", flush=True)
            r = _real_h3d(x1[i], L, skeleton).cpu()                      # (L-1, 263)
            recon_feats.append(r)
            recon_lens.append(min(r.shape[0], max_len))

            # Upstream (canonical) extractor on the same clip, for comparison.
            try:
                tpr = decode(x1[i][:L])
                u = tplusr_to_h3d_features_upstream(
                    tpr.translation, tpr.quaternions, skeleton).cpu()
                upstream_feats.append(u)
                upstream_lens.append(min(u.shape[0], max_len))
            except Exception as e:  # noqa: BLE001
                n_upstream_fail += 1
                if n_upstream_fail == 1:
                    print(f"[verify] upstream extractor failed (1st): {e!r}", flush=True)

            npy = vecs_dir / f"{cid}.npy"
            if not npy.exists():
                n_missing += 1
                continue
            c = torch.from_numpy(np.load(npy)).float()                   # (T, 263)
            canon_feats.append(c)
            canon_lens.append(min(c.shape[0], max_len))

            off, d = _best_offset_diff(r.numpy(), c.numpy())             # (L,263)
            offsets_used.append(off)
            for name, a, b in BLOCKS:
                block_err[name].append(float(d[:, a:b].mean()))
        if len(recon_feats) >= n_clips:
            break

    print(f"\n[verify] collected {len(recon_feats)} recon clips, "
          f"{len(canon_feats)} matched canonical ({n_missing} missing npy)", flush=True)

    rng = np.random.default_rng(0)

    # (A) ISOLATION ----------------------------------------------------------
    div_recon = None
    if recon_feats:
        emb = evaluator.encode_motion(_pad_stack(recon_feats),
                                      torch.tensor(recon_lens)).cpu().numpy()
        div_recon = diversity(emb, diversity_times=min(300, len(emb) * (len(emb) - 1)), rng=rng)
    div_canon = None
    if canon_feats:
        embc = evaluator.encode_motion(_pad_stack(canon_feats),
                                       torch.tensor(canon_lens)).cpu().numpy()
        div_canon = diversity(embc, diversity_times=min(300, len(embc) * (len(embc) - 1)), rng=rng)

    div_upstream = None
    if upstream_feats:
        embu = evaluator.encode_motion(_pad_stack(upstream_feats),
                                       torch.tensor(upstream_lens)).cpu().numpy()
        div_upstream = diversity(embu, diversity_times=min(300, len(embu) * (len(embu) - 1)),
                                 rng=rng)

    print("\n================ ISOLATION (diversity through the evaluator) ===========")
    print(f"  custom  _real_h3d (eval path)      : {div_recon}")
    print(f"  upstream process_file (canonical)  : {div_upstream}  "
          f"({n_upstream_fail} clip(s) failed)")
    print(f"  canonical new_joint_vecs           : {div_canon}")
    print("  reference (paper 'Div' real)       : ~9.555")
    print("  -> upstream ~9.5 => fix is to use the upstream extractor in eval")
    print("  -> upstream ~6.6 => bug is upstream of features (packed translation scale)")

    # (C) DISTRIBUTION vs evaluator's expected mean/std (no canonical npy needed).
    # The evaluator's mean.npy/std.npy ARE the canonical per-channel stats of the
    # 263-D space. If a reconstruction matches it, per-channel mean sits ~0 sigma
    # off and std-ratio ~1.
    def _dist_report(label: str, feats: list[torch.Tensor]) -> None:
        allfeat = torch.cat(feats, dim=0)                    # (sum L, 263)
        mu, sd = allfeat.mean(0), allfeat.std(0)
        em = evaluator._mean.detach().cpu()
        es = evaluator._std.detach().cpu()
        z_off = (mu - em) / es.clamp_min(1e-8)
        scale = sd / es.clamp_min(1e-8)
        print(f"\n========= DISTRIBUTION [{label}] vs evaluator mean/std ==========")
        print("  per block:  |mean offset| (eval-sigmas, ~0 good)   std_ratio (~1 good)")
        for name, a, b in BLOCKS:
            print(f"  {name}: mean_off={float(z_off[a:b].abs().mean()):.3f}   "
                  f"std_ratio={float(scale[a:b].mean()):.3f}")

    _dist_report("custom _real_h3d", recon_feats)
    if upstream_feats:
        _dist_report("upstream process_file", upstream_feats)
    print("\n  (std_ratio ~1 across blocks => that extractor matches the evaluator's")
    print("   space. Whichever does should be the one eval uses.)")

    # (B) PER-BLOCK DIFF -----------------------------------------------------
    if canon_feats:
        print("\n================ PER-BLOCK |recon - canonical| (mean abs) ==============")
        from collections import Counter
        print(f"  frame offset used (recon vs canon), histogram: {Counter(offsets_used)}")
        for name, _, _ in BLOCKS:
            errs = block_err[name]
            print(f"  {name}: mean={np.mean(errs):.5f}  p95={np.percentile(errs, 95):.5f}")
        print("  (a block that's ~0 matches canonical; a large one is where the")
        print("   reconstruction diverges — start there.)")
    else:
        print("\n[verify] no canonical npy matched — set +verify.h3d_vecs to the dir that")
        print("         holds <clip_id>.npy (HumanML3D new_joint_vecs) to enable the diff.")


if __name__ == "__main__":
    main()
