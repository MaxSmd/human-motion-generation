"""No-train sanity check for the eval pipeline.

Loads the *real* HumanML3D test motions from the packed dataset, runs them
through `representation.to_h3d_features` → Guo motion encoder, encodes the
*real* captions through the Guo text encoder, and reports:

    diversity_real, R@1/2/3, MM-Dist, FID(real-half-A, real-half-B)

These are the "GT row" of every HumanML3D paper table. If your decode +
evaluator pipeline is correct, the numbers should match (within a few %):

    diversity_real ≈ 9.503
    R@1 / R@2 / R@3 ≈ 0.511 / 0.703 / 0.797
    MM-Dist        ≈ 2.974
    FID(GT, GT)    ≈ 0.002

If they don't match, there is no point training further — the bug is in the
eval/decode path, not in the model. Fix this *before* committing more GPU hours.

Run example:
    python -m rmg.scripts.sanity_eval \\
        +eval.evaluator=real \\
        +eval.max_clips=-1
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
from rmg.eval import (
    RandomGuoEvaluator,
    RealGuoEvaluator,
    diversity,
    fid,
    mm_distance,
    r_precision,
)
from rmg.representation import (
    Representation,
    Skeleton,
    build_representation,
)
from rmg.utils import set_seed

import zipfile


def _load_pretagged_text_lookup(
    humanml3d_repo: str | Path, whole_clip_only: bool = True,
) -> dict[str, list[tuple[str, list[str]]]]:
    """Build {clip_id: [(caption, [word/POS, ...]), ...]} from texts.zip.

    Each line in HumanML3D's `texts.zip` is
        `<caption>#<word/POS word/POS ...>#<start>#<end>`
    where `start`/`end` are in seconds. When `start != 0` or `end != 0`, the
    caption describes a SUB-PORTION of the clip (not the whole motion). The
    Guo evaluator pairs caption with motion at *matching* frame indices; we
    feed the full motion in sanity_eval, so we must only evaluate against
    whole-clip captions (start = end = 0). Set `whole_clip_only=False` only
    if you handle sub-clip slicing yourself.

    Also includes the *_VIP semantic tags HumanML3D ships pre-tagged — fresh
    spaCy POS tagging at eval time loses these and silently halves R-precision.
    """
    p = Path(humanml3d_repo) / "HumanML3D" / "texts.zip"
    out: dict[str, list[tuple[str, list[str]]]] = {}
    n_total = 0
    n_kept = 0
    with zipfile.ZipFile(p) as zf:
        for name in zf.namelist():
            if not name.endswith(".txt"):
                continue
            clip_id = Path(name).stem
            captions: list[tuple[str, list[str]]] = []
            for line in zf.read(name).decode("utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split("#")
                if len(parts) < 4:
                    continue
                n_total += 1
                cap = parts[0].strip()
                tokens = parts[1].strip().split()
                try:
                    start = float(parts[2])
                    end = float(parts[3])
                except ValueError:
                    continue
                if whole_clip_only and (start != 0.0 or end != 0.0):
                    continue
                if cap and tokens:
                    captions.append((cap, tokens))
                    n_kept += 1
            if captions:
                out[clip_id] = captions
    print(f"[sanity_eval] text lookup: kept {n_kept}/{n_total} caption lines "
          f"(whole_clip_only={whole_clip_only}) across {len(out)} clips", flush=True)
    return out


# HumanML3D published GT-row reference values (Guo et al. 2022 Table 1, used in
# every follow-up paper's "Real motions" row). Yardstick for this sanity check.
PAPER_GT = {
    "diversity_real": 9.503,
    "r_precision_real": [0.511, 0.703, 0.797],
    "mm_dist_real": 2.974,
    "fid_real_self": 0.002,
}


def _build_dataset(cfg: DictConfig, split: str, representation: Representation) -> HumanML3DDataset:
    return HumanML3DDataset(
        root=cfg.data.root,
        split=split,
        max_seq_len=cfg.data.max_seq_len,
        min_seq_len=cfg.data.min_seq_len,
        mirror_augment=False,
        zip_name=cfg.data.zip_name,
        splits_name=cfg.data.splits_name,
        offsets_name=cfg.data.offsets_name,
        representation=representation,
    )


def _load_target_offsets(cfg: DictConfig) -> Skeleton:
    p = Path(cfg.data.root) / cfg.data.offsets_name
    offs = torch.load(p, weights_only=True)
    return Skeleton(offsets=offs)


def _build_evaluator(cfg: DictConfig, device: torch.device):
    if cfg.eval.evaluator == "real":
        return RealGuoEvaluator(
            text_to_motion_repo=cfg.eval.text_to_motion_repo,
            humanml3d_repo=cfg.eval.humanml3d_repo,
            device=device,
        )
    return RandomGuoEvaluator()


def _fmt_row(label: str, got, expected) -> str:
    """One line: '<label>  got=<v>  paper=<v>  Δ=<...>'."""
    if isinstance(got, (list, tuple)):
        got_s = "[" + ", ".join(f"{x:.4f}" for x in got) + "]"
        exp_s = "[" + ", ".join(f"{x:.4f}" for x in expected) + "]"
        delta = [g - e for g, e in zip(got, expected)]
        delta_s = "[" + ", ".join(f"{d:+.4f}" for d in delta) + "]"
    else:
        got_s = f"{got:.4f}"
        exp_s = f"{expected:.4f}"
        delta_s = f"{got - expected:+.4f}"
    return f"  {label:18s}  got={got_s:34s}  paper={exp_s:34s}  Δ={delta_s}"


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    # Inject sanity-eval defaults under cfg.eval (overridable at CLI with `+eval.*`).
    eval_cfg = OmegaConf.create({
        "split": "test",
        "evaluator": "real",                  # real | random
        "text_to_motion_repo": "external/text-to-motion",
        "humanml3d_repo": "external/HumanML3D",
        "batch_size": 32,
        "max_clips": -1,
        "diversity_times": 300,
        "seed": 0,
        "output_file": "sanity_eval.json",
    })
    cfg.eval = OmegaConf.merge(eval_cfg, cfg.get("eval", OmegaConf.create({})))
    set_seed(int(cfg.eval.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sanity_eval] device={device}  evaluator={cfg.eval.evaluator}  "
          f"split={cfg.eval.split}  max_clips={cfg.eval.max_clips}", flush=True)

    rep_kwargs = {k: v for k, v in dict(cfg.representation).items() if k not in ("name",)}
    representation = build_representation(cfg.representation.name, **rep_kwargs)
    print(f"[sanity_eval] representation={representation.name} "
          f"ambient_dim={representation.ambient_dim}", flush=True)

    ds = _build_dataset(cfg, split=cfg.eval.split, representation=representation)
    loader = DataLoader(
        ds, batch_size=cfg.eval.batch_size, shuffle=False,
        collate_fn=collate, num_workers=0, drop_last=False,
    )
    print(f"[sanity_eval] {cfg.eval.split} clips: {len(ds)}", flush=True)

    skeleton = _load_target_offsets(cfg)
    evaluator = _build_evaluator(cfg, device)

    pretagged_lookup = _load_pretagged_text_lookup(cfg.eval.humanml3d_repo)

    real_motion_feats: list[np.ndarray] = []
    text_feats: list[np.ndarray] = []
    n_seen = 0
    n_text_misses = 0
    cap_rng = np.random.default_rng(int(cfg.eval.seed))

    t0 = time.perf_counter()
    for batch in tqdm(loader, desc="decoding real motions"):
        x1 = batch.x1.to(device)
        lengths = batch.lengths

        # Decode each clip's real (T+R) → 263-D HumanML3D feature.
        feats: list[torch.Tensor] = []
        for i in range(x1.shape[0]):
            L = int(lengths[i].item())
            feats.append(representation.to_h3d_features(x1[i, :L], skeleton))

        # Pad to (max-1) for batch encoding (263-D feature is defined over T-1 frames).
        Tmax = int(lengths.max().item())
        padded = torch.zeros(len(feats), Tmax - 1, 263)
        for i, f in enumerate(feats):
            padded[i, : f.shape[0]] = f

        m_emb = evaluator.encode_motion(padded, lengths - 1)

        # Pick a WHOLE-CLIP pre-tagged caption per clip in the batch. This
        # overrides whatever caption the dataset's random.choice handed us
        # (the dataset doesn't filter sub-clip captions). Falls back to spaCy
        # if no whole-clip caption exists for the clip (extremely rare).
        tokens_per_clip: list[list[str]] = []
        fallback_texts: list[str] = []
        fallback_slots: list[int] = []
        for i, cid in enumerate(batch.clip_ids):
            caps = pretagged_lookup.get(cid, [])
            if caps:
                # Random whole-clip caption; seeded for reproducibility.
                cap, toks = caps[int(cap_rng.integers(len(caps)))]
                tokens_per_clip.append(toks)
            else:
                tokens_per_clip.append([])  # placeholder
                fallback_slots.append(i)
                fallback_texts.append(batch.texts[i])

        if fallback_slots:
            n_text_misses += len(fallback_slots)
            fallback_emb = evaluator.encode_text_from_strings(fallback_texts)
            good_slots = [i for i in range(len(tokens_per_clip)) if i not in set(fallback_slots)]
            if good_slots:
                good_emb = evaluator.encode_text_from_tokens(
                    [tokens_per_clip[i] for i in good_slots]
                )
                t_emb = torch.zeros(
                    len(tokens_per_clip), good_emb.shape[-1], device=good_emb.device
                )
                for j, i in enumerate(good_slots):
                    t_emb[i] = good_emb[j]
                for j, i in enumerate(fallback_slots):
                    t_emb[i] = fallback_emb[j].to(t_emb.device)
            else:
                t_emb = fallback_emb
        else:
            t_emb = evaluator.encode_text_from_tokens(tokens_per_clip)

        real_motion_feats.append(m_emb.cpu().numpy())
        text_feats.append(t_emb.cpu().numpy())

        n_seen += x1.shape[0]
        if cfg.eval.max_clips > 0 and n_seen >= cfg.eval.max_clips:
            break
    if n_text_misses:
        print(f"[sanity_eval] WARN: {n_text_misses}/{n_seen} clips had no whole-clip "
              f"caption; fell back to spaCy for those.", flush=True)

    M = np.concatenate(real_motion_feats, axis=0)
    T = np.concatenate(text_feats, axis=0)
    print(f"[sanity_eval] encoded {len(M)} clips in {time.perf_counter() - t0:.1f}s "
          f"(motion {M.shape}, text {T.shape})", flush=True)

    rng = np.random.default_rng(int(cfg.eval.seed))
    results: dict[str, object] = {
        "n_clips": int(len(M)),
        "split": str(cfg.eval.split),
        "evaluator": str(cfg.eval.evaluator),
        "representation": str(representation.name),
    }

    # Diversity on the *real* motion features. This is the headline diagnostic.
    results["diversity_real"] = diversity(M, diversity_times=int(cfg.eval.diversity_times), rng=rng)
    # R-precision pairs each real motion with its real caption — should match
    # the "Real motions" row of any HumanML3D paper.
    results["r_precision_real"] = r_precision(T, M, top_k=3, rng=rng).tolist()
    results["mm_dist_real"] = mm_distance(T, M)
    # FID(real, real) — split features in half, compute FID across the split.
    # A correct pipeline gives ~0 here (Frechet distance of two samples of
    # the same distribution); large values indicate normalization mismatch
    # or non-Gaussian-friendly features.
    half = len(M) // 2
    results["fid_real_self"] = fid(M[:half], M[half:])

    # ---- Report side-by-side ----
    print("\n=== sanity_eval results (GT row of HumanML3D evaluation) ===", flush=True)
    print(_fmt_row("diversity_real",   results["diversity_real"],   PAPER_GT["diversity_real"]))
    print(_fmt_row("R@1/R@2/R@3",       results["r_precision_real"], PAPER_GT["r_precision_real"]))
    print(_fmt_row("MM-Dist (real)",    results["mm_dist_real"],     PAPER_GT["mm_dist_real"]))
    print(_fmt_row("FID(real, real)",   results["fid_real_self"],    PAPER_GT["fid_real_self"]))

    # Verdict — quick visual signal.
    print("\n=== verdict ===", flush=True)
    div_ok = abs(results["diversity_real"] - PAPER_GT["diversity_real"]) < 1.5
    r3_ok = abs(results["r_precision_real"][2] - PAPER_GT["r_precision_real"][2]) < 0.1
    mm_ok = abs(results["mm_dist_real"] - PAPER_GT["mm_dist_real"]) < 1.0
    fid_ok = results["fid_real_self"] < 0.5
    for name, ok in [
        ("diversity_real within 1.5 of paper", div_ok),
        ("R@3 within 0.10 of paper",            r3_ok),
        ("MM-Dist within 1.0 of paper",         mm_ok),
        ("FID(real, real) < 0.5",               fid_ok),
    ]:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    all_ok = div_ok and r3_ok and mm_ok and fid_ok
    print(f"\nOverall: {'PASS — eval pipeline looks correct' if all_ok else 'FAIL — DO NOT TRAIN until fixed'}",
          flush=True)

    # ---- Save ----
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg.eval.output_file
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[sanity_eval] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
