"""Gold-standard test: feed our packed data through UPSTREAM's exact eval
pipeline (`Text2MotionDatasetV2` + `final_evaluations.evaluate_matching_score`)
and report R@k / MM-Dist.

If this yields paper-level R@k on our data → the gap is in our sanity_eval
wrapper (eval protocol differs).
If it yields the same low R@k → our packed data is subtly different from
what HumanML3D ships, even though `process_file` output is byte-equal.

Steps:
  1. Decode every clip in our packed zip → 263-D feature, save as
     `<staging>/new_joint_vecs/<clip>.npy` (the format upstream expects).
  2. Extract HumanML3D's texts.zip to `<staging>/texts/`.
  3. Copy test.txt to `<staging>/test.txt`.
  4. Build upstream's `Text2MotionDatasetV2` with our motion + their texts.
  5. Iterate with DataLoader(batch_size=32), call `get_co_embeddings`,
     compute matching-score / R@k exactly like `final_evaluations.py`.
  6. Also: re-do section 4 from diagnose_text_pairing.py (WordVectorizer
     coverage) with the texts.zip subdir prefix bug fixed.

No spaCy, no our-wrapper text encoding — purely upstream's protocol.

Run via slurm/diagnose_text.sbatch (CPU is fine; the only GPU usage is the
Guo eval model forward — already mounted).
"""

from __future__ import annotations

import io
import shutil
import sys
import time
import types
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def _section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def _decode_clips_to_upstream_format(packed_zip: Path, out_motion_dir: Path) -> int:
    """For every clip in our packed zip, decode (translation, quats) → 263-D
    feature via the same path sanity_eval uses, save as <out>/<clip>.npy."""
    from rmg.representation import Skeleton, build_representation

    rep = build_representation("tr")
    target_offsets = torch.load(
        packed_zip.parent / "target_offsets.pt", weights_only=True
    ).float()
    skel = Skeleton(offsets=target_offsets)

    out_motion_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    t0 = time.perf_counter()
    with zipfile.ZipFile(packed_zip) as zf:
        names = zf.namelist()
        for name in names:
            clip_id = name[:-3] if name.endswith(".pt") else name
            blob = torch.load(io.BytesIO(zf.read(name)), weights_only=False)
            trans = blob["translation"].float()
            quats = blob["quats"].float()
            # Same encoding the dataset uses: pack into (T, 91)
            x1 = torch.cat([trans, quats.reshape(quats.shape[0], -1)], dim=-1)
            feats = rep.to_h3d_features(x1, skel).numpy().astype(np.float32)
            np.save(out_motion_dir / f"{clip_id}.npy", feats)
            n += 1
            if n % 2000 == 0:
                print(f"  decoded {n}/{len(names)} clips "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"  decoded all {n} clips in {time.perf_counter() - t0:.0f}s", flush=True)
    return n


def _extract_texts_zip(texts_zip: Path, out_text_dir: Path) -> int:
    """Extract every .txt file out of texts.zip into out_text_dir (flat)."""
    out_text_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(texts_zip) as zf:
        for name in zf.namelist():
            if not name.endswith(".txt"):
                continue
            data = zf.read(name)
            target = out_text_dir / Path(name).name  # flatten
            target.write_bytes(data)
            n += 1
    print(f"  extracted {n} text files to {out_text_dir}", flush=True)
    return n


def _vocab_coverage_check(texts_zip: Path, w_vec) -> None:
    """Section 4 done right: walk every .txt entry in texts.zip (with the
    subdir prefix correctly handled), look up each word in WordVectorizer's
    GloVe vocab, report miss rate."""
    n_tokens = 0
    n_unk = 0
    unk_examples: list[str] = []
    with zipfile.ZipFile(texts_zip) as zf:
        # Just sample 500 files for speed; representative.
        names = [n for n in zf.namelist() if n.endswith(".txt")][:500]
        for name in names:
            for line in zf.read(name).decode().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split("#")
                if len(parts) < 2:
                    continue
                for tok in parts[1].strip().split():
                    n_tokens += 1
                    word, _, _ = tok.partition("/")
                    if word not in w_vec.word2vec:
                        n_unk += 1
                        if len(unk_examples) < 30:
                            unk_examples.append(tok)
    print(f"  scanned {n_tokens} tokens from {len(names)} text files", flush=True)
    print(f"  in vocab:   {n_tokens - n_unk}  ({100*(n_tokens-n_unk)/max(n_tokens,1):.2f}%)",
          flush=True)
    print(f"  unk:        {n_unk}  ({100*n_unk/max(n_tokens,1):.2f}%)", flush=True)
    if unk_examples:
        print(f"  first 30 unk tokens: {unk_examples}", flush=True)


def _build_upstream_opt(motion_dir: Path, text_dir: Path, text_to_motion_repo: Path) -> types.SimpleNamespace:
    opt = types.SimpleNamespace()
    opt.dataset_name = "t2m"
    opt.max_motion_length = 196
    opt.max_text_len = 20
    opt.unit_length = 4
    opt.motion_dir = str(motion_dir)
    opt.text_dir = str(text_dir)
    opt.joints_num = 22
    opt.dim_pose = 263
    opt.checkpoints_dir = str(text_to_motion_repo / "checkpoints")
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.dim_word = 300
    opt.dim_pos_ohot = 15
    opt.dim_motion_hidden = 1024
    opt.dim_text_hidden = 512
    opt.dim_coemb_hidden = 512
    opt.dim_movement_enc_hidden = 512
    opt.dim_movement_latent = 512
    return opt


def _count_nan_in_motion_dir(motion_dir: Path) -> tuple[int, list[str]]:
    """Scan all .npy files for NaN/inf; return (count, list of bad clip names)."""
    bad = []
    for npy in sorted(motion_dir.glob("*.npy")):
        arr = np.load(npy)
        if not np.isfinite(arr).all():
            bad.append(npy.stem)
    return len(bad), bad


def _run_eval(label: str, motion_dir: Path, text_dir: Path, split_file: Path,
              text_to_motion_repo: Path, humanml3d_repo: Path) -> None:
    """Run upstream's Text2MotionDatasetV2 + matching-score loop on the given
    motion_dir / text_dir / split_file. NaN-robust: drops NaN rows from R@k."""
    from data.dataset import Text2MotionDatasetV2  # type: ignore
    from networks.evaluator_wrapper import EvaluatorModelWrapper  # type: ignore
    from utils.word_vectorizer import WordVectorizer  # type: ignore

    print(f"\n----- {label} -----", flush=True)
    print(f"  motion_dir: {motion_dir}", flush=True)
    print(f"  text_dir:   {text_dir}", flush=True)
    print(f"  split:      {split_file}", flush=True)

    # Spot-check 3 files: what shape and dtype are these?
    sample_files = sorted(motion_dir.glob("*.npy"))[:3]
    print(f"  sample file shapes:", flush=True)
    for f in sample_files:
        try:
            arr = np.load(f)
            print(f"    {f.name}: shape={arr.shape}  dtype={arr.dtype}  "
                  f"finite={bool(np.isfinite(arr).all())}", flush=True)
        except Exception as e:
            print(f"    {f.name}: FAILED TO LOAD ({type(e).__name__}: {e})", flush=True)

    # Verify a few IDs from the split actually have a matching file
    split_ids = [l.strip() for l in Path(split_file).read_text().splitlines() if l.strip()]
    missing = [sid for sid in split_ids[:20] if not (motion_dir / f"{sid}.npy").exists()]
    if missing:
        print(f"  first-20 split IDs missing in motion_dir: {missing}", flush=True)
    text_missing = [sid for sid in split_ids[:20] if not (text_dir / f"{sid}.txt").exists()]
    if text_missing:
        print(f"  first-20 split IDs missing in text_dir: {text_missing}", flush=True)

    n_bad, bad = _count_nan_in_motion_dir(motion_dir)
    print(f"  NaN/inf motion .npy files: {n_bad}", flush=True)
    if n_bad and n_bad <= 20:
        print(f"  bad clips: {bad}", flush=True)

    opt = _build_upstream_opt(motion_dir, text_dir, text_to_motion_repo)
    w_vec = WordVectorizer(str(text_to_motion_repo / "glove"), "our_vab")
    mean = np.load(text_to_motion_repo / "checkpoints/t2m/Comp_v6_KLD01/meta/mean.npy")
    std  = np.load(text_to_motion_repo / "checkpoints/t2m/Comp_v6_KLD01/meta/std.npy")
    try:
        dataset = Text2MotionDatasetV2(opt, mean, std, str(split_file), w_vec)
    except ValueError as e:
        print(f"  Text2MotionDatasetV2 ctor failed: {e}", flush=True)
        print(f"  (likely: all motion files filtered out — wrong shape, wrong content, "
              f"or text files missing)", flush=True)
        return
    if len(dataset) == 0:
        print(f"  dataset is EMPTY — no valid (motion, text) pairs constructed", flush=True)
        return
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
    print(f"  dataset: {len(dataset)} entries  loader: {len(loader)} batches", flush=True)

    eval_wrapper = EvaluatorModelWrapper(opt)

    def _encode_motion(motions, m_lens):
        m_sort = torch.argsort(m_lens, descending=True)
        m_inv = torch.empty_like(m_sort)
        m_inv[m_sort] = torch.arange(m_sort.numel())
        sorted_emb = eval_wrapper.get_motion_embeddings(motions[m_sort], m_lens[m_sort])
        return sorted_emb[m_inv]

    def _encode_text(word_embs, pos_ohots, sent_lens):
        t_sort = torch.argsort(sent_lens, descending=True)
        t_inv = torch.empty_like(t_sort)
        t_inv[t_sort] = torch.arange(t_sort.numel())
        with torch.no_grad():
            emb = eval_wrapper.text_encoder(
                word_embs[t_sort].to(opt.device).float(),
                pos_ohots[t_sort].to(opt.device).float(),
                sent_lens[t_sort],
            )
        return emb[t_inv]

    all_text = []
    all_motion = []
    with torch.no_grad():
        for batch in loader:
            word_embs, pos_ohots, _, sent_lens, motions, m_lens, _ = batch
            text_emb = _encode_text(word_embs, pos_ohots, sent_lens).cpu().numpy()
            motion_emb = _encode_motion(motions, m_lens).cpu().numpy()
            all_text.append(text_emb)
            all_motion.append(motion_emb)
    T = np.concatenate(all_text, axis=0)
    M = np.concatenate(all_motion, axis=0)

    # NaN-robust R@k + MM-Dist on paired (T, M)
    good = np.isfinite(T).all(-1) & np.isfinite(M).all(-1)
    n_dropped = int((~good).sum())
    if n_dropped:
        print(f"  dropping {n_dropped}/{len(T)} pairs with NaN/inf embeddings", flush=True)
    T = T[good]; M = M[good]
    n = len(T)
    bs = 32
    nb = n // bs
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    T = T[perm]; M = M[perm]
    top_k = np.zeros(3, dtype=np.float64)
    mm_sum = 0.0
    for b in range(nb):
        s = slice(b * bs, (b + 1) * bs)
        diff = T[s][:, None, :] - M[s][None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        order = np.argsort(dist, axis=1)
        for k in range(3):
            top_k[k] += ((order[:, : k + 1] == np.arange(bs)[:, None]).any(axis=1)).sum()
        mm_sum += float(np.trace(dist))
    r_prec = top_k / (nb * bs)
    mm_dist = mm_sum / (nb * bs)
    # Diversity on motion embeddings
    n_pairs = min(300, len(M) // 2)
    ia = rng.choice(len(M), n_pairs, replace=False)
    ib = rng.choice(len(M), n_pairs, replace=False)
    diversity = float(np.linalg.norm(M[ia] - M[ib], axis=-1).mean())

    print(f"  pairs used:        {nb * bs}", flush=True)
    print(f"  MM-Dist:           {mm_dist:.4f}   paper=2.974", flush=True)
    print(f"  R@1 / R@2 / R@3:   {r_prec[0]:.4f}  {r_prec[1]:.4f}  {r_prec[2]:.4f}",
          flush=True)
    print(f"                 paper:  0.5110  0.7030  0.7970", flush=True)
    print(f"  diversity_real:    {diversity:.4f}   paper=9.503", flush=True)


def main() -> None:
    text_to_motion_repo = REPO / "external" / "text-to-motion"
    humanml3d_repo = REPO / "external" / "HumanML3D"
    packed_zip = REPO / "external" / "data" / "humanml3d_packed" / "humanml3d.zip"
    staging = Path("/tmp/rmg_upstream_eval_staging")
    # Don't blow away staging — decoding 27k clips takes ~16 min. If we want
    # a fresh decode (e.g. after re-prep), `rm -rf /tmp/rmg_upstream_eval_staging`.
    staging.mkdir(parents=True, exist_ok=True)
    motion_dir = staging / "new_joint_vecs"
    text_dir = staging / "texts"
    has_motions = motion_dir.exists() and any(motion_dir.glob("*.npy"))
    has_texts = text_dir.exists() and any(text_dir.glob("*.txt"))
    if has_motions:
        print(f"[upstream_eval] staging motions already present, skipping decode")
    if has_texts:
        print(f"[upstream_eval] staging texts already present, skipping extract")

    sys.path.insert(0, str(text_to_motion_repo))
    from utils.word_vectorizer import WordVectorizer
    from data.dataset import Text2MotionDatasetV2
    from networks.evaluator_wrapper import EvaluatorModelWrapper

    # ------------------------------------------------------------------
    _section("0. WordVectorizer vocab coverage (section 4, fixed)")
    # ------------------------------------------------------------------
    w_vec = WordVectorizer(str(text_to_motion_repo / "glove"), "our_vab")
    _vocab_coverage_check(humanml3d_repo / "HumanML3D" / "texts.zip", w_vec)

    # ------------------------------------------------------------------
    _section("1. Decode our packed clips → 263-D .npy (upstream format)")
    # ------------------------------------------------------------------
    if has_motions:
        n_motions = len(list(motion_dir.glob("*.npy")))
        print(f"  skipped: {n_motions} npy files already in {motion_dir}", flush=True)
    else:
        n_motions = _decode_clips_to_upstream_format(packed_zip, motion_dir)

    # ------------------------------------------------------------------
    _section("2. Extract texts.zip → flat .txt directory")
    # ------------------------------------------------------------------
    if has_texts:
        n_texts = len(list(text_dir.glob("*.txt")))
        print(f"  skipped: {n_texts} txt files already in {text_dir}", flush=True)
    else:
        n_texts = _extract_texts_zip(
            humanml3d_repo / "HumanML3D" / "texts.zip", text_dir
        )

    split_file = humanml3d_repo / "HumanML3D" / "test.txt"

    # ------------------------------------------------------------------
    _section("3. Upstream pipeline on OUR data")
    # ------------------------------------------------------------------
    _run_eval(
        "OUR packed data → upstream pipeline",
        motion_dir=motion_dir,
        text_dir=text_dir,
        split_file=split_file,
        text_to_motion_repo=text_to_motion_repo,
        humanml3d_repo=humanml3d_repo,
    )

    # ------------------------------------------------------------------
    _section("4. Upstream pipeline on UPSTREAM's shipped data (if available)")
    # ------------------------------------------------------------------
    upstream_motion_dir = humanml3d_repo / "HumanML3D" / "new_joint_vecs"
    if upstream_motion_dir.exists() and any(upstream_motion_dir.glob("*.npy")):
        _run_eval(
            "UPSTREAM shipped data → upstream pipeline (the true reference)",
            motion_dir=upstream_motion_dir,
            text_dir=text_dir,
            split_file=split_file,
            text_to_motion_repo=text_to_motion_repo,
            humanml3d_repo=humanml3d_repo,
        )
    else:
        print(f"  upstream's new_joint_vecs/ not found at {upstream_motion_dir}", flush=True)
        print(f"  → cannot test upstream's pipeline on upstream's data as ground truth",
              flush=True)
        print(f"  → either run external/HumanML3D/motion_representation.ipynb to "
              f"generate it, or download the precomputed features from HumanML3D's "
              f"Google Drive", flush=True)

    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
