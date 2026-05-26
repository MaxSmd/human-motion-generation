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


def _build_upstream_opt(staging: Path, text_to_motion_repo: Path) -> types.SimpleNamespace:
    opt = types.SimpleNamespace()
    opt.dataset_name = "t2m"
    opt.max_motion_length = 196
    opt.max_text_len = 20
    opt.unit_length = 4
    opt.motion_dir = str(staging / "new_joint_vecs")
    opt.text_dir = str(staging / "texts")
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

    # ------------------------------------------------------------------
    _section("3. Build upstream Text2MotionDatasetV2")
    # ------------------------------------------------------------------
    opt = _build_upstream_opt(staging, text_to_motion_repo)
    mean = np.load(
        text_to_motion_repo / "checkpoints" / "t2m" / "Comp_v6_KLD01" / "meta" / "mean.npy"
    )
    std = np.load(
        text_to_motion_repo / "checkpoints" / "t2m" / "Comp_v6_KLD01" / "meta" / "std.npy"
    )
    split_file = str(humanml3d_repo / "HumanML3D" / "test.txt")
    print(f"  motion_dir: {opt.motion_dir}  ({n_motions} npy files)", flush=True)
    print(f"  text_dir:   {opt.text_dir}    ({n_texts} txt files)", flush=True)
    print(f"  split:      {split_file}", flush=True)

    dataset = Text2MotionDatasetV2(opt, mean, std, split_file, w_vec)
    print(f"  dataset:    {len(dataset)} entries "
          f"(after pointer + sub-clip expansion)", flush=True)

    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0,
                        drop_last=True)
    print(f"  loader:     {len(loader)} batches × 32", flush=True)

    # ------------------------------------------------------------------
    _section("4. Eval with upstream EvaluatorModelWrapper + matching-score loop")
    # ------------------------------------------------------------------
    eval_wrapper = EvaluatorModelWrapper(opt)

    matching_score_sum = 0.0
    top_k_count = np.zeros(3, dtype=np.float64)
    all_size = 0
    all_motion_emb = []

    def _encode_motion(motions, m_lens):
        """Same as upstream.get_motion_embeddings but with sort + unsort so we
        can pair embeddings with text by original-batch index."""
        m_sort = torch.argsort(m_lens, descending=True)
        m_inv = torch.empty_like(m_sort)
        m_inv[m_sort] = torch.arange(m_sort.numel())
        sorted_emb = eval_wrapper.get_motion_embeddings(motions[m_sort], m_lens[m_sort])
        return sorted_emb[m_inv]

    def _encode_text(word_embs, pos_ohots, sent_lens):
        """text_encoder needs cap_lens sorted desc (pack_padded_sequence with
        enforce_sorted=True). Sort, encode, undo."""
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

    t0 = time.perf_counter()
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            word_embs, pos_ohots, _, sent_lens, motions, m_lens, _ = batch
            motion_emb = _encode_motion(motions, m_lens)
            text_emb = _encode_text(word_embs, pos_ohots, sent_lens)
            tn = text_emb.cpu().numpy()
            mn = motion_emb.cpu().numpy()
            B = tn.shape[0]
            # Euclidean distance matrix (B, B)
            diff = tn[:, None, :] - mn[None, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            matching_score_sum += float(np.trace(dist))
            order = np.argsort(dist, axis=1)
            for k in range(3):
                top_k_count[k] += float(((order[:, : k + 1] ==
                                          np.arange(B)[:, None]).any(axis=1)).sum())
            all_size += B
            all_motion_emb.append(mn)
            if (idx + 1) % 20 == 0:
                print(f"  batch {idx+1}/{len(loader)} ({time.perf_counter()-t0:.0f}s)",
                      flush=True)

    matching_score = matching_score_sum / max(all_size, 1)
    r_precision = top_k_count / max(all_size, 1)
    all_motion_emb_np = np.concatenate(all_motion_emb, axis=0)

    # Diversity on motion embeddings (same convention as our metrics.diversity)
    rng = np.random.default_rng(0)
    n_pairs = min(300, all_motion_emb_np.shape[0] // 2)
    idx_a = rng.choice(all_motion_emb_np.shape[0], n_pairs, replace=False)
    idx_b = rng.choice(all_motion_emb_np.shape[0], n_pairs, replace=False)
    diversity = float(
        np.linalg.norm(all_motion_emb_np[idx_a] - all_motion_emb_np[idx_b], axis=-1).mean()
    )

    print(f"\n  ALL_SIZE: {all_size}", flush=True)
    print(f"  Matching Score (MM-Dist):   {matching_score:.4f}   "
          f"paper={2.974:.4f}", flush=True)
    print(f"  R-precision  (top-1..top-3): "
          f"{r_precision[0]:.4f}  {r_precision[1]:.4f}  {r_precision[2]:.4f}",
          flush=True)
    print(f"                          paper:  0.5110  0.7030  0.7970", flush=True)
    print(f"  Diversity (motion-only):    {diversity:.4f}   paper={9.503:.4f}",
          flush=True)

    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
