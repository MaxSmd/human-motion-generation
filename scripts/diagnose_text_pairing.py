"""Localize the residual R@k / MM-Dist gap by checking what upstream's
own data pipeline assumes about (text, motion) pairing.

Specifically:
  1. Does HumanML3D's texts.zip ship M-prefixed mirrored captions?
     - If yes, our prep correctly looks them up.
     - If no, M-clip captions are systematically wrong (L/R-swapped motion
       paired with original "wave left hand" text) → ~50% of test pairs
       are mislabeled → explains MM-Dist and R@k gap.

  2. Spot-check 5 captions: how many tokens do they have, do our pre-tagged
     tokens match upstream's expected format, what proportion go to `unk` in
     the GloVe vocab.

  3. Count whole-clip vs sub-clip captions and motion length distribution —
     gives a sense of how much sub-clip evaluation matters.

No GPU, no model load — runs in seconds.

Usage:
    python scripts/diagnose_text_pairing.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    humanml3d_repo = REPO / "external" / "HumanML3D"
    text_to_motion_repo = REPO / "external" / "text-to-motion"
    packed_root = REPO / "external" / "data" / "humanml3d_packed"

    texts_zip = humanml3d_repo / "HumanML3D" / "texts.zip"

    # ------------------------------------------------------------------
    section("1. M-prefixed (mirrored) caption files in texts.zip")
    # ------------------------------------------------------------------
    with zipfile.ZipFile(texts_zip) as zf:
        all_txt = [n for n in zf.namelist() if n.endswith(".txt")]
        m_txt = [n for n in all_txt if Path(n).stem.startswith("M")]
        reg_txt = [n for n in all_txt if not Path(n).stem.startswith("M")]
        print(f"total .txt files in texts.zip: {len(all_txt)}")
        print(f"  regular:                     {len(reg_txt)}")
        print(f"  M-prefixed (mirrored):       {len(m_txt)}")

        # Spot-check 3 clip pairs to see whether mirrored captions differ
        # (i.e. upstream did left↔right text swap) or are identical.
        if m_txt:
            print("\nSpot-check 5 (regular, mirrored) caption pairs:")
            samples = ["000021", "000019", "000022", "000026", "000050"]
            for cid in samples:
                reg_name = f"{cid}.txt"
                mir_name = f"M{cid}.txt"
                if reg_name in all_txt and mir_name in all_txt:
                    reg_first = zf.read(reg_name).decode().splitlines()[0]
                    mir_first = zf.read(mir_name).decode().splitlines()[0]
                    reg_cap = reg_first.split("#")[0]
                    mir_cap = mir_first.split("#")[0]
                    same = "IDENTICAL" if reg_cap == mir_cap else "DIFFERENT"
                    print(f"  {cid}: {same}")
                    print(f"    reg : {reg_cap}")
                    print(f"    mir : {mir_cap}")

    # ------------------------------------------------------------------
    section("2. Packed dataset — do M-prefixed clips have their own captions?")
    # ------------------------------------------------------------------
    import io
    import torch

    packed_zip = packed_root / "humanml3d.zip"
    with zipfile.ZipFile(packed_zip) as zf:
        names = zf.namelist()
        m_clips = [n for n in names if Path(n).stem.startswith("M")]
        print(f"packed clips: total={len(names)}  M-prefixed={len(m_clips)}")

        # For 3 M-clips, compare their stored captions to the regular clip's.
        # If identical → our prep used the original captions for the mirror
        # (because M-captions weren't found in texts_by_clip).
        print("\nFor 3 M-clips: are stored captions the same as the regular clip's?")
        for cid in ["000021", "000019", "000022"]:
            reg_name = f"{cid}.pt"
            mir_name = f"M{cid}.pt"
            if reg_name in names and mir_name in names:
                reg_blob = torch.load(io.BytesIO(zf.read(reg_name)), weights_only=False)
                mir_blob = torch.load(io.BytesIO(zf.read(mir_name)), weights_only=False)
                same = set(reg_blob["texts"]) == set(mir_blob["texts"])
                print(f"  {cid}: {'SAME captions' if same else 'DIFFERENT captions'}")
                print(f"    reg [0]: {reg_blob['texts'][0][:90]}")
                print(f"    mir [0]: {mir_blob['texts'][0][:90]}")

    # ------------------------------------------------------------------
    section("3. Sub-clip vs whole-clip caption counts (eval-pool impact)")
    # ------------------------------------------------------------------
    n_total = 0
    n_whole = 0
    n_sub = 0
    n_lines_bad = 0
    with zipfile.ZipFile(texts_zip) as zf:
        for name in all_txt:
            for line in zf.read(name).decode().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split("#")
                if len(parts) < 4:
                    n_lines_bad += 1
                    continue
                try:
                    s = float(parts[2])
                    e = float(parts[3])
                except ValueError:
                    n_lines_bad += 1
                    continue
                n_total += 1
                if s == 0.0 and e == 0.0:
                    n_whole += 1
                else:
                    n_sub += 1
    print(f"total caption lines: {n_total}    bad-format: {n_lines_bad}")
    print(f"  whole-clip (start=end=0): {n_whole}  ({100*n_whole/n_total:.1f}%)")
    print(f"  sub-clip  (s≠0 or e≠0):   {n_sub}    ({100*n_sub/n_total:.1f}%)")

    # ------------------------------------------------------------------
    section("4. WordVectorizer coverage on 100 test captions")
    # ------------------------------------------------------------------
    # If many words fall back to 'unk', text embeddings degrade systematically.
    sys.path.insert(0, str(text_to_motion_repo))
    try:
        from utils.word_vectorizer import WordVectorizer
    except ImportError as e:
        print(f"could not import WordVectorizer: {e}")
        return

    w_vec = WordVectorizer(str(text_to_motion_repo / "glove"), "our_vab")
    test_ids_path = humanml3d_repo / "HumanML3D" / "test.txt"
    test_ids = [l.strip() for l in test_ids_path.read_text().splitlines() if l.strip()][:100]

    n_tokens = 0
    n_unk = 0
    unk_words = []
    with zipfile.ZipFile(texts_zip) as zf:
        for tid in test_ids:
            txt_name = f"{tid}.txt"
            if txt_name not in zf.namelist():
                continue
            for line in zf.read(txt_name).decode().splitlines():
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
                        if len(unk_words) < 20:
                            unk_words.append(tok)

    print(f"tokens scanned: {n_tokens}")
    print(f"  in vocab:    {n_tokens - n_unk}  ({100*(n_tokens-n_unk)/max(n_tokens,1):.1f}%)")
    print(f"  unk (→ fallback): {n_unk}  ({100*n_unk/max(n_tokens,1):.2f}%)")
    if unk_words:
        print(f"  first 20 unk tokens: {unk_words}")

    print("\n[done]")


if __name__ == "__main__":
    main()
