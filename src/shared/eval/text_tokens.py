"""Caption → pre-tagged word/POS token lookup from HumanML3D's texts.zip.

The Guo text encoder was trained on HumanML3D's *pre-tagged* tokens, which
include custom semantic tags (`left/Loc_VIP`, `walk/VERB`, ...) that fresh
spaCy POS tagging never reproduces — encoding captions via spaCy
(`encode_text_from_strings`) silently halves R-precision (see
guo_evaluator.encode_text_from_tokens). This module recovers the ground-truth
tokens for eval: each `texts/<clip_id>.txt` line inside texts.zip is

    <caption>#<word/POS word/POS ...>#<start>#<end>

so we index tokens by (clip_id, normalized caption) and look batches up.
"""

from __future__ import annotations

import zipfile
from pathlib import Path


def _norm(caption: str) -> str:
    return " ".join(caption.strip().lower().split())


class CaptionTokenLookup:
    """Lazy per-clip loader of HumanML3D pre-tagged caption tokens."""

    def __init__(self, humanml3d_repo: str | Path) -> None:
        p = Path(humanml3d_repo) / "HumanML3D" / "texts.zip"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found — need the HumanML3D submodule's texts.zip for "
                "pre-tagged word/POS tokens."
            )
        self._zf = zipfile.ZipFile(p)
        # texts.zip may nest entries under a "texts/" prefix; index once.
        self._names = {
            Path(n).stem: n for n in self._zf.namelist() if n.endswith(".txt")
        }
        self._cache: dict[str, dict[str, list[str]]] = {}

    def _load_clip(self, clip_id: str) -> dict[str, tuple[list[str], float, float]]:
        cached = self._cache.get(clip_id)
        if cached is not None:
            return cached
        out: dict[str, tuple[list[str], float, float]] = {}
        name = self._names.get(clip_id)
        if name is not None:
            for line in self._zf.read(name).decode("utf-8").splitlines():
                parts = line.split("#")
                if len(parts) >= 2 and parts[1].strip():
                    try:
                        start = float(parts[2]) if len(parts) > 2 and parts[2].strip() else 0.0
                        end = float(parts[3]) if len(parts) > 3 and parts[3].strip() else 0.0
                    except ValueError:
                        start = end = 0.0
                    out.setdefault(_norm(parts[0]), (parts[1].strip().split(" "), start, end))
        self._cache[clip_id] = out
        return out

    def get(self, clip_id: str, caption: str) -> list[str] | None:
        """Pre-tagged tokens for this caption, or None if not found."""
        hit = self._load_clip(clip_id).get(_norm(caption))
        return hit[0] if hit is not None else None

    def get_span(self, clip_id: str, caption: str) -> tuple[float, float] | None:
        """(start, end) seconds for this caption per texts.zip; (0, 0) = whole
        clip. In official HumanML3D, nonzero-span captions belong to a
        SEPARATE segmented clip id — a parent clip carrying them indicates the
        pack bundled sub-segment captions, which mispairs text and motion."""
        hit = self._load_clip(clip_id).get(_norm(caption))
        return (hit[1], hit[2]) if hit is not None else None

    def whole_clip_captions(self, clip_id: str) -> list[str]:
        """Captions of this clip whose span is (0, 0) (describe the whole clip)."""
        return [c for c, (_t, s, e) in self._load_clip(clip_id).items() if s == 0.0 and e == 0.0]


def encode_texts_prefer_tokens(
    evaluator,
    lookup: "CaptionTokenLookup | None",
    clip_ids: list[str],
    texts: list[str],
):
    """Text embeddings via the evaluator, preferring pre-tagged tokens.

    Items whose (clip_id, caption) resolve in `lookup` go through
    `encode_text_from_tokens` (ground-truth *_VIP POS tags); the rest fall back
    to `encode_text_from_strings` (spaCy). Results are merged back into input
    order. Returns (embeddings, n_fallback).
    """
    import torch

    from_tokens = getattr(evaluator, "encode_text_from_tokens", None)
    if lookup is None or from_tokens is None:
        return evaluator.encode_text_from_strings(texts), len(texts)

    resolved = [lookup.get(cid, cap) for cid, cap in zip(clip_ids, texts)]
    tok_idx = [i for i, r in enumerate(resolved) if r is not None]
    str_idx = [i for i, r in enumerate(resolved) if r is None]

    parts: list[tuple[list[int], torch.Tensor]] = []
    if tok_idx:
        parts.append((tok_idx, from_tokens([resolved[i] for i in tok_idx])))
    if str_idx:
        parts.append((str_idx, evaluator.encode_text_from_strings([texts[i] for i in str_idx])))

    ref = parts[0][1]
    out = torch.empty(len(texts), ref.shape[-1], dtype=ref.dtype, device=ref.device)
    for idx, emb in parts:
        out[torch.as_tensor(idx, device=ref.device)] = emb
    return out, len(str_idx)
