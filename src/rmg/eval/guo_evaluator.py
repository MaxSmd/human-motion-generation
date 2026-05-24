"""Wrapper around the Guo et al. text-to-motion evaluator (CVPR 2022).

The evaluator's three networks (movement, motion, text) live in
`external/text-to-motion/networks/modules.py`. We:
  1. Add `external/text-to-motion` to `sys.path` so the module imports work.
  2. Construct a fake `opt` namespace with the constants the upstream code reads.
  3. Load the pretrained weights from `text_mot_match/model/finest.tar`.

Pretrained weights are NOT vendored (~hundreds of MB, distributed via Google
Drive). Download per `external/text-to-motion/README.md` and place under:

    external/text-to-motion/checkpoints/t2m/text_mot_match/model/finest.tar

Also required:
  - `external/text-to-motion/glove/` (already vendored — small).
  - `external/HumanML3D/HumanML3D/{Mean,Std}.npy` (vendored).

`RandomGuoEvaluator` is a stub for tests: returns deterministic random
embeddings of the same shape, with no checkpoint dependency.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class GuoEvaluator(Protocol):
    motion_dim: int
    text_dim: int

    def encode_motion(self, motion_features: Tensor, lengths: Tensor) -> Tensor:
        """(B, T, 263) + (B,) lengths → (B, motion_dim) motion embedding."""

    def encode_text_from_strings(self, texts: list[str]) -> Tensor:
        """list[str] → (B, text_dim) text embedding."""


# ---------------------------------------------------------------------------
# Real evaluator (loads upstream networks)
# ---------------------------------------------------------------------------


@dataclass
class _GuoOpts:
    """Mirrors the `opt` namespace the upstream `EvaluatorModelWrapper` reads."""

    checkpoints_dir: Path           # contains <dataset_name>/text_mot_match/model/finest.tar
    dataset_name: str = "t2m"
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    dim_pose: int = 263             # HumanML3D
    dim_word: int = 300             # GloVe
    dim_pos_ohot: int = 15          # POS_enumerator size
    dim_motion_hidden: int = 1024
    dim_text_hidden: int = 512
    dim_coemb_hidden: int = 512
    dim_movement_enc_hidden: int = 512
    dim_movement_latent: int = 512
    max_motion_length: int = 196
    max_text_len: int = 20
    unit_length: int = 4


class RealGuoEvaluator:
    motion_dim = 512
    text_dim = 512

    def __init__(
        self,
        text_to_motion_repo: str | Path = "external/text-to-motion",
        humanml3d_repo: str | Path = "external/HumanML3D",
        device: str | torch.device = "cpu",
    ) -> None:
        repo = Path(text_to_motion_repo).resolve()
        if not (repo / "networks" / "modules.py").exists():
            raise FileNotFoundError(f"text-to-motion submodule not initialized at {repo}")
        sys.path.insert(0, str(repo))

        from networks.evaluator_wrapper import EvaluatorModelWrapper
        from utils.word_vectorizer import WordVectorizer

        self._device = torch.device(device)
        opt = _GuoOpts(checkpoints_dir=repo / "checkpoints", device=self._device)
        # EvaluatorModelWrapper expects opt.checkpoints_dir to be a string-coercible path
        opt_ns = type("Opt", (), opt.__dict__)()  # convert dataclass → simple object
        opt_ns.checkpoints_dir = str(opt.checkpoints_dir)
        self._wrapper = EvaluatorModelWrapper(opt_ns)

        self._word_vec = WordVectorizer(str(repo / "glove"), "our_vab")

        # Mean / Std for 263-D feature normalization. CRITICAL: must use the
        # files shipped *with the evaluator checkpoint* (Comp_v6_KLD01/meta/),
        # NOT the HumanML3D repo's Mean.npy/Std.npy. They differ — the
        # evaluator was trained on a specific normalization, and feeding
        # features normalized with different stats gives a 2-3× scale
        # mismatch in embeddings (diversity_real ≈ 4 instead of ~9.5).
        meta = repo / "checkpoints" / opt.dataset_name / "Comp_v6_KLD01" / "meta"
        mean_path = meta / "mean.npy"
        std_path = meta / "std.npy"
        if not mean_path.exists():
            # Fallback to HumanML3D repo (kept for tests / partial setups).
            print(f"[guo] WARN: {mean_path} not found, falling back to HumanML3D Mean.npy", flush=True)
            h3d = Path(humanml3d_repo)
            mean_path = h3d / "HumanML3D" / "Mean.npy"
            std_path = h3d / "HumanML3D" / "Std.npy"
        self._mean = torch.from_numpy(np.load(mean_path)).float().to(self._device)
        self._std = torch.from_numpy(np.load(std_path)).float().to(self._device)
        print(f"[guo] loaded normalization from {mean_path}", flush=True)

    # ------------------------------------------------------------ helpers

    def normalize(self, motion_263: Tensor) -> Tensor:
        return (motion_263 - self._mean) / self._std.clamp_min(1e-8)

    def _tokenize_for_text_enc(self, texts: list[str]) -> tuple[Tensor, Tensor, Tensor]:
        """Convert plain strings → (word_embs, pos_ohot, cap_lens) the
        upstream BiGRU text encoder expects. Mirrors `WordVectorizer` + spaCy
        POS tagging used in upstream eval scripts."""
        try:
            import spacy  # type: ignore
        except ImportError as e:
            raise ImportError(
                "RealGuoEvaluator needs spaCy + en_core_web_sm. Install via "
                "containers/requirements.txt; download the model with "
                "`python -m spacy download en_core_web_sm`."
            ) from e
        nlp = spacy.load("en_core_web_sm")

        max_len = 20
        word_embs = torch.zeros(len(texts), max_len + 2, 300)
        pos_ohot = torch.zeros(len(texts), max_len + 2, 15)
        cap_lens = torch.zeros(len(texts), dtype=torch.long)
        for i, text in enumerate(texts):
            doc = nlp(text.lower())
            tokens = [t.text for t in doc][:max_len]
            poss = [t.pos_ for t in doc][:max_len]
            # Upstream's WordVectorizer splits the "word/POS" token on '/' and
            # expects exactly 2 parts, so the word itself must not contain '/'.
            # Captions like "turn left/right" would otherwise break it.
            tokens = (
                ["sos/OTHER"]
                + [f"{w.replace('/', '')}/{p}" for w, p in zip(tokens, poss)]
                + ["eos/OTHER"]
            )
            for j, tok in enumerate(tokens):
                vec, pos = self._word_vec[tok]
                word_embs[i, j] = torch.from_numpy(vec)
                pos_ohot[i, j] = torch.from_numpy(pos)
            cap_lens[i] = len(tokens)
        return word_embs.to(self._device), pos_ohot.to(self._device), cap_lens

    # ------------------------------------------------------------ public API

    def encode_motion(self, motion_features: Tensor, lengths: Tensor) -> Tensor:
        x = self.normalize(motion_features.to(self._device).float())
        return self._wrapper.get_motion_embeddings(x, lengths.to(self._device).long())

    def encode_text_from_strings(self, texts: list[str]) -> Tensor:
        word_embs, pos_ohot, cap_lens = self._tokenize_for_text_enc(texts)
        # The upstream wrapper does paired encoding only (text + motion).
        # Pull the text encoder directly. Upstream's forward calls
        # `pack_padded_sequence(..., enforce_sorted=True)` (the default), which
        # requires lengths sorted descending — sort, encode, undo the sort.
        sort_idx = torch.argsort(cap_lens, descending=True)
        inv_idx = torch.empty_like(sort_idx)
        inv_idx[sort_idx] = torch.arange(sort_idx.numel(), device=sort_idx.device)
        with torch.no_grad():
            emb = self._wrapper.text_encoder(
                word_embs[sort_idx], pos_ohot[sort_idx], cap_lens[sort_idx],
            )
        return emb[inv_idx]


# ---------------------------------------------------------------------------
# Stub evaluator for tests
# ---------------------------------------------------------------------------


class RandomGuoEvaluator:
    """Deterministic-per-input stub. Hashes inputs, emits Gaussian features.
    Used by `tests/test_eval.py` so the metrics pipeline can be exercised
    without the real (huge) checkpoints."""

    def __init__(self, motion_dim: int = 512, text_dim: int = 512, seed: int = 0) -> None:
        self.motion_dim = motion_dim
        self.text_dim = text_dim
        self._seed = seed

    def encode_motion(self, motion_features: Tensor, lengths: Tensor) -> Tensor:
        out = torch.empty(motion_features.shape[0], self.motion_dim)
        # Hash a deterministic summary of each motion + length to seed.
        for i in range(motion_features.shape[0]):
            sig = int(motion_features[i].sum().item() * 1e6) ^ int(lengths[i].item()) ^ self._seed
            g = torch.Generator().manual_seed(sig & 0xFFFFFFFF)
            out[i] = torch.randn(self.motion_dim, generator=g)
        return out

    def encode_text_from_strings(self, texts: list[str]) -> Tensor:
        out = torch.empty(len(texts), self.text_dim)
        for i, t in enumerate(texts):
            g = torch.Generator().manual_seed((hash(t) ^ self._seed) & 0xFFFFFFFF)
            out[i] = torch.randn(self.text_dim, generator=g)
        return out
