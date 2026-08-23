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


def _upstream_align_indices(m_lens: Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Upstream's length ordering (descending, for pack_padded_sequence) + its inverse.

    `np.argsort(x)[::-1]` is *not* a descending sort: on tied values it emits the
    tie group in reverse. Evaluation truncates every long clip to a common frame
    cap, so one huge tie group is the norm — assuming this permutation is a no-op
    silently mis-pairs each tied motion with another clip's caption.
    """
    align = np.argsort(m_lens.detach().cpu().numpy())[::-1].copy()
    inv = np.empty_like(align)
    inv[align] = np.arange(align.size)
    return align, inv


def _resolve_evaluator_assets(
    text_to_motion_repo: str | Path,
    humanml3d_repo: str | Path,
    checkpoints_dir: str | Path | None,
    normalization_name: str,
) -> tuple[Path, Path, Path]:
    """Resolve one matching checkpoint and its paired motion normalization."""
    repo = Path(text_to_motion_repo).resolve()
    explicit_checkpoint_root = checkpoints_dir is not None
    checkpoint_root = (
        Path(checkpoints_dir).resolve()
        if explicit_checkpoint_root
        else repo / "checkpoints"
    )
    checkpoint = checkpoint_root / "t2m" / "text_mot_match" / "model" / "finest.tar"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Guo evaluator checkpoint not found: {checkpoint}")

    meta = checkpoint_root / "t2m" / normalization_name / "meta"
    mean_path = meta / "mean.npy"
    std_path = meta / "std.npy"
    if mean_path.is_file() and std_path.is_file():
        return checkpoint_root, mean_path, std_path

    # Preserve the old partial-setup fallback only for the legacy default.
    # An explicitly selected MoMask evaluator must fail rather than silently
    # mixing its checkpoint with unrelated HumanML3D statistics.
    if explicit_checkpoint_root or normalization_name != "Comp_v6_KLD01":
        raise FileNotFoundError(
            "Guo evaluator normalization not found: "
            f"expected {mean_path} and {std_path}"
        )

    h3d = Path(humanml3d_repo)
    fallback_mean = h3d / "HumanML3D" / "Mean.npy"
    fallback_std = h3d / "HumanML3D" / "Std.npy"
    if not fallback_mean.is_file() or not fallback_std.is_file():
        raise FileNotFoundError(
            "Guo evaluator normalization not found in either evaluator assets "
            f"({meta}) or HumanML3D ({fallback_mean.parent})"
        )
    print(
        f"[guo] WARN: {mean_path} not found, falling back to {fallback_mean}",
        flush=True,
    )
    return checkpoint_root, fallback_mean, fallback_std


def _normalize_motion_batch(
    motion: Tensor,
    lengths: Tensor | None,
    mean: Tensor,
    std: Tensor,
) -> Tensor:
    """Normalize valid frames and keep temporal padding at normalized zero."""
    x = (motion - mean.to(device=motion.device, dtype=motion.dtype)) / std.to(
        device=motion.device,
        dtype=motion.dtype,
    ).clamp_min(1e-8)
    if lengths is None:
        return x
    if motion.ndim != 3 or lengths.ndim != 1 or lengths.shape[0] != motion.shape[0]:
        raise ValueError(
            "length-aware normalization expects motion (B,T,D) and lengths (B,)"
        )
    lengths = lengths.to(device=motion.device, dtype=torch.long)
    if bool((lengths < 0).any()) or bool((lengths > motion.shape[1]).any()):
        raise ValueError(f"motion lengths must be within [0, {motion.shape[1]}]")
    valid = torch.arange(motion.shape[1], device=motion.device).unsqueeze(0) < lengths.unsqueeze(1)
    return x.masked_fill(~valid.unsqueeze(-1), 0.0)


def _denormalize_motion_batch(
    motion: Tensor,
    lengths: Tensor | None,
    mean: Tensor,
    std: Tensor,
) -> Tensor:
    """Invert evaluator normalization and keep temporal padding at zero."""
    x = motion * std.to(device=motion.device, dtype=motion.dtype) + mean.to(
        device=motion.device,
        dtype=motion.dtype,
    )
    if lengths is None:
        return x
    if motion.ndim != 3 or lengths.ndim != 1 or lengths.shape[0] != motion.shape[0]:
        raise ValueError(
            "length-aware denormalization expects motion (B,T,D) and lengths (B,)"
        )
    lengths = lengths.to(device=motion.device, dtype=torch.long)
    if bool((lengths < 0).any()) or bool((lengths > motion.shape[1]).any()):
        raise ValueError(f"motion lengths must be within [0, {motion.shape[1]}]")
    valid = torch.arange(motion.shape[1], device=motion.device).unsqueeze(0) < lengths.unsqueeze(1)
    return x.masked_fill(~valid.unsqueeze(-1), 0.0)


class RealGuoEvaluator:
    motion_dim = 512
    text_dim = 512
    protocol_version = "guo-v2-normalize-valid-then-zero-pad"

    def __init__(
        self,
        text_to_motion_repo: str | Path = "external/text-to-motion",
        humanml3d_repo: str | Path = "external/HumanML3D",
        device: str | torch.device = "cpu",
        checkpoints_dir: str | Path | None = None,
        normalization_name: str = "Comp_v6_KLD01",
    ) -> None:
        repo = Path(text_to_motion_repo).resolve()
        if not (repo / "networks" / "modules.py").exists():
            raise FileNotFoundError(f"text-to-motion submodule not initialized at {repo}")
        sys.path.insert(0, str(repo))

        from networks.evaluator_wrapper import EvaluatorModelWrapper
        from utils.word_vectorizer import WordVectorizer

        self._device = torch.device(device)
        checkpoint_root, mean_path, std_path = _resolve_evaluator_assets(
            text_to_motion_repo=repo,
            humanml3d_repo=humanml3d_repo,
            checkpoints_dir=checkpoints_dir,
            normalization_name=normalization_name,
        )
        self.checkpoints_dir = checkpoint_root
        self.normalization_name = normalization_name
        self.normalization_path = mean_path
        opt = _GuoOpts(checkpoints_dir=checkpoint_root, device=self._device)
        # EvaluatorModelWrapper expects opt.checkpoints_dir to be a string-coercible path
        opt_ns = type("Opt", (), opt.__dict__)()  # convert dataclass → simple object
        opt_ns.checkpoints_dir = str(opt.checkpoints_dir)
        self._wrapper = EvaluatorModelWrapper(opt_ns)

        self._word_vec = WordVectorizer(str(repo / "glove"), "our_vab")

        # The matching checkpoint and its experiment-specific normalization
        # must stay paired; mixing them changes the evaluator embedding space.
        self._mean = torch.from_numpy(np.load(mean_path)).float().to(self._device)
        self._std = torch.from_numpy(np.load(std_path)).float().to(self._device)
        print(
            f"[guo] checkpoints={checkpoint_root} normalization={normalization_name} "
            f"mean={mean_path} protocol={self.protocol_version}",
            flush=True,
        )

    # ------------------------------------------------------------ helpers

    def normalize(self, motion_263: Tensor, lengths: Tensor | None = None) -> Tensor:
        return _normalize_motion_batch(motion_263, lengths, self._mean, self._std)

    def denormalize(self, motion_263: Tensor, lengths: Tensor | None = None) -> Tensor:
        """Invert evaluator normalization while keeping temporal padding at zero."""
        return _denormalize_motion_batch(motion_263, lengths, self._mean, self._std)

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
        # Upstream `get_motion_embeddings` reorders by length and returns
        # embeddings in *that* order ("the results does not following the order
        # of inputs"). Anything pairing motion with text by index (R@k, mm_dist)
        # breaks unless the exact permutation is undone. Reproduce its ordering
        # and invert it, encoding the way `get_co_embeddings` does.
        m_lens = lengths.to(self._device).long()
        x = self.normalize(motion_features.to(self._device).float(), m_lens)
        align, inv = _upstream_align_indices(m_lens)
        align_t = torch.as_tensor(align, device=self._device)
        inv_t = torch.as_tensor(inv, device=self._device)
        unit = self._wrapper.opt.unit_length
        with torch.no_grad():
            movements = self._wrapper.movement_encoder(x[align_t][..., :-4]).detach()
            emb = self._wrapper.motion_encoder(movements, m_lens[align_t] // unit)
        return emb[inv_t]

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

    def encode_text_from_tokens(self, tokens_per_caption: list[list[str]]) -> Tensor:
        """Encode using HumanML3D's *pre-tagged* word/POS tokens directly.

        Each element of `tokens_per_caption` is a list of "word/POS" strings
        already in the format `WordVectorizer` expects (e.g. `"left/Loc_VIP"`,
        `"walks/VERB"`, `"a/DET"`). This bypasses spaCy entirely and preserves
        HumanML3D's custom *_VIP semantic tags that the Guo text encoder was
        trained on — fresh spaCy POS tagging never produces those, which
        silently halves R-precision.

        Pull the upstream tokens from `external/HumanML3D/HumanML3D/texts.zip`
        (each line is `<caption>#<word/POS word/POS ...>#<start>#<end>`).
        """
        word_embs, pos_ohot, cap_lens_dev = self._vectorize_caption_tokens(tokens_per_caption)
        # Same sort/unsort dance as encode_text_from_strings.
        sort_idx = torch.argsort(cap_lens_dev, descending=True)
        inv_idx = torch.empty_like(sort_idx)
        inv_idx[sort_idx] = torch.arange(sort_idx.numel(), device=sort_idx.device)
        with torch.no_grad():
            emb = self._wrapper.text_encoder(
                word_embs[sort_idx], pos_ohot[sort_idx], cap_lens_dev[sort_idx],
            )
        return emb[inv_idx]

    def _vectorize_caption_tokens(
        self,
        tokens_per_caption: list[list[str]],
    ) -> tuple[Tensor, Tensor, Tensor]:
        max_len = 20
        B = len(tokens_per_caption)
        word_embs = torch.zeros(B, max_len + 2, 300)
        pos_ohot = torch.zeros(B, max_len + 2, 15)
        cap_lens = torch.zeros(B, dtype=torch.long)
        for i, raw_tokens in enumerate(tokens_per_caption):
            toks = list(raw_tokens)[:max_len]
            full = ["sos/OTHER"] + toks + ["eos/OTHER"]
            for j, tok in enumerate(full):
                vec, pos = self._word_vec[tok]
                word_embs[i, j] = torch.from_numpy(vec)
                pos_ohot[i, j] = torch.from_numpy(pos)
            cap_lens[i] = len(full)
        word_embs = word_embs.to(self._device)
        pos_ohot = pos_ohot.to(self._device)
        cap_lens_dev = cap_lens.to(self._device)
        return word_embs, pos_ohot, cap_lens_dev

    def encode_co_embeddings_from_tokens(
        self,
        motion_features: Tensor,
        lengths: Tensor,
        tokens_per_caption: list[list[str]],
    ) -> tuple[Tensor, Tensor]:
        """Run the upstream paired evaluator path exactly for one retrieval batch."""
        if motion_features.shape[0] != len(tokens_per_caption):
            raise ValueError("motion and caption-token batch sizes must match")
        word_embs, pos_ohot, cap_lens = self._vectorize_caption_tokens(tokens_per_caption)
        m_lens = lengths.to(self._device).long()
        motion = self.normalize(motion_features.to(self._device).float(), m_lens)
        # Upstream collate_fn sorts by sentence length before get_co_embeddings.
        text_order = torch.argsort(cap_lens, descending=True, stable=True)
        with torch.no_grad():
            text_emb, motion_emb = self._wrapper.get_co_embeddings(
                word_embs[text_order],
                pos_ohot[text_order],
                cap_lens[text_order],
                motion[text_order],
                m_lens[text_order],
            )
        return text_emb, motion_emb


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
