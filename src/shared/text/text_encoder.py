"""Text encoders for conditioning. Designed so the diffusion model is decoupled
from the encoder choice: the encoder runs once outside the training step, the
trainer/sampler only see a `(B, text_dim)` tensor.

For HumanML3D the paper uses **Qwen3-Embedding-0.6B** (1024-d) — paper §4.1,
§D.1. For MotionMillion they use Qwen3-1.7B (decoder LM) with MM-DiT; that
path is out of scope for the HumanML3D milestone but the abstract base class
leaves room for it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class TextEncoder(ABC):
    text_dim: int

    @abstractmethod
    def encode(self, texts: list[str], device: torch.device | None = None) -> Tensor:
        """Encode a list of strings → (B, text_dim) tensor."""


class Qwen3EmbeddingEncoder(TextEncoder):
    """Wrapper around `Qwen/Qwen3-Embedding-0.6B`. Lazy-imports `transformers`
    so the rest of the codebase (and the test suite) doesn't depend on it.

    The HF model returns hidden states; Qwen3-Embedding's recommended pooling
    is the *last* non-pad token's hidden state, then L2-normalized. We follow
    that convention; the paper §D.1 just says "encoded hidden states".
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        cache_dir: str | None = None,
        max_length: int = 256,
        l2_normalize: bool = True,
    ) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Qwen3EmbeddingEncoder needs `transformers`. "
                "Install it inside the enroot image (see containers/Dockerfile)."
            ) from e
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self._model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        self.text_dim = int(self._model.config.hidden_size)
        self.max_length = max_length
        self.l2_normalize = l2_normalize

    @torch.no_grad()
    def encode(self, texts: list[str], device: torch.device | None = None) -> Tensor:
        if device is not None:
            self._model.to(device)
        toks = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        if device is not None:
            toks = {k: v.to(device) for k, v in toks.items()}
        out = self._model(**toks)
        hidden = out.last_hidden_state  # (B, L, H)
        attn = toks["attention_mask"]  # (B, L)
        last_idx = attn.sum(dim=1) - 1  # last non-pad position
        idx = torch.arange(hidden.size(0), device=hidden.device)
        emb = hidden[idx, last_idx]  # (B, H)
        if self.l2_normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb


class RandomTextEncoder(TextEncoder):
    """Deterministic-per-text random encoder for tests and dry runs. Hashes the
    text string, seeds an RNG with it, and emits a fixed-dim Gaussian vector.
    Two calls on the same text return the same embedding."""

    def __init__(self, text_dim: int = 1024) -> None:
        self.text_dim = text_dim

    def encode(self, texts: list[str], device: torch.device | None = None) -> Tensor:
        out = torch.empty(len(texts), self.text_dim)
        for i, text in enumerate(texts):
            seed = hash(text) & 0xFFFFFFFF
            g = torch.Generator().manual_seed(seed)
            out[i] = torch.randn(self.text_dim, generator=g)
        if device is not None:
            out = out.to(device)
        return out
