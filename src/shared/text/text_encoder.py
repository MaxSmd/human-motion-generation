"""Text encoders for model-agnostic conditioning.

The encoder runs outside the training step; trainers and samplers only consume a
`(B, text_dim)` tensor. Keeping these classes in `shared` lets rmg, mardm, and
MoMask reuse the same conditioning implementations.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import torch
from torch import Tensor


class TextEncoder(ABC):
    text_dim: int

    @abstractmethod
    def encode(self, texts: list[str], device: torch.device | None = None) -> Tensor:
        """Encode a list of strings into a `(B, text_dim)` tensor."""


def _local_snapshot(model_name: str, cache_dir: str | None) -> str | None:
    """Resolve `model_name` to its fully-downloaded local HF-cache snapshot, or
    None if it isn't cached. Handing `from_pretrained` a local *directory*
    (instead of a hub id) makes the load fully offline: transformers skips every
    Hub API round-trip — including tokenization_utils_base's
    `_patch_mistral_regex` → `model_info()` lookup, which turns a transient Hub
    outage into a hard crash at tokenizer load even though all files are cached
    (seen 2026-07-15: eval job died on a hub 504 for Qwen3-Embedding-0.6B)."""
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_name, cache_dir=cache_dir, local_files_only=True)
    except Exception:
        return None  # not (fully) cached — caller falls back to the network path


class Qwen3EmbeddingEncoder(TextEncoder):
    """Wrapper around `Qwen/Qwen3-Embedding-0.6B`.

    The HF model returns hidden states; Qwen3-Embedding's recommended pooling is
    the last non-pad token's hidden state, then L2-normalized.
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
                "Install it inside the enroot image or local environment."
            ) from e
        source = _local_snapshot(model_name, cache_dir) or model_name
        self._tokenizer = AutoTokenizer.from_pretrained(source, cache_dir=cache_dir)
        self._model = AutoModel.from_pretrained(source, cache_dir=cache_dir)
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
        hidden = out.last_hidden_state
        attn = toks["attention_mask"]
        last_idx = attn.sum(dim=1) - 1
        idx = torch.arange(hidden.size(0), device=hidden.device)
        emb = hidden[idx, last_idx]
        if self.l2_normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb


class CLIPTextEncoder(TextEncoder):
    """Frozen CLIP text encoder for MoMask-style conditioning.

    Prefers OpenAI's `clip` package, matching the official MoMask code path. If
    that package is unavailable, it can fall back to Hugging Face Transformers.
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        cache_dir: str | None = None,
        max_length: int = 77,
        l2_normalize: bool = True,
        backend: str = "auto",
    ) -> None:
        if backend not in {"auto", "openai", "transformers"}:
            raise ValueError("backend must be one of: auto, openai, transformers")
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.l2_normalize = l2_normalize
        self._backend = backend
        self._device = torch.device("cpu")
        self._clip = None
        self._tokenizer = None
        self._model = None

        if backend in {"auto", "openai"}:
            try:
                import clip  # type: ignore

                self._clip = clip
                self._model, _ = clip.load(model_name, device=self._device, jit=False, download_root=cache_dir)
                self._model.eval()
                for p in self._model.parameters():
                    p.requires_grad_(False)
                with torch.no_grad():
                    probe = self._model.encode_text(clip.tokenize(["probe"], truncate=True).to(self._device))
                self.text_dim = int(probe.shape[-1])
                self._backend = "openai"
                return
            except ImportError:
                if backend == "openai":
                    raise

        openai_to_hf = {
            "RN50": "openai/clip-rn50",
            "RN101": "openai/clip-rn101",
            "RN50x4": "openai/clip-rn50x4",
            "RN50x16": "openai/clip-rn50x16",
            "RN50x64": "openai/clip-rn50x64",
            "ViT-B/32": "openai/clip-vit-base-patch32",
            "ViT-B/16": "openai/clip-vit-base-patch16",
            "ViT-L/14": "openai/clip-vit-large-patch14",
            "ViT-L/14@336px": "openai/clip-vit-large-patch14-336",
        }
        hf_name = openai_to_hf.get(model_name, model_name)
        try:
            from transformers import AutoTokenizer, CLIPModel
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "CLIPTextEncoder needs either OpenAI `clip` or Hugging Face `transformers`."
            ) from e
        self._tokenizer = AutoTokenizer.from_pretrained(hf_name, cache_dir=cache_dir)
        self._model = CLIPModel.from_pretrained(hf_name, cache_dir=cache_dir)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        self.text_dim = int(getattr(self._model.config, "projection_dim", 512))
        self._backend = "transformers"

    @torch.no_grad()
    def encode(self, texts: list[str], device: torch.device | None = None) -> Tensor:
        if device is not None and device != self._device:
            self._device = torch.device(device)
            if self._model is not None:
                self._model.to(self._device)
        if self._backend == "openai":
            if self._clip is None or self._model is None:
                raise RuntimeError("OpenAI CLIP backend is not initialized")
            toks = self._clip.tokenize(texts, truncate=True).to(self._device)
            emb = self._model.encode_text(toks).float()
        else:
            if self._tokenizer is None or self._model is None:
                raise RuntimeError("Transformers CLIP backend is not initialized")
            toks = self._tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            toks = {k: v.to(self._device) for k, v in toks.items()}
            emb = self._model.get_text_features(**toks).float()
        if self.l2_normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb


class RandomTextEncoder(TextEncoder):
    """Deterministic-per-text random encoder for tests and dry runs."""

    def __init__(self, text_dim: int = 1024) -> None:
        self.text_dim = text_dim

    def encode(self, texts: list[str], device: torch.device | None = None) -> Tensor:
        out = torch.empty(len(texts), self.text_dim)
        for i, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], byteorder="little") & 0xFFFFFFFF
            g = torch.Generator().manual_seed(seed)
            out[i] = torch.randn(self.text_dim, generator=g)
        if device is not None:
            out = out.to(device)
        return out
