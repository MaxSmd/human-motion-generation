"""Shared, model-agnostic text-conditioning encoders.

The diffusion/flow model is decoupled from the encoder: the encoder runs once
outside the training step and the trainer/sampler only see a `(B, text_dim)`
tensor. Lives in `shared` so every model package (rmg, mardm, …) reuses the
same encoders without importing another model.
"""

from .text_encoder import Qwen3EmbeddingEncoder, RandomTextEncoder, TextEncoder

__all__ = [
    "TextEncoder",
    "Qwen3EmbeddingEncoder",
    "RandomTextEncoder",
]
