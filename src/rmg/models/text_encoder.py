"""Backward-compatible re-export shim.

The text encoders were lifted to `shared.text` (model-agnostic, reused by
mardm). This module re-exports them so existing `rmg.models.text_encoder`
imports keep working unchanged.
"""

from shared.text.text_encoder import (
    Qwen3EmbeddingEncoder,
    RandomTextEncoder,
    TextEncoder,
)

__all__ = [
    "TextEncoder",
    "Qwen3EmbeddingEncoder",
    "RandomTextEncoder",
]
