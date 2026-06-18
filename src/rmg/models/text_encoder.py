"""Backward-compatible re-export shim.

The text encoders live in `shared.text` so rmg, mardm, and MoMask can reuse the
same model-agnostic implementations. This module keeps older
`rmg.models.text_encoder` imports working unchanged.
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
