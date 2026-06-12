"""MARDM model components.

Reuse `rmg.models.text_encoder.Qwen3EmbeddingEncoder` for text conditioning so
all three methods share the same text representation; pass its per-caption
feature tensor into `MARDM` as `cond`.
"""

from .autoencoder import AE, AEConfig
from .diffmlps import DiffMLPs
from .mardm import MARDM, MARDMConfig

__all__ = ["AE", "AEConfig", "DiffMLPs", "MARDM", "MARDMConfig"]
