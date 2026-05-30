"""Task entry points for MARDM: generation + sampling orchestration."""

from .generation import generate_h3d_features, sample_latents

__all__ = ["generate_h3d_features", "sample_latents"]
