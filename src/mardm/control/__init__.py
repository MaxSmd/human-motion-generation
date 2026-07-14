"""Spatial joint control for MARDM (MaskControl translated to continuous MAR).

Phase 1 (zero-training): inference-time z-optimization guidance on an existing
checkpoint — `guidance.generate_guided` + the differentiable control loss in
`losses`. Nothing here modifies or is required by the base MARDM model.
"""

from .guidance import GuidanceConfig, generate_guided
from .losses import ControlSignal, control_loss, control_metrics, latents_to_joints

__all__ = [
    "ControlSignal",
    "GuidanceConfig",
    "control_loss",
    "control_metrics",
    "generate_guided",
    "latents_to_joints",
]
