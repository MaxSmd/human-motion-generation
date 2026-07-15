"""Spatial joint control for MARDM (MaskControl translated to continuous MAR).

Phase 1 (zero-training): inference-time z-optimization guidance on an existing
checkpoint — `guidance.generate_guided` + the differentiable control loss in
`losses`. Nothing here modifies or is required by the base MARDM model.
"""

from .guidance import GuidanceConfig, generate_guided
from .losses import ControlSignal, control_loss, control_metrics, latents_to_joints
from .regularizer import ControlMARDM, control_forward_loss, control_signal_features

__all__ = [
    "ControlMARDM",
    "ControlSignal",
    "GuidanceConfig",
    "control_forward_loss",
    "control_loss",
    "control_metrics",
    "control_signal_features",
    "generate_guided",
    "latents_to_joints",
]
