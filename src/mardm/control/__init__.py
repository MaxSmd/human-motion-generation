"""Spatial joint control for MARDM (MaskControl translated to continuous MAR).

Phase 1 (zero-training): inference-time z-optimization guidance on an existing
checkpoint — `guidance.generate_guided` + the differentiable control loss in
`losses`. Nothing here modifies or is required by the base MARDM model.
"""

from .guidance import GuidanceConfig, generate_guided
from .losses import (
    ControlSignal,
    control_loss,
    control_metrics,
    dynamics_loss,
    foot_skate_loss,
    latents_to_joints,
    motion_metrics,
)
from .root_edit import root_edit_essential

__all__ = [
    "ControlSignal",
    "GuidanceConfig",
    "control_loss",
    "control_metrics",
    "dynamics_loss",
    "foot_skate_loss",
    "generate_guided",
    "latents_to_joints",
    "motion_metrics",
    "root_edit_essential",
]
