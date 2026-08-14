"""MoMask reproduction components (Guo et al. 2024).

Data, text-encoder and evaluator utilities live in `shared`; this package imports
no other model package. Use `shared.data.H3D263Dataset` for MoMask training data
(standard 263-D HumanML3D features), or `shared.data.CanonicalHumanML3DDataset`
to train on the official `new_joint_vecs` distribution the Guo evaluator expects.
"""

from .models import (
    CodebookResidualTransformer,
    MaskedMotionTransformer,
    MotionRVQVAE,
    ResidualTransformer,
    ResidualVectorQuantizer,
    TokenTransformerConfig,
)
from .constraints import (
    BendAngleConstraint,
    JointPositionConstraint,
    LatentRefinementConfig,
    LatentRefinementResult,
    bend_angle_loss,
    bend_angle_violation,
    bend_angles_from_joints,
    joint_position_error,
    joint_position_loss,
    refine_motion_latents,
)

__all__ = [
    "ResidualVectorQuantizer",
    "MotionRVQVAE",
    "TokenTransformerConfig",
    "MaskedMotionTransformer",
    "ResidualTransformer",
    "CodebookResidualTransformer",
    "JointPositionConstraint",
    "BendAngleConstraint",
    "LatentRefinementConfig",
    "LatentRefinementResult",
    "bend_angles_from_joints",
    "joint_position_loss",
    "joint_position_error",
    "bend_angle_loss",
    "bend_angle_violation",
    "refine_motion_latents",
]
