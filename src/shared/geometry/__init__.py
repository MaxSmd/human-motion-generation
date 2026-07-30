"""Shared HumanML3D geometry: the SMPL 22-joint skeleton + forward kinematics,
quaternion ops, and the 263-D HumanML3D feature conversion (`process_file`
parity). Model-agnostic — every model derives its features from packed
`{translation, quats}` clips through this. Imports no model package.
"""

from .humanml3d_io import (
    DEFAULT_FOOT_THRESHOLD,
    FACE_JOINT_INDX,
    H3D_FEATURE_DIM,
    quat_between,
    quat_inv,
    quat_to_cont6d,
    recover_joints_from_ric,
    tplusr_to_h3d_features_with_quats,
    tplusr_to_joints,
)
from .humanml3d_upstream import (
    positions_to_h3d_features_upstream,
    tplusr_to_h3d_features_upstream,
)
from .quaternions import (
    make_continuous,
    normalize_quaternions,
    quat_continuity,
    quat_to_upper_hemisphere,
)
from .skeleton import (
    FOOT_CONTACT_IDX,
    FOOT_LEFT_IDX,
    FOOT_RIGHT_IDX,
    JOINT_NAMES,
    NUM_JOINTS,
    PARENTS,
    ROOT_JOINT,
    Skeleton,
    forward_kinematics,
    quat_mul,
    quat_rotate,
    quat_to_rotmat,
    t_pose_joints,
)

__all__ = [
    "NUM_JOINTS", "ROOT_JOINT", "PARENTS", "JOINT_NAMES",
    "FOOT_LEFT_IDX", "FOOT_RIGHT_IDX", "FOOT_CONTACT_IDX",
    "Skeleton", "forward_kinematics", "t_pose_joints",
    "quat_mul", "quat_rotate", "quat_to_rotmat",
    "H3D_FEATURE_DIM", "FACE_JOINT_INDX", "DEFAULT_FOOT_THRESHOLD",
    "quat_inv", "quat_between", "quat_to_cont6d",
    "tplusr_to_joints", "tplusr_to_h3d_features_with_quats", "recover_joints_from_ric",
    "tplusr_to_h3d_features_upstream", "positions_to_h3d_features_upstream",
    "quat_continuity", "quat_to_upper_hemisphere", "make_continuous", "normalize_quaternions",
]
