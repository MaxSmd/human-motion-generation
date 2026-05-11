"""SMPL 22-joint skeleton: topology + forward kinematics.

HumanML3D uses the first 22 joints of SMPL (pelvis + body, no hands/face).
The kinematic tree (`PARENTS[i]` = parent of joint i, root has -1) and joint
names are hard-coded SMPL constants. T-pose offsets are *not* hard-coded —
they live in the SMPL `.npz` body model and must be loaded by the caller (or
derived once and shipped with the data submodule).

For eval against the Guo et al. evaluator we need a single canonical skeleton;
HumanML3D itself uses a per-clip skeleton (recomputed from each subject's
betas). For RMG the standard practice is to use the SMPL neutral mean shape's
offsets — provided by `load_smpl_neutral_offsets()` once the SMPL model is on
disk.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# SMPL kinematic tree, joints 0..21 (HumanML3D's 22-joint subset).
PARENTS: tuple[int, ...] = (
    -1,  # 0  pelvis (root)
    0,   # 1  L_Hip
    0,   # 2  R_Hip
    0,   # 3  Spine1
    1,   # 4  L_Knee
    2,   # 5  R_Knee
    3,   # 6  Spine2
    4,   # 7  L_Ankle
    5,   # 8  R_Ankle
    6,   # 9  Spine3
    7,   # 10 L_Foot (toe)
    8,   # 11 R_Foot (toe)
    9,   # 12 Neck
    9,   # 13 L_Collar
    9,   # 14 R_Collar
    12,  # 15 Head
    13,  # 16 L_Shoulder
    14,  # 17 R_Shoulder
    16,  # 18 L_Elbow
    17,  # 19 R_Elbow
    18,  # 20 L_Wrist
    19,  # 21 R_Wrist
)
NUM_JOINTS = 22
ROOT_JOINT = 0

JOINT_NAMES: tuple[str, ...] = (
    "pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
)

# Foot indices used by HumanML3D's foot-contact feature.
FOOT_LEFT_IDX = (7, 10)   # L_Ankle, L_Foot
FOOT_RIGHT_IDX = (8, 11)  # R_Ankle, R_Foot
FOOT_CONTACT_IDX = (*FOOT_LEFT_IDX, *FOOT_RIGHT_IDX)


@dataclass
class Skeleton:
    """T-pose offsets `offsets[j] = position_j - position_parent(j)` in R^3.

    For root, offset is its absolute position at rest (typically zero).
    """

    offsets: Tensor  # (J, 3)
    parents: tuple[int, ...] = PARENTS

    def __post_init__(self) -> None:
        if self.offsets.shape != (NUM_JOINTS, 3):
            raise ValueError(f"offsets must be (22, 3), got {tuple(self.offsets.shape)}")


# ---------------------------------------------------------------------------
# Quaternion utilities (operate on (..., 4) tensors with [w, x, y, z] order).
# Same convention as the SMPL "axis-angle → quaternion" mapping used elsewhere.
# ---------------------------------------------------------------------------


def quat_mul(a: Tensor, b: Tensor) -> Tensor:
    """Hamilton product q_a ⊗ q_b. Both are (..., 4) with [w, x, y, z]."""
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_rotate(q: Tensor, v: Tensor) -> Tensor:
    """Rotate vector(s) `v ∈ R^3` by unit quaternion(s) q. Shapes broadcast."""
    qw = q[..., 0:1]
    qv = q[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)


def quat_to_rotmat(q: Tensor) -> Tensor:
    """(..., 4) [w, x, y, z] → (..., 3, 3) rotation matrices."""
    w, x, y, z = q.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    R = torch.stack(
        [
            1 - 2 * (yy + zz),  2 * (xy - wz),     2 * (xz + wy),
            2 * (xy + wz),      1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy),      2 * (yz + wx),     1 - 2 * (xx + yy),
        ],
        dim=-1,
    )
    return R.reshape(*q.shape[:-1], 3, 3)


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------


def forward_kinematics(
    skeleton: Skeleton,
    quats: Tensor,
    translation: Tensor,
) -> Tensor:
    """Compute world-space joint positions from per-joint local rotations.

    Args:
        skeleton: rest-pose offsets and parents.
        quats: (*, J, 4) per-joint *local* unit quaternions. quats[..., 0, :]
            is the global root orientation; quats[..., j, :] for j>0 are local
            rotations relative to each joint's parent (SMPL convention, which
            matches the paper's §3.1: `q_1` = global orientation, `q_{j>=2}` =
            local joint rotations).
        translation: (*, 3) global root translation.

    Returns:
        joints: (*, J, 3) world-space positions.
    """
    if quats.shape[-2:] != (NUM_JOINTS, 4):
        raise ValueError(f"quats must be (..., 22, 4), got {tuple(quats.shape)}")
    if translation.shape[-1] != 3:
        raise ValueError(f"translation must be (..., 3), got {tuple(translation.shape)}")

    offsets = skeleton.offsets.to(dtype=quats.dtype, device=quats.device)
    parents = skeleton.parents

    leading = quats.shape[:-2]
    # Per-joint global rotation as a quaternion: chain along parent path.
    # We build this iteratively in joint order; root rotation == quats[..., 0, :].
    global_quats: list[Tensor] = [None] * NUM_JOINTS  # type: ignore[list-item]
    positions: list[Tensor] = [None] * NUM_JOINTS    # type: ignore[list-item]
    for j in range(NUM_JOINTS):
        local_q = quats[..., j, :]
        if j == ROOT_JOINT:
            global_quats[j] = local_q
            positions[j] = translation
        else:
            p = parents[j]
            gq = quat_mul(global_quats[p], local_q)
            global_quats[j] = gq
            # Rotate the offset by the *parent's* global rotation, add to parent's position.
            off_j = offsets[j].expand(*leading, 3)
            positions[j] = positions[p] + quat_rotate(global_quats[p], off_j)
    return torch.stack(positions, dim=-2)


def t_pose_joints(skeleton: Skeleton, batch_shape: tuple[int, ...] = ()) -> Tensor:
    """Joint positions at T-pose (identity rotations everywhere, zero translation)."""
    quats = torch.zeros(*batch_shape, NUM_JOINTS, 4, dtype=skeleton.offsets.dtype)
    quats[..., 0] = 1.0  # identity quaternion
    trans = torch.zeros(*batch_shape, 3, dtype=skeleton.offsets.dtype)
    return forward_kinematics(skeleton, quats, trans)
