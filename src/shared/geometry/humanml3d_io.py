"""HumanML3D 263-D feature conversion (paper §D.3).

Mirrors `EricGuo5513/HumanML3D/motion_representation.ipynb::process_file` step
for step. Bit-comparable agreement with the upstream pipeline is essential for
fair comparison against the Guo et al. evaluator.

Layout of the 263-D feature (per frame, in the upstream order):
  [0]      root Y-axis rotation velocity (frame-to-frame)
  [1:3]    root XZ linear velocity in the canonical (face-+Z) frame
  [3]      root Y (height)
  [4:67]   joint positions of joints 1..21 in canonical frame ("ric"), 21*3=63
  [67:193] 6D rotation of joints 1..21 ("rot"), 21*6=126
  [193:259] joint velocities of all 22 joints in canonical frame, 22*3=66
  [259:263] foot contacts (L_Ankle, L_Foot, R_Ankle, R_Foot)

The feature is defined for `T-1` output frames (because everything is velocity-
diffed except the rotation/position parts which are also trimmed by 1).
"""

from __future__ import annotations

import torch
from torch import Tensor

from .skeleton import (
    FOOT_LEFT_IDX,
    FOOT_RIGHT_IDX,
    NUM_JOINTS,
    Skeleton,
    forward_kinematics,
    quat_mul,
    quat_rotate,
    quat_to_rotmat,
)

# HumanML3D's `face_joint_indx` for 22-joint SMPL: (R_Hip, L_Hip, R_Shoulder, L_Shoulder).
FACE_JOINT_INDX = (2, 1, 17, 16)
DEFAULT_FOOT_THRESHOLD = 0.002  # squared velocity (meters²/frame²)
H3D_FEATURE_DIM = 263

_EPS = 1e-7


# ---------------------------------------------------------------------------
# Quaternion helpers (HumanML3D convention: [w, x, y, z], unit quats)
# ---------------------------------------------------------------------------


def quat_inv(q: Tensor) -> Tensor:
    """Inverse of a unit quaternion: conjugate."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_between(v1: Tensor, v2: Tensor) -> Tensor:
    """Shortest-rotation unit quaternion taking v1 → v2 (both already unit)."""
    dot = (v1 * v2).sum(dim=-1, keepdim=True)
    cross = torch.cross(v1, v2, dim=-1)
    # w = sqrt(|v1|^2 |v2|^2) + dot — for unit vectors, w = 1 + dot.
    w = 1.0 + dot
    q = torch.cat([w, cross], dim=-1)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(_EPS)


def quat_to_cont6d(q: Tensor) -> Tensor:
    """Unit quaternion (..., 4) → continuous 6D rotation (..., 6).

    Matches HumanML3D `quaternion_to_cont6d`: cat of first two *columns* of the
    rotation matrix.
    """
    R = quat_to_rotmat(q)  # (..., 3, 3)
    # First two columns: R[..., :, 0], R[..., :, 1] each shape (..., 3).
    return torch.cat([R[..., 0], R[..., 1]], dim=-1)


# ---------------------------------------------------------------------------
# T+R → joint positions
# ---------------------------------------------------------------------------


def tplusr_to_joints(translation: Tensor, quaternions: Tensor, skeleton: Skeleton) -> Tensor:
    """RMG → joint positions via SMPL FK."""
    return forward_kinematics(skeleton, quaternions, translation)


# ---------------------------------------------------------------------------
# Joint positions → 263-D HumanML3D features
# ---------------------------------------------------------------------------


def _foot_contacts(positions: Tensor, threshold: float) -> tuple[Tensor, Tensor]:
    """Per-foot binary contact signal of length (T-1).

    `positions`: (T, J, 3) — already in the canonical frame.
    Returns (feet_l, feet_r) each of shape (T-1, 2).
    """
    fid_l = torch.tensor(FOOT_LEFT_IDX, device=positions.device)
    fid_r = torch.tensor(FOOT_RIGHT_IDX, device=positions.device)

    diff = positions[1:] - positions[:-1]  # (T-1, J, 3)
    feet_l = (diff[:, fid_l] ** 2).sum(dim=-1)  # (T-1, 2)
    feet_r = (diff[:, fid_r] ** 2).sum(dim=-1)
    return (feet_l < threshold).to(positions.dtype), (feet_r < threshold).to(positions.dtype)


def _canonicalize_first_frame(positions: Tensor) -> tuple[Tensor, Tensor]:
    """Floor + XZ-origin + face-+Z normalization of the first frame.

    Returns (positions_canon, root_quat_init): positions in the canonical
    frame, and the per-frame Y-axis rotation quaternion that was applied
    (here, broadcast across the sequence — it's a single rotation but the
    upstream code stores it as (T, J, 4); for our shape we just keep (T, 4)
    since rotating per frame is what matters downstream).
    """
    positions = positions.clone()

    # Floor
    floor_y = positions[..., 1].amin()
    positions[..., 1] = positions[..., 1] - floor_y

    # XZ origin (use first-frame pelvis XZ)
    root_xz = torch.stack(
        [positions[0, 0, 0], torch.zeros_like(positions[0, 0, 0]), positions[0, 0, 2]], dim=-1
    )
    positions = positions - root_xz

    # Face Z+ (first frame). Across vector = (R_Hip - L_Hip) + (R_Shoulder - L_Shoulder).
    r_hip, l_hip, sdr_r, sdr_l = FACE_JOINT_INDX
    init = positions[0]
    across = (init[r_hip] - init[l_hip]) + (init[sdr_r] - init[sdr_l])
    across = across / across.norm().clamp_min(_EPS)

    # forward = up × across (right-handed frame; rotation is around Y).
    up = torch.tensor([0.0, 1.0, 0.0], dtype=positions.dtype, device=positions.device)
    forward_init = torch.cross(up, across, dim=-1)
    forward_init = forward_init / forward_init.norm().clamp_min(_EPS)

    target = torch.tensor([0.0, 0.0, 1.0], dtype=positions.dtype, device=positions.device)
    root_quat_init = quat_between(forward_init, target)  # (4,)

    # Apply the same Y-rotation to every joint position in every frame.
    rq = root_quat_init.expand(*positions.shape[:-1], 4)
    positions = quat_rotate(rq, positions)
    return positions, root_quat_init


def _ik_quaternions_from_positions(
    positions: Tensor, skeleton: Skeleton
) -> Tensor:
    """Inverse kinematics: per-frame world joint positions → per-joint local
    quaternions. Mirrors HumanML3D `Skeleton.inverse_kinematics_np`.

    Implementation note: this is the *upstream* IK that builds the feature's
    `cont6d`. Since RMG already provides per-joint local quaternions, this
    routine is only needed when going `joint_positions → 263-D` from raw
    AMASS data; the RMG → H3D path uses the model's quaternions directly via
    `tplusr_to_h3d_features_with_quats`.
    """
    raise NotImplementedError(
        "IK is only needed for the AMASS preprocess path. The RMG → H3D path "
        "uses the model's per-joint quaternions directly. See "
        "`tplusr_to_h3d_features_with_quats` for the in-model conversion, and "
        "`shared.data.prepare_humanml3d` for the upstream IK that turns raw "
        "joint positions into the dataset's quaternions."
    )


def tplusr_to_h3d_features_with_quats(
    translation: Tensor,
    quaternions: Tensor,
    skeleton: Skeleton,
    feet_thre: float = DEFAULT_FOOT_THRESHOLD,
) -> Tensor:
    """RMG (T+R) → 263-D HumanML3D feature for a single (T, ...) sequence.

    Bypasses the upstream IK step (we already have the quaternions); applies
    the same canonicalization (floor/XZ-origin/face-+Z) and feature extraction
    as the upstream `process_file`.

    Args:
        translation: (T, 3) root translation.
        quaternions: (T, J, 4) per-joint local quaternions, root at index 0.
        skeleton: rest-pose offsets.

    Returns:
        feature: (T-1, 263) tensor.
    """
    if translation.dim() != 2 or translation.shape[-1] != 3:
        raise ValueError(f"translation must be (T, 3), got {tuple(translation.shape)}")
    if quaternions.dim() != 3 or quaternions.shape[1:] != (NUM_JOINTS, 4):
        raise ValueError(f"quaternions must be (T, 22, 4), got {tuple(quaternions.shape)}")

    # 1. FK → world joint positions (T, J, 3).
    positions = forward_kinematics(skeleton, quaternions, translation)

    # 2. Floor + XZ-origin + face-Z+ — and rotate ALL quaternions by the same
    #    initial Y-rotation so that the rotation features are in the same frame.
    positions, root_quat_init = _canonicalize_first_frame(positions)
    # rotate the root quaternion by root_quat_init: q'_root = root_quat_init ⊗ q_root.
    rotated_quats = quaternions.clone()
    rqi = root_quat_init.expand_as(quaternions[:, 0, :])
    rotated_quats[:, 0, :] = quat_mul(rqi, quaternions[:, 0, :])

    return _features_from_positions_and_quats(positions, rotated_quats, feet_thre)


def _features_from_positions_and_quats(
    positions: Tensor, quaternions: Tensor, feet_thre: float
) -> Tensor:
    """Common feature extraction once positions are in the canonical frame
    and the per-joint quaternions have been rotated to match.

    `positions`: (T, J, 3); `quaternions`: (T, J, 4) with root quat in canonical frame.
    """
    T, J, _ = positions.shape
    device, dtype = positions.device, positions.dtype

    # --- Foot contacts (T-1, 2 each) ---
    feet_l, feet_r = _foot_contacts(positions, feet_thre)

    # --- Root rotation series r_rot (T, 4) and angular velocity ---
    r_rot = quaternions[:, 0, :].clone()  # (T, 4) — global root orientation in canonical frame
    # Angular velocity quaternion: r_velocity_quat[t] = r_rot[t+1] * r_rot[t]^-1
    r_vel_quat = quat_mul(r_rot[1:], quat_inv(r_rot[:-1]))  # (T-1, 4)
    # Y-axis rotation velocity (scalar per frame).
    # Following upstream: r_velocity = arcsin(r_vel_quat.y). For a pure-Y
    # rotation by angle θ, q = [cos(θ/2), 0, sin(θ/2), 0]; arcsin(sin(θ/2)) = θ/2.
    # The factor of 1/2 is consistent with how `recover_root_rot_pos` cumsums it.
    r_velocity = torch.asin(r_vel_quat[:, 2:3])  # (T-1, 1)

    # --- Root linear velocity in canonical frame ---
    raw_vel = positions[1:, 0] - positions[:-1, 0]  # (T-1, 3) world XYZ root vel
    # Rotate by r_rot[1:] to get into canonical frame.
    lin_vel_rot = quat_rotate(r_rot[1:], raw_vel)  # (T-1, 3)
    l_velocity = lin_vel_rot[:, [0, 2]]  # (T-1, 2)

    # --- Root height (length T, then trimmed to T-1 in the final concat) ---
    root_y = positions[:, 0, 1:2]  # (T, 1)

    # --- root_data: (T-1, 4) ---
    root_data = torch.cat([r_velocity, l_velocity, root_y[:-1]], dim=-1)

    # --- 6D rotation for joints 1..J-1: (T, (J-1)*6), then trim to T-1 ---
    cont6d = quat_to_cont6d(quaternions)  # (T, J, 6)
    rot_data = cont6d[:, 1:, :].reshape(T, -1)  # (T, (J-1)*6)

    # --- RIC: rotation-invariant local positions ---
    # Subtract root XZ from every joint, then rotate by r_rot to align facing.
    pos_local = positions.clone()
    pos_local[..., 0] -= positions[:, 0:1, 0]
    pos_local[..., 2] -= positions[:, 0:1, 2]
    r_rot_per_joint = r_rot.unsqueeze(1).expand(T, J, 4)
    pos_local = quat_rotate(r_rot_per_joint, pos_local)  # (T, J, 3)
    ric_data = pos_local[:, 1:, :].reshape(T, -1)  # (T, (J-1)*3)

    # --- Local velocities of ALL joints in canonical frame ---
    raw_global_vel = positions[1:] - positions[:-1]  # (T-1, J, 3)
    local_vel = quat_rotate(r_rot[:-1].unsqueeze(1).expand(T - 1, J, 4), raw_global_vel)
    local_vel = local_vel.reshape(T - 1, -1)  # (T-1, J*3)

    # --- Concatenate (all trimmed to T-1) ---
    out = torch.cat(
        [
            root_data,                  # (T-1, 4)
            ric_data[:-1],              # (T-1, (J-1)*3)
            rot_data[:-1],              # (T-1, (J-1)*6)
            local_vel,                  # (T-1, J*3)
            feet_l,                     # (T-1, 2)
            feet_r,                     # (T-1, 2)
        ],
        dim=-1,
    )
    if out.shape[-1] != H3D_FEATURE_DIM:
        raise AssertionError(f"H3D feature dim {out.shape[-1]} != expected {H3D_FEATURE_DIM}")
    return out


# ---------------------------------------------------------------------------
# Recovery: 263-D → joint positions  (mirrors `recover_from_ric`)
# Used for: (a) reconstruction sanity checks, (b) shipping samples to the
# Guo et al. evaluator (which expects joint coordinates).
# ---------------------------------------------------------------------------


def _recover_root_rot_pos(data: Tensor) -> tuple[Tensor, Tensor]:
    """data: (..., T, 263) → (r_rot_quat, r_pos), both with leading (..., T)."""
    rot_vel = data[..., 0]  # (..., T)
    r_rot_ang = torch.zeros_like(rot_vel)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    leading = data.shape[:-1]
    r_rot_quat = torch.zeros(*leading, 4, device=data.device, dtype=data.dtype)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(*leading, 3, device=data.device, dtype=data.dtype)
    r_pos[..., 1:, 0] = data[..., :-1, 1]
    r_pos[..., 1:, 2] = data[..., :-1, 2]
    r_pos = quat_rotate(quat_inv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_joints_from_ric(data: Tensor, joints_num: int = NUM_JOINTS) -> Tensor:
    """263-D → (..., T, J, 3) joint positions in world frame."""
    r_rot_quat, r_pos = _recover_root_rot_pos(data)
    start, end = 4, 4 + (joints_num - 1) * 3
    positions = data[..., start:end]
    leading_t = positions.shape[:-1]
    positions = positions.reshape(*leading_t, joints_num - 1, 3)

    rq = quat_inv(r_rot_quat).unsqueeze(-2).expand(*positions.shape[:-1], 4)
    positions = quat_rotate(rq, positions)
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    return positions
