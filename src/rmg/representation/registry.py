"""Representation registry for the paper's Figure 3 ablations.

The training pipeline trains a single Riemannian-flow-matching DiT on the
manifold of the *chosen* representation. To support the ablations, all the
representation-specific choices are encapsulated in a `Representation` class:

    Representation.manifold          → ProductManifold
    Representation.prior_mu          → reference point (rest pose)
    Representation.ambient_dim       → flat input/output dim of the network
    Representation.encode_clip(...)  → (T, 3 + ...) flat tensor on the manifold
    Representation.to_h3d_features(...) → 263-D HumanML3D feature

Implemented variants:
    "tr"  — T + R  (paper's main result, FID 0.043)
    "tp"  — T + P
    "trp" — T + R + P (with the convention that recovery for evaluation goes
            through R; the "recover by P" variant is selected via
            `decode_via='preshape'` at evaluation time)

The dT / dR variants from Figure 3b are *negative-result* ablations (they
underperform T+R consistently). They are documented as config stubs but not
implemented end-to-end — the data pipeline would need to compute and store
temporal differences, and recovery via integration would need additional
care. See `configs/representation/{dt_plus_r,t_plus_dr}.yaml`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

from ..manifolds import (
    Euclidean,
    Manifold,
    PreShape,
    ProductManifold,
    Sphere,
    joints_to_preshape,
)
from shared.geometry.humanml3d_io import (
    _features_from_positions_and_quats,
    _canonicalize_first_frame,
    quat_mul,
)
from shared.geometry.skeleton import NUM_JOINTS, Skeleton, forward_kinematics
from .tplusr import normalize_quaternions


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


@dataclass
class Representation(ABC):
    name: str
    num_joints: int = NUM_JOINTS

    @property
    @abstractmethod
    def ambient_dim(self) -> int: ...

    @abstractmethod
    def build_manifold(self) -> Manifold: ...

    @abstractmethod
    def prior_mu(self, dtype: torch.dtype = torch.float32) -> Tensor: ...

    @abstractmethod
    def encode_clip(
        self,
        translation: Tensor,            # (T, 3)
        quaternions: Tensor,            # (T, J, 4)
        skeleton: Skeleton | None = None,
    ) -> Tensor:                        # (T, ambient_dim)
        ...

    @abstractmethod
    def to_h3d_features(
        self,
        flat: Tensor,                   # (T, ambient_dim)
        skeleton: Skeleton,
    ) -> Tensor:                        # (T-1, 263)
        ...


# ---------------------------------------------------------------------------
# T + R   (default; matches the rest of the codebase)
# ---------------------------------------------------------------------------


@dataclass
class TRRepresentation(Representation):
    name: str = "tr"

    @property
    def ambient_dim(self) -> int:
        return 3 + 4 * self.num_joints

    def build_manifold(self) -> Manifold:
        # Sphere factors are unit quaternions → enable the double-cover quotient
        # so CFM paths stay off the antipodal cut locus.
        return ProductManifold(
            [Euclidean(3)]
            + [Sphere(3, antipodal_quotient=True) for _ in range(self.num_joints)]
        )

    def prior_mu(self, dtype: torch.dtype = torch.float32) -> Tensor:
        rest_T = torch.zeros(3, dtype=dtype)
        rest_q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype)
        return torch.cat([rest_T] + [rest_q] * self.num_joints, dim=-1)

    def encode_clip(self, translation, quaternions, skeleton=None):
        quats = normalize_quaternions(quaternions)
        return torch.cat([translation, quats.reshape(*quats.shape[:-2], -1)], dim=-1)

    def to_h3d_features(self, flat, skeleton):
        T = flat.shape[0]
        translation = flat[:, :3]
        quats = flat[:, 3:].reshape(T, self.num_joints, 4)
        # Route through upstream's `process_file` for bit-comparable agreement
        # with the Guo evaluator's training distribution. Our hand-written
        # reimplementation diverges from upstream by ~110% on cont6d because
        # upstream re-runs IK with `smooth_forward=True` *inside* the feature
        # extractor — see shared/geometry/humanml3d_upstream.py.
        from shared.geometry.humanml3d_upstream import tplusr_to_h3d_features_upstream
        return tplusr_to_h3d_features_upstream(translation, quats, skeleton)


# ---------------------------------------------------------------------------
# T + P  (translation + pre-shape, no rotations)
# ---------------------------------------------------------------------------


@dataclass
class TPRepresentation(Representation):
    name: str = "tp"

    @property
    def ambient_dim(self) -> int:
        return 3 + 3 * self.num_joints  # T (3) + flattened pre-shape (J*3)

    def build_manifold(self) -> Manifold:
        return ProductManifold([Euclidean(3), PreShape(self.num_joints, dim=3)])

    def prior_mu(self, dtype: torch.dtype = torch.float32) -> Tensor:
        # Center of T = 0; "T-pose" pre-shape is computed from a canonical
        # skeleton at training-script construction time. We return a zero-vector
        # placeholder here and rely on `set_prior_mu_from_skeleton` to fill it.
        return torch.zeros(self.ambient_dim, dtype=dtype)

    def prior_mu_from_skeleton(self, skeleton: Skeleton, dtype: torch.dtype = torch.float32) -> Tensor:
        from shared.geometry.skeleton import t_pose_joints
        J = t_pose_joints(skeleton).to(dtype)             # (J, 3)
        pshape = joints_to_preshape(J).reshape(-1)         # (J*3,)
        rest_T = torch.zeros(3, dtype=dtype)
        return torch.cat([rest_T, pshape], dim=-1)

    def encode_clip(self, translation, quaternions, skeleton=None):
        if skeleton is None:
            raise ValueError("TPRepresentation.encode_clip requires `skeleton` (for FK).")
        joints = forward_kinematics(skeleton, normalize_quaternions(quaternions), translation)  # (T, J, 3)
        pshape = joints_to_preshape(joints)                                                     # (T, J*3)
        return torch.cat([translation, pshape], dim=-1)

    def to_h3d_features(self, flat, skeleton):
        # Recovery via pre-shape rescaling: pre-shape carries unit-Frobenius
        # joint configuration; we restore scale using the canonical skeleton's
        # T-pose Frobenius norm. Then build identity quaternions and run the
        # standard §D.3 pipeline.
        from shared.geometry.skeleton import t_pose_joints
        T = flat.shape[0]
        translation = flat[:, :3]
        pshape = flat[:, 3:].reshape(T, self.num_joints, 3)         # (T, J, 3)
        # rescale pre-shape so it has the canonical body's Frobenius norm
        canon = t_pose_joints(skeleton)                             # (J, 3)
        canon_centered = canon - canon.mean(dim=0, keepdim=True)
        scale = canon_centered.flatten().norm()
        joints = pshape * scale + translation.unsqueeze(1)          # (T, J, 3)

        positions, _ = _canonicalize_first_frame(joints)
        # Build identity quaternions for the rotation features. The
        # 263-D feature still has a rotation component; for T+P we don't have
        # rotations, so we use identities (paper "recovered by P" path).
        quats = torch.zeros(T, self.num_joints, 4, dtype=positions.dtype, device=positions.device)
        quats[..., 0] = 1.0
        return _features_from_positions_and_quats(positions, quats, feet_thre=0.002)


# ---------------------------------------------------------------------------
# T + R + P  (full)
# ---------------------------------------------------------------------------


@dataclass
class TRPRepresentation(Representation):
    name: str = "trp"
    decode_via: str = "rotation"  # 'rotation' (recovered by R) | 'preshape' (recovered by P)

    @property
    def ambient_dim(self) -> int:
        return 3 + 4 * self.num_joints + 3 * self.num_joints

    def build_manifold(self) -> Manifold:
        return ProductManifold(
            [Euclidean(3)]
            + [Sphere(3, antipodal_quotient=True) for _ in range(self.num_joints)]
            + [PreShape(self.num_joints, dim=3)]
        )

    def prior_mu(self, dtype: torch.dtype = torch.float32) -> Tensor:
        rest_T = torch.zeros(3, dtype=dtype)
        rest_q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype)
        return torch.cat([rest_T] + [rest_q] * self.num_joints
                         + [torch.zeros(3 * self.num_joints, dtype=dtype)], dim=-1)

    def prior_mu_from_skeleton(self, skeleton: Skeleton, dtype: torch.dtype = torch.float32) -> Tensor:
        from shared.geometry.skeleton import t_pose_joints
        rest_T = torch.zeros(3, dtype=dtype)
        rest_q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype)
        J = t_pose_joints(skeleton).to(dtype)
        pshape = joints_to_preshape(J).reshape(-1)
        return torch.cat([rest_T] + [rest_q] * self.num_joints + [pshape], dim=-1)

    def encode_clip(self, translation, quaternions, skeleton=None):
        if skeleton is None:
            raise ValueError("TRPRepresentation.encode_clip requires `skeleton`.")
        quats = normalize_quaternions(quaternions)
        joints = forward_kinematics(skeleton, quats, translation)
        pshape = joints_to_preshape(joints)
        return torch.cat([translation, quats.reshape(*quats.shape[:-2], -1), pshape], dim=-1)

    def to_h3d_features(self, flat, skeleton):
        T = flat.shape[0]
        translation = flat[:, :3]
        quats = flat[:, 3 : 3 + 4 * self.num_joints].reshape(T, self.num_joints, 4)
        if self.decode_via == "rotation":
            from shared.geometry.humanml3d_upstream import tplusr_to_h3d_features_upstream
            return tplusr_to_h3d_features_upstream(translation, quats, skeleton)
        elif self.decode_via == "preshape":
            tp_repr = TPRepresentation(num_joints=self.num_joints)
            tp_flat = torch.cat([translation, flat[:, 3 + 4 * self.num_joints :]], dim=-1)
            return tp_repr.to_h3d_features(tp_flat, skeleton)
        raise ValueError(f"unknown decode_via {self.decode_via!r}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_REGISTRY: dict[str, type[Representation]] = {
    "tr": TRRepresentation,
    "tp": TPRepresentation,
    "trp": TRPRepresentation,
}


def build_representation(name: str, num_joints: int = NUM_JOINTS, **kwargs) -> Representation:
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown representation {name!r} — available: {sorted(_REGISTRY)} "
            f"(dT/dR ablations are config stubs only; see "
            f"src/rmg/configs/representation/{{dt_plus_r,t_plus_dr}}.yaml)"
        )
    return _REGISTRY[name](num_joints=num_joints, **kwargs)
