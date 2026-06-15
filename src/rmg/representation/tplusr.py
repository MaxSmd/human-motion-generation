"""T+R representation: pack/unpack between structured (translation, quaternions)
and the flat (D = 3 + 4·J) tensor that the network and ProductManifold see.

The flat layout matches `ProductManifold([Euclidean(3), Sphere(3) × J])`:
  flat[..., 0:3]                  = translation
  flat[..., 3 + 4*j : 3 + 4*(j+1)] = quaternion for joint j
where j == 0 is the global root orientation and j ∈ {1, …, J-1} are local
per-joint rotations (paper §3.1, SMPL convention).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from shared.geometry.quaternions import make_continuous, normalize_quaternions  # noqa: F401
from shared.geometry.skeleton import NUM_JOINTS


@dataclass
class TPlusR:
    translation: Tensor   # (..., 3)
    quaternions: Tensor   # (..., J, 4)

    @property
    def num_joints(self) -> int:
        return self.quaternions.shape[-2]


def tplusr_dim(num_joints: int = NUM_JOINTS) -> int:
    return 3 + 4 * num_joints


def encode(tpr: TPlusR) -> Tensor:
    """Pack into the flat (..., 3 + 4*J) tensor."""
    leading = tpr.translation.shape[:-1]
    if tpr.quaternions.shape[:-2] != leading:
        raise ValueError(
            f"translation leading shape {leading} != quaternions leading shape "
            f"{tpr.quaternions.shape[:-2]}"
        )
    quats_flat = tpr.quaternions.reshape(*leading, -1)
    return torch.cat([tpr.translation, quats_flat], dim=-1)


def decode(flat: Tensor, num_joints: int = NUM_JOINTS) -> TPlusR:
    """Unpack the flat (..., 3 + 4*J) tensor into structured (translation, quaternions)."""
    expected = tplusr_dim(num_joints)
    if flat.shape[-1] != expected:
        raise ValueError(f"flat last dim {flat.shape[-1]} != expected {expected}")
    translation = flat[..., :3]
    quats = flat[..., 3:].reshape(*flat.shape[:-1], num_joints, 4)
    return TPlusR(translation=translation, quaternions=quats)


# `normalize_quaternions` / `make_continuous` were lifted to
# `shared.geometry.quaternions` (model-agnostic, reused by mardm). Re-exported
# here so `rmg.representation` and `rmg.data` imports keep working unchanged.
