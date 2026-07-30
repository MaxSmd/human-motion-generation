"""In-process motion rendering: (joints) → media file on disk.

The animation itself now lives in `shared.render` (a torch-free module shared
with the cluster's `rmg.scripts.visualize`) so both paths produce the identical
multi-panel styling. This module re-exports the renderer and adds
`decode_to_joints` — flat manifold sample → (T, 22, 3) world joints via the
representation's rotations + forward kinematics — the common "decode → FK" step
every backend feature runs before rendering.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from shared.geometry import Skeleton, forward_kinematics
from shared.render import ffmpeg_available, render_joints, resolve_format  # noqa: F401
from rmg.representation.tplusr import decode as tplusr_decode

__all__ = ["ffmpeg_available", "resolve_format", "render_joints", "decode_to_joints"]


def decode_to_joints(
    flat: Tensor,                       # (T, ambient_dim)
    skeleton: Skeleton,
    representation_name: str = "tr",
) -> np.ndarray:                        # (T, 22, 3) float32
    """Decode a flat manifold sample to world-space joint positions.

    T+R / T+R+P carry per-joint quaternions in dims [3 : 3 + 4*J]; we take the
    translation + rotations and run forward kinematics. (T+P has no rotations;
    that path would need pre-shape rescaling — not wired here yet.)
    """
    if representation_name not in ("tr", "trp"):
        # PLACEHOLDER: T+P (pre-shape) decode goes through a different recovery
        # path (see TPRepresentation.to_h3d_features). All current runs are T+R.
        raise NotImplementedError(
            f"decode_to_joints not implemented for representation {representation_name!r}; "
            "only 'tr'/'trp' (rotation-based) are wired."
        )
    tpr = tplusr_decode(flat.float())  # slices [:3] and [3:3+4*J] internally
    joints = forward_kinematics(
        skeleton, tpr.quaternions.float(), tpr.translation.float()
    )
    return joints.cpu().numpy().astype(np.float32)
