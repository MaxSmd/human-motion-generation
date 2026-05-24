"""Upstream-compatible 263-D feature extraction.

Our hand-written `tplusr_to_h3d_features_with_quats` reuses the model's stored
quaternions for the cont6d feature block. Those quaternions come from our
prepare-time IK with `smooth_forward=False`. Upstream's `process_file`, by
contrast, runs IK with `smooth_forward=True` (Gaussian-smooths the forward axis
over 20 frames) *inside* the feature extractor — this changes the per-frame
root quaternion, which propagates through the IK chain into *every* joint's
local quaternion. The result is that our cont6d differs from upstream's by
~100% (verified with `scripts/diagnose_h3d_conversion.py`), and the Guo et al.
evaluator (trained on the upstream features) treats our 263-D vectors as
nonsense — diversity-on-real comes out at ~4 instead of the paper's ~9.5.

The fix is to feed *joint positions* (not stored quaternions) into upstream's
`process_file`. We FK our stored quats to positions, hand the positions to the
upstream function, get back the same 263-D vector the Guo evaluator was trained
on.

The model's training is unaffected — it still learns to predict T+R quaternions
directly. Only the evaluation-side decode path changes.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .skeleton import Skeleton, forward_kinematics


# ---------------------------------------------------------------------------
# Lazy loader for upstream `process_file`
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_upstream_process_file(repo_root: str | None = None):
    """Load upstream `process_file` from the HumanML3D submodule notebook.

    Cached so the (heavy) execs only happen once per process.
    """
    if repo_root is None:
        # Walk up from this file to find the repo root.
        here = Path(__file__).resolve()
        for parent in [here, *here.parents]:
            if (parent / "external" / "HumanML3D" / "common").exists():
                repo_root_path = parent
                break
        else:
            raise FileNotFoundError(
                "Could not locate the repo root containing external/HumanML3D. "
                "Pass repo_root explicitly."
            )
    else:
        repo_root_path = Path(repo_root)

    hml3d = repo_root_path / "external" / "HumanML3D"
    if not (hml3d / "common" / "skeleton.py").exists():
        raise FileNotFoundError(
            f"HumanML3D submodule not initialized at {hml3d}. Run "
            f"`git submodule update --init -- external/HumanML3D` first."
        )

    sys.path.insert(0, str(hml3d))

    # Polyfill removed numpy aliases.
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]

    from common.skeleton import Skeleton as UpSkeleton  # type: ignore
    import common.quaternion as Q  # type: ignore
    from paramUtil import t2m_raw_offsets, t2m_kinematic_chain  # type: ignore

    nb = json.loads((hml3d / "motion_representation.ipynb").read_text())
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]

    ns: dict = {
        "np": np,
        "torch": torch,
        "Skeleton": UpSkeleton,
        "n_raw_offsets": torch.from_numpy(t2m_raw_offsets),
        "kinematic_chain": t2m_kinematic_chain,
        "face_joint_indx": [2, 1, 17, 16],
        "fid_l": [7, 10],
        "fid_r": [8, 11],
        "l_idx1": 5,
        "l_idx2": 8,
        "r_hip": 2,
        "l_hip": 1,
    }
    for k in dir(Q):
        if not k.startswith("_"):
            ns[k] = getattr(Q, k)

    exec("".join(code_cells[1]["source"]), ns)  # uniform_skeleton
    exec("".join(code_cells[2]["source"]), ns)  # process_file
    return ns


# ---------------------------------------------------------------------------
# Public: T+R → 263-D via upstream
# ---------------------------------------------------------------------------


def tplusr_to_h3d_features_upstream(
    translation: Tensor,
    quaternions: Tensor,
    skeleton: Skeleton,
    feet_thre: float = 0.002,
    repo_root: str | None = None,
) -> Tensor:
    """T+R → 263-D HumanML3D feature, using upstream's `process_file`.

    Bit-comparable agreement with the Guo evaluator's training distribution.

    Args:
        translation: (T, 3) root translation.
        quaternions: (T, J, 4) per-joint local quaternions.
        skeleton: rest-pose offsets used for FK + as `tgt_offsets` for upstream.
        feet_thre: foot-contact squared-velocity threshold (upstream default 0.002).
        repo_root: optional explicit repo root (otherwise auto-located).

    Returns:
        (T-1, 263) tensor on CPU.
    """
    # FK on our stored quats → world joint positions.
    positions = forward_kinematics(
        skeleton, quaternions.float(), translation.float()
    ).detach().cpu().numpy().astype(np.float32)  # (T, J, 3)

    ns = _load_upstream_process_file(repo_root)
    ns["tgt_offsets"] = skeleton.offsets.detach().cpu().float()  # (J, 3)
    process_file = ns["process_file"]

    # Upstream returns (data, ground_positions, positions, l_velocity)
    data, _, _, _ = process_file(positions, feet_thre)
    return torch.from_numpy(np.asarray(data, dtype=np.float32))
