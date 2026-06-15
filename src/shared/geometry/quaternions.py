"""Model-agnostic quaternion sign-convention helpers.

These resolve the q ↔ -q double-cover ambiguity of unit quaternions, both
per-frame (upper hemisphere) and across a temporal sequence (sign continuity).
Used by rmg's manifold/representation code and by mardm's essential
representation; kept here in `shared` so every model package can reuse them
without importing another model.
"""

from __future__ import annotations

import torch
from torch import Tensor


def quat_to_upper_hemisphere(q: Tensor) -> Tensor:
    """Flip sign so that q_0 ≥ 0 (paper App. A: avoids the q ↔ -q ambiguity)."""
    sign = torch.where(q[..., :1] < 0, -torch.ones_like(q[..., :1]), torch.ones_like(q[..., :1]))
    return q * sign


def quat_continuity(q: Tensor, dim: int = -2) -> Tensor:
    """Propagate sign continuity across a sequence of quaternions (along `dim`).

    For adjacent frames q_t, q_{t+1}: if <q_t, q_{t+1}> < 0, flip q_{t+1}'s sign.
    Done as a python loop over the temporal axis (small, e.g. 196 frames).
    """
    if dim < 0:
        dim = q.dim() + dim
    if q.shape[dim] < 2:
        return q
    out = q.clone()
    # iterate from index 1 along `dim`
    prev = out.index_select(dim, torch.tensor([0], device=q.device)).squeeze(dim)
    pieces = [prev]
    for i in range(1, q.shape[dim]):
        cur = out.index_select(dim, torch.tensor([i], device=q.device)).squeeze(dim)
        dot = (prev * cur).sum(dim=-1, keepdim=True)
        cur = torch.where(dot < 0, -cur, cur)
        pieces.append(cur)
        prev = cur
    return torch.stack(pieces, dim=dim)


def normalize_quaternions(quats: Tensor) -> Tensor:
    """Project to unit-norm and pick the upper hemisphere (q_w ≥ 0)."""
    q = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return quat_to_upper_hemisphere(q)


def make_continuous(quats: Tensor, time_dim: int) -> Tensor:
    """Resolve sign ambiguity along the temporal axis.

    HumanML3D / AMASS frames are independent quaternions; per-frame upper-
    hemisphere restriction can flip the sign between adjacent frames even
    though they represent close rotations. This pass propagates sign so that
    `<q_t, q_{t+1}> ≥ 0` everywhere — important for the geodesic path used
    during flow-matching to actually be the *short* arc.

    Operates per-joint independently along `time_dim`.
    """
    # quat_continuity does the loop; broadcast-friendly.
    return quat_continuity(quats, dim=time_dim)
