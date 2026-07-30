"""Root-channel reparameterization: closed-form waypoint satisfaction for the pelvis.

Dims 0:4 of the 67-D essential vector are (root angular velocity, root linear
velocity x/z in the root's own frame, root height), and the pelvis world
trajectory is exactly their cumulative integral — see
`shared.geometry.humanml3d_io._recover_root_rot_pos`. The remaining 63 dims are
joint positions *relative to the root*, so they are invariant under any edit to
0:4: articulation, and therefore gait, is preserved by construction.

That makes pelvis control a linear problem instead of an optimization one.
Waypoint `p(τ_i) = target_i` constrains a prefix sum of the (rotated) velocity
channels, so the minimum-norm correction is piecewise constant: within each
inter-keyframe segment, spread the residual evenly. One pass, no gradients, no
diffusion calls — control at the cost of unguided sampling.

The catch is that moving the root while freezing the local pose desynchronizes
the legs from the ground: the closed-form solution is exact on the waypoints
and introduces foot skate proportional to how far it had to move the pelvis.
`skate_iters > 0` runs a short gradient polish over the four root channels only
(everything else stays frozen) trading a little waypoint error for less
sliding — this is where the geometric foot term from `losses.foot_skate_loss`
earns its keep.

Scope: joint 0 only. Any other constrained joint is ignored here — the root
channels cannot place an ankle without changing the pose.
"""

from __future__ import annotations

import torch
from torch import Tensor

from shared.geometry import quat_rotate, recover_joints_from_ric
from shared.geometry.humanml3d_io import _recover_root_rot_pos

from ..representation import denormalize
from .losses import ControlSignal, control_loss, foot_skate_loss


def _normalize(feats: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """Inverse of `representation.denormalize` (which has no public partner)."""
    return (feats - mean.to(feats)) / std.to(feats).clamp_min(1e-8)


def _segment_correction(residual: Tensor, frames: Tensor, t: int) -> Tensor:
    """Minimum-norm per-frame velocity correction hitting `residual` at `frames`.

    residual: (k, C) the amount p(τ_i) must change by; frames: (k,) sorted,
    strictly positive. Returns (T, C) — constant on each segment
    (τ_{i-1}, τ_i], zero after the last keyframe, so the tail keeps the
    velocity it already had rather than snapping back.
    """
    out = torch.zeros(t, residual.shape[-1], device=residual.device, dtype=residual.dtype)
    prev_f, prev_r = 0, torch.zeros_like(residual[0])
    for i in range(frames.shape[0]):
        f = int(frames[i])
        # p[f] = sum of velocities stored at indices [0, f), so the segment that
        # moves the pelvis from keyframe i-1 to keyframe i is [prev_f, f).
        out[prev_f:f] = (residual[i] - prev_r) / max(f - prev_f, 1)
        prev_f, prev_r = f, residual[i]
    return out


def _ramp_correction(residual: Tensor, frames: Tensor, t: int) -> Tensor:
    """Piecewise-linear correction equal to `residual` AT `frames` (height channel).

    Root height is stored absolutely, not as a velocity, so the smooth
    minimum-‖Δc‖² correction is a linear interpolation between the keyframe
    residuals, held constant outside the first/last keyframe.
    """
    idx = torch.arange(t, device=residual.device, dtype=residual.dtype)
    f = frames.to(residual.dtype)
    r = residual.squeeze(-1)
    if f.shape[0] == 1:
        return r.expand(t).unsqueeze(-1).clone()
    out = torch.zeros(t, device=residual.device, dtype=residual.dtype)
    out[:] = r[0]
    for i in range(1, f.shape[0]):
        lo, hi = f[i - 1], f[i]
        seg = (idx >= lo) & (idx <= hi)
        w = (idx - lo) / (hi - lo).clamp(min=1.0)
        out = torch.where(seg, r[i - 1] + w * (r[i] - r[i - 1]), out)
    out = torch.where(idx > f[-1], r[-1], out)
    return out.unsqueeze(-1)


@torch.no_grad()
def _closed_form(essential: Tensor, control: ControlSignal) -> Tensor:
    """Apply the minimum-norm root-channel edit. `essential` is DENORMALIZED (B, T, 67)."""
    out = essential.clone()
    b, t, _ = out.shape
    r_rot_quat, r_pos = _recover_root_rot_pos(out)             # (B, T, 4), (B, T, 3)
    tc = min(t, control.targets.shape[1])
    for i in range(b):
        # Frame 0 is the origin by construction (p(0) = 0), so a waypoint there
        # is already satisfied and carries no correction.
        frames = torch.nonzero(control.mask[i, :tc, 0], as_tuple=False).flatten()
        frames = frames[frames > 0]
        if frames.numel() == 0:
            continue
        frames, _ = torch.sort(frames)
        residual = control.targets[i, frames, 0] - r_pos[i, frames]   # (k, 3)

        # xz: correct the WORLD-frame velocity, then rotate into the root frame
        # the channels are stored in (world = R(q⁻¹)·local ⟹ local = R(q)·world;
        # q is a pure yaw rotation, so the y component is untouched).
        # Index care: recovery reads p[t] = Σ_{s<t} R(q_{s+1}⁻¹)·data[s, 1:3],
        # i.e. the velocity stored at s is rotated by the NEXT frame's yaw.
        d_world = _segment_correction(residual[:, [0, 2]], frames, t)  # (T, 2)
        d3 = torch.zeros(t, 3, device=out.device, dtype=out.dtype)
        d3[:, 0], d3[:, 2] = d_world[:, 0], d_world[:, 1]
        d_local = quat_rotate(r_rot_quat[i, 1:], d3[:-1])
        out[i, :-1, 1] += d_local[:, 0]
        out[i, :-1, 2] += d_local[:, 2]

        # height: stored absolutely, ramp between keyframes.
        out[i, :, 3] += _ramp_correction(residual[:, 1:2], frames, t).squeeze(-1)
    return out


def root_edit_essential(essential: Tensor, control: ControlSignal, mean: Tensor,
                        std: Tensor, *, skate_iters: int = 0, skate_lr: float = 0.01,
                        skate_weight: float = 1.0, skate_height: float = 0.05,
                        verbose: bool = False) -> Tensor:
    """Edit the root channels so the pelvis hits its waypoints. Normalized in/out.

    `essential`: (B, T, 67) as returned by `AE.decode`. Returns the same layout
    so the caller's downstream path (FID features, joint recovery) is unchanged.
    With `skate_iters > 0`, polishes the four root channels against
    control error + `skate_weight`·foot-skate; the 63 local-position dims are
    never touched in either stage.
    """
    control = control.to(essential.device)
    if control.num_constraints == 0:
        return essential
    mean, std = mean.to(essential.device), std.to(essential.device)
    dn = denormalize(essential, mean, std)
    edited = _closed_form(dn, control)

    if skate_iters > 0:
        # enable_grad so the polish still runs when a caller wraps the decode
        # path in torch.no_grad (both call sites do).
        with torch.enable_grad():
            root = edited[..., :4].detach().clone().requires_grad_(True)
            rest = edited[..., 4:].detach()
            optimizer = torch.optim.Adam([root], lr=skate_lr)
            for it in range(skate_iters):
                joints = recover_joints_from_ric(torch.cat([root, rest], dim=-1))
                loss = control_loss(joints, control) + skate_weight * foot_skate_loss(
                    joints, height=skate_height)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if verbose and (it == 0 or it == skate_iters - 1):
                    print(f"[root-edit] polish it {it + 1}/{skate_iters} "
                          f"L={float(loss.detach()):.4f}", flush=True)
        edited = torch.cat([root.detach(), rest], dim=-1)

    return _normalize(edited, mean, std)
