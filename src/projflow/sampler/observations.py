"""Time-varying observation model for motion inpainting (ProjFlow §4.3.1).

Sparse hard keyframes are densified with *pseudo-observations*: per-joint linear
interpolation between the nearest keyframes (copying the nearest one outside
the observed range). They are active only within a shrinking temporal radius
ℓ(t) of a keyframe ("dynamic masking", Eq. 14) and carry a variance derived
from a trust score that decays with sampling time and local curvature
("adaptive variance", Eq. 15–18, supp. B.2).

Ported from upstream's projflow_helpers.py (same formulas and constants);
tensors are (B, D, L, J).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from projflow.sampler.metric import KinematicMetric


def halo_radius(step: int, num_steps: int, ell_min: float, ell_max: float) -> float:
    """ℓ(t): linear from ell_max at the first step to ell_min at the last."""
    if num_steps <= 1:
        return float(ell_min)
    u = step / (num_steps - 1)
    return float(ell_max - (ell_max - ell_min) * u)


def _nearest_keyframes(observed: Tensor) -> tuple[Tensor, Tensor]:
    """Index of the nearest observed frame at or before / at or after each frame.

    observed: (B, L, J) bool. Returns (prev, next), each (B, L, J) long, −1 if none.
    """
    B, L, J = observed.shape
    t = torch.arange(L, device=observed.device).view(1, L, 1).expand(B, L, J)
    prev = torch.where(observed, t, torch.full_like(t, -1)).cummax(dim=1).values
    next_rev = torch.where(observed.flip(1), t, torch.full_like(t, -1)).cummax(dim=1).values
    nxt = torch.where(next_rev.flip(1) >= 0, (L - 1 - next_rev).flip(1), torch.full_like(t, -1))
    return prev, nxt


def pseudo_observations(hard_mask: Tensor, hard_value: Tensor, radius: float) -> tuple[Tensor, Tensor]:
    """Interpolated targets y_src and the halo selector M_halo (hard rows excluded).

    Returns (y_src, halo), both (B, D, L, J); halo is 0/1.
    """
    B, D, L, J = hard_value.shape
    dtype = hard_value.dtype
    hard = hard_mask > 0.5
    observed = hard.any(dim=1)                                      # (B, L, J)
    prev, nxt = _nearest_keyframes(observed)

    def gather(idx: Tensor) -> Tensor:
        return torch.gather(hard_value, 2, idx.clamp(min=0).view(B, 1, L, J).expand(B, D, L, J))

    y_prev, y_next = gather(prev), gather(nxt)
    prev_ok = (prev >= 0).view(B, 1, L, J)
    next_ok = (nxt >= 0).view(B, 1, L, J)
    both = prev_ok & next_ok

    t = torch.arange(L, device=hard_value.device, dtype=dtype).view(1, 1, L, 1)
    span = (nxt - prev).clamp(min=1).to(dtype).view(B, 1, L, J)
    alpha = ((t - prev.to(dtype).view(B, 1, L, J)) / span).clamp(0, 1)

    y_src = torch.zeros_like(hard_value)
    y_src = torch.where(both, (1.0 - alpha) * y_prev + alpha * y_next, y_src)
    y_src = torch.where(prev_ok & ~both, y_prev, y_src)
    y_src = torch.where(next_ok & ~both, y_next, y_src)

    frames = torch.arange(L, device=hard_value.device).view(1, L, 1)
    thr = max(float(radius), 1e-6)
    near = ((prev >= 0) & ((frames - prev).to(dtype) < thr)) | ((nxt >= 0) & ((nxt - frames).to(dtype) < thr))
    halo = (~observed & near).to(dtype).view(B, 1, L, J).expand(B, D, L, J)
    halo = halo * (~hard).to(dtype)
    return y_src.contiguous(), halo.contiguous()


@dataclass(frozen=True)
class TrustSchedule:
    """Adaptive-variance hyperparameters (supp. Table 5)."""

    tau_min: float = 0.1
    c0: float = 3.0
    lambda_s: float = 1.0
    p: float = 2.0
    pi_min: float = 0.02
    pi_max: float = 1.0


def curvature(x1_hat: Tensor, metric: KinematicMetric) -> Tensor:
    """s_n = ||x_{n+1} − 2x_n + x_{n−1}||_R per frame (Eq. 17), edge-replicated. (B, L)."""
    x_prev = torch.cat([x1_hat[:, :, :1], x1_hat[:, :, :-1]], dim=2)
    x_next = torch.cat([x1_hat[:, :, 1:], x1_hat[:, :, -1:]], dim=2)
    acc = x_next - 2.0 * x1_hat + x_prev
    energy = torch.einsum("bdlj,jk,bdlk->bl", acc, metric.energy_joint.to(acc), acc).clamp_min(0.0)
    return torch.sqrt(energy + 1e-12)


def frame_trust(t: float, curv: Tensor, schedule: TrustSchedule) -> Tensor:
    """π_n = clip(τ(t) c0 / (1 + λ_s (s_n / s_med)^p), π_min, π_max) (Eq. 15–16). (B, L)."""
    s_med = torch.quantile(curv, 0.5, dim=1, keepdim=True, interpolation="lower").clamp_min(1e-6)
    tau = schedule.tau_min + (1.0 - schedule.tau_min) * (1.0 - t)
    pi = tau * schedule.c0 / (1.0 + schedule.lambda_s * (curv / s_med).pow(schedule.p))
    return pi.clamp(schedule.pi_min, schedule.pi_max)


def joint_trust(pi_frame: Tensor, halo_any: Tensor, q_joint: Tensor, schedule: TrustSchedule) -> Tensor:
    """Split each frame's trust over its halo joints ∝ q_j (supp. Eq. 52). (B, 1, L, J)."""
    B, L = pi_frame.shape
    J = q_joint.numel()
    q = q_joint.view(1, 1, J).to(pi_frame)
    denom = (halo_any.to(pi_frame.dtype) * q).sum(dim=2, keepdim=True).clamp_min(1e-12)
    pi = (pi_frame.view(B, L, 1) * q / denom).clamp(schedule.pi_min, schedule.pi_max)
    return pi.view(B, 1, L, J)


def trust_to_variance(pi_joint: Tensor, metric: KinematicMetric, hard_mask: Tensor, selected: Tensor) -> Tensor:
    """σ²_i = r_i (1/π_i − 1) on pseudo-observation rows, 0 on hard rows (Eq. 18). (B, D, L, J)."""
    B, D, L, J = hard_mask.shape
    hard = hard_mask > 0.5
    halo = (selected > 0.5) & ~hard
    r = metric.diag_rinv_joint.view(1, 1, 1, J).to(hard_mask)
    sigma2 = r * (1.0 / pi_joint.expand(B, D, L, J).clamp_min(1e-12) - 1.0)
    return torch.where(halo, sigma2, torch.zeros_like(sigma2)).contiguous()
