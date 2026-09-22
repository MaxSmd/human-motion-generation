"""ProjFlow sampling loop (Watanabe et al., CVPR 2026, §4.2, Fig. 2).

Per Euler step on the rectified-flow path x_t = t x₁ + (1 − t) x₀:

  1. clean endpoint  x̂₁ = x_t + (1 − t) v_θ(x_t, t),  x̂₀ = x_t − t v_θ   (Eq. 5)
  2. project x̂₁ onto the (time-varying) observations under R           (Eq. 7)
  3. recompose x_{t+Δt} = (t+Δt) x̂₁* + (1 − t − Δt) x̃₀ with
     x̃₀ = √(1−η) x̂₀ + √η ε,  η = 1 − σ_{t+Δt}  (FlowDPS mixing)           (Eq. 9–10)

Faithful to upstream Sampler.sample_projflow (uniform grid, NUM_STEPS − 1 model
calls, x̂₀ as the mixed noise), with two changes that do not alter the math:
the state carries only the conditional half of the CFG batch (upstream carries
both halves but the model only ever reads the first), and upstream's single
`use_projflow` switch is split into the three Table 3 ablations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor

from projflow.sampler.metric import KinematicMetric, project_clean_endpoint
from projflow.sampler.observations import (
    TrustSchedule,
    curvature,
    frame_trust,
    halo_radius,
    joint_trust,
    pseudo_observations,
    trust_to_variance,
)

VelocityFn = Callable[[Tensor, Tensor], Tensor]   # (x_t (B,D,L,J), t (B,)) -> v
NoiseFn = Callable[[Tensor], Tensor]              # template (B,D,L,J) -> ε ~ N(0, I)


@dataclass(frozen=True)
class ProjFlowConfig:
    """Defaults are the paper's (supp. Table 5); the switches give Table 3's rows."""

    num_steps: int = 100
    w_kin: float = 10.0
    ridge: float = 1.0
    ell_min: float = 3.0
    ell_max: float = 10.0
    trust: TrustSchedule = field(default_factory=TrustSchedule)
    use_kinematic_metric: bool = True   # False: "Euclid. (R=I)"
    use_pseudo_obs: bool = True         # False: "Plain masking"
    use_noise_mixing: bool = True       # False: "No noise (η_t = 0)"

    @classmethod
    def upstream_off(cls, **kw) -> "ProjFlowConfig":
        """Upstream's `use_projflow=False`: all three components disabled."""
        return cls(use_kinematic_metric=False, use_pseudo_obs=False, use_noise_mixing=False, **kw)


def projflow_sample(
    velocity: VelocityFn,
    x: Tensor,
    hard_mask: Tensor,
    hard_value: Tensor,
    cfg: ProjFlowConfig = ProjFlowConfig(),
    noise: NoiseFn | None = None,
) -> Tensor:
    """Run the ProjFlow sampler from initial noise `x` (B, D, L, J).

    `hard_mask` / `hard_value` (B, D, L, J) are the keyframe observations in the
    model's (normalised) space. `noise` draws the mixing ε (default randn_like).
    Returns the final sample (B, D, L, J); hard rows hold `hard_value` exactly.
    """
    if cfg.num_steps < 2:
        raise ValueError("num_steps must be >= 2")
    B = x.shape[0]
    device, dtype = x.device, x.dtype
    hard_mask = hard_mask.to(device=device, dtype=dtype)
    hard_value = hard_value.to(device=device, dtype=dtype)
    hard = hard_mask > 0.5
    noise = noise or torch.randn_like
    build = KinematicMetric.kinematic if cfg.use_kinematic_metric else KinematicMetric.euclidean
    metric = build(cfg.w_kin, cfg.ridge, device=device, dtype=dtype)

    t_grid = torch.linspace(0.0, 1.0, cfg.num_steps, device=device, dtype=dtype)
    for k in range(cfg.num_steps - 1):
        t_k, t_next = float(t_grid[k]), float(t_grid[k + 1])
        v = velocity(x, torch.full((B,), t_k, device=device, dtype=dtype))
        x1_hat = x + (1.0 - t_k) * v
        x0_hat = x - t_k * v

        if cfg.use_pseudo_obs:
            radius = halo_radius(k, cfg.num_steps, cfg.ell_min, cfg.ell_max)
            y_src, halo = pseudo_observations(hard_mask, hard_value, radius)
            selected = (hard | (halo > 0.5)).to(dtype)
            targets = torch.where(hard, hard_value, torch.where(halo > 0.5, y_src, hard_value))
            pi_frame = frame_trust(t_k, curvature(x1_hat, metric), cfg.trust)
            pi_joint = joint_trust(pi_frame, halo[:, 0] > 0.5, metric.joint_weights_q, cfg.trust)
            sigma2 = trust_to_variance(pi_joint, metric, hard_mask, selected)
        else:
            selected, targets, sigma2 = hard.to(dtype), hard_value, None

        x1_proj = project_clean_endpoint(x1_hat, selected, targets, metric, sigma2)

        sigma_next = 1.0 - t_next
        if cfg.use_noise_mixing:
            eta = 1.0 - sigma_next
            x0_mix = (1.0 - eta) ** 0.5 * x0_hat + eta**0.5 * noise(x0_hat)
        else:
            x0_mix = x0_hat
        x = t_next * x1_proj + sigma_next * x0_mix
    return x
