"""Riemannian Euler ODE solver for sampling (paper eq. 5).

  x_{t+h} = Exp_{x_t}( h · Π_{T_{x_t}M} v_θ(x_t, t, cond) )

CFG: run the model twice per step (cond + uncond) and combine ambient
velocities before projection — the projection is linear so this is equivalent
to combining post-projection.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..manifolds import Manifold
from .prior import WrappedGaussianPrior


@dataclass
class SamplerCfg:
    num_steps: int = 50
    guidance_scale: float = 6.5  # paper's best ω on HumanML3D


class RiemannianEulerSampler:
    def __init__(
        self,
        manifold: Manifold,
        prior: WrappedGaussianPrior,
        cfg: SamplerCfg | None = None,
    ) -> None:
        self.manifold = manifold
        self.prior = prior
        self.cfg = cfg or SamplerCfg()

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, int],
        cond: Tensor | None = None,
        mask: Tensor | None = None,
        guidance_scale: float | None = None,
        num_steps: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        return_trajectory: bool = False,
        generator: torch.Generator | None = None,
        fixed_values: Tensor | None = None,
        fixed_mask: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Generate (B, T, D) samples by integrating the learned velocity from t=0 to t=1.

        Args:
            shape: (B, T) — batch size and sequence length.
            cond: per-batch conditioning (e.g. text features). None = unconditional.
            guidance_scale: CFG scale ω. None ⇒ uses cfg.guidance_scale.
            num_steps: ODE steps. None ⇒ uses cfg.num_steps.
            fixed_values / fixed_mask: optional sampling-time constraints. Where
                `fixed_mask` is True, the state is overwritten with `fixed_values`
                after every ODE step (inpainting). For RMG these pin whole S^3
                joint factors to a target quaternion — see `flow.constraints`.
                Both broadcast against (B, T, D) (e.g. pass (T, D)).
        """
        B, T = shape
        n = num_steps if num_steps is not None else self.cfg.num_steps
        omega = guidance_scale if guidance_scale is not None else self.cfg.guidance_scale
        device = device or (cond.device if isinstance(cond, Tensor) else torch.device("cpu"))
        dtype = dtype or (cond.dtype if isinstance(cond, Tensor) else torch.float32)

        # Optional inpainting constraints: move to device/dtype once, then apply
        # at init and after every step so the model conditions on the pinned
        # coordinates from the very first integration step.
        apply_constraint = fixed_mask is not None and fixed_values is not None
        if apply_constraint:
            fixed_mask = fixed_mask.to(device=device)
            fixed_values = fixed_values.to(device=device, dtype=dtype)

        # initial state x_0 ~ prior
        x = self.prior.sample((B, T), device=device, dtype=dtype, generator=generator)
        if apply_constraint:
            x = torch.where(fixed_mask, fixed_values, x)

        # Full ODE integration over [0, 1]. The closed-form CFM target γ̇(t)
        # is finite at both endpoints, so no eps-offset is needed at inference
        # (the t_eps trim during training is to avoid the 1/(1-t) factor in the
        # log-form target — different concern).
        ts = torch.linspace(0.0, 1.0, n + 1, device=device, dtype=dtype)
        traj = [x] if return_trajectory else None

        do_cfg = cond is not None and abs(omega - 1.0) > 1e-8

        for i in range(n):
            t_i = ts[i]
            h = ts[i + 1] - t_i
            t_batch = t_i.expand(B)

            if do_cfg:
                drop_false = torch.zeros(B, dtype=torch.bool, device=device)
                drop_true = torch.ones(B, dtype=torch.bool, device=device)
                v_cond = model(x, t_batch, cond=cond, drop_cond_mask=drop_false, mask=mask)
                v_uncond = model(x, t_batch, cond=cond, drop_cond_mask=drop_true, mask=mask)
                v_amb = v_uncond + omega * (v_cond - v_uncond)
            else:
                drop_mask = torch.zeros(B, dtype=torch.bool, device=device) if cond is not None else None
                v_amb = model(x, t_batch, cond=cond, drop_cond_mask=drop_mask, mask=mask)

            v_t = self.manifold.project_tangent(x, v_amb)
            x = self.manifold.exp(x, h * v_t)

            if apply_constraint:
                x = torch.where(fixed_mask, fixed_values, x)

            if traj is not None:
                traj.append(x)

        if traj is not None:
            return x, torch.stack(traj, dim=0)  # (n+1, B, T, D)
        return x


# ---------------------------------------------------------------------------
# Utility: an "oracle" model that returns the exact CFM target. Used in tests
# to verify the sampler can recover x_1 given the true velocity field.
# ---------------------------------------------------------------------------


class OracleVelocity(nn.Module):
    """Returns the exact conditional CFM target for fixed (x_0, x_1).

    cond is ignored; drop_cond_mask is ignored. Used for sampler unit tests:
    integrating from x_0 with this 'model' must recover x_1.
    """

    def __init__(self, manifold: Manifold, x0: Tensor, x1: Tensor) -> None:
        super().__init__()
        self.manifold = manifold
        self.register_buffer("x0", x0)
        self.register_buffer("x1", x1)

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        *,
        cond: Tensor | None = None,
        drop_cond_mask: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        # The CFM target depends only on (x_0, x_1, t) — not on the current x_t.
        # That's because the conditional flow {x_t}_{t∈[0,1]} traces a single
        # geodesic, so given (x_0, x_1, t) the target is fully determined.
        return self.manifold.cfm_target_velocity(self.x0, self.x1, t)
