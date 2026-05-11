"""Exponential moving average of model parameters (paper §4.1)."""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import torch
from torch import Tensor, nn


class EMA:
    """Holds shadow weights `w_ema = decay * w_ema + (1-decay) * w` for every
    floating-point parameter and buffer of `model`. State-dict friendly so the
    EMA snapshot rides along with the training checkpoint.

    Usage:
        ema = EMA(model, decay=0.9999)
        ...
        loss.backward(); opt.step()
        ema.update(model)
        ...
        # at sampling time:
        with ema.swapped(model):
            samples = sampler.sample(model, ...)
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.decay = decay
        # Track every floating-point tensor (params + buffers) by name so that
        # both BN running stats and learnable params are EMA'd consistently.
        self.shadow: dict[str, Tensor] = {
            k: v.detach().clone()
            for k, v in self._floating_state(model).items()
        }
        # Backup buffer used by `swapped`.
        self._backup: dict[str, Tensor] | None = None

    @staticmethod
    def _floating_state(model: nn.Module) -> dict[str, Tensor]:
        return {k: v for k, v in model.state_dict().items() if v.is_floating_point()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        cur = self._floating_state(model)
        for k, ema_t in self.shadow.items():
            ema_t.mul_(self.decay).add_(cur[k].detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Overwrite `model`'s floating-point state with the EMA shadow."""
        sd = model.state_dict()
        for k, ema_t in self.shadow.items():
            sd[k].copy_(ema_t)

    class _Swap:
        def __init__(self, ema: "EMA", model: nn.Module) -> None:
            self.ema = ema
            self.model = model

        def __enter__(self) -> nn.Module:
            assert self.ema._backup is None, "swapped contexts cannot be nested"
            self.ema._backup = {
                k: v.detach().clone() for k, v in self.ema._floating_state(self.model).items()
            }
            self.ema.copy_to(self.model)
            return self.model

        def __exit__(self, *exc) -> None:
            assert self.ema._backup is not None
            sd = self.model.state_dict()
            for k, v in self.ema._backup.items():
                sd[k].copy_(v)
            self.ema._backup = None

    def swapped(self, model: nn.Module) -> "EMA._Swap":
        """Context manager that swaps EMA weights into `model` for inference,
        then restores the live training weights on exit."""
        return EMA._Swap(self, model)

    def state_dict(self) -> dict:
        return {"decay": float(self.decay), "shadow": self.shadow}

    def load_state_dict(self, sd: dict) -> None:
        self.decay = float(sd["decay"])
        self.shadow = sd["shadow"]
