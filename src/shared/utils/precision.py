"""Auto-pick training precision based on the local CUDA device capability."""

from __future__ import annotations

import warnings

import torch


def resolve_precision(requested: str, *, device: torch.device | None = None) -> str:
    # bf16 has no tensor-core path on pre-Ampere (Turing, sm_75 and earlier) and
    # silently falls back to fp32 emulation that is ~50x slower than the fp16
    # tensor-core path. "auto" picks fp16 there, bf16 on Ampere+. Explicit
    # settings are honored but flagged when the device disagrees.
    if not torch.cuda.is_available():
        return "fp32" if requested == "auto" else requested

    device = device or torch.device("cuda", 0)
    cap = torch.cuda.get_device_properties(device).major
    name = torch.cuda.get_device_name(device)

    if requested == "auto":
        chosen = "bf16" if cap >= 8 else "fp16"
        print(f"[precision] auto → {chosen} (GPU: {name}, sm_{cap}x)")
        return chosen

    if requested == "bf16" and cap < 8:
        warnings.warn(
            f"train.precision=bf16 on {name} (sm_{cap}x, pre-Ampere): bf16 has "
            "no tensor-core support here; expect ~50x slowdown vs fp16. Use "
            "'fp16' or 'auto'.",
            stacklevel=2,
        )
    return requested
