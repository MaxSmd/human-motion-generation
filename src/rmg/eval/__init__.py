"""Backward-compatible re-export shim for evaluator utilities.

The evaluator and metrics live in `shared.eval` so multiple model packages can
reuse them. This package keeps older `rmg.eval` imports working.
"""

from shared.eval import (
    GuoEvaluator,
    RandomGuoEvaluator,
    RealGuoEvaluator,
    calculate_activation_statistics,
    calculate_fid,
    diversity,
    fid,
    mm_distance,
    multimodality,
    r_precision,
    r_precision_batch,
)

__all__ = [
    "GuoEvaluator",
    "RealGuoEvaluator",
    "RandomGuoEvaluator",
    "calculate_activation_statistics",
    "calculate_fid",
    "fid",
    "r_precision",
    "r_precision_batch",
    "mm_distance",
    "diversity",
    "multimodality",
]
