"""Runtime shims that let the pinned upstream ProjFlow code (written against
torch 2.2 / numpy 1.21) run unmodified in our torch 2.9 / numpy 2 container.

Loaded automatically by Python when this directory is on PYTHONPATH; only the
slurm/projflow/*upstream* jobs put it there. The submodule itself is never
edited, so Stage A stays a faithful run of the published code.
"""
import numpy as np

# utils/quaternion.py evaluates np.finfo(np.float) at import time; the aliases
# were removed in numpy 1.24.
for _name, _type in (("float", float), ("int", int), ("bool", bool), ("object", object)):
    if not hasattr(np, _name):
        setattr(np, _name, _type)

# torch>=2.6 defaults torch.load(weights_only=True); the upstream generator and
# evaluator checkpoints are full pickles.
try:
    import torch

    _orig_load = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    torch.load = _load
except ImportError:
    pass
