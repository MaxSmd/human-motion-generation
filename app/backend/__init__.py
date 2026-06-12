"""RMG interactive backend (FastAPI).

A thin in-process wrapper over the PyTorch training/inference code in `src/rmg`
that keeps the model warm in memory and renders motion clips to media files for
the React frontend. See `plan.md` for the full design.
"""
