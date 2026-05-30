"""MARDM data: 67-D essential dataset (reuses rmg's packed loader + collate)."""

from rmg.data.humanml3d import collate

from .dataset import EssentialDataset

__all__ = ["EssentialDataset", "collate"]
