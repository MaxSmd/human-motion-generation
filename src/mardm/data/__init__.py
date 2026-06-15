"""MARDM data: 67-D essential dataset (reuses the shared packed loader + collate)."""

from shared.data import collate

from .dataset import EssentialDataset

__all__ = ["EssentialDataset", "collate"]
