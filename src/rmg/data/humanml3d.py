"""HumanML3D dataset reader for rmg — the shared loader with rmg's T+R encoding.

The packed format and all loading / subset / mirror / crop / pad logic live in
`shared.data.humanml3d` (model-agnostic, reused by mardm and MoMask). rmg's only
specialization is defaulting the representation to T+R (the paper's main
result); other models supply their own — token-based models want
`shared.data.H3D263Dataset`. `HumanML3DSample`, `CollatedBatch` and `collate`
are re-exported from `shared.data` so existing imports are unchanged.
"""

from __future__ import annotations

from pathlib import Path

from shared.data import (
    ClipRepresentation,
    CollatedBatch,
    HumanML3DSample,
    collate,
)
from shared.data import HumanML3DDataset as _BaseHumanML3DDataset

from ..representation import TRRepresentation

__all__ = ["HumanML3DDataset", "HumanML3DSample", "CollatedBatch", "collate"]


class HumanML3DDataset(_BaseHumanML3DDataset):
    """Shared packed loader with rmg's T+R representation as the default.

    Identical public signature and behavior to the original rmg dataset; the
    representation defaults to `TRRepresentation()` (paper main result).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        max_seq_len: int = 196,
        min_seq_len: int = 40,
        mirror_augment: bool = False,
        zip_name: str = "humanml3d.zip",
        splits_name: str = "splits.json",
        offsets_name: str = "target_offsets.pt",
        representation: ClipRepresentation | None = None,
        subset_fraction: float = 1.0,
        subset_seed: int = 0,
        subset_n: int = 0,
        preload: bool = False,
        canonicalize_crops: bool = False,
    ) -> None:
        super().__init__(
            root,
            split=split,
            max_seq_len=max_seq_len,
            min_seq_len=min_seq_len,
            zip_name=zip_name,
            splits_name=splits_name,
            offsets_name=offsets_name,
            mirror_augment=mirror_augment,
            representation=representation or TRRepresentation(),
            subset_fraction=subset_fraction,
            subset_seed=subset_seed,
            subset_n=subset_n,
            preload=preload,
            canonicalize_crops=canonicalize_crops,
        )
