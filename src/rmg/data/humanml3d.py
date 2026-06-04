"""HumanML3D dataset reader for rmg and MoMask-compatible 263-D features.

The packed format and loading / subset / mirror / crop / pad logic live in
`shared.data.humanml3d`. This module only provides rmg's default T+R
representation and a compatibility `output_mode="h3d_263"` path used by
MoMask.
"""

from __future__ import annotations

from pathlib import Path

from torch import Tensor

from shared.data import (
    ClipRepresentation,
    CollatedBatch,
    HumanML3DSample,
    collate,
)
from shared.data import HumanML3DDataset as _BaseHumanML3DDataset
from shared.geometry import Skeleton

from ..representation import H3D_FEATURE_DIM, TRRepresentation, tplusr_to_h3d_features_with_quats

__all__ = ["HumanML3DDataset", "HumanML3DSample", "CollatedBatch", "collate"]


class _HumanML3D263Representation:
    """Adapter that converts packed T+R clips into standard HumanML3D 263-D features."""

    def encode_clip(self, translation: Tensor, quaternions: Tensor, skeleton: Skeleton | None = None) -> Tensor:
        if skeleton is None:
            raise FileNotFoundError(
                "output_mode='h3d_263' requires target_offsets.pt so clips can be "
                "converted into standard HumanML3D 263-D features."
            )
        x = tplusr_to_h3d_features_with_quats(translation, quaternions, skeleton)
        if x.shape[-1] != H3D_FEATURE_DIM:
            raise AssertionError(f"expected {H3D_FEATURE_DIM}-D H3D features, got {x.shape[-1]}")
        return x


class HumanML3DDataset(_BaseHumanML3DDataset):
    """Shared packed loader with rmg's T+R representation as the default.

    Set `output_mode="h3d_263"` for token-based baselines such as MoMask.
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
        output_mode: str = "representation",
        subset_fraction: float = 1.0,
        subset_seed: int = 0,
        subset_n: int = 0,
    ) -> None:
        if output_mode not in ("representation", "h3d_263"):
            raise ValueError("output_mode must be 'representation' or 'h3d_263'")
        if output_mode == "h3d_263":
            if representation is not None:
                raise ValueError("pass either representation or output_mode='h3d_263', not both")
            representation = _HumanML3D263Representation()
        else:
            representation = representation or TRRepresentation()

        super().__init__(
            root,
            split=split,
            max_seq_len=max_seq_len,
            min_seq_len=min_seq_len,
            zip_name=zip_name,
            splits_name=splits_name,
            offsets_name=offsets_name,
            mirror_augment=mirror_augment,
            representation=representation,
            subset_fraction=subset_fraction,
            subset_seed=subset_seed,
            subset_n=subset_n,
        )
