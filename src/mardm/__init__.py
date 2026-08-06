"""MARDM reproduction.

Placeholder package — fill in as you implement.

Suggested top-level entry points (mirror `rmg`'s shape so cross-method tooling
stays uniform):

    from mardm.models     import MARDMModel
    from mardm.masking    import cosine_schedule, lengths_to_mask
    from mardm.generation import generate_h3d_features, sample_latents

Shared, model-agnostic code lives in `shared` (don't duplicate it):

    from shared.eval     import RealGuoEvaluator, fid, r_precision, ...
    from shared.utils    import EMA, Logger, save_checkpoint, ...
    from shared.text     import Qwen3EmbeddingEncoder, RandomTextEncoder
    from shared.geometry import Skeleton, make_continuous, normalize_quaternions
    from shared.data     import HumanML3DDataset, collate

The shared `HumanML3DDataset` is representation-agnostic: pass it this model's
`EssentialRepresentation` (see `mardm.data`) to get 67-D essential features.

If you find yourself copy-pasting from `rmg/`, lift the shared piece into
`shared/` instead.
"""
