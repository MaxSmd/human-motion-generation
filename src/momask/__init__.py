"""MoMask reproduction (Guo et al. 2024).

Placeholder package — fill in as you implement.

Suggested top-level entry points (mirror `rmg`'s shape so cross-method tooling
stays uniform):

    from momask.models      import ResidualVQ, MaskedTransformer
    from momask.training    import VQTrainerCfg, MaskedTrainerCfg
    from momask.tasks       import generation as momask_generation

Shared utilities live in `rmg` and are intentionally NOT duplicated here:

    from rmg.data           import HumanML3DDataset, collate
    from rmg.eval           import RealGuoEvaluator, fid, r_precision, ...
    from rmg.utils          import EMA, Logger, save_checkpoint, ...

If you find yourself copy-pasting from `rmg/`, lift the shared piece into
`rmg/` (or a new `common/` package) instead.
"""
