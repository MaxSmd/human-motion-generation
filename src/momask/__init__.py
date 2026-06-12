"""MoMask reproduction (Guo et al. 2024).

Placeholder package — fill in as you implement.

Suggested top-level entry points (mirror `rmg`'s shape so cross-method tooling
stays uniform):

    from momask.models      import ResidualVQ, MaskedTransformer
    from momask.training    import VQTrainerCfg, MaskedTrainerCfg
    from momask.tasks       import generation as momask_generation

Shared, model-agnostic code lives in `common` (don't duplicate it):

    from common.eval   import RealGuoEvaluator, fid, r_precision, ...
    from common.utils  import EMA, Logger, save_checkpoint, ...

The HumanML3D loader (`rmg.data`) bakes in rmg's representation; build your
own dataset/encoding for this model, reusing `common` where you can

If you find yourself copy-pasting from `rmg/`, lift the shared piece into
`rmg/` (or a new `common/` package) instead.
"""
