"""MARDM reproduction.

Placeholder package — fill in as you implement.

Suggested top-level entry points (mirror `rmg`'s shape so cross-method tooling
stays uniform):

    from mardm.models     import MARDMModel
    from mardm.training   import MARDMTrainerCfg
    from mardm.tasks      import generation as mardm_generation

Shared, model-agnostic code lives in `shared` (don't duplicate it):

    from shared.eval   import RealGuoEvaluator, fid, r_precision, ...
    from shared.utils  import EMA, Logger, save_checkpoint, ...

The HumanML3D loader (`rmg.data`) bakes in rmg's representation; build your
own dataset/encoding for this model, reusing `shared` where you can

If you find yourself copy-pasting from `rmg/`, lift the shared piece into
`rmg/` (or a new `common/` package) instead.
"""
