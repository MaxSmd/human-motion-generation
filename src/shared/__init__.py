"""Shared, model-agnostic code: the Guo evaluator + metrics (`shared.eval`) and
training utilities (`shared.utils`). Imported by every model package; never
imports a model. (The HumanML3D loader stays in `rmg.data` — it bakes in rmg's
representation, so it is not model-agnostic yet.)

Named `shared`, not `common`, to avoid colliding with HumanML3D's upstream
`common` package (`common.skeleton`, `common.quaternion`)."""
