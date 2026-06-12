"""Shared, model-agnostic code: the Guo evaluator + metrics (`common.eval`) and
training utilities (`common.utils`). Imported by every model package; never
imports a model. (The HumanML3D loader stays in `rmg.data` — it bakes in rmg's
representation, so it is not model-agnostic yet.)"""
