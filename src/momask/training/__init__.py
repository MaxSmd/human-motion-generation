"""MoMask training loops (VQ + masked transformer + residual transformer).

TODO: implement the three-stage training:
  1. Train ResidualVQ on the 263-D feature space (reconstruction loss)
  2. Train masked generator transformer on VQ tokens (text-conditioned MLM)
  3. Train residual transformer on residual codebook tokens

Mirror `src/rmg/scripts/train.py` for the run-loop shape:
checkpointing (`rmg.utils.save_checkpoint`), EMA, Hydra config, AMP, resume.
"""
