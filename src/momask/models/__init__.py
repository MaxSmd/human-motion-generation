"""MoMask model components (residual VQ-VAE + masked transformer).

TODO:
  - ResidualVQ encoder/decoder over 263-D HumanML3D features
  - Masked generator transformer (text-conditioned)
  - Residual transformer (refines residual codebook tokens)

Keep tensors on (B, T, D) layout to match the rest of the repo. Reuse
`rmg.models.text_encoder.Qwen3EmbeddingEncoder` for text conditioning so all
three methods use the same text representation.
"""
