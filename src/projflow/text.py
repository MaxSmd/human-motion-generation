"""CLIP ViT-B/32 text embeddings for ACMDM conditioning.

Upstream calls OpenAI's `clip` package (`clip.load("ViT-B/32")`,
`convert_weights` → fp16 linear layers, `encode_text(...).float()`). We use the
same weights through `transformers` (already a dependency) so the port needs no
git-only package; `dtype=float16` mirrors upstream's half-precision encoder.
projflow.scripts.check_port measures the gap to OpenAI's implementation.
"""

from __future__ import annotations

import torch
from torch import Tensor

CLIP_REPO = "openai/clip-vit-base-patch32"


class ClipTextEncoder:
    """Projected CLIP text features (B, 512), returned as float32."""

    def __init__(self, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float16,
                 repo: str = CLIP_REPO):
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer

        self.device = torch.device(device)
        if self.device.type == "cpu" and dtype == torch.float16:
            dtype = torch.float32  # half-precision CLIP is a GPU path
        self.tokenizer = CLIPTokenizer.from_pretrained(repo)
        self.model = CLIPTextModelWithProjection.from_pretrained(repo, torch_dtype=dtype).to(self.device).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, texts: list[str]) -> Tensor:
        tok = self.tokenizer(texts, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        out = self.model(input_ids=tok.input_ids.to(self.device))
        return out.text_embeds.float()
