"""MoMask smoke tests over the shared HumanML3D packed dataset path."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from momask.models import (
    MaskedMotionTransformer,
    MotionRVQVAE,
    ResidualTransformer,
    TokenTransformerConfig,
)
from momask.tasks import generate_h3d263
from rmg.data import HumanML3DDataset, collate
from rmg.representation import H3D_FEATURE_DIM, NUM_JOINTS


def _toy_offsets() -> torch.Tensor:
    o = torch.zeros(NUM_JOINTS, 3, dtype=torch.float32)
    o[1] = torch.tensor([0.10, -0.05, 0.0])
    o[2] = torch.tensor([-0.10, -0.05, 0.0])
    o[3] = torch.tensor([0.0, 0.10, 0.0])
    o[4] = torch.tensor([0.0, -0.40, 0.0])
    o[5] = torch.tensor([0.0, -0.40, 0.0])
    o[6] = torch.tensor([0.0, 0.10, 0.0])
    o[7] = torch.tensor([0.0, -0.40, 0.0])
    o[8] = torch.tensor([0.0, -0.40, 0.0])
    o[9] = torch.tensor([0.0, 0.10, 0.0])
    o[10] = torch.tensor([0.0, -0.05, 0.10])
    o[11] = torch.tensor([0.0, -0.05, 0.10])
    o[12] = torch.tensor([0.0, 0.20, 0.0])
    o[13] = torch.tensor([0.05, 0.10, 0.0])
    o[14] = torch.tensor([-0.05, 0.10, 0.0])
    o[15] = torch.tensor([0.0, 0.10, 0.0])
    o[16] = torch.tensor([0.10, 0.0, 0.0])
    o[17] = torch.tensor([-0.10, 0.0, 0.0])
    o[18] = torch.tensor([0.30, 0.0, 0.0])
    o[19] = torch.tensor([-0.30, 0.0, 0.0])
    o[20] = torch.tensor([0.30, 0.0, 0.0])
    o[21] = torch.tensor([-0.30, 0.0, 0.0])
    return o


def _clip(T: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    translation = torch.zeros(T, 3)
    translation[:, 0] = torch.linspace(0.0, 0.4, T)
    translation[:, 1] = 0.9
    translation += torch.randn(T, 3, generator=g) * 0.01
    quats = torch.randn(T, NUM_JOINTS, 4, generator=g)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    return {"translation": translation, "quats": quats, "texts": [f"motion {seed}"]}


def _build_dataset(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    clips = {"000000": _clip(32, 0), "000001": _clip(36, 1)}
    with zipfile.ZipFile(root / "humanml3d.zip", "w", compression=zipfile.ZIP_STORED) as zf:
        for cid, blob in clips.items():
            buf = io.BytesIO()
            torch.save(blob, buf)
            zf.writestr(f"{cid}.pt", buf.getvalue())
    (root / "splits.json").write_text(json.dumps({"train": list(clips), "val": [], "test": []}))
    torch.save(_toy_offsets(), root / "target_offsets.pt")


def test_shared_dataset_can_emit_h3d_263_for_momask(tmp_path: Path) -> None:
    _build_dataset(tmp_path)
    ds = HumanML3DDataset(
        tmp_path,
        split="train",
        min_seq_len=10,
        max_seq_len=40,
        mirror_augment=False,
        output_mode="h3d_263",
    )
    sample = ds[0]
    assert sample.x1.shape == (31, H3D_FEATURE_DIM)
    assert sample.length == 31
    assert torch.isfinite(sample.x1).all()

    batch = next(iter(DataLoader(ds, batch_size=2, collate_fn=collate)))
    assert batch.x1.shape[-1] == H3D_FEATURE_DIM
    assert batch.mask.shape == batch.x1.shape[:2]


def test_momask_models_tokenize_predict_and_decode() -> None:
    torch.manual_seed(0)
    B, T = 2, 12
    x = torch.randn(B, T, H3D_FEATURE_DIM)
    mask = torch.ones(B, T, dtype=torch.bool)

    vqvae = MotionRVQVAE(
        input_dim=H3D_FEATURE_DIM,
        hidden_dim=32,
        latent_dim=16,
        num_quantizers=3,
        codebook_size=8,
        downsample=1,
        num_res_blocks=1,
    )
    out = vqvae(x, mask=mask)
    assert out.recon.shape == x.shape
    assert out.tokens.shape == (B, 3, T)
    assert torch.isfinite(out.loss)

    cfg = TokenTransformerConfig(
        vocab_size=8,
        text_dim=20,
        hidden_dim=32,
        depth=1,
        num_heads=4,
        ffn_dim=64,
        max_seq_len=T,
        dropout=0.0,
    )
    cond = torch.randn(B, cfg.text_dim)
    masked = MaskedMotionTransformer(cfg)
    residual = ResidualTransformer(cfg, num_quantizers=3)

    base_loss = masked.training_loss(out.tokens[:, 0], cond=cond, valid_mask=mask)
    res_loss = residual.training_loss(out.tokens, target_level=1, cond=cond, valid_mask=mask)
    assert torch.isfinite(base_loss)
    assert torch.isfinite(res_loss)

    base = masked.generate(cond=cond, seq_len=T, steps=2, guidance_scale=1.0, mask=mask)
    tokens = residual.generate_residuals(base, cond=cond, guidance_scale=1.0, mask=mask)
    recon = vqvae.decode_from_tokens(tokens, target_len=T)
    assert tokens.shape == (B, 3, T)
    assert recon.shape == x.shape

    ragged_mask = mask.clone()
    ragged_mask[0, -3:] = False
    ragged_base = masked.generate(cond=cond, seq_len=T, steps=2, guidance_scale=1.0, mask=ragged_mask)
    assert (ragged_base[0, -3:] == 0).all()
    ragged_tokens = residual.generate_residuals(ragged_base, cond=cond, guidance_scale=1.0, mask=ragged_mask)
    assert ragged_tokens.shape == (B, 3, T)


def test_generation_helper_returns_h3d_features() -> None:
    torch.manual_seed(1)
    T = 8
    vqvae = MotionRVQVAE(
        input_dim=H3D_FEATURE_DIM,
        hidden_dim=16,
        latent_dim=8,
        num_quantizers=2,
        codebook_size=6,
        num_res_blocks=1,
    )
    cfg = TokenTransformerConfig(
        vocab_size=6,
        text_dim=12,
        hidden_dim=16,
        depth=1,
        num_heads=4,
        ffn_dim=32,
        max_seq_len=T,
        dropout=0.0,
    )
    cond = torch.randn(1, cfg.text_dim)
    motion = generate_h3d263(
        vqvae=vqvae,
        masked_transformer=MaskedMotionTransformer(cfg),
        residual_transformer=ResidualTransformer(cfg, num_quantizers=2),
        cond=cond,
        seq_len=T,
        steps=2,
        guidance_scale=1.0,
        target_len=T,
    )
    assert motion.shape == (1, T, H3D_FEATURE_DIM)
