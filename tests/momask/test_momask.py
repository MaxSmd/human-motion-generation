"""MoMask smoke tests over the shared HumanML3D packed dataset path."""

from __future__ import annotations

import io
import json
import types
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from momask.models import (
    MaskedMotionTransformer,
    MotionRVQVAE,
    ResidualTransformer,
    TokenTransformerConfig,
)
from momask.data_utils import normalize_motion, token_mask_from_frame_mask
from momask.scripts.train_momask_smoke import (
    FixedWindowTensorBatcher,
    H3DNormalizer,
    cycle_loader,
    stack_token_cache,
)
from momask.tasks import generate_h3d263
from shared.data import CanonicalHumanML3DWindowDataset, H3D263Dataset, collate
from shared.geometry import H3D_FEATURE_DIM, NUM_JOINTS


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
    ds = H3D263Dataset(
        tmp_path,
        split="train",
        min_seq_len=10,
        max_seq_len=40,
        mirror_augment=False,
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
    assert torch.isfinite(out.velocity_loss)

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
    full_base_loss = masked.training_loss(out.tokens[:, 0], cond=cond, valid_mask=mask, force_full_mask=True)
    res_loss = residual.training_loss(out.tokens, target_level=1, cond=cond, valid_mask=mask)
    assert torch.isfinite(base_loss)
    assert torch.isfinite(full_base_loss)
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


def test_cycle_loader_restarts_iterable_instead_of_replaying_first_epoch() -> None:
    class FreshEpochs:
        def __init__(self) -> None:
            self.epoch = 0

        def __iter__(self):
            self.epoch += 1
            yield self.epoch

    batches = cycle_loader(FreshEpochs())
    assert next(batches) == 1
    assert next(batches) == 2


def test_canonical_window_dataset_preloads_and_indexes_all_windows(tmp_path: Path, monkeypatch) -> None:
    canonical = tmp_path / "new_joint_vecs"
    canonical.mkdir()
    (tmp_path / "splits.json").write_text(
        json.dumps({"train": ["000001", "000002"], "val": [], "test": []})
    )
    first = np.arange(8 * H3D_FEATURE_DIM, dtype=np.float32).reshape(8, H3D_FEATURE_DIM)
    second = np.arange(6 * H3D_FEATURE_DIM, dtype=np.float32).reshape(6, H3D_FEATURE_DIM)
    np.save(canonical / "000001.npy", first)
    np.save(canonical / "000002.npy", second)

    ds = CanonicalHumanML3DWindowDataset(
        tmp_path,
        canonical,
        window_size=4,
        window_stride=2,
        preload=True,
        subset_n=999999,
    )
    assert ds.num_clips == 2
    assert len(ds) == 5

    def unexpected_load(*_args, **_kwargs):
        raise AssertionError("preloaded dataset reopened a NumPy file")

    monkeypatch.setattr(np, "load", unexpected_load)
    sample = ds[2]
    assert sample.clip_id == "000001:4"
    assert torch.equal(sample.x1, torch.from_numpy(first[4:8]))


def test_fixed_window_tensor_batcher_preserves_window_contents(tmp_path: Path) -> None:
    canonical = tmp_path / "new_joint_vecs"
    canonical.mkdir()
    (tmp_path / "splits.json").write_text(
        json.dumps({"train": ["000001"], "val": [], "test": []})
    )
    motion = np.arange(7 * H3D_FEATURE_DIM, dtype=np.float32).reshape(7, H3D_FEATURE_DIM)
    np.save(canonical / "000001.npy", motion)
    ds = CanonicalHumanML3DWindowDataset(
        tmp_path,
        canonical,
        window_size=4,
        window_stride=1,
        preload=True,
        subset_n=999999,
    )
    batcher = FixedWindowTensorBatcher(
        ds,
        batch_size=4,
        device=torch.device("cpu"),
        normalizer=H3DNormalizer.identity(),
        seed=0,
    )
    x, mask = batcher.next()
    expected = {tuple(motion[start : start + 4, 0]) for start in range(4)}
    actual = {tuple(window[:, 0].tolist()) for window in x}
    assert actual == expected
    assert mask.all()


def test_all_valid_mask_matches_unmasked_vq_path() -> None:
    torch.manual_seed(0)
    model = MotionRVQVAE(
        input_dim=H3D_FEATURE_DIM,
        hidden_dim=16,
        latent_dim=8,
        num_quantizers=2,
        codebook_size=8,
        downsample=4,
        num_res_blocks=1,
        quantize_dropout_prob=0.0,
    ).eval()
    x = torch.randn(2, 64, H3D_FEATURE_DIM)
    with torch.no_grad():
        unmasked = model(x)
        masked = model(x, mask=torch.ones(2, 64, dtype=torch.bool))
    assert torch.equal(unmasked.tokens, masked.tokens)
    assert torch.allclose(unmasked.recon, masked.recon)
    assert torch.allclose(unmasked.loss, masked.loss)


def test_token_mask_uses_floor_convolution_lengths() -> None:
    mask = torch.zeros(3, 12, dtype=torch.bool)
    mask[0, :12] = True
    mask[1, :11] = True
    mask[2, :7] = True
    token_mask = token_mask_from_frame_mask(mask, token_len=3, downsample=4)
    assert token_mask.sum(dim=1).tolist() == [3, 2, 1]


def test_normalization_keeps_padding_zero() -> None:
    class ShiftNormalizer:
        def transform(self, x: torch.Tensor) -> torch.Tensor:
            return x + 5.0

    x = torch.zeros(1, 3, 2)
    mask = torch.tensor([[True, True, False]])
    normalized = normalize_motion(x, mask, ShiftNormalizer())
    assert torch.equal(normalized[0, :2], torch.full((2, 2), 5.0))
    assert torch.equal(normalized[0, 2], torch.zeros(2))


def test_stack_token_cache_pads_variable_batch_lengths() -> None:
    cached = [
        {
            "tokens": torch.ones(1, 2, 3, dtype=torch.long),
            "token_mask": torch.ones(1, 3, dtype=torch.bool),
            "cond": torch.ones(1, 4),
            "texts": ["a"],
        },
        {
            "tokens": torch.full((1, 2, 5), 2, dtype=torch.long),
            "token_mask": torch.ones(1, 5, dtype=torch.bool),
            "cond": torch.ones(1, 4),
            "texts": ["b"],
        },
    ]
    stacked = stack_token_cache(cached, torch.device("cpu"), max_token_len=5)
    assert stacked["tokens"].shape == (2, 2, 5)
    assert not stacked["token_mask"][0, 3:].any()
    assert (stacked["tokens"][0, :, 3:] == 0).all()


def test_force_full_mask_replaces_every_valid_input_token() -> None:
    cfg = TokenTransformerConfig(
        vocab_size=8,
        text_dim=4,
        hidden_dim=8,
        depth=1,
        num_heads=2,
        ffn_dim=16,
        max_seq_len=4,
        dropout=0.0,
    )
    model = MaskedMotionTransformer(cfg)
    seen: dict[str, torch.Tensor] = {}

    def fake_forward(self, tokens, **_kwargs):
        seen["tokens"] = tokens.detach().clone()
        return torch.zeros(*tokens.shape, cfg.vocab_size, requires_grad=True)

    model.forward = types.MethodType(fake_forward, model)
    tokens = torch.tensor([[1, 2, 3, 4]])
    valid = torch.tensor([[True, True, True, False]])
    loss = model.training_loss(tokens, valid_mask=valid, force_full_mask=True)
    assert torch.isfinite(loss)
    assert (seen["tokens"][valid] == model.mask_token_id).all()
    assert seen["tokens"][0, 3] == tokens[0, 3]


def test_decode_from_tokens_zeros_invalid_latents() -> None:
    vqvae = MotionRVQVAE(
        input_dim=H3D_FEATURE_DIM,
        hidden_dim=8,
        latent_dim=4,
        num_quantizers=2,
        codebook_size=3,
        num_res_blocks=1,
    )
    for codebook in vqvae.quantizer.codebooks:
        codebook.weight.data.fill_(1.0)

    def identity_decode(self, latents, target_len=None):
        return latents

    vqvae.decode_latents = types.MethodType(identity_decode, vqvae)
    tokens = torch.zeros(1, 2, 4, dtype=torch.long)
    mask = torch.tensor([[True, True, False, False]])
    latents = vqvae.decode_from_tokens(tokens, token_mask=mask)
    assert (latents[0, :2] == 2.0).all()
    assert (latents[0, 2:] == 0.0).all()
