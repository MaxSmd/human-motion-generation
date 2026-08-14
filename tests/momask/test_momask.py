"""MoMask smoke tests over the shared HumanML3D packed dataset path."""

from __future__ import annotations

import io
import json
import random
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from momask.models import (
    CodebookResidualTransformer,
    MaskedMotionTransformer,
    MotionRVQVAE,
    ResidualVectorQuantizer,
    ResidualTransformer,
    TokenTransformerConfig,
)
from momask.data_utils import normalize_motion, token_mask_from_frame_mask
from momask.scripts.assemble_token_checkpoints import assemble_token_checkpoints
from momask.scripts.train_momask import (
    FixedWindowTensorBatcher,
    H3DNormalizer,
    best_validation_metadata_matches,
    configure_token_training_stage,
    cycle_loader,
    evaluate_tokens,
    expected_token_validation_metric_name,
    resolve_token_stage,
    stack_token_cache,
    token_validation_metric,
)
from momask.tasks import generate_h3d263
from shared.data import (
    CanonicalHumanML3DText2MotionDataset,
    CanonicalHumanML3DWindowDataset,
    H3D263Dataset,
    collate,
)
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


def test_token_training_stages_select_only_the_requested_model() -> None:
    masked = torch.nn.Linear(3, 4)
    residual = torch.nn.Linear(4, 5)

    params = configure_token_training_stage(masked, residual, "masked")
    assert masked.training
    assert not residual.training
    assert all(parameter.requires_grad for parameter in masked.parameters())
    assert all(not parameter.requires_grad for parameter in residual.parameters())
    assert {id(parameter) for parameter in params} == {
        id(parameter) for parameter in masked.parameters()
    }

    params = configure_token_training_stage(masked, residual, "residual")
    assert not masked.training
    assert residual.training
    assert all(not parameter.requires_grad for parameter in masked.parameters())
    assert all(parameter.requires_grad for parameter in residual.parameters())
    assert {id(parameter) for parameter in params} == {
        id(parameter) for parameter in residual.parameters()
    }


def test_token_stage_optimizer_updates_only_the_selected_model() -> None:
    torch.manual_seed(11)
    masked = torch.nn.Linear(3, 3)
    residual = torch.nn.Linear(3, 3)
    residual_before = {name: value.detach().clone() for name, value in residual.state_dict().items()}

    params = configure_token_training_stage(masked, residual, "masked")
    optimizer = torch.optim.AdamW(params, lr=1e-2)
    loss = masked(torch.ones(2, 3)).sum()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert all(torch.equal(residual.state_dict()[name], value) for name, value in residual_before.items())


def test_token_validation_metric_is_stage_specific() -> None:
    metrics = {"base_ce": 0.5, "residual_ce": 0.25, "residual_sampled_ce": 0.2}
    assert token_validation_metric("masked", metrics) == ("base_ce", 0.5)
    assert token_validation_metric("residual", metrics) == ("residual_sampled_ce", 0.2)
    assert token_validation_metric("joint", metrics) == ("combined_ce", 0.7)
    assert expected_token_validation_metric_name("masked", "paper", "codebook") == "base_ce"
    assert (
        expected_token_validation_metric_name("residual", "paper", "codebook")
        == "residual_sampled_ce"
    )


def test_old_or_missing_best_validation_metadata_is_rejected() -> None:
    current = {
        "best_val_metric_name": "base_ce",
        "token_validation_version": 1,
    }
    assert best_validation_metadata_matches(current, "base_ce", best_path_exists=True)
    assert not best_validation_metadata_matches(current, "base_ce", best_path_exists=False)
    assert not best_validation_metadata_matches(current, "residual_sampled_ce", True)
    assert not best_validation_metadata_matches(
        {"best_val_metric_name": "base_ce"},
        "base_ce",
        True,
    )


def test_paper_residual_validation_reports_sampled_training_objective() -> None:
    torch.manual_seed(9)
    cfg = TokenTransformerConfig(
        vocab_size=8,
        text_dim=4,
        code_dim=6,
        hidden_dim=12,
        depth=1,
        num_heads=3,
        ffn_dim=24,
        max_seq_len=5,
        dropout=0.0,
        architecture="paper",
    )
    masked = MaskedMotionTransformer(cfg)
    residual = CodebookResidualTransformer(
        cfg,
        num_quantizers=3,
        code_dim=6,
        share_weight=True,
    )
    cached = [
        {
            "tokens": torch.randint(0, cfg.vocab_size, (2, 3, 5)),
            "token_mask": torch.ones(2, 5, dtype=torch.bool),
            "cond": torch.randn(2, cfg.text_dim),
            "texts": ["first", "second"],
        }
    ]

    metrics = evaluate_tokens(
        masked,
        residual,
        cached,
        torch.device("cpu"),
        max_batches=1,
        generation_steps=1,
        token_stage="residual",
    )

    assert torch.isfinite(torch.tensor(metrics["residual_sampled_ce"]))
    assert torch.isfinite(torch.tensor(metrics["residual_ce"]))


def test_legacy_freeze_masked_flag_resolves_to_residual_stage() -> None:
    args = types.SimpleNamespace(token_stage="joint", freeze_masked_transformer=True)
    assert resolve_token_stage(args) == "residual"
    assert args.token_stage == "residual"


def test_independent_token_checkpoints_assemble_trained_components() -> None:
    common = {
        "normalizer": {"mean": torch.tensor([1.0]), "std": torch.tensor([2.0])},
        "vqvae": {"weight": torch.tensor([3.0])},
    }
    masked = {
        **common,
        "step": 10,
        "args": {
            "token_stage": "masked",
            "max_seq_len": 196,
            "transformer_arch": "paper",
            "residual_arch": "codebook",
        },
        "checkpoint_role": "best_validation",
        "token_validation_version": 1,
        "best_val_metric_name": "base_ce",
        "best_val_metric": 0.2,
        "val_eval": {"base_ce": 0.2},
        "masked_transformer": {"weight": torch.tensor([4.0])},
        "residual_transformer": {"weight": torch.tensor([-1.0])},
        "token_optimizer": {"state": "masked"},
        "token_eval": {"base_ce": 0.1},
        "sample_generated": torch.tensor([0.0]),
    }
    residual = {
        **common,
        "step": 12,
        "args": {
            "token_stage": "residual",
            "max_seq_len": 196,
            "transformer_arch": "paper",
            "residual_arch": "codebook",
        },
        "checkpoint_role": "best_validation",
        "token_validation_version": 1,
        "best_val_metric_name": "residual_sampled_ce",
        "best_val_metric": 0.1,
        "val_eval": {"residual_sampled_ce": 0.1},
        "masked_transformer": {"weight": torch.tensor([-2.0])},
        "residual_transformer": {"weight": torch.tensor([5.0])},
        "token_optimizer": {"state": "residual"},
    }

    assembled = assemble_token_checkpoints(masked, residual)
    assert torch.equal(assembled["masked_transformer"]["weight"], torch.tensor([4.0]))
    assert torch.equal(assembled["residual_transformer"]["weight"], torch.tensor([5.0]))
    assert assembled["args"]["token_stage"] == "assembled"
    assert assembled["component_steps"] == {"masked": 10, "residual": 12}
    assert assembled["component_validation"]["masked"] == {
        "name": "base_ce",
        "value": 0.2,
        "step": 10,
    }
    assert assembled["component_validation"]["residual"] == {
        "name": "residual_sampled_ce",
        "value": 0.1,
        "step": 12,
    }
    assert assembled["checkpoint_role"] == "assembled"
    assert "token_optimizer" not in assembled
    assert "token_eval" not in assembled
    assert "best_val_metric" not in assembled
    assert "val_eval" not in assembled
    assert "sample_generated" not in assembled


def test_assembly_rejects_unvalidated_component_checkpoints() -> None:
    common = {
        "normalizer": {"mean": torch.tensor([1.0])},
        "vqvae": {"weight": torch.tensor([2.0])},
        "masked_transformer": {"weight": torch.tensor([3.0])},
        "residual_transformer": {"weight": torch.tensor([4.0])},
    }
    masked = {**common, "args": {"token_stage": "masked"}}
    residual = {**common, "args": {"token_stage": "residual"}}

    with pytest.raises(ValueError, match="best-validation"):
        assemble_token_checkpoints(masked, residual)


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


def test_paper_text_motion_dataset_preserves_caption_spans_and_196_padding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    canonical = tmp_path / "new_joint_vecs"
    canonical.mkdir()
    (tmp_path / "splits.json").write_text(
        json.dumps({"train": ["000001", "000002"], "val": [], "test": []})
    )
    motion = np.arange(100 * H3D_FEATURE_DIM, dtype=np.float32).reshape(100, H3D_FEATURE_DIM)
    np.save(canonical / "000001.npy", motion)
    np.save(canonical / "000002.npy", np.zeros((200, H3D_FEATURE_DIM), dtype=np.float32))
    with zipfile.ZipFile(tmp_path / "texts.zip", "w") as zf:
        zf.writestr(
            "texts/000001.txt",
            "whole caption#whole/OTHER#0#0\n"
            "another whole caption#another/OTHER#0#0\n"
            "segment caption#segment/OTHER#1.0#3.5\n",
        )
        zf.writestr("texts/000002.txt", "excluded#excluded/OTHER#0#0\n")

    ds = CanonicalHumanML3DText2MotionDataset(
        tmp_path,
        canonical,
        tmp_path / "texts.zip",
        max_seq_len=196,
        min_seq_len=40,
        unit_length=4,
        subset_n=999999,
    )
    assert ds.num_clips == 1
    assert len(ds) == 2

    monkeypatch.setattr(random, "random", lambda: 1.0)
    monkeypatch.setattr(random, "randint", lambda _low, _high: 0)
    monkeypatch.setattr(random, "choice", lambda values: values[0])
    def unexpected_load(*_args, **_kwargs):
        raise AssertionError("preloaded text-motion dataset reopened a NumPy file")

    monkeypatch.setattr(np, "load", unexpected_load)

    full_idx = next(i for i, entry in enumerate(ds.entries) if entry.clip_id == "000001")
    segment_idx = next(i for i, entry in enumerate(ds.entries) if ":segment" in entry.clip_id)
    full = ds[full_idx]
    segment = ds[segment_idx]
    assert full.length == 100
    assert full.x1.shape == (196, H3D_FEATURE_DIM)
    assert full.text == "whole caption"
    assert torch.equal(full.x1[:100], torch.from_numpy(motion))
    assert not full.x1[100:].any()
    assert segment.length == 48
    assert segment.text == "segment caption"
    assert torch.equal(segment.x1[:48], torch.from_numpy(motion[20:68]))
    assert not segment.x1[48:].any()


def test_paper_transformers_use_official_shapes_and_sample_one_residual_level() -> None:
    torch.manual_seed(7)
    cfg = TokenTransformerConfig(
        vocab_size=8,
        text_dim=6,
        code_dim=12,
        hidden_dim=8,
        depth=1,
        num_heads=2,
        ffn_dim=16,
        max_seq_len=6,
        dropout=0.0,
        architecture="paper",
    )
    masked = MaskedMotionTransformer(cfg)
    residual = CodebookResidualTransformer(
        cfg,
        num_quantizers=4,
        code_dim=cfg.code_dim,
        share_weight=True,
    )
    assert masked.token_embed.embedding_dim == cfg.code_dim
    assert "pos_embed" not in dict(masked.named_parameters())
    assert isinstance(masked.norm, torch.nn.Identity)

    tokens = torch.randint(0, cfg.vocab_size, (3, 4, 6))
    valid = torch.tensor(
        [
            [True, True, True, True, True, True],
            [True, True, True, True, False, False],
            [True, True, True, False, False, False],
        ]
    )
    cond = torch.randn(3, cfg.text_dim)
    base_loss = masked.training_loss(tokens[:, 0], cond=cond, valid_mask=valid)
    residual_loss, levels = residual.sampled_training_loss(
        tokens,
        cond=cond,
        valid_mask=valid,
    )
    assert torch.isfinite(base_loss)
    assert torch.isfinite(residual_loss)
    assert levels.shape == (3,)
    assert ((levels >= 1) & (levels < 4)).all()


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


def test_rvq_has_one_straight_through_gradient_path_per_level() -> None:
    torch.manual_seed(0)
    quantizer = ResidualVectorQuantizer(
        num_quantizers=3,
        codebook_size=8,
        dim=4,
        quantize_dropout_prob=0.0,
    ).train()
    z = torch.randn(2, 5, 4, requires_grad=True)
    out = quantizer(z)
    out.quantized.sum().backward()

    assert torch.allclose(z.grad, torch.full_like(z, 3.0))
    assert out.perplexity_per_level.shape == (3,)
    assert out.active_codes_per_level.shape == (3,)
    assert (out.active_codes_per_level > 0).all()


def test_rvq_gumbel_sampling_uses_multiple_equal_distance_codes() -> None:
    torch.manual_seed(0)
    quantizer = ResidualVectorQuantizer(
        num_quantizers=1,
        codebook_size=8,
        dim=4,
        quantize_dropout_prob=0.0,
        sample_codebook_temp=0.5,
    ).train()
    quantizer.codebooks[0].weight.data.zero_()
    out = quantizer(torch.zeros(16, 8, 4))

    assert out.active_codes_per_level[0] > 1
    assert out.perplexity_per_level[0] > 1.0


def test_rvq_dropout_does_not_initialize_inactive_ema_levels(monkeypatch) -> None:
    quantizer = ResidualVectorQuantizer(
        num_quantizers=3,
        codebook_size=4,
        dim=3,
        quantize_dropout_prob=1.0,
        use_ema=True,
    ).train()
    monkeypatch.setattr(
        torch,
        "randint",
        lambda _low, _high, _size: torch.tensor(1),
    )
    out = quantizer(torch.randn(2, 4, 3))

    assert quantizer.ema_initialized.tolist() == [True, False, False]
    assert out.perplexity_per_level[1:].tolist() == [0.0, 0.0]
    assert out.active_codes_per_level[1:].tolist() == [0, 0]


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
