"""Render MoMask trajectory constraints as a side-by-side GIF.

The GIF compares:
  real motion | unconstrained generation | trajectory-projected generation

The target trajectory is drawn as a dashed black line with ordered anchor
markers. This script generates a fresh sample from a checkpoint, so run it
through Slurm rather than on the head node.
"""

from __future__ import annotations

import argparse
import math
import random
import textwrap
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation, PillowWriter
from torch import Tensor

from momask.models import (
    CodebookResidualTransformer,
    MaskedMotionTransformer,
    MotionRVQVAE,
    ResidualTransformer,
    TokenTransformerConfig,
)
from shared.data import H3D263Dataset, collate
from shared.text import CLIPTextEncoder, RandomTextEncoder, TextEncoder
from shared.geometry import H3D_FEATURE_DIM, PARENTS, quat_rotate, recover_joints_from_ric

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)


class H3DNormalizer:
    def __init__(self, mean: Tensor, std: Tensor) -> None:
        self.mean = mean.float()
        self.std = std.float().clamp_min(1e-6)

    @classmethod
    def from_state_dict(cls, state: dict[str, Tensor]) -> "H3DNormalizer":
        return cls(state["mean"], state["std"])

    def transform(self, x: Tensor) -> Tensor:
        return (x - self.mean.to(x.device, x.dtype)) / self.std.to(x.device, x.dtype)

    def inverse(self, x: Tensor) -> Tensor:
        return x * self.std.to(x.device, x.dtype) + self.mean.to(x.device, x.dtype)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--min-seq-len", type=int, default=40)
    p.add_argument("--generation-steps", type=int, default=None)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--topk-filter-thres", type=float, default=1.0)
    p.add_argument("--sample-tokens", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--remask-kept-tokens", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--anchor-stride", type=int, default=20)
    p.add_argument("--constraint-variant", choices=["projected", "vq", "both"], default="both")
    p.add_argument("--real-h3d-dir", default=None)
    p.add_argument("--model-input-source", choices=["auto", "packed", "canonical"], default="auto")
    p.add_argument("--auto-sample-moving", type=int, default=0)
    p.add_argument("--min-root-span", type=float, default=0.75)
    p.add_argument("--max-frames", type=int, default=100)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--text-encoder", choices=["checkpoint", "random", "clip"], default="checkpoint")
    p.add_argument("--clip-model", default=None)
    p.add_argument("--clip-cache-dir", default=None)
    p.add_argument("--clip-backend", choices=["auto", "openai", "transformers"], default="auto")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def ckpt_args(ckpt: dict) -> dict:
    args = ckpt.get("args", {})
    if not isinstance(args, dict):
        raise ValueError("checkpoint does not contain an args dict")
    return args


def build_text_encoder(args: argparse.Namespace, saved_args: dict) -> TextEncoder:
    kind = saved_args.get("text_encoder", "random") if args.text_encoder == "checkpoint" else args.text_encoder
    if kind == "random":
        return RandomTextEncoder(text_dim=int(saved_args.get("text_dim", 64)))
    if kind == "clip":
        return CLIPTextEncoder(
            model_name=args.clip_model or str(saved_args.get("clip_model", "ViT-B/32")),
            cache_dir=args.clip_cache_dir,
            backend=args.clip_backend,
        )
    raise ValueError(f"unknown text encoder: {kind}")


def build_models(ckpt: dict, device: torch.device):
    a = ckpt_args(ckpt)
    vqvae = MotionRVQVAE(
        input_dim=H3D_FEATURE_DIM,
        hidden_dim=int(a.get("vq_hidden_dim", 64)),
        latent_dim=int(a.get("vq_latent_dim", 32)),
        num_quantizers=int(a.get("num_quantizers", 3)),
        codebook_size=int(a.get("codebook_size", 64)),
        downsample=int(a.get("downsample", 1)),
        num_res_blocks=int(a.get("vq_res_blocks", 1)),
        commitment_weight=float(a.get("vq_commitment_weight", 0.25)),
        quantize_dropout_prob=float(a.get("quantize_dropout", 0.2)),
        velocity_loss_weight=float(a.get("vq_velocity_weight", 0.0)),
        explicit_loss_weight=float(a.get("vq_explicit_weight", 0.0)),
        use_ema_quantizer=bool(a.get("vq_use_ema", False)),
        ema_decay=float(a.get("vq_ema_decay", 0.99)),
        codebook_sample_temp=float(a.get("vq_codebook_sample_temp", 0.0)),
    ).to(device)
    cfg = TokenTransformerConfig(
        vocab_size=int(a.get("codebook_size", 64)),
        text_dim=int(a.get("text_dim", 64)),
        hidden_dim=int(a.get("transformer_hidden_dim", 64)),
        depth=int(a.get("transformer_depth", 2)),
        num_heads=int(a.get("transformer_heads", 4)),
        ffn_dim=int(a.get("transformer_ffn_dim", 128)),
        max_seq_len=math.ceil(int(a.get("max_seq_len", 80)) / int(a.get("downsample", 1))),
        dropout=float(a.get("transformer_dropout", 0.0)),
    )
    masked = MaskedMotionTransformer(cfg).to(device)
    if a.get("residual_arch", "simple") == "codebook":
        residual = CodebookResidualTransformer(
            cfg,
            num_quantizers=int(a.get("num_quantizers", 3)),
            code_dim=int(a.get("vq_latent_dim", 32)),
            share_weight=bool(a.get("residual_share_weight", False)),
        ).to(device)
    else:
        residual = ResidualTransformer(
            cfg,
            num_quantizers=int(a.get("num_quantizers", 3)),
            separate_level_heads=not bool(a.get("shared_residual_head", False)),
        ).to(device)
    vqvae.load_state_dict(ckpt["vqvae"])
    masked.load_state_dict(ckpt["masked_transformer"])
    residual.load_state_dict(ckpt["residual_transformer"])
    vqvae.eval()
    masked.eval()
    residual.eval()
    return vqvae, masked, residual


def load_canonical_sample(h3d_dir: str | Path, clip_id: str, max_len: int) -> Tensor:
    root = Path(h3d_dir)
    candidates = [clip_id]
    if clip_id.startswith("M") and len(clip_id) > 1:
        candidates.append(clip_id[1:])
    path = next((root / f"{name}.npy" for name in candidates if (root / f"{name}.npy").exists()), None)
    if path is None:
        raise FileNotFoundError(f"canonical HumanML3D feature not found for {clip_id} under {root}")
    arr = torch.from_numpy(np.load(path).astype("float32"))
    if arr.ndim != 2 or arr.shape[1] != H3D_FEATURE_DIM:
        raise ValueError(f"{path} must have shape (T, {H3D_FEATURE_DIM}), got {tuple(arr.shape)}")
    return arr[:max_len]


def token_mask_from_frame_mask(mask: Tensor, token_len: int) -> Tensor:
    if mask.shape[1] == token_len:
        return mask
    pooled = F.adaptive_max_pool1d(mask.float().unsqueeze(1), token_len).squeeze(1)
    return pooled > 0.5


def root_xz(motion: Tensor) -> Tensor:
    return recover_joints_from_ric(motion.float())[:, :, 0, :][:, :, [0, 2]]


def root_xz_from_joints(joints: Tensor) -> Tensor:
    return joints[:, 0, :][:, [0, 2]]


def trajectory_span(root: Tensor) -> float:
    extent = root.amax(dim=0) - root.amin(dim=0)
    return float(extent.norm())


def root_quat_from_features(motion: Tensor) -> Tensor:
    rot_vel = motion[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)
    quat = torch.zeros(*motion.shape[:-1], 4, device=motion.device, dtype=motion.dtype)
    quat[..., 0] = torch.cos(r_rot_ang)
    quat[..., 2] = torch.sin(r_rot_ang)
    return quat


def interpolate_anchor_trajectory(real_root: Tensor, length: int, anchor_stride: int) -> tuple[Tensor, Tensor]:
    target = real_root.clone()
    anchors = list(range(0, length, max(1, anchor_stride)))
    if anchors[-1] != length - 1:
        anchors.append(length - 1)
    anchor_mask = torch.zeros(real_root.shape[0], dtype=torch.bool, device=real_root.device)
    anchor_mask[anchors] = True
    for left, right in zip(anchors[:-1], anchors[1:]):
        span = max(1, right - left)
        alpha = torch.linspace(0, 1, span + 1, device=real_root.device, dtype=real_root.dtype).unsqueeze(-1)
        target[left : right + 1] = (1 - alpha) * real_root[left] + alpha * real_root[right]
    return target, anchor_mask


def project_root_trajectory(motion: Tensor, target_root: Tensor, length: int) -> Tensor:
    out = motion.clone()
    quat = root_quat_from_features(out.unsqueeze(0))[0]
    target3 = torch.zeros(target_root.shape[0], 3, device=out.device, dtype=out.dtype)
    target3[:, 0] = target_root[:, 0]
    target3[:, 2] = target_root[:, 1]
    # HumanML3D stores root XZ velocity rotated by the destination frame's
    # root orientation, matching the dataset feature conversion.
    local_delta = quat_rotate(quat[1:], target3[1:] - target3[:-1])
    out[: length - 1, 1] = local_delta[: length - 1, 0]
    out[: length - 1, 2] = local_delta[: length - 1, 2]
    return out


@torch.no_grad()
def generate_full(
    vqvae: MotionRVQVAE,
    masked: MaskedMotionTransformer,
    residual: ResidualTransformer,
    normalizer: H3DNormalizer,
    cond: Tensor,
    real_x: Tensor,
    frame_mask: Tensor,
    steps: int,
    guidance_scale: float,
    temperature: float,
    topk_filter_thres: float,
    sample_tokens: bool,
    remask_kept_tokens: bool,
) -> Tensor:
    x_norm = normalizer.transform(real_x)
    true_tokens = vqvae.encode_to_tokens(x_norm)
    token_mask = token_mask_from_frame_mask(frame_mask, true_tokens.shape[-1])
    base = masked.generate(
        cond=cond,
        seq_len=true_tokens.shape[-1],
        steps=steps,
        guidance_scale=guidance_scale,
        temperature=temperature,
        topk_filter_thres=topk_filter_thres,
        sample=sample_tokens,
        remask_kept_tokens=remask_kept_tokens,
        mask=token_mask,
    )
    tokens = residual.generate_residuals(
        base,
        cond=cond,
        guidance_scale=guidance_scale,
        temperature=temperature,
        topk_filter_thres=topk_filter_thres,
        sample=sample_tokens,
        mask=token_mask,
    )
    return normalizer.inverse(vqvae.decode_from_tokens(tokens, target_len=real_x.shape[1]))


@torch.no_grad()
def vq_project(vqvae: MotionRVQVAE, normalizer: H3DNormalizer, motion: Tensor, frame_mask: Tensor) -> Tensor:
    return normalizer.inverse(vqvae(normalizer.transform(motion), mask=frame_mask).recon)


def axis_limits(joints_list: list[Tensor], target: Tensor) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    all_joints = torch.stack(joints_list, dim=0)
    root_target = torch.zeros(target.shape[0], 3)
    root_target[:, 0] = target[:, 0].cpu()
    root_target[:, 2] = target[:, 1].cpu()
    points = torch.cat([all_joints.reshape(-1, 3), root_target], dim=0)
    mins = points.amin(dim=0)
    maxs = points.amax(dim=0)
    center = (mins + maxs) / 2
    radius = float((maxs - mins).max().clamp_min(1.0) / 2) * 1.15
    return tuple((float(c - radius), float(c + radius)) for c in center)  # type: ignore[return-value]


def root_error(joints_or_frame: Tensor, target: Tensor, frame: int) -> float:
    if joints_or_frame.ndim == 3:
        root = joints_or_frame[frame, 0, [0, 2]]
    else:
        root = joints_or_frame[0, [0, 2]]
    return float((root - target[frame].cpu()).norm())


def anchor_style(anchor_mask: Tensor) -> tuple[np.ndarray, np.ndarray]:
    anchor_frames = torch.nonzero(anchor_mask, as_tuple=False).flatten().cpu().numpy()
    if len(anchor_frames) == 0:
        return anchor_frames, np.zeros((0, 4), dtype=np.float32)
    colors = plt.cm.viridis(np.linspace(0.12, 0.92, len(anchor_frames)))
    return anchor_frames, colors


def anchor_label(order: int, frame_idx: int, num_anchors: int) -> str:
    if order == 0:
        return f"start\nf{frame_idx}"
    if order == num_anchors - 1:
        return f"end\nf{frame_idx}"
    return f"f{frame_idx}"


def draw_pose_frame(ax, joints_seq: Tensor, target: Tensor, anchor_mask: Tensor, title: str, limits, frame: int) -> None:
    ax.cla()
    joints = joints_seq[frame]
    root_path = root_xz_from_joints(joints_seq)
    for j, p in enumerate(PARENTS):
        if p < 0:
            continue
        xs = [joints[p, 0], joints[j, 0]]
        ys = [joints[p, 2], joints[j, 2]]
        zs = [joints[p, 1], joints[j, 1]]
        ax.plot(xs, ys, zs, color="#2563eb", linewidth=2.3)
    ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1], color="#dc2626", s=16, depthshade=True)
    ax.scatter(joints[0, 0], joints[0, 2], joints[0, 1], color="#f97316", s=58, label="current root")
    ax.plot(
        root_path[: frame + 1, 0],
        root_path[: frame + 1, 1],
        torch.zeros(frame + 1),
        color="#f97316",
        linewidth=2.6,
        alpha=0.95,
        label="root trail",
    )
    ax.plot(
        target[:, 0].cpu(),
        target[:, 1].cpu(),
        torch.zeros(target.shape[0]),
        color="black",
        linestyle="--",
        linewidth=2.2,
        label="target root path",
    )
    anchors = target[anchor_mask]
    anchor_frames, anchor_colors = anchor_style(anchor_mask)
    ax.scatter(
        anchors[:, 0].cpu(),
        anchors[:, 1].cpu(),
        torch.zeros(anchors.shape[0]),
        color=anchor_colors,
        edgecolor="white",
        linewidth=0.8,
        s=74,
        label="ordered anchors",
    )
    for order, (frame_idx, anchor) in enumerate(zip(anchor_frames, anchors)):
        if order in {0, len(anchor_frames) - 1} or len(anchor_frames) <= 6:
            ax.text(
                float(anchor[0]),
                float(anchor[1]),
                0.05,
                anchor_label(order, int(frame_idx), len(anchor_frames)).replace("\n", " "),
                color="#111827",
                fontsize=8,
                weight="bold",
            )
    err = root_error(joints_seq, target, min(frame, target.shape[0] - 1))
    ax.set_title(f"{title} | frame {frame}\nroot-target error: {err:.3f} m", pad=14)
    ax.set_xlabel("X position (m)", labelpad=8)
    ax.set_ylabel("Z position (m)", labelpad=8)
    ax.set_zlabel("Height Y (m)", labelpad=8)
    body_min = joints.amin(dim=0)
    body_max = joints.amax(dim=0)
    root = joints[0]
    body_radius = float((body_max - body_min).max().clamp_min(0.9)) * 0.9
    ax.set_xlim(float(root[0] - body_radius), float(root[0] + body_radius))
    ax.set_ylim(float(root[2] - body_radius), float(root[2] + body_radius))
    ax.set_zlim(float(body_min[1] - 0.25), float(body_max[1] + 0.25))
    ax.grid(True, linewidth=0.8, alpha=0.55)
    ax.xaxis.pane.set_alpha(0.08)
    ax.yaxis.pane.set_alpha(0.08)
    ax.zaxis.pane.set_alpha(0.04)
    ax.view_init(elev=18, azim=-75)
    try:
        ax.set_box_aspect((1.0, 1.0, 0.9))
    except AttributeError:
        pass


def draw_path_frame(ax, joints_seq: Tensor, target: Tensor, anchor_mask: Tensor, title: str, limits, frame: int) -> None:
    ax.cla()
    root_path = root_xz_from_joints(joints_seq)
    anchors = target[anchor_mask]
    anchor_frames, anchor_colors = anchor_style(anchor_mask)
    ax.plot(
        target[:, 0].cpu(),
        target[:, 1].cpu(),
        color="black",
        linestyle="--",
        linewidth=2.4,
        label="constraint target",
    )
    ax.scatter(
        anchors[:, 0].cpu(),
        anchors[:, 1].cpu(),
        color=anchor_colors,
        edgecolor="white",
        linewidth=1.0,
        s=88,
        zorder=4,
        label="ordered anchors",
    )
    if anchors.shape[0] > 1:
        anchor_np = anchors.cpu().numpy()
        for left, right in zip(anchor_np[:-1], anchor_np[1:]):
            dx = float(right[0] - left[0])
            dy = float(right[1] - left[1])
            ax.annotate(
                "",
                xy=(float(left[0] + dx * 0.72), float(left[1] + dy * 0.72)),
                xytext=(float(left[0] + dx * 0.28), float(left[1] + dy * 0.28)),
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#16a34a",
                    "lw": 1.8,
                    "alpha": 0.85,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
                zorder=3,
            )
        for order, (frame_idx, anchor) in enumerate(zip(anchor_frames, anchor_np)):
            offset_y = 8 if order % 2 == 0 else -15
            ax.annotate(
                anchor_label(order, int(frame_idx), len(anchor_frames)),
                xy=(float(anchor[0]), float(anchor[1])),
                xytext=(8, offset_y),
                textcoords="offset points",
                fontsize=8,
                weight="bold",
                color="#111827",
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#16a34a", "alpha": 0.9},
                zorder=6,
            )
    ax.plot(root_path[:, 0], root_path[:, 1], color="#9ca3af", linewidth=1.3, alpha=0.45, label="full root path")
    ax.plot(root_path[: frame + 1, 0], root_path[: frame + 1, 1], color="#2563eb", linewidth=3.0, label="root trail")
    ax.scatter(
        root_path[frame, 0],
        root_path[frame, 1],
        color="#f97316",
        edgecolor="white",
        linewidth=0.9,
        s=92,
        zorder=5,
        label="current root",
    )
    err = root_error(joints_seq, target.cpu(), min(frame, target.shape[0] - 1))
    ax.set_title(f"{title} top view\nroot-target error: {err:.3f} m", pad=10)
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Z position (m)")
    xlim, ylim, _ = limits
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.75, alpha=0.45)


def select_sample(ds: H3D263Dataset, start_idx: int, max_frames: int, min_root_span: float, auto: bool):
    if not auto:
        return start_idx, ds[start_idx], None
    best_idx = start_idx
    best_sample = ds[start_idx]
    best_span = -1.0
    for offset in range(len(ds)):
        idx = (start_idx + offset) % len(ds)
        sample = ds[idx]
        length = min(sample.length, max_frames)
        root = root_xz(sample.x1[:length].unsqueeze(0))[0]
        span = trajectory_span(root)
        if span > best_span:
            best_idx = idx
            best_sample = sample
            best_span = span
        if span >= min_root_span:
            return idx, sample, span
    return best_idx, best_sample, best_span


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)
    ckpt = torch_load(ckpt_path, map_location=device)
    saved_args = ckpt_args(ckpt)
    normalizer = H3DNormalizer.from_state_dict(ckpt["normalizer"])
    steps = int(args.generation_steps or saved_args.get("generation_steps", 10))
    max_seq_len = int(args.max_seq_len or saved_args.get("max_seq_len", 80))
    real_h3d_dir = Path(args.real_h3d_dir) if args.real_h3d_dir else None
    if real_h3d_dir is not None and not real_h3d_dir.exists():
        raise FileNotFoundError(f"canonical HumanML3D new_joint_vecs dir not found: {real_h3d_dir}")
    model_input_source = args.model_input_source
    if model_input_source == "auto":
        model_input_source = "canonical" if saved_args.get("canonical_h3d_dir") and real_h3d_dir is not None else "packed"
    if model_input_source == "canonical" and real_h3d_dir is None:
        raise ValueError("--model-input-source canonical requires --real-h3d-dir")
    text_encoder = build_text_encoder(args, saved_args)
    vqvae, masked, residual = build_models(ckpt, device)

    ds = H3D263Dataset(
        root=args.data_root,
        split=args.split,
        max_seq_len=max_seq_len,
        min_seq_len=args.min_seq_len,
        mirror_augment=False,
    )
    if not 0 <= args.sample < len(ds):
        raise ValueError(f"--sample must be in [0, {len(ds) - 1}]")
    sample_idx, sample, selected_span = select_sample(
        ds,
        args.sample,
        args.max_frames,
        args.min_root_span,
        bool(args.auto_sample_moving),
    )
    if sample_idx != args.sample:
        print(
            f"[constraints-viz] auto selected sample={sample_idx} "
            f"root_span={selected_span:.3f}m from requested sample={args.sample}"
        )
    batch = collate([sample])
    real_x = batch.x1.to(device)
    frame_mask = batch.mask.to(device)
    length = min(int(batch.lengths[0].item()), args.max_frames)
    text = batch.texts[0]
    cond = text_encoder.encode([text], device=device)

    if model_input_source == "canonical":
        canonical = load_canonical_sample(real_h3d_dir, batch.clip_ids[0], max_len=real_x.shape[1])  # type: ignore[arg-type]
        canonical = canonical[: args.max_frames]
        real_x = canonical.unsqueeze(0).to(device)
        length = int(canonical.shape[0])
        frame_mask = torch.ones(1, length, dtype=torch.bool, device=device)

    generated = generate_full(
        vqvae,
        masked,
        residual,
        normalizer,
        cond,
        real_x,
        frame_mask,
        steps,
        args.guidance_scale,
        args.temperature,
        args.topk_filter_thres,
        args.sample_tokens,
        args.remask_kept_tokens,
    )[0, :length]
    real = real_x[0, :length]
    target, anchor_mask = interpolate_anchor_trajectory(root_xz(real.unsqueeze(0))[0], length, args.anchor_stride)
    projected = project_root_trajectory(generated, target, length)
    constrained_vq = vq_project(vqvae, normalizer, projected.unsqueeze(0), frame_mask)[0, :length]

    series = [
        ("ground truth", real),
        ("generated", generated),
    ]
    if args.constraint_variant in {"projected", "both"}:
        series.append(("constraint exact", projected))
    if args.constraint_variant in {"vq", "both"}:
        series.append(("constraint + VQ", constrained_vq))
    joints_by_name = [(name, recover_joints_from_ric(feat.unsqueeze(0))[0].float().cpu()) for name, feat in series]
    limits = axis_limits([j for _, j in joints_by_name], target.cpu())

    default_run_dir = ckpt_path.parent.parent if ckpt_path.parent.name == "checkpoints" else ckpt_path.parent
    out_dir = Path(args.out_dir) if args.out_dir else default_run_dir / "constraints" / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18, 10.6))
    constraint_text = (
        f"Constraint: root XZ trajectory must pass through ordered anchors every "
        f"{args.anchor_stride} frames; labels show anchor frame/time order; dashed line is the interpolated target path; "
        f"exact projection should hit anchors; VQ projection trades exact control for realism."
    )
    title = f"Sample {sample_idx} | Seed {args.seed} | Prompt: {text}\n{constraint_text}"
    fig.suptitle("\n".join(textwrap.wrap(title, width=132)), fontsize=15, fontweight="semibold", y=0.98)
    n_cols = len(joints_by_name)
    grid = fig.add_gridspec(2, n_cols, height_ratios=[2.1, 1.0], hspace=0.22, wspace=0.16)
    pose_axes = [fig.add_subplot(grid[0, i], projection="3d") for i in range(n_cols)]
    path_axes = [fig.add_subplot(grid[1, i]) for i in range(n_cols)]
    fig.subplots_adjust(top=0.84, left=0.045, right=0.985, bottom=0.075)

    def update(i: int):
        for ax, (name, joints) in zip(pose_axes, joints_by_name):
            draw_pose_frame(ax, joints, target.cpu(), anchor_mask.cpu(), name, limits, i)
        for ax, (name, joints) in zip(path_axes, joints_by_name):
            draw_path_frame(ax, joints, target.cpu(), anchor_mask.cpu(), name, limits, i)
        handles, labels = path_axes[-1].get_legend_handles_labels()
        if handles:
            path_axes[-1].legend(handles, labels, loc="upper right", framealpha=0.92, fontsize=9)
        return []

    update(0)
    png = out_dir / f"constraint_sample_{sample_idx:04d}_seed{args.seed:03d}_frame0.png"
    gif = out_dir / f"constraint_sample_{sample_idx:04d}_seed{args.seed:03d}.gif"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    anim = FuncAnimation(fig, update, frames=length, interval=1000 / args.fps, blit=False)
    anim.save(gif, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"[constraints-viz] wrote {png}")
    print(f"[constraints-viz] wrote {gif}")


if __name__ == "__main__":
    main()
