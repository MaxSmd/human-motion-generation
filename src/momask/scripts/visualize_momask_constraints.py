"""Render MoMask trajectory or latent pose constraints as GIFs.

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
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Polygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from torch import Tensor

from momask.models import (
    CodebookResidualTransformer,
    MaskedMotionTransformer,
    MotionRVQVAE,
    ResidualTransformer,
    TokenTransformerConfig,
)
from momask.constraints import (
    BendAngleConstraint,
    JointPositionConstraint,
    LatentRefinementConfig,
    ParentRelativeJointConstraint,
    bend_angles_from_joints,
    build_parent_relative_joint_constraint,
    parent_relative_targets_world,
    refine_motion_latents,
)
from momask.data_utils import normalize_motion, token_mask_from_frame_mask
from momask.scene_constraints import (
    RoomGeometryConstraint,
    SceneObstacle,
    sample_body_points,
    scene_clearance_violation_components,
)
from momask.scripts.evaluate_momask_constraints import (
    angle_triplets_from_centers,
    build_angle_constraint,
    build_joint_constraint,
    build_scene_constraint,
    parse_csv_ints,
)
from shared.data import H3D263Dataset, collate
from shared.text import CLIPTextEncoder, RandomTextEncoder, TextEncoder
from shared.geometry import H3D_FEATURE_DIM, JOINT_NAMES, PARENTS, quat_rotate, recover_joints_from_ric

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
    p.add_argument(
        "--constraint-variant",
        choices=[
            "projected",
            "vq",
            "both",
            "joint",
            "angle",
            "joint-angle",
            "body-fixed",
            "scene",
            "all",
        ],
        default="both",
    )
    p.add_argument("--joint-ids", default="20,21")
    p.add_argument(
        "--joint-target-space",
        choices=["root-relative", "global"],
        default="root-relative",
    )
    p.add_argument("--angle-joints", default="4,5,18,19")
    p.add_argument("--angle-tolerance-deg", type=float, default=5.0)
    p.add_argument("--body-fixed-joint-ids", default="17,19,21")
    p.add_argument("--body-reference-frame", type=int, default=0)
    p.add_argument("--constraint-text", default=None)
    p.add_argument("--body-output-prefix", default="body_fixed")
    p.add_argument("--refinement-steps", type=int, default=50)
    p.add_argument("--refinement-lr", type=float, default=0.01)
    p.add_argument("--position-weight", type=float, default=1.0)
    p.add_argument(
        "--torso-relative-weight",
        type=float,
        default=1.0,
        help="Legacy fallback for --parent-relative-weight.",
    )
    p.add_argument("--parent-relative-weight", type=float, default=None)
    p.add_argument("--angle-weight", type=float, default=1.0)
    p.add_argument("--latent-weight", type=float, default=0.01)
    p.add_argument("--dynamics-weight", type=float, default=0.1)
    p.add_argument("--root-weight", type=float, default=0.1)
    p.add_argument("--bone-weight", type=float, default=0.1)
    p.add_argument("--max-delta-norm", type=float, default=1.0)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--scene-weight", type=float, default=10.0)
    p.add_argument("--scene-peak-weight", type=float, default=50.0)
    p.add_argument("--scene-peak-temperature", type=float, default=0.01)
    p.add_argument("--scene-penalty-growth", type=float, default=1.0)
    p.add_argument("--scene-penalty-interval", type=int, default=50)
    p.add_argument("--scene-max-penalty-scale", type=float, default=1.0)
    p.add_argument("--scene-violation-tolerance", type=float, default=0.005)
    p.add_argument("--scene-regularization-floor", type=float, default=1.0)
    p.add_argument("--scene-root-weight", type=float, default=0.0)
    p.add_argument("--scene-room-width", type=float, default=6.0)
    p.add_argument("--scene-room-depth", type=float, default=8.0)
    p.add_argument("--scene-room-height", type=float, default=3.0)
    p.add_argument("--scene-spawn-x", type=float, default=0.0)
    p.add_argument("--scene-spawn-z", type=float, default=-2.0)
    p.add_argument("--scene-spawn-yaw", type=float, default=0.0)
    p.add_argument("--scene-padding", type=float, default=0.02)
    p.add_argument("--scene-body-radius", type=float, default=0.05)
    p.add_argument("--scene-bone-samples", type=int, default=2)
    p.add_argument("--scene-swept-samples", type=int, default=3)
    p.add_argument(
        "--scene-obstacle-kind",
        choices=["box", "sphere", "cylinder"],
        default="box",
    )
    p.add_argument("--scene-obstacle-x", type=float, default=0.0)
    p.add_argument("--scene-obstacle-y", type=float, default=0.5)
    p.add_argument("--scene-obstacle-z", type=float, default=-1.0)
    p.add_argument("--scene-obstacle-width", type=float, default=1.0)
    p.add_argument("--scene-obstacle-height", type=float, default=1.0)
    p.add_argument("--scene-obstacle-depth", type=float, default=0.4)
    p.add_argument("--scene-obstacle-radius", type=float, default=0.4)
    p.add_argument("--scene-obstacle-yaw", type=float, default=0.0)
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
            l2_normalize=bool(saved_args.get("clip_l2_normalize", True)),
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
        architecture=str(a.get("vq_arch", "simple")),
    ).to(device)
    cfg = TokenTransformerConfig(
        vocab_size=int(a.get("codebook_size", 64)),
        text_dim=int(a.get("text_dim", 64)),
        code_dim=int(a.get("vq_latent_dim", 32)),
        hidden_dim=int(a.get("transformer_hidden_dim", 64)),
        depth=int(a.get("transformer_depth", 2)),
        num_heads=int(a.get("transformer_heads", 4)),
        ffn_dim=int(a.get("transformer_ffn_dim", 128)),
        max_seq_len=math.ceil(int(a.get("max_seq_len", 80)) / int(a.get("downsample", 1))),
        dropout=float(a.get("transformer_dropout", 0.0)),
        architecture=str(a.get("transformer_arch", "legacy")),
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
    residual: ResidualTransformer | CodebookResidualTransformer,
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
) -> tuple[Tensor, Tensor, Tensor]:
    x_norm = normalize_motion(real_x, frame_mask, normalizer)
    true_tokens = vqvae.encode_to_tokens(x_norm)
    token_mask = token_mask_from_frame_mask(frame_mask, true_tokens.shape[-1], vqvae.downsample)
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
    motion = normalizer.inverse(
        vqvae.decode_from_tokens(tokens, target_len=real_x.shape[1], token_mask=token_mask)
    )
    return motion, tokens, token_mask


@torch.no_grad()
def vq_project(vqvae: MotionRVQVAE, normalizer: H3DNormalizer, motion: Tensor, frame_mask: Tensor) -> Tensor:
    normalized = normalize_motion(motion, frame_mask, normalizer)
    return normalizer.inverse(vqvae(normalized, mask=frame_mask).recon)


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


def control_error_series(
    joints: Tensor,
    *,
    position_targets: Tensor | None,
    joint_ids: list[int],
    angle_triplets: Tensor | None,
    angle_target: Tensor | None,
) -> tuple[Tensor | None, Tensor | None]:
    joint_error = None
    if position_targets is not None:
        joint_error = torch.linalg.vector_norm(
            joints[:, joint_ids] - position_targets[:, joint_ids], dim=-1
        ).mean(dim=-1) * 100.0
    angle_error = None
    if angle_triplets is not None and angle_target is not None:
        angles = bend_angles_from_joints(joints.unsqueeze(0), angle_triplets)[0]
        angle_error = torch.rad2deg((angles - angle_target).abs()).mean(dim=-1)
    return joint_error, angle_error


def natural_joint_names(joint_ids: list[int]) -> str:
    names = [
        JOINT_NAMES[joint].replace("L_", "left ").replace("R_", "right ").replace("_", " ").lower()
        for joint in joint_ids
    ]
    if len(names) == 1:
        return f"the {names[0]}"
    if len(names) == 2:
        return f"the {names[0]} and {names[1]}"
    return "the " + ", ".join(names[:-1]) + f", and {names[-1]}"


def draw_latent_pose_frame(
    ax,
    *,
    joints_seq: Tensor,
    position_targets: Tensor | None,
    joint_ids: list[int],
    angle_triplets: Tensor | None,
    anchor_mask: Tensor,
    joint_error: Tensor | None,
    angle_error: Tensor | None,
    title: str,
    frame: int,
    dense_constraint: bool,
) -> None:
    ax.cla()
    joints = joints_seq[frame]
    highlighted_bones: set[tuple[int, int]] = set()
    if angle_triplets is not None:
        for parent, center, child in angle_triplets.tolist():
            highlighted_bones.add(tuple(sorted((parent, center))))
            highlighted_bones.add(tuple(sorted((center, child))))
    for joint, parent in enumerate(PARENTS):
        if parent < 0:
            continue
        highlighted = tuple(sorted((parent, joint))) in highlighted_bones
        ax.plot(
            [joints[parent, 0], joints[joint, 0]],
            [joints[parent, 2], joints[joint, 2]],
            [joints[parent, 1], joints[joint, 1]],
            color="#16a34a" if highlighted else "#2563eb",
            linewidth=4.0 if highlighted else 2.3,
            alpha=0.95,
        )
    ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1], color="#dc2626", s=16, depthshade=True)
    if joint_ids:
        selected = joints[joint_ids]
        ax.scatter(
            selected[:, 0], selected[:, 2], selected[:, 1],
            color="#f97316", edgecolor="white", linewidth=0.8, s=62, label="controlled joints",
        )
    active_anchor = bool(anchor_mask[frame])
    if active_anchor and angle_triplets is not None:
        centers = angle_triplets[:, 1]
        selected = joints[centers]
        ax.scatter(
            selected[:, 0], selected[:, 2], selected[:, 1],
            marker="D", color="#22c55e", edgecolor="#052e16", linewidth=0.8,
            s=66, depthshade=False, label="active angle joints",
        )
    if active_anchor and position_targets is not None:
        targets = position_targets[frame, joint_ids]
        ax.scatter(
            targets[:, 0], targets[:, 2], targets[:, 1],
            marker="*", color="#22c55e", edgecolor="#052e16", linewidth=0.9,
            s=180, depthshade=False, label="active joint targets",
        )
        for joint_id, target in zip(joint_ids, targets):
            current = joints[joint_id]
            ax.plot(
                [current[0], target[0]], [current[2], target[2]], [current[1], target[1]],
                color="#22c55e", linestyle="--", linewidth=2.0,
            )
            ax.text(
                float(target[0]), float(target[2]), float(target[1] + 0.05),
                JOINT_NAMES[joint_id].replace("_", " "), fontsize=8, color="#14532d", weight="bold",
            )

    status = "CONSTRAINT ACTIVE" if dense_constraint else (
        "ACTIVE KEYFRAME" if active_anchor else "between keyframes"
    )
    measurements: list[str] = []
    if joint_error is not None:
        measurements.append(f"joint-pose error {float(joint_error[frame]):.1f} cm")
    if angle_error is not None:
        measurements.append(f"angle error {float(angle_error[frame]):.1f} deg")
    subtitle = " | ".join(measurements)
    ax.set_title(f"{title} | frame {frame} | {status}\n{subtitle}", pad=13)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_zlabel("Y height (m)")
    body_min = joints.amin(dim=0)
    body_max = joints.amax(dim=0)
    root = joints[0]
    extra_points = [joints]
    if active_anchor and position_targets is not None:
        extra_points.append(position_targets[frame, joint_ids])
    visible = torch.cat(extra_points, dim=0)
    horizontal_radius = float(
        torch.stack(
            [
                (visible[:, 0] - root[0]).abs().max(),
                (visible[:, 2] - root[2]).abs().max(),
                visible.new_tensor(0.8),
            ]
        ).max()
    ) * 1.15
    ax.set_xlim(float(root[0] - horizontal_radius), float(root[0] + horizontal_radius))
    ax.set_ylim(float(root[2] - horizontal_radius), float(root[2] + horizontal_radius))
    ax.set_zlim(float(body_min[1] - 0.25), float(body_max[1] + 0.25))
    ax.grid(True, linewidth=0.8, alpha=0.5)
    ax.xaxis.pane.set_alpha(0.08)
    ax.yaxis.pane.set_alpha(0.08)
    ax.zaxis.pane.set_alpha(0.04)
    ax.view_init(elev=17, azim=-72)
    try:
        ax.set_box_aspect((1.0, 1.0, 0.9))
    except AttributeError:
        pass


def draw_control_error_frame(
    ax,
    angle_ax,
    *,
    joint_error: Tensor | None,
    angle_error: Tensor | None,
    anchor_mask: Tensor,
    angle_tolerance_deg: float,
    joint_ylim: float,
    angle_ylim: float,
    title: str,
    frame: int,
) -> None:
    ax.cla()
    angle_ax.cla()
    anchor_frames = torch.nonzero(anchor_mask, as_tuple=False).flatten()
    completed = anchor_frames <= frame
    if joint_error is not None:
        values = joint_error[anchor_frames]
        ax.plot(anchor_frames, values, color="#93c5fd", linewidth=1.4, alpha=0.75)
        ax.scatter(
            anchor_frames[completed],
            values[completed],
            color="#2563eb",
            s=36,
            label="joint-pose error",
        )
        ax.axhline(5.0, color="#2563eb", linestyle="--", linewidth=1.0, alpha=0.65, label="5 cm")
        ax.axhline(10.0, color="#60a5fa", linestyle=":", linewidth=1.0, alpha=0.65, label="10 cm")
        ax.set_ylabel("Joint-pose error (cm)", color="#1d4ed8")
        ax.tick_params(axis="y", colors="#1d4ed8")
        ax.set_ylim(0.0, joint_ylim)
    else:
        ax.set_yticks([])
    if angle_error is not None:
        values = angle_error[anchor_frames]
        angle_ax.plot(anchor_frames, values, color="#86efac", linewidth=1.4, alpha=0.75)
        angle_ax.scatter(anchor_frames[completed], values[completed], color="#16a34a", s=36, label="bend error")
        angle_ax.axhline(
            angle_tolerance_deg, color="#16a34a", linestyle="--", linewidth=1.0,
            alpha=0.7, label=f"{angle_tolerance_deg:g} deg tolerance",
        )
        angle_ax.set_ylabel("Bend error (deg)", color="#15803d")
        angle_ax.tick_params(axis="y", colors="#15803d")
        angle_ax.set_ylim(0.0, angle_ylim)
    else:
        angle_ax.set_yticks([])
    ax.axvline(frame, color="#111827", linewidth=1.2, alpha=0.8)
    ax.set_xlim(0, max(1, anchor_mask.shape[0] - 1))
    ax.set_xlabel("Frame (dots are active constraint frames)")
    ax.set_title(f"{title}: control error", pad=8, fontsize=11)
    ax.grid(True, axis="x", linewidth=0.7, alpha=0.35)


def _draw_scene_box(ax, obstacle: SceneObstacle) -> None:
    assert obstacle.size is not None
    width, height, depth = obstacle.size
    local = np.array(
        [
            [sx * width / 2.0, sy * height / 2.0, sz * depth / 2.0]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float32,
    )
    yaw = math.radians(obstacle.yaw_degrees)
    c, s = math.cos(yaw), math.sin(yaw)
    x = c * local[:, 0] + s * local[:, 2]
    z = -s * local[:, 0] + c * local[:, 2]
    center = np.asarray(obstacle.center, dtype=np.float32)
    world = np.stack((x, local[:, 1], z), axis=-1) + center
    faces = [
        [0, 1, 3, 2],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 3, 7, 5],
    ]
    vertices = [
        [(world[i, 0], world[i, 2], world[i, 1]) for i in face] for face in faces
    ]
    ax.add_collection3d(
        Poly3DCollection(
            vertices,
            facecolors="#f59e0b",
            edgecolors="#92400e",
            linewidths=1.1,
            alpha=0.38,
        )
    )


def _draw_scene_obstacle(ax, obstacle: SceneObstacle) -> None:
    if obstacle.kind == "box":
        _draw_scene_box(ax, obstacle)
        return
    center_x, center_y, center_z = obstacle.center
    if obstacle.kind == "sphere":
        assert obstacle.radius is not None
        azimuth = np.linspace(0.0, 2.0 * np.pi, 24)
        polar = np.linspace(0.0, np.pi, 14)
        x = center_x + obstacle.radius * np.outer(np.cos(azimuth), np.sin(polar))
        z = center_z + obstacle.radius * np.outer(np.sin(azimuth), np.sin(polar))
        y = center_y + obstacle.radius * np.outer(np.ones_like(azimuth), np.cos(polar))
    else:
        assert obstacle.radius is not None and obstacle.height is not None
        azimuth = np.linspace(0.0, 2.0 * np.pi, 24)
        vertical = np.linspace(-obstacle.height / 2.0, obstacle.height / 2.0, 8)
        x = center_x + obstacle.radius * np.outer(np.cos(azimuth), np.ones_like(vertical))
        z = center_z + obstacle.radius * np.outer(np.sin(azimuth), np.ones_like(vertical))
        y = center_y + np.outer(np.ones_like(azimuth), vertical)
    ax.plot_surface(x, z, y, color="#f59e0b", edgecolor="#92400e", alpha=0.38)


def draw_scene_geometry(ax, constraint: RoomGeometryConstraint) -> None:
    width, depth, height = constraint.room_size
    half_w, half_d = width / 2.0, depth / 2.0
    floor = [
        [
            (-half_w, -half_d, 0.0),
            (half_w, -half_d, 0.0),
            (half_w, half_d, 0.0),
            (-half_w, half_d, 0.0),
        ]
    ]
    ax.add_collection3d(
        Poly3DCollection(
            floor,
            facecolors="#e5e7eb",
            edgecolors="#6b7280",
            linewidths=0.9,
            alpha=0.18,
        )
    )
    corners = [
        (x, z, y)
        for x in (-half_w, half_w)
        for z in (-half_d, half_d)
        for y in (0.0, height)
    ]
    for start, end in (
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ):
        ax.plot(
            [corners[start][0], corners[end][0]],
            [corners[start][1], corners[end][1]],
            [corners[start][2], corners[end][2]],
            color="#9ca3af",
            linewidth=0.8,
            alpha=0.55,
        )
    for obstacle in constraint.obstacles:
        _draw_scene_obstacle(ax, obstacle)


def scene_axis_limits(
    series: list[tuple[str, Tensor]],
    constraint: RoomGeometryConstraint,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    width, depth, height = constraint.room_size
    points = torch.cat([joints.reshape(-1, 3) for _, joints in series], dim=0)
    margin = 0.3
    xlim = (
        min(-width / 2.0, float(points[:, 0].min())) - margin,
        max(width / 2.0, float(points[:, 0].max())) + margin,
    )
    zlim = (
        min(-depth / 2.0, float(points[:, 2].min())) - margin,
        max(depth / 2.0, float(points[:, 2].max())) + margin,
    )
    ylim = (
        min(0.0, float(points[:, 1].min())) - 0.15,
        max(height, float(points[:, 1].max())) + 0.15,
    )
    return xlim, zlim, ylim


def draw_scene_pose_frame(
    ax,
    *,
    name: str,
    joints_seq: Tensor,
    body_points: Tensor,
    components: dict[str, Tensor],
    constraint: RoomGeometryConstraint,
    limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    frame: int,
) -> None:
    ax.cla()
    draw_scene_geometry(ax, constraint)
    joints = joints_seq[frame]
    for joint, parent in enumerate(PARENTS):
        if parent < 0:
            continue
        ax.plot(
            [joints[parent, 0], joints[joint, 0]],
            [joints[parent, 2], joints[joint, 2]],
            [joints[parent, 1], joints[joint, 1]],
            color="#2563eb",
            linewidth=2.4,
            alpha=0.95,
        )
    points = body_points[frame]
    violation = components["combined"][frame]
    colliding = violation > 0.0
    clear = ~colliding
    if bool(clear.any()):
        safe_points = points[clear]
        ax.scatter(
            safe_points[:, 0],
            safe_points[:, 2],
            safe_points[:, 1],
            color="#60a5fa",
            s=10,
            alpha=0.55,
            depthshade=False,
            label="clear body samples",
        )
    if bool(colliding.any()):
        collision_points = points[colliding]
        ax.scatter(
            collision_points[:, 0],
            collision_points[:, 2],
            collision_points[:, 1],
            color="#dc2626",
            edgecolor="white",
            linewidth=0.6,
            s=48,
            depthshade=False,
            label="colliding body samples",
        )
    root_path = joints_seq[:, 0][:, [0, 2]]
    ax.plot(
        root_path[: frame + 1, 0],
        root_path[: frame + 1, 1],
        torch.zeros(frame + 1),
        color="#111827",
        linewidth=2.0,
        alpha=0.8,
    )
    source_values = {
        source: float(values[frame].max())
        for source, values in components.items()
        if source != "combined" and float(values[frame].max()) > 0.0
    }
    source_text = ", ".join(
        f"{source} {value * 100:.1f} cm"
        for source, value in source_values.items()
    )
    if not source_text:
        source_text = "clear"
    ax.set_title(
        f"{name} | frame {frame}\n"
        f"max {float(violation.max()) * 100:.1f} cm | "
        f"points {int(colliding.sum())}/{points.shape[0]} | {source_text}",
        pad=12,
    )
    ax.set_xlabel("Scene X (m)")
    ax.set_ylabel("Scene Z (m)")
    ax.set_zlabel("Height Y (m)")
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.grid(True, linewidth=0.7, alpha=0.4)
    ax.view_init(elev=20, azim=-68)
    try:
        ax.set_box_aspect(
            (
                limits[0][1] - limits[0][0],
                limits[1][1] - limits[1][0],
                limits[2][1] - limits[2][0],
            )
        )
    except AttributeError:
        pass


def draw_scene_violation_frame(
    ax,
    *,
    name: str,
    components: dict[str, Tensor],
    frame: int,
    ymax_cm: float,
) -> None:
    ax.cla()
    colors = {
        "combined": "#111827",
        "obstacle": "#f59e0b",
        "floor": "#16a34a",
        "wall": "#2563eb",
        "ceiling": "#7c3aed",
    }
    frames = np.arange(components["combined"].shape[0])
    for source in ("combined", "obstacle", "floor", "wall", "ceiling"):
        values = components[source].amax(dim=-1).numpy() * 100.0
        ax.plot(
            frames,
            values,
            color=colors[source],
            linewidth=2.2 if source == "combined" else 1.4,
            alpha=0.95 if source == "combined" else 0.75,
            label=source,
        )
    ax.axvline(frame, color="#dc2626", linewidth=1.2, alpha=0.85)
    ax.set_xlim(0, max(1, len(frames) - 1))
    ax.set_ylim(0.0, ymax_cm)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Maximum penetration (cm)")
    ax.set_title(f"{name}: collision source", fontsize=11)
    ax.grid(True, linewidth=0.7, alpha=0.35)
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)


def draw_scene_top_view_frame(
    ax,
    *,
    name: str,
    joints_seq: Tensor,
    constraint: RoomGeometryConstraint,
    limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    frame: int,
) -> None:
    ax.cla()
    width, depth, _ = constraint.room_size
    ax.add_patch(
        Rectangle(
            (-width / 2.0, -depth / 2.0),
            width,
            depth,
            facecolor="#f3f4f6",
            edgecolor="#6b7280",
            linewidth=1.0,
        )
    )
    for obstacle in constraint.obstacles:
        if obstacle.kind == "box":
            assert obstacle.size is not None
            half_width = obstacle.size[0] / 2.0
            half_depth = obstacle.size[2] / 2.0
            local = np.array(
                [
                    [-half_width, -half_depth],
                    [half_width, -half_depth],
                    [half_width, half_depth],
                    [-half_width, half_depth],
                ]
            )
            yaw = math.radians(obstacle.yaw_degrees)
            rotation = np.array(
                [[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]]
            )
            polygon = local @ rotation.T
            polygon += np.array([obstacle.center[0], obstacle.center[2]])
            patch = Polygon(
                polygon,
                closed=True,
                facecolor="#f59e0b",
                edgecolor="#92400e",
                linewidth=1.2,
                alpha=0.5,
            )
        else:
            assert obstacle.radius is not None
            patch = Circle(
                (obstacle.center[0], obstacle.center[2]),
                obstacle.radius,
                facecolor="#f59e0b",
                edgecolor="#92400e",
                linewidth=1.2,
                alpha=0.5,
            )
        ax.add_patch(patch)
    root_path = joints_seq[:, 0][:, [0, 2]]
    ax.plot(
        root_path[:, 0],
        root_path[:, 1],
        color="#9ca3af",
        linewidth=1.2,
        alpha=0.65,
        label="full pelvis path",
    )
    ax.plot(
        root_path[: frame + 1, 0],
        root_path[: frame + 1, 1],
        color="#2563eb",
        linewidth=2.8,
        label="pelvis trail",
    )
    ax.scatter(
        root_path[frame, 0],
        root_path[frame, 1],
        color="#dc2626",
        edgecolor="white",
        linewidth=0.7,
        s=54,
        zorder=5,
    )
    ax.set_title(f"{name}: top view | pelvis trajectory")
    ax.set_xlabel("Scene X (m)")
    ax.set_ylabel("Scene Z (m)")
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.7, alpha=0.35)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)


def render_scene_constraint_comparison(
    *,
    series: list[tuple[str, Tensor]],
    constraint: RoomGeometryConstraint,
    prompt: str,
    sample_idx: int,
    seed: int,
    fps: int,
    out_dir: Path,
) -> None:
    diagnostics = []
    for _, joints in series:
        diagnostics.append(
            (
                sample_body_points(joints.unsqueeze(0), constraint.bone_samples)[0],
                {
                    key: value[0].detach().cpu()
                    for key, value in scene_clearance_violation_components(
                        joints.unsqueeze(0), constraint
                    ).items()
                },
            )
        )
    limits = scene_axis_limits(series, constraint)
    ymax_cm = max(
        [float(components["combined"].max()) * 100.0 for _, components in diagnostics]
        + [5.0]
    ) * 1.15
    n_cols = len(series)
    fig = plt.figure(figsize=(max(14.0, 7.0 * n_cols), 13.0))
    obstacle_description = ", ".join(
        f"{obstacle.kind} at ({obstacle.center[0]:g}, {obstacle.center[1]:g}, "
        f"{obstacle.center[2]:g})"
        for obstacle in constraint.obstacles
    ) or "no obstacle"
    title = (
        f"Sample {sample_idx} | Seed {seed} | Prompt: {prompt}\n"
        f"Scene constraint (not part of the text): avoid {obstacle_description}; "
        "red points violate clearance."
    )
    fig.suptitle(
        "\n".join(textwrap.wrap(title, width=150)),
        fontsize=14,
        fontweight="semibold",
        y=0.985,
    )
    grid = fig.add_gridspec(
        3,
        n_cols,
        height_ratios=[2.4, 1.0, 1.0],
        hspace=0.28,
        wspace=0.2,
    )
    pose_axes = [fig.add_subplot(grid[0, col], projection="3d") for col in range(n_cols)]
    path_axes = [fig.add_subplot(grid[1, col]) for col in range(n_cols)]
    metric_axes = [fig.add_subplot(grid[2, col]) for col in range(n_cols)]
    fig.subplots_adjust(top=0.84, left=0.055, right=0.97, bottom=0.08)

    def update(frame: int):
        for index, ((name, joints), (points, components)) in enumerate(
            zip(series, diagnostics)
        ):
            draw_scene_pose_frame(
                pose_axes[index],
                name=name,
                joints_seq=joints,
                body_points=points,
                components=components,
                constraint=constraint,
                limits=limits,
                frame=frame,
            )
            draw_scene_violation_frame(
                metric_axes[index],
                name=name,
                components=components,
                frame=frame,
                ymax_cm=ymax_cm,
            )
            draw_scene_top_view_frame(
                path_axes[index],
                name=name,
                joints_seq=joints,
                constraint=constraint,
                limits=limits,
                frame=frame,
            )
        return []

    snapshot_frame = int(diagnostics[0][1]["combined"].amax(dim=-1).argmax())
    update(snapshot_frame)
    prefix = f"scene_constraint_sample_{sample_idx:04d}_seed{seed:03d}"
    png = out_dir / f"{prefix}_frame{snapshot_frame:04d}.png"
    gif = out_dir / f"{prefix}.gif"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    animation = FuncAnimation(
        fig,
        update,
        frames=series[0][1].shape[0],
        interval=1000 / fps,
        blit=False,
    )
    animation.save(gif, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"[constraints-viz] wrote {png}")
    print(f"[constraints-viz] wrote {gif}")


def render_latent_constraint_comparison(
    *,
    series: list[tuple[str, Tensor, Tensor | None]],
    joint_ids: list[int],
    angle_triplets: Tensor | None,
    angle_target: Tensor | None,
    anchor_mask: Tensor,
    angle_tolerance_deg: float,
    anchor_stride: int,
    constraint_text_override: str | None,
    dense_constraint: bool,
    output_prefix: str,
    prompt: str,
    sample_idx: int,
    seed: int,
    fps: int,
    out_dir: Path,
) -> None:
    errors = [
        control_error_series(
            joints,
            position_targets=position_targets,
            joint_ids=joint_ids,
            angle_triplets=angle_triplets,
            angle_target=angle_target,
        )
        for _, joints, position_targets in series
    ]
    anchor_values = anchor_mask.bool()
    joint_max = max(
        [float(err[anchor_values].max()) for err, _ in errors if err is not None] + [10.0]
    )
    angle_max = max(
        [float(err[anchor_values].max()) for _, err in errors if err is not None] + [angle_tolerance_deg]
    )
    joint_ylim = joint_max * 1.2 + 1.0
    angle_ylim = angle_max * 1.2 + 1.0
    n_cols = len(series)
    fig = plt.figure(figsize=(max(13.5, 4.5 * n_cols), 9.8))
    rules: list[str] = []
    if joint_ids:
        rules.append(f"keep {natural_joint_names(joint_ids)} at their reference positions")
    if angle_triplets is not None:
        angle_centers = [int(joint) for joint in angle_triplets[:, 1]]
        rules.append(
            f"keep {natural_joint_names(angle_centers)} within +/-{angle_tolerance_deg:g} deg "
            "of their reference bend"
        )
    constraint_text = constraint_text_override or (
        f"Constraint: {'; '.join(rules)} at keyframes every {anchor_stride} frames. "
        "Green markers show the active targets."
    )
    title = f"Sample {sample_idx} | Seed {seed} | Prompt: {prompt}\n{constraint_text}"
    fig.suptitle("\n".join(textwrap.wrap(title, width=150)), fontsize=14, fontweight="semibold", y=0.985)
    grid = fig.add_gridspec(2, n_cols, height_ratios=[2.3, 1.0], hspace=0.24, wspace=0.32)
    pose_axes = [fig.add_subplot(grid[0, col], projection="3d") for col in range(n_cols)]
    error_axes = [fig.add_subplot(grid[1, col]) for col in range(n_cols)]
    angle_axes = [ax.twinx() for ax in error_axes]
    fig.subplots_adjust(top=0.82, left=0.045, right=0.96, bottom=0.08)

    def update(frame: int):
        for idx, ((name, joints, targets), (joint_error, angle_error)) in enumerate(zip(series, errors)):
            draw_latent_pose_frame(
                pose_axes[idx],
                joints_seq=joints,
                position_targets=targets,
                joint_ids=joint_ids,
                angle_triplets=angle_triplets,
                anchor_mask=anchor_mask,
                joint_error=joint_error,
                angle_error=angle_error,
                title=name,
                frame=frame,
                dense_constraint=dense_constraint,
            )
            draw_control_error_frame(
                error_axes[idx],
                angle_axes[idx],
                joint_error=joint_error,
                angle_error=angle_error,
                anchor_mask=anchor_mask,
                angle_tolerance_deg=angle_tolerance_deg,
                joint_ylim=joint_ylim,
                angle_ylim=angle_ylim,
                title=name,
                frame=frame,
            )
        return []

    snapshot_frame = 0
    if dense_constraint and len(errors) > 1 and errors[1][0] is not None:
        snapshot_frame = int(torch.argmax(errors[1][0]).item())
    update(snapshot_frame)
    png = out_dir / (
        f"{output_prefix}_sample_{sample_idx:04d}_seed{seed:03d}_frame{snapshot_frame:04d}.png"
    )
    gif = out_dir / f"{output_prefix}_sample_{sample_idx:04d}_seed{seed:03d}.gif"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    held_frames: list[int] = []
    anchor_hold = max(2, int(round(fps * 0.3)))
    for frame in range(anchor_mask.shape[0]):
        held_frames.append(frame)
        if bool(anchor_mask[frame]) and not dense_constraint:
            held_frames.extend([frame] * anchor_hold)
    animation = FuncAnimation(fig, update, frames=held_frames, interval=1000 / fps, blit=False)
    animation.save(gif, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"[constraints-viz] wrote {png}")
    print(f"[constraints-viz] wrote {gif}")


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
    if args.parent_relative_weight is None:
        args.parent_relative_weight = args.torso_relative_weight
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

    generated_batch, generated_tokens, token_mask = generate_full(
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
    )
    generated = generated_batch[0, :length]
    real = real_x[0, :length]
    target, anchor_mask = interpolate_anchor_trajectory(root_xz(real.unsqueeze(0))[0], length, args.anchor_stride)

    default_run_dir = ckpt_path.parent.parent if ckpt_path.parent.name == "checkpoints" else ckpt_path.parent
    out_dir = Path(args.out_dir) if args.out_dir else default_run_dir / "constraints" / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    latent_joint = args.constraint_variant in {"joint", "joint-angle", "all"}
    latent_angle = args.constraint_variant in {"angle", "joint-angle", "all"}
    latent_body_fixed = args.constraint_variant in {"body-fixed", "all"}
    latent_scene = args.constraint_variant in {"scene", "all"}
    trajectory_mode = args.constraint_variant in {"projected", "vq", "both", "all"}
    if latent_joint or latent_angle or latent_body_fixed or latent_scene:
        joint_ids = parse_csv_ints(args.joint_ids, name="--joint-ids") if latent_joint else []
        angle_centers = parse_csv_ints(args.angle_joints, name="--angle-joints") if latent_angle else []
        body_fixed_joint_ids = (
            parse_csv_ints(args.body_fixed_joint_ids, name="--body-fixed-joint-ids")
            if latent_body_fixed
            else []
        )
        scene_constraint = build_scene_constraint(args) if latent_scene else None
        triplets = angle_triplets_from_centers(angle_centers, device) if latent_angle else None
        decoded_length = int(
            (token_mask[0].sum() * vqvae.downsample).clamp(max=length).item()
        )
        if decoded_length < 1:
            raise RuntimeError("generated token mask contains no valid decoded frames")
        _, latent_anchor_mask = interpolate_anchor_trajectory(
            root_xz(real.unsqueeze(0))[0], decoded_length, args.anchor_stride
        )
        latent_anchor_mask_batch = latent_anchor_mask.unsqueeze(0)
        constraint_frame_mask = frame_mask[:, :length]
        generated_for_constraints = generated_batch[:, :length]
        decoded_frame_mask = constraint_frame_mask & (
            torch.arange(length, device=device).unsqueeze(0) < decoded_length
        )
        with torch.no_grad():
            real_joints = recover_joints_from_ric(real.unsqueeze(0).float())
            generated_joints = recover_joints_from_ric(generated_for_constraints.float())
            initial_latents = vqvae.quantizer.decode(generated_tokens)

        position_constraint: JointPositionConstraint | None = None
        parent_relative_constraint: ParentRelativeJointConstraint | None = None
        angle_constraint: BendAngleConstraint | None = None
        angle_target: Tensor | None = None
        joint_motion: Tensor | None = None
        angle_motion: Tensor | None = None
        body_fixed_motion: Tensor | None = None
        scene_result = None
        if latent_joint:
            position_constraint = build_joint_constraint(
                real.unsqueeze(0),
                generated_for_constraints,
                real_joints,
                generated_joints,
                latent_anchor_mask_batch,
                joint_ids,
                target_space=args.joint_target_space,
            )
            joint_result = refine_motion_latents(
                vqvae,
                initial_latents,
                mean=normalizer.mean,
                std=normalizer.std,
                target_len=length,
                token_mask=token_mask,
                frame_mask=constraint_frame_mask,
                position_constraint=position_constraint,
                config=LatentRefinementConfig(
                    steps=args.refinement_steps,
                    learning_rate=args.refinement_lr,
                    position_weight=args.position_weight,
                    angle_weight=0.0,
                    latent_weight=args.latent_weight,
                    dynamics_weight=args.dynamics_weight,
                    root_weight=args.root_weight,
                    bone_weight=args.bone_weight,
                    max_delta_norm=args.max_delta_norm,
                    grad_clip_norm=args.grad_clip_norm,
                ),
            )
            joint_motion = joint_result.motion
            print(f"[constraints-viz] joint refinement metrics={joint_result.metrics}", flush=True)
        if latent_angle:
            assert triplets is not None
            angle_constraint, angle_target = build_angle_constraint(
                real_joints,
                latent_anchor_mask_batch,
                triplets,
                args.angle_tolerance_deg,
            )
            angle_result = refine_motion_latents(
                vqvae,
                initial_latents,
                mean=normalizer.mean,
                std=normalizer.std,
                target_len=length,
                token_mask=token_mask,
                frame_mask=constraint_frame_mask,
                angle_constraint=angle_constraint,
                config=LatentRefinementConfig(
                    steps=args.refinement_steps,
                    learning_rate=args.refinement_lr,
                    position_weight=0.0,
                    angle_weight=args.angle_weight,
                    latent_weight=args.latent_weight,
                    dynamics_weight=args.dynamics_weight,
                    root_weight=args.root_weight,
                    bone_weight=args.bone_weight,
                    max_delta_norm=args.max_delta_norm,
                    grad_clip_norm=args.grad_clip_norm,
                ),
            )
            angle_motion = angle_result.motion
            print(f"[constraints-viz] angle refinement metrics={angle_result.metrics}", flush=True)
        if latent_body_fixed:
            parent_relative_constraint = build_parent_relative_joint_constraint(
                generated_joints,
                decoded_frame_mask,
                body_fixed_joint_ids,
                reference_frame=args.body_reference_frame,
            )
            body_fixed_result = refine_motion_latents(
                vqvae,
                initial_latents,
                mean=normalizer.mean,
                std=normalizer.std,
                target_len=length,
                token_mask=token_mask,
                frame_mask=constraint_frame_mask,
                parent_relative_constraint=parent_relative_constraint,
                config=LatentRefinementConfig(
                    steps=args.refinement_steps,
                    learning_rate=args.refinement_lr,
                    position_weight=0.0,
                    torso_relative_weight=0.0,
                    parent_relative_weight=args.parent_relative_weight,
                    angle_weight=0.0,
                    latent_weight=args.latent_weight,
                    dynamics_weight=args.dynamics_weight,
                    root_weight=args.root_weight,
                    bone_weight=args.bone_weight,
                    max_delta_norm=args.max_delta_norm,
                    grad_clip_norm=args.grad_clip_norm,
                ),
            )
            body_fixed_motion = body_fixed_result.motion
            print(
                f"[constraints-viz] body-fixed refinement metrics={body_fixed_result.metrics}",
                flush=True,
            )
        if latent_scene:
            assert scene_constraint is not None
            scene_result = refine_motion_latents(
                vqvae,
                initial_latents,
                mean=normalizer.mean,
                std=normalizer.std,
                target_len=length,
                token_mask=token_mask,
                frame_mask=decoded_frame_mask,
                scene_constraint=scene_constraint,
                config=LatentRefinementConfig(
                    steps=args.refinement_steps,
                    learning_rate=args.refinement_lr,
                    position_weight=0.0,
                    torso_relative_weight=0.0,
                    parent_relative_weight=0.0,
                    angle_weight=0.0,
                    scene_weight=args.scene_weight,
                    scene_peak_weight=args.scene_peak_weight,
                    scene_peak_temperature=args.scene_peak_temperature,
                    scene_penalty_growth=args.scene_penalty_growth,
                    scene_penalty_interval=args.scene_penalty_interval,
                    scene_max_penalty_scale=args.scene_max_penalty_scale,
                    scene_violation_tolerance=args.scene_violation_tolerance,
                    scene_regularization_floor=args.scene_regularization_floor,
                    latent_weight=args.latent_weight,
                    dynamics_weight=args.dynamics_weight,
                    root_weight=args.scene_root_weight,
                    bone_weight=args.bone_weight,
                    max_delta_norm=args.max_delta_norm,
                    grad_clip_norm=args.grad_clip_norm,
                ),
            )
            print(
                f"[constraints-viz] scene refinement metrics={scene_result.metrics}",
                flush=True,
            )

        real_joints_cpu = real_joints[0, :decoded_length].detach().cpu()
        if latent_joint or latent_angle:
            position_targets = (
                None
                if position_constraint is None
                else position_constraint.targets[0].detach().cpu()
            )
            rendered_joint_ids = joint_ids if position_constraint is not None else []
            rendered_triplets = None if triplets is None else triplets.detach().cpu()
            rendered_angle_target = None if angle_target is None else angle_target[0].detach().cpu()
            latent_series: list[tuple[str, Tensor, Tensor | None]] = [
                (
                    "ground truth reference",
                    real_joints_cpu,
                    real_joints_cpu if position_constraint is not None else None,
                ),
                (
                    "generated",
                    generated_joints[0, :decoded_length].detach().cpu(),
                    None if position_targets is None else position_targets[:decoded_length],
                ),
            ]
            if joint_motion is not None:
                latent_series.append(
                    (
                        "joint-position refined",
                        recover_joints_from_ric(joint_motion.float())[0, :decoded_length]
                        .detach()
                        .cpu(),
                        position_targets[:decoded_length] if position_targets is not None else None,
                    )
                )
            if angle_motion is not None:
                latent_series.append(
                    (
                        "bend-angle refined",
                        recover_joints_from_ric(angle_motion.float())[0, :decoded_length]
                        .detach()
                        .cpu(),
                        position_targets[:decoded_length] if position_targets is not None else None,
                    )
                )
            render_latent_constraint_comparison(
                series=latent_series,
                joint_ids=rendered_joint_ids,
                angle_triplets=rendered_triplets,
                angle_target=None
                if rendered_angle_target is None
                else rendered_angle_target[:decoded_length],
                anchor_mask=latent_anchor_mask[:decoded_length].detach().cpu(),
                angle_tolerance_deg=args.angle_tolerance_deg,
                anchor_stride=args.anchor_stride,
                constraint_text_override=None,
                dense_constraint=False,
                output_prefix="joint_angle_constraint",
                prompt=text,
                sample_idx=sample_idx,
                seed=args.seed,
                fps=args.fps,
                out_dir=out_dir,
            )

        if body_fixed_motion is not None and parent_relative_constraint is not None:
            generated_joints_valid = generated_joints[:, :decoded_length]
            body_fixed_joints = recover_joints_from_ric(body_fixed_motion.float())[:, :decoded_length]
            body_constraint_valid = ParentRelativeJointConstraint(
                reference_bone_offsets=parent_relative_constraint.reference_bone_offsets,
                mask=parent_relative_constraint.mask[:, :decoded_length],
                origin_joint=parent_relative_constraint.origin_joint,
                left_joint=parent_relative_constraint.left_joint,
                right_joint=parent_relative_constraint.right_joint,
                up_joint=parent_relative_constraint.up_joint,
            )
            generated_targets = parent_relative_targets_world(
                generated_joints_valid,
                body_constraint_valid,
            )
            body_fixed_targets = parent_relative_targets_world(
                body_fixed_joints,
                body_constraint_valid,
            )
            body_series = [
                ("ground truth (comparison only)", real_joints_cpu, None),
                (
                    "generated",
                    generated_joints_valid[0].detach().cpu(),
                    generated_targets[0].detach().cpu(),
                ),
                (
                    f"{natural_joint_names(body_fixed_joint_ids)} fixed",
                    body_fixed_joints[0].detach().cpu(),
                    body_fixed_targets[0].detach().cpu(),
                ),
            ]
            render_latent_constraint_comparison(
                series=body_series,
                joint_ids=body_fixed_joint_ids,
                angle_triplets=None,
                angle_target=None,
                anchor_mask=decoded_frame_mask[0, :decoded_length].detach().cpu(),
                angle_tolerance_deg=args.angle_tolerance_deg,
                anchor_stride=1,
                constraint_text_override=args.constraint_text
                or (
                    f"Constraint: keep {natural_joint_names(body_fixed_joint_ids)} fixed "
                    "relative to their anatomical parents throughout the motion. Green "
                    "markers show the moving parent-relative targets."
                ),
                dense_constraint=True,
                output_prefix=args.body_output_prefix,
                prompt=text,
                sample_idx=sample_idx,
                seed=args.seed,
                fps=args.fps,
                out_dir=out_dir,
            )

        if scene_result is not None:
            assert scene_constraint is not None
            assert scene_result.initial_scene_joints is not None
            assert scene_result.scene_joints is not None
            render_scene_constraint_comparison(
                series=[
                    (
                        "unconstrained generation",
                        scene_result.initial_scene_joints[0, :decoded_length].detach().cpu(),
                    ),
                    (
                        "scene-constrained generation",
                        scene_result.scene_joints[0, :decoded_length].detach().cpu(),
                    ),
                ],
                constraint=scene_constraint,
                prompt=text,
                sample_idx=sample_idx,
                seed=args.seed,
                fps=args.fps,
                out_dir=out_dir,
            )

    if not trajectory_mode:
        return

    projected = project_root_trajectory(generated, target, length)
    constrained_vq = vq_project(vqvae, normalizer, projected.unsqueeze(0), frame_mask)[0, :length]

    series = [
        ("ground truth", real),
        ("generated", generated),
    ]
    if args.constraint_variant in {"projected", "both", "all"}:
        series.append(("constraint exact", projected))
    if args.constraint_variant in {"vq", "both", "all"}:
        series.append(("constraint + VQ", constrained_vq))
    joints_by_name = [(name, recover_joints_from_ric(feat.unsqueeze(0))[0].float().cpu()) for name, feat in series]
    limits = axis_limits([j for _, j in joints_by_name], target.cpu())

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
