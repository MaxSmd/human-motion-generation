"""Evaluate inference-time trajectory, joint-position, and bend constraints.

This is a first constraint baseline, intentionally separate from the standard
FID evaluator:

  1. Generate unconstrained MoMask motion.
  2. Build sparse root-XZ anchors from the paired real test motion.
  3. Interpolate those anchors into a target trajectory.
  4. Project generated root velocity features to follow that target.
  5. Optionally re-encode/decode through the RVQ-VAE to pull the result back
     toward the learned motion-token manifold.
  6. Refine the same generated continuous RVQ latents against sparse joint,
     bend-angle, or dense parent-relative targets while keeping MoMask's weights
     frozen.

The paired real trajectory and pose are oracle targets used only for measuring
whether the constraint mechanism can satisfy known motion constraints.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    parent_relative_joint_errors,
    refine_motion_latents,
)
from momask.data_utils import normalize_motion, token_mask_from_frame_mask
from momask.scene_constraints import (
    RoomGeometryConstraint,
    SceneObstacle,
    scene_clearance_violations,
)
from shared.data import H3D263Dataset, collate
from shared.text import CLIPTextEncoder, RandomTextEncoder, TextEncoder
from shared.geometry import (
    H3D_FEATURE_DIM,
    NUM_JOINTS,
    PARENTS,
    quat_inv,
    quat_rotate,
    recover_joints_from_ric,
)
from shared.eval import RandomGuoEvaluator, RealGuoEvaluator, diversity, fid, mm_distance, r_precision


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
    p.add_argument("--data-root", default=os.environ.get("RMG_DATA_ROOT", "external/data/humanml3d_packed"))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--max-clips", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--min-seq-len", type=int, default=40)
    p.add_argument("--generation-steps", type=int, default=None)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--topk-filter-thres", type=float, default=1.0)
    p.add_argument("--sample", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--remask-kept-tokens", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--anchor-stride", type=int, default=20, help="Frames between root-XZ trajectory anchors.")
    p.add_argument(
        "--latent-variants",
        default="joint,angle",
        help="Comma-separated latent-refinement variants: joint, angle, body-fixed, scene, or none.",
    )
    p.add_argument(
        "--joint-ids",
        default="20,21",
        help="Comma-separated HumanML3D joint ids constrained at sparse anchors (default: both wrists).",
    )
    p.add_argument(
        "--joint-target-space",
        choices=["root-relative", "global"],
        default="root-relative",
        help="Align GT joint targets to the generated pelvis path, or keep canonical global coordinates.",
    )
    p.add_argument(
        "--angle-joints",
        default="4,5,18,19",
        help="Comma-separated bend centers (default: knees and elbows).",
    )
    p.add_argument(
        "--body-fixed-joint-ids",
        default="17,19,21",
        help="Dense parent-relative joints (default: right shoulder, elbow, and wrist).",
    )
    p.add_argument(
        "--body-reference-frame",
        type=int,
        default=0,
        help="Generated frame whose parent-to-joint vectors are held fixed.",
    )
    p.add_argument("--angle-tolerance-deg", type=float, default=5.0)
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
    p.add_argument("--text-encoder", choices=["checkpoint", "random", "clip"], default="checkpoint")
    p.add_argument("--clip-model", default=None)
    p.add_argument("--clip-cache-dir", default=None)
    p.add_argument("--clip-backend", choices=["auto", "openai", "transformers"], default="auto")
    p.add_argument("--evaluator", choices=["real", "random"], default="real")
    p.add_argument("--text-to-motion-repo", default="external/text-to-motion")
    p.add_argument("--humanml3d-repo", default="external/HumanML3D")
    p.add_argument(
        "--real-h3d-dir",
        default=None,
        help="Optional canonical HumanML3D new_joint_vecs dir for real features, targets, and canonical checkpoints.",
    )
    p.add_argument(
        "--model-input-source",
        choices=["auto", "packed", "canonical"],
        default="auto",
        help="Motion features used for token length/masks and constraint targets.",
    )
    p.add_argument(
        "--humanml3d-texts-zip",
        default=None,
        help="Optional direct path to HumanML3D/HumanML3D/texts.zip for VIP caption tokens.",
    )
    p.add_argument(
        "--vip-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use HumanML3D's original word/POS tokens from texts.zip for R-Precision/MM-Dist.",
    )
    p.add_argument("--diversity-times", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default=None)
    return p.parse_args()


def parse_csv_ints(value: str, *, name: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of integers") from exc
    if not result:
        raise ValueError(f"{name} must contain at least one value")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate values")
    return result


def parse_latent_variants(value: str) -> tuple[str, ...]:
    variants = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if variants == ("none",) or not variants:
        return ()
    unknown = set(variants) - {"joint", "angle", "body-fixed", "scene"}
    if unknown:
        raise ValueError(f"unknown latent constraint variants: {sorted(unknown)}")
    if len(set(variants)) != len(variants):
        raise ValueError("--latent-variants must not contain duplicates")
    return variants


def build_scene_constraint(args: argparse.Namespace) -> RoomGeometryConstraint:
    center = (
        args.scene_obstacle_x,
        args.scene_obstacle_y,
        args.scene_obstacle_z,
    )
    if args.scene_obstacle_kind == "box":
        obstacle = SceneObstacle.box(
            center=center,
            size=(
                args.scene_obstacle_width,
                args.scene_obstacle_height,
                args.scene_obstacle_depth,
            ),
            yaw_degrees=args.scene_obstacle_yaw,
        )
    elif args.scene_obstacle_kind == "sphere":
        obstacle = SceneObstacle.sphere(center=center, radius=args.scene_obstacle_radius)
    else:
        obstacle = SceneObstacle.cylinder(
            center=center,
            radius=args.scene_obstacle_radius,
            height=args.scene_obstacle_height,
        )
    return RoomGeometryConstraint(
        room_size=(
            args.scene_room_width,
            args.scene_room_depth,
            args.scene_room_height,
        ),
        obstacles=(obstacle,),
        spawn=(args.scene_spawn_x, args.scene_spawn_z, args.scene_spawn_yaw),
        padding=args.scene_padding,
        body_radius=args.scene_body_radius,
        bone_samples=args.scene_bone_samples,
    )


def angle_triplets_from_centers(centers: list[int], device: torch.device) -> Tensor:
    children: dict[int, list[int]] = {joint: [] for joint in range(NUM_JOINTS)}
    for child, parent in enumerate(PARENTS):
        if parent >= 0:
            children[parent].append(child)
    triplets: list[tuple[int, int, int]] = []
    for center in centers:
        if not 0 <= center < NUM_JOINTS:
            raise ValueError(f"angle joint id {center} is outside [0, {NUM_JOINTS - 1}]")
        parent = PARENTS[center]
        if parent < 0 or len(children[center]) != 1:
            raise ValueError(
                f"angle joint {center} must have exactly one parent and one child; "
                f"found parent={parent}, children={children[center]}"
            )
        triplets.append((parent, center, children[center][0]))
    return torch.tensor(triplets, dtype=torch.long, device=device)


def torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def require_path(path: str | Path, what: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


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


def build_evaluator(args: argparse.Namespace, device: torch.device):
    if args.evaluator == "random":
        return RandomGuoEvaluator()
    require_path(Path(args.text_to_motion_repo) / "networks" / "evaluator_wrapper.py", "text-to-motion evaluator wrapper")
    require_path(
        Path(args.text_to_motion_repo) / "checkpoints" / "t2m" / "text_mot_match" / "model" / "finest.tar",
        "Guo evaluator checkpoint",
    )
    return RealGuoEvaluator(
        text_to_motion_repo=args.text_to_motion_repo,
        humanml3d_repo=args.humanml3d_repo,
        device=device,
    )


def load_caption_tokens(texts_zip: str | Path) -> dict[str, dict[str, list[str]]]:
    """clip_id -> {caption_text: [word/POS tokens]} from HumanML3D's texts.zip."""
    zp = Path(texts_zip)
    if not zp.exists():
        raise FileNotFoundError(f"HumanML3D texts.zip not found: {zp}")
    out: dict[str, dict[str, list[str]]] = {}
    with zipfile.ZipFile(zp) as zf:
        for name in zf.namelist():
            if not name.endswith(".txt"):
                continue
            cid = Path(name).stem
            cap_to_tokens: dict[str, list[str]] = {}
            for line in zf.read(name).decode("utf-8").splitlines():
                parts = line.strip().split("#")
                if len(parts) < 2:
                    continue
                cap = parts[0].strip()
                tokens = parts[1].strip().split()
                if cap and tokens:
                    cap_to_tokens[cap] = tokens
            if cap_to_tokens:
                out[cid] = cap_to_tokens
    print(f"[constraints] loaded VIP caption tokens for {len(out)} clips from {zp}", flush=True)
    return out


def resolve_texts_zip(args: argparse.Namespace) -> Path:
    if args.humanml3d_texts_zip:
        return Path(args.humanml3d_texts_zip)
    repo = Path(args.humanml3d_repo)
    candidates = [
        repo / "HumanML3D" / "texts.zip",
        repo / "texts.zip",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def encode_text_batch(
    evaluator,
    texts: list[str],
    clip_ids: list[str],
    caption_tokens: dict[str, dict[str, list[str]]] | None,
) -> tuple[np.ndarray, int]:
    if caption_tokens is None:
        return evaluator.encode_text_from_strings(texts).cpu().numpy(), 0
    tokens = [caption_tokens.get(cid, {}).get(text) for cid, text in zip(clip_ids, texts)]
    missing_idxs = [i for i, tok in enumerate(tokens) if tok is None]
    present_idxs = [i for i, tok in enumerate(tokens) if tok is not None]
    out = np.empty((len(texts), evaluator.text_dim), dtype=np.float32)
    if present_idxs:
        present_tokens = [tokens[i] for i in present_idxs]
        out[present_idxs] = evaluator.encode_text_from_tokens(present_tokens).cpu().numpy()
    if missing_idxs:
        missing_texts = [texts[i] for i in missing_idxs]
        out[missing_idxs] = evaluator.encode_text_from_strings(missing_texts).cpu().numpy()
    return out, len(missing_idxs)


def load_canonical_motion_batch(
    h3d_dir: str | Path,
    clip_ids: list[str],
    max_len: int,
) -> tuple[Tensor, Tensor, int]:
    root = Path(h3d_dir)
    feats: list[Tensor] = []
    missing = 0
    for cid in clip_ids:
        candidates = [cid]
        if cid.startswith("M") and len(cid) > 1:
            candidates.append(cid[1:])
        path = next((root / f"{name}.npy" for name in candidates if (root / f"{name}.npy").exists()), None)
        if path is None:
            missing += 1
            feats.append(torch.zeros(1, H3D_FEATURE_DIM))
            continue
        arr = np.load(path).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != H3D_FEATURE_DIM:
            raise ValueError(f"canonical HumanML3D feature must be (T, {H3D_FEATURE_DIM}), got {arr.shape}: {path}")
        arr = arr[:max_len]
        if len(arr) < 1:
            missing += 1
            feats.append(torch.zeros(1, H3D_FEATURE_DIM))
            continue
        feats.append(torch.from_numpy(arr))

    tmax = max(f.shape[0] for f in feats)
    out = torch.zeros(len(feats), tmax, H3D_FEATURE_DIM)
    lengths = torch.zeros(len(feats), dtype=torch.long)
    for i, feat in enumerate(feats):
        out[i, : feat.shape[0]] = feat
        lengths[i] = feat.shape[0]
    return out, lengths, missing


def root_xz(motion: Tensor) -> Tensor:
    return recover_joints_from_ric(motion.float())[:, :, 0, :][:, :, [0, 2]]


def root_quat_from_features(motion: Tensor) -> Tensor:
    rot_vel = motion[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)
    quat = torch.zeros(*motion.shape[:-1], 4, device=motion.device, dtype=motion.dtype)
    quat[..., 0] = torch.cos(r_rot_ang)
    quat[..., 2] = torch.sin(r_rot_ang)
    return quat


def interpolate_anchor_trajectory(real_root: Tensor, lengths: Tensor, anchor_stride: int) -> tuple[Tensor, Tensor]:
    """Return dense target root-XZ path and sparse anchor mask."""
    B, T, _ = real_root.shape
    target = real_root.clone()
    anchor_mask = torch.zeros(B, T, dtype=torch.bool, device=real_root.device)
    for b in range(B):
        L = int(lengths[b].item())
        if L <= 0:
            continue
        anchors = list(range(0, L, max(1, anchor_stride)))
        if anchors[-1] != L - 1:
            anchors.append(L - 1)
        anchor_mask[b, anchors] = True
        for left, right in zip(anchors[:-1], anchors[1:]):
            span = max(1, right - left)
            alpha = torch.linspace(0, 1, span + 1, device=real_root.device, dtype=real_root.dtype).unsqueeze(-1)
            target[b, left : right + 1] = (1 - alpha) * real_root[b, left] + alpha * real_root[b, right]
    return target, anchor_mask


def project_root_trajectory(motion: Tensor, target_root: Tensor, lengths: Tensor) -> Tensor:
    """Edit HumanML3D root velocity channels so recovered root follows target."""
    out = motion.clone()
    quat = root_quat_from_features(out)
    target3 = torch.zeros(*target_root.shape[:-1], 3, device=out.device, dtype=out.dtype)
    target3[..., 0] = target_root[..., 0]
    target3[..., 2] = target_root[..., 1]
    # HumanML3D stores root XZ velocity rotated by the destination frame's
    # root orientation, matching `_features_from_positions_and_quats`.
    local_delta = quat_rotate(quat[:, 1:], target3[:, 1:] - target3[:, :-1])
    for b in range(out.shape[0]):
        L = int(lengths[b].item())
        if L <= 1:
            continue
        out[b, : L - 1, 1] = local_delta[b, : L - 1, 0]
        out[b, : L - 1, 2] = local_delta[b, : L - 1, 2]
        out[b, :L, 3] = motion[b, :L, 3]
    return out


def trajectory_errors(motion: Tensor, target_root: Tensor, anchor_mask: Tensor, lengths: Tensor) -> dict[str, float]:
    pred = root_xz(motion)
    valid = torch.zeros_like(anchor_mask)
    for b, L in enumerate(lengths.tolist()):
        valid[b, : int(L)] = True
    dense_err = ((pred - target_root).norm(dim=-1)[valid]).mean()
    anchor_valid = anchor_mask & valid
    anchor_err = ((pred - target_root).norm(dim=-1)[anchor_valid]).mean()
    final_errs = []
    for b, L in enumerate(lengths.tolist()):
        L = int(L)
        if L > 0:
            final_errs.append((pred[b, L - 1] - target_root[b, L - 1]).norm())
    final_err = torch.stack(final_errs).mean() if final_errs else dense_err.new_tensor(float("nan"))
    return {
        "traj_dense_l2": float(dense_err.detach().cpu()),
        "traj_anchor_l2": float(anchor_err.detach().cpu()),
        "traj_final_l2": float(final_err.detach().cpu()),
    }


def build_joint_constraint(
    real_motion: Tensor,
    generated_motion: Tensor,
    real_joints: Tensor,
    generated_joints: Tensor,
    anchor_mask: Tensor,
    joint_ids: list[int],
    *,
    target_space: str,
) -> JointPositionConstraint:
    for joint_id in joint_ids:
        if not 0 <= joint_id < NUM_JOINTS:
            raise ValueError(f"joint id {joint_id} is outside [0, {NUM_JOINTS - 1}]")
    targets = real_joints.clone()
    if target_space == "root-relative":
        # Convert GT pelvis-relative offsets back into the root-facing frame,
        # then rotate them into the generated heading and place them on the
        # generated pelvis path. Translation alone is wrong when headings differ.
        real_root_quat = root_quat_from_features(real_motion)
        generated_root_quat = root_quat_from_features(generated_motion)
        real_to_root = real_root_quat.unsqueeze(-2).expand(-1, -1, NUM_JOINTS, -1)
        root_to_generated = quat_inv(generated_root_quat).unsqueeze(-2).expand_as(real_to_root)
        root_facing_offsets = quat_rotate(real_to_root, real_joints - real_joints[:, :, :1])
        targets = generated_joints[:, :, :1] + quat_rotate(
            root_to_generated,
            root_facing_offsets,
        )
    elif target_space != "global":
        raise ValueError(f"unknown joint target space: {target_space}")
    mask = torch.zeros(
        *real_joints.shape[:-1], dtype=torch.bool, device=real_joints.device
    )
    mask[:, :, joint_ids] = anchor_mask.unsqueeze(-1)
    return JointPositionConstraint(targets=targets, mask=mask)


def build_angle_constraint(
    real_joints: Tensor,
    anchor_mask: Tensor,
    triplets: Tensor,
    tolerance_degrees: float,
) -> tuple[BendAngleConstraint, Tensor]:
    if not 0.0 <= tolerance_degrees < 180.0:
        raise ValueError("--angle-tolerance-deg must lie in [0, 180)")
    target = bend_angles_from_joints(real_joints, triplets)
    tolerance = math.radians(tolerance_degrees)
    mask = anchor_mask.unsqueeze(-1).expand_as(target)
    return (
        BendAngleConstraint(
            triplets=triplets,
            mask=mask,
            min_radians=(target - tolerance).clamp_min(0.0),
            max_radians=(target + tolerance).clamp_max(math.pi),
        ),
        target,
    )


@torch.no_grad()
def control_statistics(
    joints: Tensor,
    *,
    position_constraint: JointPositionConstraint | None,
    parent_relative_constraint: ParentRelativeJointConstraint | None,
    angle_constraint: BendAngleConstraint | None,
    angle_target: Tensor | None,
    angle_tolerance_degrees: float,
) -> dict[str, float]:
    stats: dict[str, float] = {}
    if position_constraint is not None:
        active = position_constraint.mask.to(joints.device, torch.bool)
        delta = joints - position_constraint.targets.to(joints.device, joints.dtype)
        if position_constraint.axis_mask is not None:
            axis_mask = position_constraint.axis_mask.to(joints.device, torch.bool)
            delta = delta * axis_mask.to(delta.dtype)
            active = active & axis_mask.any(dim=-1)
        errors = torch.linalg.vector_norm(delta, dim=-1)[active]
        stats["joint_error_sum_m"] = float(errors.sum().cpu())
        stats["joint_within_5cm_count"] = float((errors <= 0.05).sum().cpu())
        stats["joint_within_10cm_count"] = float((errors <= 0.10).sum().cpu())
        stats["joint_constraint_count"] = float(errors.numel())
    if parent_relative_constraint is not None:
        errors = parent_relative_joint_errors(joints, parent_relative_constraint)
        stats["parent_relative_error_sum_m"] = float(errors.sum().cpu())
        stats["parent_relative_within_5cm_count"] = float((errors <= 0.05).sum().cpu())
        stats["parent_relative_within_10cm_count"] = float((errors <= 0.10).sum().cpu())
        stats["parent_relative_constraint_count"] = float(errors.numel())
    if angle_constraint is not None:
        if angle_target is None:
            raise ValueError("angle_target is required with an angle constraint")
        active = angle_constraint.mask.to(joints.device, torch.bool)
        angles = bend_angles_from_joints(joints, angle_constraint.triplets)
        errors_degrees = torch.rad2deg(
            (angles - angle_target.to(joints.device, joints.dtype)).abs()
        )[active]
        stats["angle_error_sum_deg"] = float(errors_degrees.sum().cpu())
        stats["angle_within_tolerance_count"] = float(
            (errors_degrees <= angle_tolerance_degrees + 1e-5).sum().cpu()
        )
        stats["angle_constraint_count"] = float(errors_degrees.numel())
    return stats


def accumulate_statistics(total: dict[str, float], current: dict[str, float]) -> None:
    for key, value in current.items():
        total[key] = total.get(key, 0.0) + value


def summarize_control_statistics(stats: dict[str, float]) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    joint_count = stats.get("joint_constraint_count", 0.0)
    if joint_count:
        result.update(
            {
                "joint_position_l2_m": stats["joint_error_sum_m"] / joint_count,
                "joint_success_5cm": stats["joint_within_5cm_count"] / joint_count,
                "joint_success_10cm": stats["joint_within_10cm_count"] / joint_count,
                "joint_constraint_count": int(joint_count),
            }
        )
    angle_count = stats.get("angle_constraint_count", 0.0)
    if angle_count:
        result.update(
            {
                "angle_abs_error_deg": stats["angle_error_sum_deg"] / angle_count,
                "angle_within_tolerance": stats["angle_within_tolerance_count"] / angle_count,
                "angle_constraint_count": int(angle_count),
            }
        )
    parent_relative_count = stats.get("parent_relative_constraint_count", 0.0)
    if parent_relative_count:
        result.update(
            {
                "parent_relative_l2_m": stats["parent_relative_error_sum_m"]
                / parent_relative_count,
                "parent_relative_success_5cm": stats["parent_relative_within_5cm_count"]
                / parent_relative_count,
                "parent_relative_success_10cm": stats["parent_relative_within_10cm_count"]
                / parent_relative_count,
                "parent_relative_constraint_count": int(parent_relative_count),
            }
        )
    return result


def accumulate_scene_statistics(
    total: dict[str, float],
    *,
    joints_world: Tensor,
    constraint: RoomGeometryConstraint,
    frame_mask: Tensor,
) -> None:
    violation = scene_clearance_violations(joints_world, constraint)
    valid_frames = frame_mask.to(device=violation.device, dtype=torch.bool)
    if valid_frames.shape != violation.shape[:2]:
        raise ValueError("scene frame mask must match scene joints")
    valid_points = valid_frames.unsqueeze(-1).expand_as(violation)
    values = violation[valid_points]
    positive = values > 0.0
    colliding_frames = (violation > 0.0).any(dim=-1) & valid_frames
    total["scene_max_violation_m"] = max(
        total.get("scene_max_violation_m", 0.0),
        float(values.max().cpu()) if values.numel() else 0.0,
    )
    total["scene_violation_sum_m"] = total.get("scene_violation_sum_m", 0.0) + float(
        values[positive].sum().cpu()
    )
    total["scene_violating_point_count"] = total.get(
        "scene_violating_point_count", 0.0
    ) + float(positive.sum().cpu())
    total["scene_valid_point_count"] = total.get("scene_valid_point_count", 0.0) + float(
        values.numel()
    )
    total["scene_colliding_frame_count"] = total.get(
        "scene_colliding_frame_count", 0.0
    ) + float(colliding_frames.sum().cpu())
    total["scene_valid_frame_count"] = total.get("scene_valid_frame_count", 0.0) + float(
        valid_frames.sum().cpu()
    )


def summarize_scene_statistics(stats: dict[str, float]) -> dict[str, float]:
    valid_points = stats.get("scene_valid_point_count", 0.0)
    valid_frames = stats.get("scene_valid_frame_count", 0.0)
    if valid_points <= 0.0 or valid_frames <= 0.0:
        return {}
    return {
        "scene_max_violation_m": stats.get("scene_max_violation_m", 0.0),
        "scene_mean_violation_m": stats.get("scene_violation_sum_m", 0.0)
        / max(stats.get("scene_violating_point_count", 0.0), 1.0),
        "scene_violating_point_fraction": stats.get("scene_violating_point_count", 0.0)
        / valid_points,
        "scene_colliding_frame_fraction": stats.get("scene_colliding_frame_count", 0.0)
        / valid_frames,
    }


@torch.no_grad()
def generate_full(
    *,
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
    sample: bool,
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
        sample=sample,
        remask_kept_tokens=remask_kept_tokens,
        mask=token_mask,
    )
    tokens = residual.generate_residuals(
        base,
        cond=cond,
        guidance_scale=guidance_scale,
        temperature=temperature,
        topk_filter_thres=topk_filter_thres,
        sample=sample,
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


def compute_quality(real: np.ndarray, gen: np.ndarray, text: np.ndarray, diversity_times: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "fid": float(fid(real, gen)),
        "r_precision": r_precision(text, gen, top_k=3, rng=rng).tolist(),
        "mm_dist": float(mm_distance(text, gen)),
        "diversity": float(diversity(gen, diversity_times=diversity_times, rng=rng)),
        "num_clips": int(gen.shape[0]),
    }


def encode_motion(evaluator, motion: Tensor, lengths: Tensor) -> np.ndarray:
    return evaluator.encode_motion(motion.cpu(), lengths.cpu()).cpu().numpy()


def main() -> None:
    args = parse_args()
    if args.parent_relative_weight is None:
        args.parent_relative_weight = args.torso_relative_weight
    latent_variants = parse_latent_variants(args.latent_variants)
    joint_ids = (
        parse_csv_ints(args.joint_ids, name="--joint-ids") if "joint" in latent_variants else []
    )
    angle_centers = (
        parse_csv_ints(args.angle_joints, name="--angle-joints")
        if "angle" in latent_variants
        else []
    )
    body_fixed_joint_ids = (
        parse_csv_ints(args.body_fixed_joint_ids, name="--body-fixed-joint-ids")
        if "body-fixed" in latent_variants
        else []
    )
    scene_constraint = build_scene_constraint(args) if "scene" in latent_variants else None
    if args.anchor_stride < 1:
        raise ValueError("--anchor-stride must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    ckpt = torch_load(require_path(args.checkpoint, "MoMask checkpoint"), map_location=device)
    saved_args = ckpt_args(ckpt)
    normalizer = H3DNormalizer.from_state_dict(ckpt["normalizer"])
    real_h3d_dir = Path(args.real_h3d_dir) if args.real_h3d_dir else None
    if real_h3d_dir is not None:
        require_path(real_h3d_dir, "canonical HumanML3D new_joint_vecs dir")
    model_input_source = args.model_input_source
    if model_input_source == "auto":
        model_input_source = "canonical" if saved_args.get("canonical_h3d_dir") and real_h3d_dir is not None else "packed"
    if model_input_source == "canonical" and real_h3d_dir is None:
        raise ValueError("--model-input-source canonical requires --real-h3d-dir")
    steps = int(args.generation_steps or saved_args.get("generation_steps", 10))
    max_seq_len = int(args.max_seq_len or saved_args.get("max_seq_len", 80))
    text_encoder = build_text_encoder(args, saved_args)
    vqvae, masked, residual = build_models(ckpt, device)
    angle_triplets = (
        angle_triplets_from_centers(angle_centers, device)
        if angle_centers
        else torch.empty(0, 3, dtype=torch.long, device=device)
    )
    evaluator = build_evaluator(args, device)
    caption_tokens = None
    if args.evaluator == "real" and args.vip_tokens:
        texts_zip = resolve_texts_zip(args)
        try:
            caption_tokens = load_caption_tokens(texts_zip)
        except FileNotFoundError as e:
            print(f"[constraints] WARN: {e}; falling back to spaCy text encoding", flush=True)

    ds = H3D263Dataset(
        root=args.data_root,
        split=args.split,
        max_seq_len=max_seq_len,
        min_seq_len=args.min_seq_len,
        mirror_augment=False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    variant_names = ["unconstrained", "traj_projected", "traj_vq"]
    if "joint" in latent_variants:
        variant_names.append("joint_latent")
    if "angle" in latent_variants:
        variant_names.append("angle_latent")
    if "body-fixed" in latent_variants:
        variant_names.append("body_fixed_latent")
    if "scene" in latent_variants:
        variant_names.append("scene_latent")
    buckets: dict[str, list[np.ndarray]] = {name: [] for name in ["real", *variant_names]}
    text_embs: list[np.ndarray] = []
    traj_sums: dict[str, dict[str, float]] = {name: {} for name in variant_names}
    control_sums: dict[str, dict[str, float]] = {name: {} for name in variant_names}
    n_seen = 0
    n_text_fallback = 0
    t0 = time.perf_counter()

    print(
        f"[constraints] ckpt={args.checkpoint} clips={len(ds)} max_clips={args.max_clips} "
        f"steps={steps} guidance={args.guidance_scale} anchor_stride={args.anchor_stride} "
        f"text_tokens={'vip' if caption_tokens is not None else 'spacy'} "
        f"real_features={'canonical' if real_h3d_dir is not None else 'packed'} "
        f"model_input={model_input_source} sample={args.sample} topk={args.topk_filter_thres}",
        flush=True,
    )
    if latent_variants:
        print(
            f"[constraints] latent_variants={','.join(latent_variants)} "
            f"joint_ids={joint_ids} joint_space={args.joint_target_space} "
            f"angle_centers={angle_centers} angle_tolerance={args.angle_tolerance_deg:g}deg "
            f"body_fixed_joint_ids={body_fixed_joint_ids} "
            f"body_reference_frame={args.body_reference_frame} "
            f"refinement_steps={args.refinement_steps} refinement_lr={args.refinement_lr:g}",
            flush=True,
        )
        if scene_constraint is not None:
            print(f"[constraints] scene={scene_constraint.as_dict()}", flush=True)

    for batch in tqdm(loader, desc="evaluate constrained MoMask"):
        take = batch.x1.shape[0]
        if args.max_clips > 0:
            take = min(take, args.max_clips - n_seen)
            if take <= 0:
                break
        real_x = batch.x1[:take].to(device)
        frame_mask = batch.mask[:take].to(device)
        lengths = batch.lengths[:take]
        texts = batch.texts[:take]
        clip_ids = batch.clip_ids[:take]
        cond = text_encoder.encode(texts, device=device)

        eval_real_x = real_x
        eval_lengths = lengths
        if real_h3d_dir is not None:
            eval_real_x, eval_lengths, n_missing_real = load_canonical_motion_batch(
                real_h3d_dir,
                clip_ids,
                max_len=real_x.shape[1],
            )
            if n_missing_real:
                raise FileNotFoundError(
                    f"{n_missing_real} canonical HumanML3D feature files missing under {real_h3d_dir}"
                )

        model_real_x = eval_real_x.to(device) if model_input_source == "canonical" else real_x
        model_lengths = eval_lengths if model_input_source == "canonical" else lengths
        model_frame_mask = torch.arange(model_real_x.shape[1], device=device).unsqueeze(0) < model_lengths.to(device).unsqueeze(1)

        gen, generated_tokens, token_mask = generate_full(
            vqvae=vqvae,
            masked=masked,
            residual=residual,
            normalizer=normalizer,
            cond=cond,
            real_x=model_real_x,
            frame_mask=model_frame_mask,
            steps=steps,
            guidance_scale=args.guidance_scale,
            temperature=args.temperature,
            topk_filter_thres=args.topk_filter_thres,
            sample=args.sample,
            remask_kept_tokens=args.remask_kept_tokens,
        )
        real_root = root_xz(model_real_x)
        target_root, anchor_mask = interpolate_anchor_trajectory(real_root, model_lengths.to(device), args.anchor_stride)
        decoded_lengths = (token_mask.sum(dim=1) * vqvae.downsample).clamp(
            max=model_real_x.shape[1]
        )
        _, latent_anchor_mask = interpolate_anchor_trajectory(
            real_root,
            decoded_lengths,
            args.anchor_stride,
        )
        projected = project_root_trajectory(gen, target_root, model_lengths.to(device))
        traj_vq = vq_project(vqvae, normalizer, projected, model_frame_mask)
        motions: dict[str, Tensor] = {
            "unconstrained": gen,
            "traj_projected": projected,
            "traj_vq": traj_vq,
        }

        position_constraint: JointPositionConstraint | None = None
        parent_relative_constraint: ParentRelativeJointConstraint | None = None
        angle_constraint: BendAngleConstraint | None = None
        angle_target: Tensor | None = None
        scene_result = None
        if latent_variants:
            with torch.no_grad():
                real_joints = recover_joints_from_ric(model_real_x.float())
                generated_joints = recover_joints_from_ric(gen.float())
                initial_latents = vqvae.quantizer.decode(generated_tokens)
            decoded_frame_mask = model_frame_mask & (
                torch.arange(model_real_x.shape[1], device=device).unsqueeze(0)
                < decoded_lengths.unsqueeze(1)
            )
            if "joint" in latent_variants:
                position_constraint = build_joint_constraint(
                    model_real_x,
                    gen,
                    real_joints,
                    generated_joints,
                    latent_anchor_mask,
                    joint_ids,
                    target_space=args.joint_target_space,
                )
                joint_result = refine_motion_latents(
                    vqvae,
                    initial_latents,
                    mean=normalizer.mean,
                    std=normalizer.std,
                    target_len=model_real_x.shape[1],
                    token_mask=token_mask,
                    frame_mask=model_frame_mask,
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
                motions["joint_latent"] = joint_result.motion
            if "angle" in latent_variants:
                angle_constraint, angle_target = build_angle_constraint(
                    real_joints,
                    latent_anchor_mask,
                    angle_triplets,
                    args.angle_tolerance_deg,
                )
                angle_result = refine_motion_latents(
                    vqvae,
                    initial_latents,
                    mean=normalizer.mean,
                    std=normalizer.std,
                    target_len=model_real_x.shape[1],
                    token_mask=token_mask,
                    frame_mask=model_frame_mask,
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
                motions["angle_latent"] = angle_result.motion
            if "body-fixed" in latent_variants:
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
                    target_len=model_real_x.shape[1],
                    token_mask=token_mask,
                    frame_mask=model_frame_mask,
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
                motions["body_fixed_latent"] = body_fixed_result.motion
            if "scene" in latent_variants:
                assert scene_constraint is not None
                scene_result = refine_motion_latents(
                    vqvae,
                    initial_latents,
                    mean=normalizer.mean,
                    std=normalizer.std,
                    target_len=model_real_x.shape[1],
                    token_mask=token_mask,
                    frame_mask=model_frame_mask,
                    scene_constraint=scene_constraint,
                    config=LatentRefinementConfig(
                        steps=args.refinement_steps,
                        learning_rate=args.refinement_lr,
                        position_weight=0.0,
                        torso_relative_weight=0.0,
                        parent_relative_weight=0.0,
                        angle_weight=0.0,
                        scene_weight=args.scene_weight,
                        latent_weight=args.latent_weight,
                        dynamics_weight=args.dynamics_weight,
                        root_weight=args.scene_root_weight,
                        bone_weight=args.bone_weight,
                        max_delta_norm=args.max_delta_norm,
                        grad_clip_norm=args.grad_clip_norm,
                    ),
                )
                motions["scene_latent"] = scene_result.motion
                assert scene_result.initial_scene_joints is not None
                assert scene_result.scene_joints is not None
                accumulate_scene_statistics(
                    control_sums["unconstrained"],
                    joints_world=scene_result.initial_scene_joints,
                    constraint=scene_constraint,
                    frame_mask=decoded_frame_mask,
                )
                accumulate_scene_statistics(
                    control_sums["scene_latent"],
                    joints_world=scene_result.scene_joints,
                    constraint=scene_constraint,
                    frame_mask=decoded_frame_mask,
                )

        for name, motion in motions.items():
            errs = trajectory_errors(motion, target_root, anchor_mask, model_lengths.to(device))
            for key, value in errs.items():
                traj_sums[name][key] = traj_sums[name].get(key, 0.0) + value * take
            if name == "unconstrained" or name.endswith("_latent"):
                accumulate_statistics(
                    control_sums[name],
                    control_statistics(
                        recover_joints_from_ric(motion.float()),
                        position_constraint=position_constraint,
                        parent_relative_constraint=parent_relative_constraint,
                        angle_constraint=angle_constraint,
                        angle_target=angle_target,
                        angle_tolerance_degrees=args.angle_tolerance_deg,
                    ),
                )

        buckets["real"].append(encode_motion(evaluator, eval_real_x, eval_lengths))
        for name, motion in motions.items():
            buckets[name].append(encode_motion(evaluator, motion, eval_lengths))
        text_np, n_missing = encode_text_batch(evaluator, texts, clip_ids, caption_tokens)
        text_embs.append(text_np)
        n_text_fallback += n_missing

        n_seen += take
        if args.max_clips > 0 and n_seen >= args.max_clips:
            break

    real = np.concatenate(buckets["real"], axis=0)
    text = np.concatenate(text_embs, axis=0)
    diag_rng = np.random.default_rng(args.seed)
    results = {
        "_meta": {
            "checkpoint": args.checkpoint,
            "split": args.split,
            "max_clips": args.max_clips,
            "num_clips": int(real.shape[0]),
            "max_seq_len": max_seq_len,
            "generation_steps": steps,
            "guidance_scale": args.guidance_scale,
            "temperature": args.temperature,
            "topk_filter_thres": args.topk_filter_thres,
            "sample": args.sample,
            "remask_kept_tokens": args.remask_kept_tokens,
            "anchor_stride": args.anchor_stride,
            "latent_variants": list(latent_variants),
            "joint_ids": joint_ids if "joint" in latent_variants else [],
            "joint_target_space": args.joint_target_space,
            "body_fixed_joint_ids": body_fixed_joint_ids
            if "body-fixed" in latent_variants
            else [],
            "body_fixed_edges": [[PARENTS[joint], joint] for joint in body_fixed_joint_ids]
            if "body-fixed" in latent_variants
            else [],
            "body_reference_frame": args.body_reference_frame,
            "body_fixed_constraint_space": "parent_bone_in_moving_torso_frame"
            if "body-fixed" in latent_variants
            else None,
            "scene": scene_constraint.as_dict() if scene_constraint is not None else None,
            "angle_centers": angle_centers if "angle" in latent_variants else [],
            "angle_triplets": angle_triplets.detach().cpu().tolist()
            if "angle" in latent_variants
            else [],
            "angle_tolerance_deg": args.angle_tolerance_deg,
            "angle_definition": "unsigned bend: 0 degrees straight, 180 degrees folded",
            "refinement": {
                "steps": args.refinement_steps,
                "learning_rate": args.refinement_lr,
                "position_weight": args.position_weight,
                "parent_relative_weight": args.parent_relative_weight,
                "angle_weight": args.angle_weight,
                "latent_weight": args.latent_weight,
                "dynamics_weight": args.dynamics_weight,
                "root_weight": args.root_weight,
                "bone_weight": args.bone_weight,
                "scene_weight": args.scene_weight,
                "scene_root_weight": args.scene_root_weight,
                "max_delta_norm": args.max_delta_norm,
                "grad_clip_norm": args.grad_clip_norm,
            },
            "real_feature_source": "canonical" if real_h3d_dir is not None else "packed",
            "real_h3d_dir": str(real_h3d_dir) if real_h3d_dir is not None else None,
            "model_input_source": model_input_source,
            "text_token_source": "vip" if caption_tokens is not None else "spacy",
            "vip_token_fallbacks": int(n_text_fallback),
            "elapsed_sec": time.perf_counter() - t0,
        },
        "_diagnostics": {
            "r_precision_real": r_precision(text, real, top_k=3, rng=diag_rng).tolist(),
            "mm_dist_real": float(mm_distance(text, real)),
            "diversity_real": float(diversity(real, diversity_times=args.diversity_times, rng=diag_rng)),
        },
    }
    if caption_tokens is not None:
        print(f"[constraints] VIP token fallbacks={n_text_fallback}", flush=True)
    print(f"\n[diagnostics]\n{json.dumps(results['_diagnostics'], indent=2)}", flush=True)
    for name in variant_names:
        emb = np.concatenate(buckets[name], axis=0)
        quality = compute_quality(real, emb, text, args.diversity_times, args.seed)
        traj = {k: v / max(n_seen, 1) for k, v in traj_sums[name].items()}
        control = summarize_control_statistics(control_sums[name])
        scene = summarize_scene_statistics(control_sums[name])
        results[name] = {**quality, **traj, **control, **scene}
        print(f"\n[{name}]\n{json.dumps(results[name], indent=2)}", flush=True)

    checkpoint_dir = Path(args.checkpoint).parent
    run_dir = checkpoint_dir.parent if checkpoint_dir.name == "checkpoints" else checkpoint_dir
    output = (
        Path(args.output)
        if args.output
        else run_dir / "constraints" / "trajectory_joint_angle.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[constraints] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
