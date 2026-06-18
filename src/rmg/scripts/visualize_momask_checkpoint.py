"""Visualize MoMask 263-D HumanML3D features from a smoke checkpoint."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from rmg.representation import PARENTS, recover_joints_from_ric


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", help="Path to momask_smoke_latest.pt")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--max-frames", type=int, default=120)
    p.add_argument("--format", choices=["gif", "mp4"], default="gif")
    p.add_argument(
        "--view",
        choices=["generated", "real", "reconstruction", "teacher_residual", "compare", "diagnostic"],
        default="generated",
        help=(
            "What to render. 'compare' shows real | reconstruction | generated; "
            "'diagnostic' also shows true-base/generated-residual."
        ),
    )
    return p.parse_args()


def _axis_limits(joints: torch.Tensor) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    mins = joints.amin(dim=tuple(range(joints.dim() - 1)))
    maxs = joints.amax(dim=tuple(range(joints.dim() - 1)))
    center = (mins + maxs) / 2
    radius = float((maxs - mins).max().clamp_min(1.0) / 2) * 1.15
    return tuple((float(c - radius), float(c + radius)) for c in center)  # type: ignore[return-value]


def _draw_frame(ax, joints, title: str, limits) -> None:
    ax.cla()
    for j, p in enumerate(PARENTS):
        if p < 0:
            continue
        xs = [joints[p, 0], joints[j, 0]]
        ys = [joints[p, 2], joints[j, 2]]
        zs = [joints[p, 1], joints[j, 1]]
        ax.plot(xs, ys, zs, color="#1f77b4", linewidth=2)
    ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1], color="#d62728", s=12)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    xlim, ylim, zlim = limits
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.view_init(elev=18, azim=-75)


def _motion_from_checkpoint(ckpt: dict, key: str) -> torch.Tensor:
    if key not in ckpt:
        raise KeyError(f"checkpoint does not contain {key!r}")
    motion = ckpt[key]
    if motion.dim() != 3 or motion.shape[-1] != 263:
        raise ValueError(f"{key} must be (B, T, 263), got {tuple(motion.shape)}")
    return motion


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    key_by_view = {
        "generated": "sample_generated",
        "real": "sample_real",
        "reconstruction": "sample_reconstruction",
        "teacher_residual": "sample_teacher_residual",
    }
    first_key = "sample_generated" if args.view in {"compare", "diagnostic"} else key_by_view[args.view]
    motion = _motion_from_checkpoint(ckpt, first_key)
    if not 0 <= args.sample < motion.shape[0]:
        raise ValueError(f"--sample must be in [0, {motion.shape[0] - 1}]")

    title_text = ""
    texts = ckpt.get("sample_texts")
    if isinstance(texts, list) and args.sample < len(texts):
        title_text = str(texts[args.sample])

    if args.view == "compare":
        series = [
            ("real", _motion_from_checkpoint(ckpt, "sample_real")),
            ("reconstruction", _motion_from_checkpoint(ckpt, "sample_reconstruction")),
            ("generated", _motion_from_checkpoint(ckpt, "sample_generated")),
        ]
    elif args.view == "diagnostic":
        series = [
            ("real", _motion_from_checkpoint(ckpt, "sample_real")),
            ("reconstruction", _motion_from_checkpoint(ckpt, "sample_reconstruction")),
            ("true base + generated residual", _motion_from_checkpoint(ckpt, "sample_teacher_residual")),
            ("generated", _motion_from_checkpoint(ckpt, "sample_generated")),
        ]
    else:
        series = [(args.view, motion)]

    joints_by_name = []
    for name, tensor in series:
        feat = tensor[args.sample, : args.max_frames]
        joints_by_name.append((name, recover_joints_from_ric(feat.unsqueeze(0))[0].float()))
    n_frames = min(j.shape[0] for _, j in joints_by_name)
    all_joints = torch.stack([j[:n_frames] for _, j in joints_by_name], dim=0)
    limits = _axis_limits(all_joints)

    fig = plt.figure(figsize=(5.5 * len(joints_by_name), 6))
    if title_text:
        fig.suptitle("\n".join(textwrap.wrap(title_text, width=110)), fontsize=11)
    axes = [fig.add_subplot(1, len(joints_by_name), i + 1, projection="3d") for i in range(len(joints_by_name))]

    def update(i: int):
        for ax, (name, joints) in zip(axes, joints_by_name):
            _draw_frame(ax, joints[i], f"{name} | frame {i}", limits)
        return []

    update(0)
    suffix = args.view
    png_path = out_dir / f"sample_{args.sample:02d}_{suffix}_frame0.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / args.fps, blit=False)
    if args.format == "gif":
        anim_path = out_dir / f"sample_{args.sample:02d}_{suffix}.gif"
        anim.save(anim_path, writer=PillowWriter(fps=args.fps))
    else:
        anim_path = out_dir / f"sample_{args.sample:02d}_{suffix}.mp4"
        anim.save(anim_path, fps=args.fps)
    plt.close(fig)

    print(f"[viz] wrote {png_path}")
    print(f"[viz] wrote {anim_path}")


if __name__ == "__main__":
    main()
