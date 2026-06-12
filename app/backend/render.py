"""In-process motion rendering: (joints) → media file on disk.

Refactored out of `src/rmg/scripts/visualize.py::_render` so the backend can render
without the Hydra/CLI wrapper. Renders (T, 22, 3) joint positions to an MP4
(when a system ffmpeg is available) or a GIF (Pillow, always available), and
always dumps the raw joints as `.npy` next to the media for re-render/debug.

Also provides `decode_to_joints` — flat manifold sample → (T, 22, 3) world
joints via the representation's rotations + forward kinematics — which is the
common "decode → FK" step every feature runs before rendering.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

# Non-interactive backend BEFORE any pyplot import.
import matplotlib

matplotlib.use("Agg")

from rmg.representation import Skeleton, forward_kinematics  # noqa: E402
from rmg.representation.tplusr import decode as tplusr_decode  # noqa: E402

# HumanML3D 22-joint kinematic chains (upstream's t2m_kinematic_chain).
_T2M_CHAINS: tuple[tuple[int, ...], ...] = (
    (0, 2, 5, 8, 11),       # right leg
    (0, 1, 4, 7, 10),       # left leg
    (0, 3, 6, 9, 12, 15),   # spine + head
    (9, 14, 17, 19, 21),    # right arm
    (9, 13, 16, 18, 20),    # left arm
)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def resolve_format(fmt: str = "auto") -> str:
    """'auto' → mp4 if ffmpeg present else gif. Otherwise honour the request,
    downgrading mp4→gif when ffmpeg is missing."""
    if fmt == "auto":
        return "mp4" if ffmpeg_available() else "gif"
    if fmt == "mp4" and not ffmpeg_available():
        return "gif"
    return fmt


def decode_to_joints(
    flat: Tensor,                       # (T, ambient_dim)
    skeleton: Skeleton,
    representation_name: str = "tr",
) -> np.ndarray:                        # (T, 22, 3) float32
    """Decode a flat manifold sample to world-space joint positions.

    T+R / T+R+P carry per-joint quaternions in dims [3 : 3 + 4*J]; we take the
    translation + rotations and run forward kinematics. (T+P has no rotations;
    that path would need pre-shape rescaling — not wired here yet.)
    """
    if representation_name not in ("tr", "trp"):
        # PLACEHOLDER: T+P (pre-shape) decode goes through a different recovery
        # path (see TPRepresentation.to_h3d_features). All current runs are T+R.
        raise NotImplementedError(
            f"decode_to_joints not implemented for representation {representation_name!r}; "
            "only 'tr'/'trp' (rotation-based) are wired."
        )
    tpr = tplusr_decode(flat.float())  # slices [:3] and [3:3+4*J] internally
    joints = forward_kinematics(
        skeleton, tpr.quaternions.float(), tpr.translation.float()
    )
    return joints.cpu().numpy().astype(np.float32)


def render_joints(
    joints: np.ndarray,                 # (T, 22, 3)
    out_path: Path,                     # media path; suffix is forced to match fmt
    title: str = "",
    fps: int = 20,
    fmt: str = "auto",
) -> tuple[Path, Path]:
    """Render joints to MP4/GIF and dump joints as `.npy`. Returns (media, npy)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = resolve_format(fmt)

    npy_path = out_path.with_suffix(".npy")
    np.save(npy_path, joints.astype(np.float32))

    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    T = joints.shape[0]
    pts = joints.reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = (lo + hi) / 2
    radius = float(np.max(hi - lo)) / 2 * 1.1 + 1e-3

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    chain_lines = [
        ax.plot([], [], [], "-o", linewidth=2, markersize=3)[0] for _ in _T2M_CHAINS
    ]
    title_text = ax.set_title("")

    def _setup_axes():
        # HumanML3D is Y-up; matplotlib's 3D viewer is Z-up, so swap Y↔Z.
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[2] - radius, center[2] + radius)
        ax.set_zlim(center[1] - radius, center[1] + radius)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("y")
        ax.view_init(elev=15, azim=-70)

    def update(t):
        _setup_axes()
        for line, chain in zip(chain_lines, _T2M_CHAINS):
            xs = joints[t, list(chain), 0]
            ys = joints[t, list(chain), 2]   # Y/Z swap for display
            zs = joints[t, list(chain), 1]
            line.set_data(xs, ys)
            line.set_3d_properties(zs)
        title_text.set_text(f"{title}\nframe {t + 1}/{T}")
        return chain_lines + [title_text]

    ani = FuncAnimation(fig, update, frames=T, interval=1000 // max(fps, 1), blit=False)

    media_path = out_path.with_suffix(f".{fmt}")
    if fmt == "mp4":
        ani.save(str(media_path), writer=FFMpegWriter(fps=fps, bitrate=2000))
    else:
        ani.save(str(media_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    return media_path, npy_path
