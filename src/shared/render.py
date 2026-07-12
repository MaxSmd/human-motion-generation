"""Shared motion renderer: (T, 22, 3) world joints → animated media on disk.

Single source of truth for the scientific multi-panel animation used by BOTH
the cluster job pipeline (`rmg.scripts.visualize`) and the backend
(`app.backend.render`). Renders four synchronized panels — three orthographic
2-D projections (Front X–Y, Side Z–Y, Top X–Z) plus a larger 3-D perspective —
to an MP4 (system ffmpeg) or GIF (Pillow, always available), and dumps the raw
joints as `.npy` next to the media for re-render/debug.

Depends only on numpy + matplotlib so it imports cleanly on a GPU node without
torch or the Hydra/CLI wrapper.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

# HumanML3D 22-joint kinematic chains (upstream's t2m_kinematic_chain).
_T2M_CHAINS: tuple[tuple[int, ...], ...] = (
    (0, 2, 5, 8, 11),       # right leg
    (0, 1, 4, 7, 10),       # left leg
    (0, 3, 6, 9, 12, 15),   # spine + head
    (9, 14, 17, 19, 21),    # right arm
    (9, 13, 16, 18, 20),    # left arm
)


# Parent/child neighbours along the kinematic chains, used to measure the
# interior bend angle at a joint straight from world positions (mirrors
# flow.constraints' bend, but position-space and torch-free).
def _chain_neighbors(joint: int) -> tuple[int, int] | None:
    """(parent, child) of `joint` along its chain, or None if it's a chain
    endpoint (root/end-effector) with no representable bend."""
    for chain in _T2M_CHAINS:
        if joint in chain:
            i = chain.index(joint)
            if 0 < i < len(chain) - 1:
                return chain[i - 1], chain[i + 1]
            return None
    return None


def _bend_series_deg(joints: np.ndarray, joint: int) -> np.ndarray | None:
    """Per-frame interior bend angle (deg, 0 = straight) at `joint`: the angle
    between the incoming bone (parent→joint) and the outgoing bone
    (joint→child). Returns None if `joint` has no representable bend."""
    nb = _chain_neighbors(joint)
    if nb is None:
        return None
    parent, child = nb
    inc = joints[:, joint] - joints[:, parent]
    out = joints[:, child] - joints[:, joint]
    inc = inc / (np.linalg.norm(inc, axis=-1, keepdims=True) + 1e-9)
    out = out / (np.linalg.norm(out, axis=-1, keepdims=True) + 1e-9)
    cos = np.clip((inc * out).sum(axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


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


# --- Scientific styling -------------------------------------------------------
# Anatomical limb colouring: warm = right side, cool = left side, pale = spine.
# Chains are ordered (right leg, left leg, spine, right arm, left arm).
_C_RIGHT = "#ff6b6b"
_C_LEFT = "#46c7c7"
_C_SPINE = "#e8edf2"
_CHAIN_COLORS = (_C_RIGHT, _C_LEFT, _C_SPINE, _C_RIGHT, _C_LEFT)

# End-effectors / root to trace as fading motion trails: joint -> colour.
_C_ROOT = "#ffd166"
_TRAIL_JOINTS: dict[int, str] = {
    0: _C_ROOT,    # pelvis (root trajectory through space)
    10: _C_LEFT,   # left foot
    11: _C_RIGHT,  # right foot
    20: _C_LEFT,   # left hand
    21: _C_RIGHT,  # right hand
}
_TRAIL_WINDOW = 18  # frames of history drawn behind each tracked joint

# Theme colours.
_BG = "#0e1116"
_PANEL = "#0e1116"
_GRID = "#2a3340"
_FG = "#9fb0c0"

# Orthographic 2-D projections: (label, horizontal-axis idx, vertical-axis idx).
# World axes are HumanML3D's X (right), Y (up), Z (forward/depth).
_VIEWS_2D = (
    ("Front  ·  X–Y", 0, 1),
    ("Side  ·  Z–Y", 2, 1),
    ("Top  ·  X–Z", 0, 2),
)

# Scene / obstacle colours.
_C_ROOM = "#46566a"     # room wireframe
_C_OBST = "#e0904a"     # obstacle edges (warm amber, distinct from the body)
_C_OBST_FILL = "#e0904a"
_C_SPAWN = "#6ee7a8"    # spawn marker / heading arrow


# --------------------------------------------------------------------------- scene geometry
# Lightweight, dependency-free projections of the `rmg.flow.scene.Scene` model
# (room box + box/sphere/cylinder obstacles) so the renderer can draw the room
# and obstacles into every panel without importing torch.


def _box_corners(cx, cy, cz, w, h, d, yaw_deg=0.0) -> np.ndarray:
    """8 world corners (8, 3) of a yawed (about Y) axis box."""
    hx, hy, hz = w / 2, h / 2, d / 2
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    out = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                lx, lz = sx * hx, sz * hz
                out.append((cx + lx * c + lz * s, cy + sy * hy, cz - lx * s + lz * c))
    return np.array(out, dtype=float)


# Edges of a box as index pairs into the 8 corners produced above.
_BOX_EDGES = (
    (0, 1), (1, 3), (3, 2), (2, 0),     # one yz face (sx=-1)
    (4, 5), (5, 7), (7, 6), (6, 4),     # opposite face (sx=+1)
    (0, 4), (1, 5), (2, 6), (3, 7),     # connectors
)


def _convex_hull(pts: np.ndarray) -> np.ndarray:
    """Monotone-chain convex hull of 2-D points (no scipy)."""
    pts = sorted(set(map(tuple, np.round(pts, 6))))
    if len(pts) <= 2:
        return np.array(pts, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def _scene_bounds(scene: dict) -> tuple[np.ndarray, np.ndarray]:
    """World-space (lo, hi) corners enclosing the room and all obstacles."""
    room = scene.get("room", {})
    w = float(room.get("width", 4.0)); d = float(room.get("depth", 4.0))
    h = float(room.get("height", 2.5))
    lo = np.array([-w / 2, 0.0, -d / 2]); hi = np.array([w / 2, h, d / 2])
    for o in scene.get("objects", []) or []:
        k = o.get("kind")
        if k == "sphere":
            c = np.array([o["x"], o["y"], o["z"]], float); r = float(o["radius"])
            lo = np.minimum(lo, c - r); hi = np.maximum(hi, c + r)
        elif k == "cylinder":
            r, hh = float(o["radius"]), float(o["height"]) / 2
            c = np.array([o["x"], o["y"], o["z"]], float)
            lo = np.minimum(lo, c - [r, hh, r]); hi = np.maximum(hi, c + [r, hh, r])
        elif k == "box":
            cn = _box_corners(o["x"], o["y"], o["z"], float(o["w"]), float(o["h"]),
                              float(o["d"]), float(o.get("rotation", 0.0)))
            lo = np.minimum(lo, cn.min(0)); hi = np.maximum(hi, cn.max(0))
    return lo, hi


def _draw_scene_2d(ax, h: int, v: int, scene: dict) -> None:
    """Draw the room outline + obstacle silhouettes into a 2-D panel (axes h, v)."""
    from matplotlib.patches import Circle, Polygon, Rectangle

    room = scene.get("room", {})
    w = float(room.get("width", 4.0)); d = float(room.get("depth", 4.0))
    hgt = float(room.get("height", 2.5))
    # Room outline: project its 8 corners and bound them in this view.
    rc = _box_corners(0.0, hgt / 2, 0.0, w, hgt, d)[:, [h, v]]
    rlo, rhi = rc.min(0), rc.max(0)
    ax.add_patch(Rectangle(rlo, *(rhi - rlo), fill=False, ec=_C_ROOM, lw=1.3,
                           ls="--", alpha=0.8, zorder=0))

    for o in scene.get("objects", []) or []:
        k = o.get("kind")
        if k == "sphere":
            ax.add_patch(Circle((o[("x", "y", "z")[h]], o[("x", "y", "z")[v]]),
                                float(o["radius"]), fc=_C_OBST_FILL, ec=_C_OBST,
                                alpha=0.22, lw=1.4, zorder=1))
        elif k == "cylinder":
            r, hh = float(o["radius"]), float(o["height"]) / 2
            if v == 1:  # vertical axis is world-Y → cylinder is a rectangle
                cx = o[("x", "y", "z")[h]]
                ax.add_patch(Rectangle((cx - r, o["y"] - hh), 2 * r, 2 * hh,
                                       fc=_C_OBST_FILL, ec=_C_OBST, alpha=0.22,
                                       lw=1.4, zorder=1))
            else:        # top view (X–Z) → circle
                ax.add_patch(Circle((o["x"], o["z"]), r, fc=_C_OBST_FILL,
                                    ec=_C_OBST, alpha=0.22, lw=1.4, zorder=1))
        elif k == "box":
            cn = _box_corners(o["x"], o["y"], o["z"], float(o["w"]), float(o["h"]),
                              float(o["d"]), float(o.get("rotation", 0.0)))
            poly = _convex_hull(cn[:, [h, v]])
            ax.add_patch(Polygon(poly, closed=True, fc=_C_OBST_FILL, ec=_C_OBST,
                                 alpha=0.22, lw=1.4, zorder=1))

    _draw_spawn_2d(ax, h, v, scene.get("spawn"))


def _draw_spawn_2d(ax, h: int, v: int, spawn) -> None:
    if not spawn:
        return
    sx = float(spawn.get("x", 0.0)); sz = float(spawn.get("z", 0.0))
    p = {0: sx, 2: sz}
    if h in p and v in p:        # top view: marker + heading arrow
        ang = np.radians(float(spawn.get("rotation", 0.0)))
        ax.plot(p[h], p[v], "o", color=_C_SPAWN, ms=5, zorder=2)
        # heading 0° faces +Z (see scene.place_motion).
        ax.annotate("", xy=(sx + 0.5 * np.sin(ang), sz + 0.5 * np.cos(ang)),
                    xytext=(sx, sz), zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=_C_SPAWN, lw=1.6))
    elif h in p:                 # front/side: a tick on the floor
        ax.plot(p[h], 0.0, "^", color=_C_SPAWN, ms=6, zorder=2)


def _draw_scene_3d(ax3d, scene: dict) -> None:
    """Draw room wireframe + obstacles in the 3-D panel (display swaps Y↔Z)."""
    room = scene.get("room", {})
    w = float(room.get("width", 4.0)); d = float(room.get("depth", 4.0))
    hgt = float(room.get("height", 2.5))
    rc = _box_corners(0.0, hgt / 2, 0.0, w, hgt, d)
    for a, b in _BOX_EDGES:
        ax3d.plot(*[[rc[a, i], rc[b, i]] for i in (0, 2, 1)],
                  color=_C_ROOM, lw=1.0, ls="--", alpha=0.7, zorder=0)

    u = np.linspace(0, 2 * np.pi, 24)
    for o in scene.get("objects", []) or []:
        k = o.get("kind")
        if k == "sphere":
            r = float(o["radius"]); pv = np.linspace(0, np.pi, 14)
            xs = o["x"] + r * np.outer(np.cos(u), np.sin(pv))
            ys = o["y"] + r * np.outer(np.ones_like(u), np.cos(pv))
            zs = o["z"] + r * np.outer(np.sin(u), np.sin(pv))
            ax3d.plot_surface(xs, zs, ys, color=_C_OBST, alpha=0.18, shade=False)
        elif k == "cylinder":
            r, hh = float(o["radius"]), float(o["height"]) / 2
            yy = np.array([o["y"] - hh, o["y"] + hh])
            xs = o["x"] + r * np.cos(u)[None, :] * np.ones((2, 1))
            zs = o["z"] + r * np.sin(u)[None, :] * np.ones((2, 1))
            ys = yy[:, None] * np.ones((1, u.size))
            ax3d.plot_surface(xs, zs, ys, color=_C_OBST, alpha=0.18, shade=False)
        elif k == "box":
            cn = _box_corners(o["x"], o["y"], o["z"], float(o["w"]), float(o["h"]),
                              float(o["d"]), float(o.get("rotation", 0.0)))
            for a, b in _BOX_EDGES:
                ax3d.plot(*[[cn[a, i], cn[b, i]] for i in (0, 2, 1)],
                          color=_C_OBST, lw=1.6, alpha=0.9)

    spawn = scene.get("spawn")
    if spawn:
        sx = float(spawn.get("x", 0.0)); sz = float(spawn.get("z", 0.0))
        ang = np.radians(float(spawn.get("rotation", 0.0)))
        ax3d.plot([sx], [sz], [0.0], "o", color=_C_SPAWN, ms=5)
        ax3d.plot([sx, sx + 0.5 * np.sin(ang)], [sz, sz + 0.5 * np.cos(ang)],
                  [0.0, 0.0], color=_C_SPAWN, lw=1.8)


# Allowed-band colour for the constraint panel (reuse the obstacle amber).
_C_BAND = "#e0904a"


def _draw_constraint_panel(fig, rect: tuple[float, float, float, float],
                           joints: np.ndarray, constraints: list[dict]):
    """Draw a full-width panel under the main viz summarising the sampling-time
    bend constraints, in the same dark scientific style.

    Each constraint dict is `{joint, name, min_deg, max_deg, frame_start,
    frame_end}` (joint = int index, window already resolved). Fixed pins
    (min ≈ max) render as text; clamped ranges render as a bend-angle-over-time
    plot with the allowed band shaded across its active window. Returns a
    per-frame `update(t)` that moves an animated cursor, or None when there is
    no time plot to scrub.
    """
    import matplotlib.pyplot as plt  # noqa: F401  (backend already selected)
    from matplotlib.patches import Rectangle

    T = joints.shape[0]
    fixed = [c for c in constraints if abs(c["max_deg"] - c["min_deg"]) < 0.5]
    clamped = [c for c in constraints if abs(c["max_deg"] - c["min_deg"]) >= 0.5]

    fig.text(rect[0], rect[1] + rect[3] + 0.012, "Constraints", color=_FG,
             fontsize=10, fontfamily="monospace", ha="left", va="bottom")

    ax = fig.add_axes(rect)
    ax.set_facecolor(_PANEL)
    for spine in ax.spines.values():
        spine.set_color(_GRID)

    if not clamped:
        # Nothing to scrub over time → a plain text card listing the pins.
        ax.axis("off")
        lines = [f"{c['name']}   pinned to {c['min_deg']:.0f}°" for c in fixed]
        ax.text(0.5, 0.5, "\n".join(lines) or "—", color="#f5f7fa", fontsize=13,
                fontfamily="monospace", ha="center", va="center",
                linespacing=1.6, transform=ax.transAxes)
        return None

    ax.set_xlim(0, max(T - 1, 1))
    ax.tick_params(colors=_FG, labelsize=7, length=2)
    ax.grid(True, color=_GRID, linewidth=0.5, alpha=0.6)
    ax.set_xlabel("frame", color=_FG, fontsize=8, fontfamily="monospace")
    ax.set_ylabel("bend  (deg)", color=_FG, fontsize=8, fontfamily="monospace")

    ymax = 90.0
    for c in clamped:
        fs, fe = c["frame_start"], c["frame_end"]
        col = _C_LEFT if c["name"].startswith("L_") else (
            _C_RIGHT if c["name"].startswith("R_") else _C_SPINE)
        # Allowed band, shaded only over the window where the clamp is active.
        ax.add_patch(Rectangle((fs, c["min_deg"]), max(fe - fs, 0),
                               c["max_deg"] - c["min_deg"], facecolor=_C_BAND,
                               edgecolor="none", alpha=0.20, zorder=1))
        ax.hlines([c["min_deg"], c["max_deg"]], fs, fe, color=_C_BAND, lw=1.0,
                  alpha=0.75, zorder=2)
        series = _bend_series_deg(joints, c["joint"])
        if series is not None:
            ax.plot(np.arange(T), series, "-", lw=1.8, color=col, zorder=3,
                    label=f"{c['name']}  clamp {c['min_deg']:.0f}–{c['max_deg']:.0f}°")
            ymax = max(ymax, float(np.nanmax(series)))
        ymax = max(ymax, c["max_deg"])
    ax.set_ylim(0, min(185.0, ymax * 1.1 + 5))

    # Any fixed pins alongside the clamp(s): note them in a corner.
    if fixed:
        txt = "    ".join(f"{c['name']} = {c['min_deg']:.0f}°" for c in fixed)
        ax.text(0.01, 0.96, txt, color="#f5f7fa", fontsize=8,
                fontfamily="monospace", ha="left", va="top", transform=ax.transAxes)

    leg = ax.legend(loc="upper right", fontsize=7, facecolor=_PANEL,
                    edgecolor=_GRID, framealpha=0.6)
    for t in leg.get_texts():
        t.set_color(_FG)

    cursor = ax.axvline(0, color=_C_ROOT, lw=1.2, alpha=0.9, zorder=4)

    def _update(t: int) -> None:
        cursor.set_xdata([t, t])

    return _update


# Energy panel + joint-glow colours ("how hard is the skeleton being punished").
_C_ENERGY = "#ff5c5c"                 # total penalty fill/line (hot)
_C_GLOW_CORE = "#ff2b2b"              # violating-joint core
_C_GLOW_HALO = "#ff7a3c"             # soft halo behind it
_C_COMPONENTS = {                     # per-term penalty lines
    "room": "#6aa9ff",
    "obstacle": _C_OBST,              # amber — matches the drawn obstacles
    "contact": "#c792ea",
    "foot_skate": "#6ee7a8",
}
_COMPONENT_LABELS = {
    "room": "room / walls",
    "obstacle": "obstacles",
    "contact": "contacts",
    "foot_skate": "foot-skate",
}


def _draw_energy_panel(fig, rect: tuple[float, float, float, float],
                       energy: dict, T: int):
    """Draw the constraint-violation PENALTY over time — the "how hard is the
    skeleton being punished" curve — in the same dark scientific style.

    `energy` is `{total: (T,), per_joint: (T, J), components: {name: (T,)}}`
    (see `flow.scene.scene_energy_series`). The per-frame total is drawn as a
    filled red area with each non-zero component term overlaid; a cursor plus a
    numeric readout track the current frame. Returns a per-frame `update(t)`.
    """
    from matplotlib.patches import Rectangle  # noqa: F401  (backend already selected)

    total = np.nan_to_num(np.asarray(energy["total"], dtype=float))
    frames = np.arange(len(total))

    fig.text(rect[0], rect[1] + rect[3] + 0.012, "Constraint penalty", color=_FG,
             fontsize=10, fontfamily="monospace", ha="left", va="bottom")

    ax = fig.add_axes(rect)
    ax.set_facecolor(_PANEL)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.tick_params(colors=_FG, labelsize=7, length=2)
    ax.grid(True, color=_GRID, linewidth=0.5, alpha=0.6)
    ax.set_xlim(0, max(T - 1, 1))
    ax.set_xlabel("frame", color=_FG, fontsize=8, fontfamily="monospace")
    ax.set_ylabel("penalty  (a.u.)", color=_FG, fontsize=8, fontfamily="monospace")

    peak = float(np.nanmax(total)) if total.size else 0.0
    ax.set_ylim(0, max(peak * 1.15, 1e-6))

    # Total penalty: filled area + line.
    ax.fill_between(frames, total, color=_C_ENERGY, alpha=0.22, zorder=1)
    ax.plot(frames, total, "-", lw=1.8, color=_C_ENERGY, zorder=3, label="total")

    # Component breakdown — only terms that actually fire, to keep it honest.
    for name, series in (energy.get("components") or {}).items():
        s = np.nan_to_num(np.asarray(series, dtype=float))
        if s.size == len(total) and float(np.nanmax(s)) > 1e-9:
            ax.plot(frames, s, "-", lw=1.0, alpha=0.85,
                    color=_C_COMPONENTS.get(name, _FG),
                    label=_COMPONENT_LABELS.get(name, name), zorder=2)

    if peak <= 1e-9:
        ax.text(0.5, 0.55, "no constraint violated — motion stays feasible",
                color="#6ee7a8", fontsize=11, fontfamily="monospace",
                ha="center", va="center", transform=ax.transAxes)
    else:
        leg = ax.legend(loc="upper right", fontsize=7, facecolor=_PANEL,
                        edgecolor=_GRID, framealpha=0.6)
        for t_ in leg.get_texts():
            t_.set_color(_FG)

    cursor = ax.axvline(0, color=_C_ROOT, lw=1.2, alpha=0.9, zorder=4)
    readout = ax.text(0.01, 0.96, "", color="#f5f7fa", fontsize=8,
                      fontfamily="monospace", ha="left", va="top",
                      transform=ax.transAxes)

    def _update(t: int) -> None:
        cursor.set_xdata([t, t])
        val = float(total[t]) if t < len(total) else 0.0
        readout.set_text(f"penalty[{t:>3}] = {val:7.3f}")

    return _update


def render_joints(
    joints: np.ndarray,                 # (T, 22, 3)
    out_path: Path,                     # media path; suffix is forced to match fmt
    title: str = "",
    fps: int = 20,
    fmt: str = "auto",
    scene: dict | None = None,          # optional room/obstacle scene (see flow.scene)
    constraints: list[dict] | None = None,  # optional sampling-time bend constraints
    energy: dict | None = None,         # optional per-frame/per-joint penalty (see flow.scene)
) -> tuple[Path, Path]:
    """Render joints to MP4/GIF and dump joints as `.npy`. Returns (media, npy).

    Lays out four synchronized panels: three orthographic 2-D projections
    (Front X–Y, Side Z–Y, Top X–Z) and a larger 3-D perspective on the right.
    All panels share equal-aspect limits, use anatomical left/right limb
    colouring, draw a ground plane, and carry fading motion trails on the root
    and the four end-effectors so the dynamics read clearly frame to frame.

    When `scene` is given (the frontend/`RMG_SCENE` dict: a `room`, a list of
    `objects` of kind box/sphere/cylinder, and a `spawn`), the room wireframe,
    obstacle silhouettes and the spawn marker are drawn into every panel and the
    axis limits are widened to frame the whole room — "obstacle mode".

    When `constraints` is given (a list of resolved bend-constraint dicts
    `{joint, name, min_deg, max_deg, frame_start, frame_end}`), a panel is added
    below the four viz panels: fixed pins render as text, clamped ranges render
    as a bend-angle-over-time plot with the allowed band shaded and a cursor that
    tracks the current frame.

    When `energy` is given (the `flow.scene.scene_energy_series` dict `{total,
    per_joint, components}` for a world-constrained sample), a second bottom
    panel plots the constraint-violation penalty over time AND every joint that
    is being penalised lights up with a red glow (sized by its violation depth)
    in all four viz panels — so you literally see where and when the skeleton is
    punished for leaving the room / hitting an obstacle / missing a contact.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = resolve_format(fmt)

    npy_path = out_path.with_suffix(".npy")
    np.save(npy_path, joints.astype(np.float32))

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    T = joints.shape[0]

    # Limits from finite points only, so a stray NaN frame can't poison them.
    pts = joints.reshape(-1, 3)
    finite_rows = np.isfinite(pts).all(axis=1)
    fpts = pts[finite_rows] if finite_rows.any() else pts
    lo, hi = fpts.min(axis=0), fpts.max(axis=0)
    if scene is not None:
        slo, shi = _scene_bounds(scene)
        lo, hi = np.minimum(lo, slo), np.maximum(hi, shi)
    center = (lo + hi) / 2
    pad = 1.1 if scene is None else 1.04  # less slack when a room frames the view
    radius = float(np.max(hi - lo)) / 2 * pad + 1e-3
    floor = float(lo[1])  # ground plane at min Y

    def _lim(axis: int) -> tuple[float, float]:
        return center[axis] - radius, center[axis] + radius

    has_constraints = bool(constraints)
    # Per-joint glow: normalise the violation depths to [0, 1] so a stray huge
    # penalty can't blow the marker sizes up. `None` ⇒ nothing to light up.
    glow = None
    if energy and energy.get("per_joint") is not None:
        pj = np.nan_to_num(np.asarray(energy["per_joint"], dtype=float))
        gmax = float(pj.max()) if pj.size else 0.0
        if gmax > 1e-9:
            glow = pj / gmax                                       # (T, J) in [0,1]
    has_energy = energy is not None
    has_bottom = has_constraints or has_energy

    fig = plt.figure(figsize=(19, 6.9 if has_bottom else 5.4))
    fig.patch.set_facecolor(_BG)
    # With a bottom panel, lift the viz row into the top of a taller figure and
    # reserve the strip between the row and the progress bar for the panel(s).
    gs_top, gs_bottom = (0.90, 0.42) if has_bottom else (0.86, 0.10)
    gs = GridSpec(
        1, 4, figure=fig, width_ratios=[1, 1, 1, 1.5],
        left=0.015, right=0.985, top=gs_top, bottom=gs_bottom, wspace=0.12,
    )

    # --- 2-D orthographic panels ---------------------------------------------
    views2d = []
    for col, (label, h, v) in enumerate(_VIEWS_2D):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(_PANEL)
        ax.set_xlim(*_lim(h))
        ax.set_ylim(*_lim(v))
        ax.set_aspect("equal")
        ax.set_title(label, color=_FG, fontsize=10, fontfamily="monospace", pad=8)
        ax.tick_params(colors=_FG, labelsize=7, length=2)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.grid(True, color=_GRID, linewidth=0.5, alpha=0.6)
        # Ground line (floor) when the vertical axis is Y (up).
        if v == 1:
            ax.axhline(floor, color=_GRID, linewidth=1.2, alpha=0.9)
        if scene is not None:
            _draw_scene_2d(ax, h, v, scene)
        lines = [
            ax.plot([], [], "-o", lw=2, ms=3, color=c, mec=c,
                    solid_capstyle="round")[0]
            for c in _CHAIN_COLORS
        ]
        trails = {
            j: ax.plot([], [], "-", lw=1.4, color=c, alpha=0.55)[0]
            for j, c in _TRAIL_JOINTS.items()
        }
        # Punishment glow: a soft halo + bright core scatter over violating
        # joints, updated per frame from the normalised per-joint penalty.
        glow_sc = None
        if glow is not None:
            halo = ax.scatter([], [], s=[], c=_C_GLOW_HALO, alpha=0.16,
                              edgecolors="none", zorder=4)
            core = ax.scatter([], [], s=[], c=_C_GLOW_CORE, alpha=0.85,
                              edgecolors="none", zorder=5)
            glow_sc = (halo, core)
        views2d.append((h, v, lines, trails, glow_sc))

    # --- 3-D perspective panel -----------------------------------------------
    ax3d = fig.add_subplot(gs[0, 3], projection="3d")
    ax3d.set_facecolor(_PANEL)
    ax3d.set_title("3D  ·  perspective", color=_FG, fontsize=10,
                   fontfamily="monospace", pad=2)
    chain_lines3d = [
        ax3d.plot([], [], [], "-o", lw=2, ms=3, color=c, mec=c,
                  solid_capstyle="round")[0]
        for c in _CHAIN_COLORS
    ]
    trails3d = {
        j: ax3d.plot([], [], [], "-", lw=1.4, color=c, alpha=0.55)[0]
        for j, c in _TRAIL_JOINTS.items()
    }
    glow3d = None
    if glow is not None:
        halo3d = ax3d.scatter([], [], [], s=[], c=_C_GLOW_HALO, alpha=0.16,
                              edgecolors="none", depthshade=False)
        core3d = ax3d.scatter([], [], [], s=[], c=_C_GLOW_CORE, alpha=0.85,
                              edgecolors="none", depthshade=False)
        glow3d = (halo3d, core3d)
    # Static ground plane (a faint grid quad) at y = floor in display space.
    gx = np.linspace(*_lim(0), 2)
    gz = np.linspace(*_lim(2), 2)
    gx, gz = np.meshgrid(gx, gz)
    ax3d.plot_surface(gx, gz, np.full_like(gx, floor), color=_GRID,
                      alpha=0.18, shade=False, zorder=0)
    if scene is not None:
        _draw_scene_3d(ax3d, scene)

    def _setup_3d():
        # HumanML3D is Y-up; matplotlib's 3-D viewer is Z-up, so swap Y↔Z.
        ax3d.set_xlim(*_lim(0))
        ax3d.set_ylim(*_lim(2))
        ax3d.set_zlim(*_lim(1))
        ax3d.set_box_aspect((1, 1, 1))
        ax3d.set_xlabel("x", color=_FG, fontsize=8)
        ax3d.set_ylabel("z", color=_FG, fontsize=8)
        ax3d.set_zlabel("y", color=_FG, fontsize=8)
        ax3d.tick_params(colors=_FG, labelsize=6)
        ax3d.locator_params(nbins=5)
        ax3d.view_init(elev=12, azim=-70)
        for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
            axis.pane.set_facecolor(_PANEL)
            axis.pane.set_edgecolor(_GRID)
            axis.pane.set_alpha(0.6)
            axis._axinfo["grid"].update(color=_GRID, linewidth=0.5)

    # --- Header + progress bar -----------------------------------------------
    head = fig.text(0.015, 0.95, title, color="#f5f7fa", fontsize=12,
                    fontweight="bold", ha="left", va="center")
    clock = fig.text(0.985, 0.95, "", color=_FG, fontsize=10,
                     fontfamily="monospace", ha="right", va="center")
    bar_bg = fig.add_axes((0.015, 0.025, 0.97, 0.012))
    bar_bg.set_xlim(0, 1)
    bar_bg.set_ylim(0, 1)
    bar_bg.axis("off")
    bar_bg.add_patch(plt.Rectangle((0, 0), 1, 1, color=_GRID, alpha=0.5))
    bar_fill = bar_bg.add_patch(plt.Rectangle((0, 0), 0, 1, color=_C_ROOT))

    # --- Bottom panel(s): bend constraints and/or the penalty curve ----------
    # One panel spans full width; when both are present they sit side by side.
    cursor_updates = []
    if has_constraints and has_energy:
        c_rect, e_rect = (0.05, 0.10, 0.41, 0.22), (0.56, 0.10, 0.41, 0.22)
    else:
        c_rect = e_rect = (0.06, 0.10, 0.90, 0.22)
    if has_constraints:
        u = _draw_constraint_panel(fig, c_rect, joints, constraints)
        if u is not None:
            cursor_updates.append(u)
    if has_energy:
        cursor_updates.append(_draw_energy_panel(fig, e_rect, energy, T))

    def _set_glow(sc, xs, ys, zs, sizes):
        """Point a (halo, core) scatter pair at the violating joints. `zs` is
        None for a 2-D panel; the halo is drawn 3× the core area."""
        if sc is None:
            return
        halo, core = sc
        if zs is None:
            xy = np.column_stack([xs, ys]) if len(xs) else np.empty((0, 2))
            halo.set_offsets(xy); core.set_offsets(xy)
        else:
            halo._offsets3d = (xs, zs, ys); core._offsets3d = (xs, zs, ys)
        halo.set_sizes(sizes * 2.4); core.set_sizes(sizes)

    def update(t):
        lo_t = max(0, t - _TRAIL_WINDOW)
        # Which joints are being punished this frame, and how big to draw them.
        if glow is not None:
            g = glow[t]
            gidx = np.nonzero(g > 0.02)[0]
            gsize = 18.0 + 130.0 * g[gidx]
        for h, v, lines, trails, glow_sc in views2d:
            for line, chain in zip(lines, _T2M_CHAINS):
                idx = list(chain)
                line.set_data(joints[t, idx, h], joints[t, idx, v])
            for j, tr in trails.items():
                tr.set_data(joints[lo_t:t + 1, j, h], joints[lo_t:t + 1, j, v])
            if glow is not None:
                _set_glow(glow_sc, joints[t, gidx, h], joints[t, gidx, v],
                          None, gsize)

        _setup_3d()
        for line, chain in zip(chain_lines3d, _T2M_CHAINS):
            idx = list(chain)
            line.set_data(joints[t, idx, 0], joints[t, idx, 2])  # Y/Z swap
            line.set_3d_properties(joints[t, idx, 1])
        for j, tr in trails3d.items():
            tr.set_data(joints[lo_t:t + 1, j, 0], joints[lo_t:t + 1, j, 2])
            tr.set_3d_properties(joints[lo_t:t + 1, j, 1])
        if glow is not None:
            _set_glow(glow3d, joints[t, gidx, 0], joints[t, gidx, 1],
                      joints[t, gidx, 2], gsize)  # xs, ys(up), zs → swapped inside

        for cu in cursor_updates:
            cu(t)

        clock.set_text(f"frame {t + 1:>3}/{T}   ·   {t / max(fps, 1):4.1f}s")
        bar_fill.set_width((t + 1) / T)
        return []

    ani = FuncAnimation(fig, update, frames=T, interval=1000 // max(fps, 1), blit=False)

    media_path = out_path.with_suffix(f".{fmt}")
    if fmt == "mp4":
        ani.save(str(media_path), writer=FFMpegWriter(fps=fps, bitrate=3500),
                 savefig_kwargs={"facecolor": _BG})
    else:
        ani.save(str(media_path), writer=PillowWriter(fps=fps),
                 savefig_kwargs={"facecolor": _BG})
    plt.close(fig)
    return media_path, npy_path
