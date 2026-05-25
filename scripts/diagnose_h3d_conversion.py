"""Element-wise comparison of our T+R → 263-D conversion against upstream's
`process_file` from external/HumanML3D/motion_representation.ipynb.

Picks one real clip from the packed dataset, runs both pipelines on the same
underlying motion, prints per-block (root/RIC/rot/vel/foot) max & mean abs diff.

Why this matters: the Guo evaluator's `diversity_real` came out ~4.05 in our
eval, but the paper measures ~9.5 on the same test clips. That gap can only
come from the 263-D features being in a different space than the evaluator was
trained on. This script localizes which block diverges.

Usage:
    python scripts/diagnose_h3d_conversion.py            # default: first train clip
    python scripts/diagnose_h3d_conversion.py --clip 000021
    python scripts/diagnose_h3d_conversion.py --n 5      # diagnose 5 different clips
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# Polyfill removed-from-numpy aliases that upstream still uses.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]

from rmg.representation.humanml3d_io import tplusr_to_h3d_features_with_quats
from rmg.representation.humanml3d_upstream import tplusr_to_h3d_features_upstream
from rmg.representation.skeleton import Skeleton as RmgSkeleton, forward_kinematics


# ---------------------------------------------------------------------------
# Upstream loader — bring in process_file from the notebook
# ---------------------------------------------------------------------------

def load_upstream_process_file(target_offsets: torch.Tensor):
    """Execute the relevant cells of motion_representation.ipynb in our process
    so we can call `process_file` directly with our globals.

    `target_offsets` (22, 3) is injected as `tgt_offsets` — upstream's
    `uniform_skeleton` expects this exact shape and definition (per-joint bone
    vector from parent), which is what we save in `target_offsets.pt`.
    """
    hml3d = REPO / "external" / "HumanML3D"
    sys.path.insert(0, str(hml3d))

    from common.skeleton import Skeleton as UpSkeleton  # type: ignore
    import common.quaternion as Q  # type: ignore
    from paramUtil import t2m_raw_offsets, t2m_kinematic_chain  # type: ignore

    # .ipynb is just JSON — no need for nbformat.
    nb = json.loads((hml3d / "motion_representation.ipynb").read_text())

    ns: dict = {
        "np": np,
        "torch": torch,
        "Skeleton": UpSkeleton,
        "n_raw_offsets": torch.from_numpy(t2m_raw_offsets),
        "kinematic_chain": t2m_kinematic_chain,
        "tgt_offsets": target_offsets,
        # Constants from cell 5's main block:
        "face_joint_indx": [2, 1, 17, 16],
        "fid_l": [7, 10],
        "fid_r": [8, 11],
        "l_idx1": 5,
        "l_idx2": 8,
        "r_hip": 2,
        "l_hip": 1,
    }
    # Inject every public symbol from common.quaternion (qbetween_np, qrot_np, etc.)
    for k in dir(Q):
        if not k.startswith("_"):
            ns[k] = getattr(Q, k)

    # Cells 1 and 2 define uniform_skeleton + process_file.
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    exec("".join(code_cells[1]["source"]), ns)  # uniform_skeleton
    exec("".join(code_cells[2]["source"]), ns)  # process_file
    return ns


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------

H3D_BLOCKS = [
    ("root_data       ", 0, 4),     # r_velocity, l_vel_xz, root_y
    ("ric (joint pos) ", 4, 4 + 21 * 3),
    ("rot (cont6d)    ", 67, 67 + 21 * 6),
    ("vel (joint vel) ", 193, 193 + 22 * 3),
    ("foot contacts   ", 259, 263),
]


def diff_report(name: str, ours: np.ndarray, theirs: np.ndarray) -> dict:
    assert ours.shape == theirs.shape, f"{name}: shape mismatch {ours.shape} vs {theirs.shape}"
    abs_diff = np.abs(ours - theirs)
    rel_scale = max(np.abs(theirs).max(), 1e-8)
    report = {
        "name": name,
        "shape": list(ours.shape),
        "ours_mean_abs": float(np.abs(ours).mean()),
        "theirs_mean_abs": float(np.abs(theirs).mean()),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "rel_max": float(abs_diff.max() / rel_scale),
    }
    return report


def _load_raw_joints_with_prep_transforms(
    clip_name: str,
    index_rows: list,
    joints_root: Path,
) -> np.ndarray | None:
    """Replay `stage_pack`'s prep transforms on the raw joints for this clip
    to produce the joint positions upstream's `process_file` would have
    consumed: subset pre-trim → index.csv slice → X-flip.

    Returns None if we can't locate the source file.
    """
    target = clip_name[:-3] if clip_name.endswith(".pt") else clip_name
    target = target + ".npy"
    match = next(((src, s, e) for (src, s, e, name) in index_rows
                  if Path(name).stem == Path(target).stem), None)
    if match is None:
        return None
    src, start, end = match
    src_rel = src.replace("./pose_data/", "")
    npy_path = joints_root / src_rel
    if not npy_path.exists():
        return None
    joints = np.load(npy_path).reshape(-1, 22, 3)

    # Mirror stage_pack's subset pre-trims.
    fps_for_trim = 20
    if "Eyes_Japan_Dataset" in src:
        joints = joints[3 * fps_for_trim:]
    elif "MPI_HDM05" in src:
        joints = joints[3 * fps_for_trim:]
    elif "TotalCapture" in src:
        joints = joints[1 * fps_for_trim:]
    elif "MPI_Limits" in src:
        joints = joints[1 * fps_for_trim:]
    elif "Transitions_mocap" in src:
        joints = joints[int(0.5 * fps_for_trim):]

    joints = joints[start:end] if end > 0 else joints[start:]

    # X-flip (skipped for humanact12, per stage_pack).
    if "humanact12" not in src:
        joints = joints.copy()
        joints[..., 0] *= -1

    return joints.astype(np.float32)


def diagnose_clip(clip_name: str, packed_zip: Path, target_offsets: torch.Tensor,
                  process_file, index_rows: list, joints_root: Path) -> dict:
    print(f"\n{'=' * 78}\nCLIP: {clip_name}\n{'=' * 78}")

    # Load packed clip (T, R).
    with zipfile.ZipFile(packed_zip) as zf:
        blob = torch.load(io.BytesIO(zf.read(clip_name)), weights_only=False)
    translation = blob["translation"].float()          # (T, 3)
    quats = blob["quats"].float()                      # (T, 22, 4)
    T = translation.shape[0]
    print(f"  frames={T}  text[0]={blob['texts'][0][:80]!r}")

    # Our FK from stored (quats, translation) → joint positions.
    skel = RmgSkeleton(offsets=target_offsets)
    positions_from_packed = forward_kinematics(skel, quats, translation).numpy().astype(np.float32)

    # Raw joints with the SAME prep transforms applied (the only honest baseline).
    positions_raw = _load_raw_joints_with_prep_transforms(clip_name, index_rows, joints_root)
    if positions_raw is None:
        print("  !! raw joints not found — skipping raw-vs-packed comparison")
        return {"clip": clip_name, "error": "raw joints missing"}
    if positions_raw.shape != positions_from_packed.shape:
        print(f"  !! shape mismatch raw={positions_raw.shape} packed_FK={positions_from_packed.shape}")
        return {"clip": clip_name, "error": "shape mismatch"}

    # ------------ Position-level comparison (the new, honest test) ------------
    pos_diff = np.abs(positions_from_packed - positions_raw)
    print(f"\n  RAW JOINTS vs PACKED→FK (joint positions, meters):")
    print(f"    mean|Δ| = {pos_diff.mean():.6f}   max|Δ| = {pos_diff.max():.6f}")
    print(f"    per-joint mean|Δ|:")
    for j in range(22):
        print(f"      joint {j:2d}: {pos_diff[:, j, :].mean():.6f}   max={pos_diff[:, j, :].max():.6f}")

    # ------------ Feature-level comparison: raw → process_file vs packed-FK → process_file ----
    feats_raw, _, _, _ = process_file(positions_raw, 0.002)
    feats_raw = np.asarray(feats_raw, dtype=np.float32)
    feats_packed, _, _, _ = process_file(positions_from_packed, 0.002)
    feats_packed = np.asarray(feats_packed, dtype=np.float32)

    print(f"\n  263-D FEATURES: process_file(raw) vs process_file(packed→FK):")
    print(f"  {'block':<18}  {'shape':>14}  {'mean|Δ|':>10}  {'rel_max':>10}")
    print(f"  {'-' * 60}")
    reports_real = []
    for name, a, b in H3D_BLOCKS:
        r = diff_report(name, feats_packed[:, a:b], feats_raw[:, a:b])
        reports_real.append(r)
        print(f"  {r['name']}  {str(r['shape']):>14}  {r['mean_abs_diff']:>10.6f}  {r['rel_max']:>10.6f}")

    return {"clip": clip_name, "blocks_real": reports_real,
            "pos_mean_abs_diff": float(pos_diff.mean()),
            "pos_max_abs_diff": float(pos_diff.max())}


def _read_index_csv_rows(humanml3d_repo: Path) -> list:
    """Parse index.csv → list of (source_npy_relpath, start, end, new_name)."""
    rows = []
    with open(humanml3d_repo / "index.csv") as f:
        next(f)  # header
        for line in f:
            cols = line.strip().split(",")
            if len(cols) != 4:
                continue
            rows.append((cols[0], int(cols[1]), int(cols[2]), cols[3]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(REPO / "external" / "data" / "humanml3d_packed"))
    ap.add_argument("--joints-root", default=str(REPO / "external" / "data" / "joints_cache"))
    ap.add_argument("--humanml3d-repo", default=str(REPO / "external" / "HumanML3D"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--clip", default=None, help="clip name like '000021.pt'")
    ap.add_argument("--n", type=int, default=3, help="how many clips to diagnose")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    joints_root = Path(args.joints_root)
    humanml3d_repo = Path(args.humanml3d_repo)
    packed_zip = data_root / "humanml3d.zip"
    splits = json.loads((data_root / "splits.json").read_text())
    target_offsets = torch.load(data_root / "target_offsets.pt", weights_only=True).float()
    print(f"[diag] target_offsets shape: {tuple(target_offsets.shape)}")
    print(f"[diag] joints_root: {joints_root}")

    index_rows = _read_index_csv_rows(humanml3d_repo)
    print(f"[diag] loaded {len(index_rows)} rows from index.csv")

    # Pick clips.
    if args.clip:
        clips = [args.clip if args.clip.endswith(".pt") else f"{args.clip}.pt"]
    else:
        names = sorted(splits[args.split])
        clips = [n if n.endswith(".pt") else f"{n}.pt" for n in names[: args.n]]
    print(f"[diag] will diagnose: {clips}")

    process_file = load_upstream_process_file(target_offsets)["process_file"]

    all_reports = []
    for clip in clips:
        try:
            all_reports.append(diagnose_clip(
                clip, packed_zip, target_offsets, process_file, index_rows, joints_root,
            ))
        except Exception as e:
            print(f"  !! exception: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            all_reports.append({"clip": clip, "error": str(e)})

    # Summary across clips: raw vs packed at the position level + per-block features.
    print(f"\n{'=' * 78}\nSUMMARY (across {len(all_reports)} clip(s)):\n{'=' * 78}")
    pos_diffs = [r["pos_mean_abs_diff"] for r in all_reports if "pos_mean_abs_diff" in r]
    if pos_diffs:
        print(f"\n  Joint positions RAW vs PACKED→FK (meters):")
        print(f"    mean|Δ|: {np.mean(pos_diffs):.6f}    max over clips: {np.max(pos_diffs):.6f}")
        print(f"    interpretation: 0 = our IK+FK lossless;   >0.01 = prep loses motion info")

    print(f"\n  263-D features RAW→process_file vs PACKED→FK→process_file:")
    block_names = [b[0] for b in H3D_BLOCKS]
    for name in block_names:
        diffs = []
        for r in all_reports:
            for b in r.get("blocks_real", []):
                if b["name"] == name:
                    diffs.append(b["mean_abs_diff"])
        if diffs:
            print(f"    {name}  mean|Δ|: {np.mean(diffs):.6f}    max: {np.max(diffs):.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
