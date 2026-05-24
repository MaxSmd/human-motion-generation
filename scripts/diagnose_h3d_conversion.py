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
from rmg.representation.skeleton import Skeleton as RmgSkeleton, forward_kinematics


# ---------------------------------------------------------------------------
# Upstream loader — bring in process_file from the notebook
# ---------------------------------------------------------------------------

def load_upstream_process_file():
    """Execute the relevant cells of motion_representation.ipynb in our process
    so we can call `process_file` directly with our globals."""
    hml3d = REPO / "external" / "HumanML3D"
    sys.path.insert(0, str(hml3d))

    import nbformat  # type: ignore
    from common.skeleton import Skeleton as UpSkeleton  # type: ignore
    import common.quaternion as Q  # type: ignore
    from paramUtil import t2m_raw_offsets, t2m_kinematic_chain  # type: ignore

    nb = nbformat.read(str(hml3d / "motion_representation.ipynb"), as_version=4)

    ns: dict = {
        "np": np,
        "torch": torch,
        "Skeleton": UpSkeleton,
        "n_raw_offsets": torch.from_numpy(t2m_raw_offsets),
        "kinematic_chain": t2m_kinematic_chain,
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
    exec(nb.cells[1].source, ns)
    exec(nb.cells[2].source, ns)
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


def diagnose_clip(clip_name: str, packed_zip: Path, target_offsets: torch.Tensor,
                  process_file) -> dict:
    print(f"\n{'=' * 78}\nCLIP: {clip_name}\n{'=' * 78}")

    # Load packed clip (T, R).
    with zipfile.ZipFile(packed_zip) as zf:
        blob = torch.load(io.BytesIO(zf.read(clip_name)), weights_only=False)
    translation = blob["translation"].float()          # (T, 3)
    quats = blob["quats"].float()                      # (T, 22, 4)
    T = translation.shape[0]
    print(f"  frames={T}  text[0]={blob['texts'][0][:80]!r}")

    # Our FK → joint positions (T, 22, 3) using stored quats + stored target offsets.
    skel = RmgSkeleton(offsets=target_offsets)
    positions = forward_kinematics(skel, quats, translation).numpy().astype(np.float32)
    print(f"  positions shape: {positions.shape}")

    # Upstream: process_file on these joint positions → 263-D
    data_up, _, _, _ = process_file(positions, 0.002)
    theirs = np.asarray(data_up, dtype=np.float32)

    # Ours: T+R → 263-D using our converter
    ours_t = tplusr_to_h3d_features_with_quats(translation, quats, skel).numpy().astype(np.float32)

    print(f"  upstream output: {theirs.shape}    ours: {ours_t.shape}")
    if ours_t.shape != theirs.shape:
        print("  !! shape mismatch — can't continue comparison")
        return {"clip": clip_name, "error": "shape mismatch"}

    # Per-block comparison.
    print(f"\n  {'block':<18}  {'shape':>14}  {'ours|x|':>10}  {'theirs|x|':>10}"
          f"  {'max|Δ|':>10}  {'mean|Δ|':>10}  {'rel_max':>10}")
    print(f"  {'-' * 88}")
    reports = []
    for name, a, b in H3D_BLOCKS:
        r = diff_report(name, ours_t[:, a:b], theirs[:, a:b])
        reports.append(r)
        print(f"  {r['name']}  {str(r['shape']):>14}  "
              f"{r['ours_mean_abs']:>10.4f}  {r['theirs_mean_abs']:>10.4f}  "
              f"{r['max_abs_diff']:>10.4f}  {r['mean_abs_diff']:>10.4f}  "
              f"{r['rel_max']:>10.4f}")
    return {"clip": clip_name, "blocks": reports}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(REPO / "external" / "data" / "humanml3d_packed"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--clip", default=None, help="clip name like '000021.pt'")
    ap.add_argument("--n", type=int, default=3, help="how many clips to diagnose")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    packed_zip = data_root / "humanml3d.zip"
    splits = json.loads((data_root / "splits.json").read_text())
    target_offsets = torch.load(data_root / "target_offsets.pt", weights_only=True).float()
    print(f"[diag] target_offsets shape: {tuple(target_offsets.shape)}")

    # Pick clips.
    if args.clip:
        clips = [args.clip if args.clip.endswith(".pt") else f"{args.clip}.pt"]
    else:
        names = sorted(splits[args.split])
        clips = [n if n.endswith(".pt") else f"{n}.pt" for n in names[: args.n]]
    print(f"[diag] will diagnose: {clips}")

    process_file = load_upstream_process_file()["process_file"]

    all_reports = []
    for clip in clips:
        try:
            all_reports.append(diagnose_clip(clip, packed_zip, target_offsets, process_file))
        except Exception as e:
            print(f"  !! exception: {type(e).__name__}: {e}")
            all_reports.append({"clip": clip, "error": str(e)})

    # Summary across clips.
    print(f"\n{'=' * 78}\nSUMMARY (mean across {len(all_reports)} clip(s)):\n{'=' * 78}")
    block_names = [b[0] for b in H3D_BLOCKS]
    for name in block_names:
        diffs = []
        for r in all_reports:
            if "blocks" not in r:
                continue
            for b in r["blocks"]:
                if b["name"] == name:
                    diffs.append(b["mean_abs_diff"])
        if diffs:
            print(f"  {name}  mean|Δ| over clips: {np.mean(diffs):.6f}  "
                  f"max: {np.max(diffs):.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
