"""Rebuild HumanML3D `new_joints/` (T, 22, 3) from canonical `new_joint_vecs/`.

The shared HumanML3D copy on the cluster was trimmed to a single `new_joints`
file, but ProjFlow's raw-joint prior trains and evaluates on exactly that
directory. HumanML3D's motion_representation notebook writes it as
`recover_from_ric(new_joint_vecs)` (float32), so we regenerate it the same way
from the canonical vectors we already rebuilt for MARDM, using upstream
ProjFlow's own `recover_from_ric`.

`--golden` names surviving original `new_joints` files; each rebuilt clip is
compared against them and the script aborts if any differs by more than
`--tol` metres.

Usage:
    python -m projflow.scripts.build_new_joints \
        --vecs-dir ~/rmg-runs/h3d-canonical/new_joint_vecs \
        --out-dir  ~/projflow-work/datasets/HumanML3D/new_joints \
        --upstream external/ProjFlow \
        --golden   /path/to/HumanML3D/new_joints/012314.npy
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

JOINTS_NUM = 22


def load_recover_from_ric(upstream: Path):
    sys.path.insert(0, str(upstream))  # motion_process imports utils.quaternion
    spec = importlib.util.spec_from_file_location("projflow_motion_process", upstream / "utils" / "motion_process.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.recover_from_ric


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vecs-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--golden", type=Path, nargs="*", default=[])
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args()

    recover_from_ric = load_recover_from_ric(args.upstream.resolve())

    def rebuild(vec_file: Path) -> np.ndarray:
        data = torch.from_numpy(np.load(vec_file)).unsqueeze(0).float()
        return recover_from_ric(data, JOINTS_NUM).squeeze(0).numpy()

    for golden in args.golden:
        ours = rebuild(args.vecs_dir / golden.name)
        ref = np.load(golden)
        if ours.shape != ref.shape:
            sys.exit(f"golden {golden.name}: shape {ours.shape} != original {ref.shape}")
        diff = float(np.abs(ours - ref).max())
        print(f"golden {golden.name}: shape {ours.shape}, max abs diff {diff:.2e} m")
        if diff > args.tol:
            sys.exit(f"golden {golden.name} differs by {diff:.2e} > tol {args.tol:.0e}; not writing new_joints")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    vec_files = sorted(args.vecs_dir.glob("*.npy"))
    for i, vec_file in enumerate(vec_files):
        out = args.out_dir / vec_file.name
        if not out.exists():
            np.save(out, rebuild(vec_file))
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(vec_files)}")
    print(f"wrote {len(vec_files)} clips to {args.out_dir}")


if __name__ == "__main__":
    main()
