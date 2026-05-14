"""Build the packed RMG dataset from AMASS + HumanML3D.

Produces, under `--output-dir`:
    humanml3d.zip       # one <clip_id>.pt per HumanML3D clip
    splits.json         # {"train": [...], "val": [...], "test": [...]}
    target_offsets.pt   # (22, 3) reference T-pose offsets, derived from a fixed clip
    meta.json           # bookkeeping (fps, num_clips, version)

Two stages, each runnable independently. Run `--help` for usage:

    raw-pose  : AMASS .npz → joint positions .npy (per AMASS file)
                Mirrors `external/HumanML3D/raw_pose_processing.ipynb`.
    pack      : .npy joints → IK quaternions → T+R clips → packed zip
                Mirrors `external/HumanML3D/motion_representation.ipynb`,
                but stops *before* the 263-D feature extraction; we keep
                (translation, quaternions) so the network sees raw T+R.

This script imports `external/HumanML3D/common/skeleton.py::Skeleton` for IK
and FK so the produced quaternions are bit-comparable with the upstream
pipeline. The IK is the same one the upstream evaluator implicitly assumes.

You will not be able to run this script unless you have:
  1. AMASS subsets downloaded (see plan / docs for the exact list of 17).
  2. SMPL+H + DMPLs body models in `--body-models`.
  3. The `external/HumanML3D` submodule initialized.
  4. The `human_body_prior` package installed (in containers/requirements.txt).
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
from tqdm import tqdm

# AMASS subset folder names (HumanML3D-canonical, after AMASS rename mapping
# documented in the project plan). Excludes TCD_handMocap; includes humanact12.
AMASS_SUBSETS: tuple[str, ...] = (
    "ACCAD", "BMLhandball", "BMLmovi", "BioMotionLab_NTroje", "CMU",
    "DFaust_67", "EKUT", "Eyes_Japan_Dataset", "HumanEva", "KIT",
    "MPI_HDM05", "MPI_Limits", "MPI_mosh", "SFU", "SSM_synced",
    "TotalCapture", "Transitions_mocap",
)

EX_FPS = 20  # HumanML3D-canonical sampling rate
TRANS_MATRIX = np.array(
    [[1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0],
     [0.0, 1.0, 0.0]],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Stage 1: AMASS .npz → joint positions .npy
# ---------------------------------------------------------------------------


def stage_raw_pose(
    amass_root: Path,
    body_models: Path,
    out_dir: Path,
    device: str = "cuda",
    num_betas: int = 10,
) -> None:
    """Run SMPL+H FK on every AMASS .npz file and save 22-joint positions.

    Mirrors `external/HumanML3D/raw_pose_processing.ipynb`. Output files are
    `out_dir/<dataset>/<subject>/<motion>_poses.npy`, each of shape (T, 22, 3),
    in HumanML3D coords (Y-up, after `TRANS_MATRIX`), at 20 fps.
    """
    try:
        from human_body_prior.body_model.body_model import BodyModel
    except ImportError:
        sys.exit(
            "stage_raw_pose needs `human_body_prior`. Install via "
            "containers/requirements.txt (cluster) or `pip install "
            "git+https://github.com/nghorbani/human_body_prior.git`."
        )

    male_bm = BodyModel(
        bm_fname=str(body_models / "smplh/male/model.npz"),
        num_betas=num_betas,
        num_dmpls=8,
        dmpl_fname=str(body_models / "dmpls/male/model.npz"),
    ).to(device)
    female_bm = BodyModel(
        bm_fname=str(body_models / "smplh/female/model.npz"),
        num_betas=num_betas,
        num_dmpls=8,
        dmpl_fname=str(body_models / "dmpls/female/model.npz"),
    ).to(device)

    paths = []
    for subset in AMASS_SUBSETS:
        d = amass_root / subset
        if not d.exists():
            print(f"[warn] missing AMASS subset {subset} at {d}", file=sys.stderr)
            continue
        for npz in d.rglob("*.npz"):
            paths.append(npz)
    print(f"[stage_raw_pose] found {len(paths)} AMASS files")

    n_skipped = 0
    for npz in tqdm(paths, desc="raw_pose"):
        try:
            rel = npz.relative_to(amass_root).with_suffix(".npy")
            save_path = out_dir / rel
            if save_path.exists():
                # Resume support: skip clips already processed in a prior run.
                # Lets prep_data.sbatch reruns avoid the ~30-60 min SMPL+H FK.
                n_skipped += 1
                continue
            bdata = np.load(npz, allow_pickle=True)
            fps = float(bdata["mocap_framerate"])
            ds = max(int(round(fps / EX_FPS)), 1)

            poses = bdata["poses"][::ds]
            trans = bdata["trans"][::ds]
            betas = np.repeat(bdata["betas"][:num_betas][None], len(trans), axis=0)
            gender = str(bdata["gender"])
            bm = male_bm if gender == "male" else female_bm

            params = {
                "root_orient": torch.tensor(poses[:, :3], dtype=torch.float32, device=device),
                "pose_body":   torch.tensor(poses[:, 3:66], dtype=torch.float32, device=device),
                "pose_hand":   torch.tensor(poses[:, 66:], dtype=torch.float32, device=device),
                "trans":       torch.tensor(trans, dtype=torch.float32, device=device),
                "betas":       torch.tensor(betas, dtype=torch.float32, device=device),
            }
            with torch.no_grad():
                body = bm(**params)
            joints = body.Jtr.detach().cpu().numpy()         # (T, 52, 3)
            joints = joints[:, :22]                           # take HumanML3D's 22
            joints = joints @ TRANS_MATRIX                    # Y-up swap

            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, joints)
        except Exception as e:
            print(f"[stage_raw_pose] failed {npz}: {e}", file=sys.stderr)
    if n_skipped:
        print(f"[stage_raw_pose] skipped {n_skipped} already-processed AMASS files")

    # HumanAct12 pass-through: clips are (T, 24, 3) joint positions already
    # (no SMPL+H needed). Slice to 22 joints to match the AMASS-derived files.
    ha12_root = amass_root / "humanact12" / "humanact12"
    if ha12_root.exists():
        out_ha12 = out_dir / "humanact12" / "humanact12"
        out_ha12.mkdir(parents=True, exist_ok=True)
        ha12_files = list(ha12_root.glob("*.npy"))
        print(f"[stage_raw_pose] passing through {len(ha12_files)} HumanAct12 clips")
        n_skipped_ha12 = 0
        for npy in tqdm(ha12_files, desc="humanact12"):
            try:
                target = out_ha12 / npy.name
                if target.exists():
                    n_skipped_ha12 += 1
                    continue
                arr = np.load(npy)
                if arr.ndim == 2:                      # (T, 24*3) → reshape
                    arr = arr.reshape(arr.shape[0], -1, 3)
                arr = arr[:, :22].astype(np.float32)   # slice to 22 joints
                np.save(target, arr)
            except Exception as e:
                print(f"[stage_raw_pose] humanact12 failed {npy.name}: {e}", file=sys.stderr)
        if n_skipped_ha12:
            print(f"[stage_raw_pose] skipped {n_skipped_ha12} already-processed HumanAct12 files")
    else:
        print(f"[stage_raw_pose] no humanact12 at {ha12_root} — skipping pass-through")


# ---------------------------------------------------------------------------
# Stage 2: joints .npy → IK → T+R → packed zip
# ---------------------------------------------------------------------------


def _import_upstream_skeleton(humanml3d_repo: Path):
    """Vendor the HumanML3D Skeleton/IK at runtime (avoids re-implementing IK).

    The upstream code was written against numpy <1.20 and uses ``np.float``,
    which was removed in numpy 1.24+. Polyfill it before import. Also silence
    upstream's repeated ``torch.cross`` deprecation warning — it fires once per
    sample inside IK and floods the log.
    """
    import warnings
    if not hasattr(np, "float"):
        np.float = float
    warnings.filterwarnings(
        "ignore",
        message=r"Using torch\.cross without specifying the dim arg is deprecated.*",
    )
    sys.path.insert(0, str(humanml3d_repo))
    from common.skeleton import Skeleton
    from paramUtil import t2m_kinematic_chain, t2m_raw_offsets
    return Skeleton, t2m_raw_offsets, t2m_kinematic_chain


def _read_index_csv(path: Path) -> list[tuple[str, int, int, str]]:
    """Parse HumanML3D index.csv → list of (source_npy_relpath, start, end, new_name)."""
    rows = []
    with open(path) as f:
        next(f)  # header
        for line in f:
            cols = line.strip().split(",")
            if len(cols) != 4:
                continue
            src = cols[0]
            start = int(cols[1])
            end = int(cols[2])
            new_name = cols[3]
            rows.append((src, start, end, new_name))
    return rows


def _read_texts(humanml3d_repo: Path) -> dict[str, list[str]]:
    """Decode HumanML3D/texts.zip → {clip_id: [caption_1, ...]}.

    HumanML3D/texts.zip contains <clip_id>.txt files where each line has the
    form `<caption>#<pos_tags>#<start>#<end>` (start=0,end=0 means full clip).
    We keep the captions (first '#'-split field) only.
    """
    out: dict[str, list[str]] = {}
    zpath = humanml3d_repo / "HumanML3D" / "texts.zip"
    with zipfile.ZipFile(zpath, "r") as zf:
        for name in zf.namelist():
            if not name.endswith(".txt"):
                continue
            clip_id = Path(name).stem
            captions = []
            for line in zf.read(name).decode("utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                cap = line.split("#", 1)[0].strip()
                if cap:
                    captions.append(cap)
            if captions:
                out[clip_id] = captions
    return out


def _read_split_ids(humanml3d_repo: Path, split: str) -> list[str]:
    p = humanml3d_repo / "HumanML3D" / f"{split}.txt"
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def stage_pack(
    joints_root: Path,
    humanml3d_repo: Path,
    output_dir: Path,
    example_id: str = "000021",
    feet_thre: float = 0.002,
) -> None:
    """Slice clips per index.csv, run upstream IK to get H3D-coord quats,
    save (translation, quats, texts) per clip into a single zip.
    """
    Skeleton, raw_offsets, kinematic_chain = _import_upstream_skeleton(humanml3d_repo)

    n_raw_offsets = torch.from_numpy(raw_offsets)
    skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")

    # ---- target T-pose offsets, derived from the canonical example clip ----
    example_path = joints_root / f"{example_id}.npy"
    if not example_path.exists():
        # The example_id in the upstream notebook is post-rename; many users
        # invoke stage_pack on pre-rename joints. Try to find it via index.csv.
        index = _read_index_csv(humanml3d_repo / "index.csv")
        match = [src for (src, _s, _e, name) in index if Path(name).stem == example_id]
        if match:
            example_path = joints_root / Path(match[0]).relative_to("./pose_data")
    example_data = np.load(example_path).reshape(-1, 22, 3)
    tgt_offsets = skel.get_offsets_joints(torch.from_numpy(example_data[0]))  # (22, 3)
    skel.set_offset(tgt_offsets)
    target_offsets_t = tgt_offsets.detach().clone().float()

    # ---- texts ----
    print("[stage_pack] reading texts ...")
    texts_by_clip = _read_texts(humanml3d_repo)

    # ---- splits ----
    splits = {
        "train": _read_split_ids(humanml3d_repo, "train"),
        "val":   _read_split_ids(humanml3d_repo, "val"),
        "test":  _read_split_ids(humanml3d_repo, "test"),
    }
    all_split_ids = set().union(*splits.values())

    # ---- iterate index.csv, slice + IK + pack ----
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_out = output_dir / "humanml3d.zip"
    n_written = 0
    n_skipped_existing = 0

    # Resume support: if a prior run was killed mid-pack, append rather than
    # rewrite. A corrupted (no-central-directory) zip from a hard-killed
    # process is detected and restarted.
    existing_ids: set[str] = set()
    zip_mode = "w"
    if zip_out.exists():
        try:
            with zipfile.ZipFile(zip_out, "r") as zf:
                existing_ids = {Path(n).stem for n in zf.namelist()}
            zip_mode = "a"
            print(f"[stage_pack] resuming: {len(existing_ids)} clips already in {zip_out.name}")
        except zipfile.BadZipFile:
            print(f"[stage_pack] {zip_out.name} is corrupted (probably killed mid-write); restarting")
            zip_out.unlink()

    print(f"[stage_pack] {'appending to' if zip_mode == 'a' else 'writing'} {zip_out}")
    with zipfile.ZipFile(zip_out, zip_mode, compression=zipfile.ZIP_STORED) as zf:
        index = _read_index_csv(humanml3d_repo / "index.csv")
        for src, start, end, new_name in tqdm(index, desc="pack"):
            clip_id = Path(new_name).stem
            if clip_id in existing_ids:
                n_skipped_existing += 1
                continue
            if clip_id not in all_split_ids:
                continue  # not part of any split
            src_rel = src.replace("./pose_data/", "")
            joints_path = joints_root / src_rel
            if not joints_path.exists():
                # silently skip missing files; user gets a count summary
                continue
            joints = np.load(joints_path).reshape(-1, 22, 3)
            joints = joints[start:end] if end > 0 else joints[start:]
            if joints.shape[0] < 40:
                continue
            joints = torch.from_numpy(joints).float()

            # IK on the (already H3D-coord) joint positions → per-joint quats.
            # `inverse_kinematics_np` returns quats in HumanML3D's [w, x, y, z]
            # convention with quaternion-discontinuity fix applied via `qfix`.
            face_joint_indx = [2, 1, 17, 16]  # R_Hip, L_Hip, R_Shoulder, L_Shoulder
            quat_params = skel.inverse_kinematics_np(
                joints.numpy(), face_joint_indx, smooth_forward=False
            )  # (T, 22, 4)
            # Root translation == joint 0 position
            translation = joints[:, 0, :].clone()  # (T, 3)
            quats = torch.from_numpy(quat_params).float()  # (T, 22, 4)

            captions = texts_by_clip.get(clip_id, [])
            if not captions:
                continue

            blob = {"translation": translation, "quats": quats, "texts": captions}
            buf = io.BytesIO()
            torch.save(blob, buf)
            zf.writestr(f"{clip_id}.pt", buf.getvalue())
            n_written += 1

    print(f"[stage_pack] wrote {n_written} new clips → {zip_out}")
    if n_skipped_existing:
        print(f"[stage_pack] skipped {n_skipped_existing} already-packed clips")

    # Filter splits to clips actually present
    present = set()
    with zipfile.ZipFile(zip_out, "r") as zf:
        for name in zf.namelist():
            present.add(Path(name).stem)
    splits_filtered = {k: [c for c in v if c in present] for k, v in splits.items()}

    (output_dir / "splits.json").write_text(json.dumps(splits_filtered))
    torch.save(target_offsets_t, output_dir / "target_offsets.pt")
    (output_dir / "meta.json").write_text(json.dumps({
        "fps": EX_FPS,
        "num_clips": n_written,
        "version": 1,
    }))
    print(f"[stage_pack] split sizes: " + ", ".join(
        f"{k}={len(v)}" for k, v in splits_filtered.items()
    ))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    rp = sub.add_parser("raw-pose", help="Stage 1: AMASS .npz → joints .npy")
    rp.add_argument("--amass-root", type=Path, required=True, help="dir containing the 17 AMASS subset folders")
    rp.add_argument("--body-models", type=Path, required=True, help="dir with smplh/{male,female}/model.npz and dmpls/...")
    rp.add_argument("--out-dir", type=Path, required=True, help="output dir for joint .npy files")
    rp.add_argument("--device", default="cuda")

    pk = sub.add_parser("pack", help="Stage 2: joints → IK → packed humanml3d.zip")
    pk.add_argument("--joints-root", type=Path, required=True, help="output dir from raw-pose stage")
    pk.add_argument("--humanml3d-repo", type=Path, default=Path("external/HumanML3D"))
    pk.add_argument("--output-dir", type=Path, default=Path("external/data/humanml3d_packed"))
    pk.add_argument("--example-id", default="000021")

    args = p.parse_args()

    if args.cmd == "raw-pose":
        stage_raw_pose(
            amass_root=args.amass_root,
            body_models=args.body_models,
            out_dir=args.out_dir,
            device=args.device,
        )
    elif args.cmd == "pack":
        stage_pack(
            joints_root=args.joints_root,
            humanml3d_repo=args.humanml3d_repo,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
