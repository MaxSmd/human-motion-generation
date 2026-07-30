"""Quantify the pack's IK→FK round-trip error against official HumanML3D files.

The eval harness featurizes REAL clips from the pack (positions → IK at pack
time → stored quats → FK at eval → process_file), while HumanML3D's official
`new_joint_vecs` were computed from the ORIGINAL positions. Real-vs-real FID
0.55 and real R@1 ~0.32 (published: 0.002 / 0.51) suggest this round trip
shifts the whole real-feature cloud. This script measures that directly on a
clip present both in the pack and as official files (012314 in our sparse
submodule checkout):

  1. |FK(pack quats) − new_joints|            → positional round-trip error (m)
  2. |to_h3d_features(pack) − new_joint_vecs| → 263-D feature error, per group
  3. Guo-embedding distance between both      → what the evaluator actually sees

Run (inside the eval container; needs the pack + text-to-motion evaluator):
    python -m rmg.scripts.eval_roundtrip_check train=rmg_mid \\
        +eval.evaluator=real +check.clip_id=012314
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from shared.data.humanml3d import read_clip
from shared.geometry import normalize_quaternions

from rmg.scripts.evaluate import _build_evaluator, _load_target_offsets
from rmg.representation import build_representation

# 263-D HumanML3D feature layout (22 joints)
GROUPS = {
    "root (rot-vel, lin-vel, height)": (0, 4),
    "ric positions": (4, 4 + 21 * 3),
    "cont6d rotations": (4 + 21 * 3, 4 + 21 * 3 + 21 * 6),
    "local velocities": (4 + 21 * 3 + 21 * 6, 4 + 21 * 3 + 21 * 6 + 22 * 3),
    "foot contacts": (259, 263),
}


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    check_cfg = OmegaConf.create({"clip_id": "012314", "pack_zip": "", "humanml3d_repo": "external/HumanML3D"})
    with open_dict(cfg):
        cfg.check = OmegaConf.merge(check_cfg, cfg.get("check", OmegaConf.create({})))
        cfg.eval = OmegaConf.merge(
            OmegaConf.create({
                "evaluator": "real",
                "text_to_motion_repo": "external/text-to-motion",
                "humanml3d_repo": "external/HumanML3D",
            }),
            cfg.get("eval", OmegaConf.create({})),
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cid = str(cfg.check.clip_id)

    h3d = Path(cfg.check.humanml3d_repo) / "HumanML3D"
    official_pos = torch.from_numpy(np.load(h3d / "new_joints" / f"{cid}.npy")).float()
    official_feat = torch.from_numpy(np.load(h3d / "new_joint_vecs" / f"{cid}.npy")).float()

    pack_zip = str(cfg.check.pack_zip) or str(Path(cfg.data.root) / "humanml3d.zip")
    with zipfile.ZipFile(pack_zip) as zf:
        translation, quats, _texts = read_clip(zf, cid)

    rep_kwargs = {k: v for k, v in dict(cfg.representation).items() if k not in ("name",)}
    representation = build_representation(cfg.representation.name, **rep_kwargs)
    skeleton = _load_target_offsets(cfg)

    # 1) positional round trip
    from shared.geometry.skeleton import forward_kinematics
    fk_pos = forward_kinematics(skeleton, normalize_quaternions(quats), translation)  # (T, 22, 3)
    T = min(fk_pos.shape[0], official_pos.shape[0])
    pos_err = (fk_pos[:T] - official_pos[:T]).norm(dim=-1)  # (T, 22) metres
    print(f"[roundtrip] clip {cid}: T_pack={fk_pos.shape[0]} T_official={official_pos.shape[0]}")
    print(f"[roundtrip] joint position error (m): mean {pos_err.mean():.4f}  "
          f"p95 {pos_err.quantile(0.95):.4f}  max {pos_err.max():.4f}")

    # 2) feature error per group
    x1 = representation.encode_clip(translation, quats, skeleton=skeleton)
    ours = representation.to_h3d_features(x1[:T], skeleton)  # (T-1, 263)
    Tf = min(ours.shape[0], official_feat.shape[0])
    d = (ours[:Tf] - official_feat[:Tf]).abs()
    scale = official_feat[:Tf].abs().mean(dim=0).clamp_min(1e-6)
    print(f"[roundtrip] 263-D |diff|: mean {d.mean():.4f}  max {d.max():.4f}")
    for name, (a, b) in GROUPS.items():
        rel = (d[:, a:b].mean() / scale[a:b].mean()).item()
        print(f"  {name:36s} mean|diff| {d[:, a:b].mean():.4f}  rel {rel:6.1%}")

    # 3) what the evaluator sees
    evaluator = _build_evaluator(cfg, device)
    lens = torch.tensor([Tf])
    e_ours = evaluator.encode_motion(ours[None, :Tf], lens)
    e_off = evaluator.encode_motion(official_feat[None, :Tf], lens)
    cos = torch.nn.functional.cosine_similarity(e_ours, e_off).item()
    l2 = (e_ours - e_off).norm().item()
    print(f"[roundtrip] Guo embedding: cosine {cos:.4f}  L2 {l2:.3f}  "
          f"(norms {e_ours.norm():.3f} / {e_off.norm():.3f})")


if __name__ == "__main__":
    main()
