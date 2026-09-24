"""Stage B validation steps 2-3: the port against upstream on the same GPU.

  model    same checkpoint, same inputs: guided velocity of our ACMDM vs
           upstream `forward_with_CFG` (should agree to float precision)
  text     our transformers CLIP (fp16 and fp32) vs upstream OpenAI CLIP (fp16)
  sampler  full 100-step ProjFlow run, same noise / keyframes / mixing ε, with
           our text embeddings fed to both (isolates the sampler + model)
  joints   our recover_joints_from_ric vs a surviving original HumanML3D
           new_joints file

Needs upstream's environment (OpenAI `clip`, timm 1.0.9, and the numpy/torch
shims): run through slurm/projflow/check_port.sbatch. Writes a JSON report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from projflow.models import ACMDM, attention_mask_from_lengths
from projflow.sampler import ProjFlowConfig, projflow_sample
from projflow.text import ClipTextEncoder
from shared.geometry.humanml3d_io import recover_joints_from_ric

TEXTS = [
    "a person walks forward and turns left",
    "someone jumps up and down twice",
    "a man waves his right hand while standing still",
    "the person crouches, then runs in a circle",
]


def diff(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.double().flatten(), b.double().flatten()
    return {
        "max_abs": float((a - b).abs().max()),
        "rel_l2": float((a - b).norm() / b.norm().clamp_min(1e-12)),
        "cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--upstream", default="external/ProjFlow")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--canonical-dir", required=True)
    p.add_argument("--golden-new-joints", nargs="*", default=[])
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=100)
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    report: dict = {}

    sys.path.insert(0, str(Path(args.upstream).resolve()))
    from models.ACMDM import ACMDM_models  # noqa: E402  (upstream)

    up = ACMDM_models["ACMDM-Raw-Flow-S-PatchSize22"](input_dim=3, cond_mode="text")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = up.load_state_dict(ckpt["ema_acmdm"], strict=False)
    assert not unexpected and all(k.startswith("clip_model.") for k in missing)
    up = up.to(device).eval()
    ours = ACMDM.from_upstream_checkpoint(args.checkpoint).to(device)
    report["params_ours_M"] = sum(p.numel() for p in ours.parameters()) / 1e6

    # --- text encoder --------------------------------------------------------
    with torch.no_grad():
        up_text = up.encode_text(TEXTS)
    enc16 = ClipTextEncoder(device, torch.float16)
    report["text_fp16"] = diff(enc16(TEXTS), up_text)
    report["text_fp32"] = diff(ClipTextEncoder(device, torch.float32)(TEXTS), up_text)

    # --- model velocity ------------------------------------------------------
    B, L = len(TEXTS), 196
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(B, 3, L, 22, device=device, generator=g)
    t = torch.rand(B, device=device, generator=g)
    lengths = torch.tensor([196, 160, 120, 64], device=device)
    attn = attention_mask_from_lengths(lengths, L)
    cond = up_text
    with torch.no_grad():
        v_up = up.forward_with_CFG(torch.cat([x, x]), torch.cat([t, t]), conds=torch.cat([cond, torch.zeros_like(cond)]),
                                   attention_mask=attn.repeat(2, 1, 1, 1), cfg=3.0)[:B]
        v_ours = ours.guided_velocity(x, t, cond, attn, 3.0)
    valid = attn[:, 0, 0][:, None, :, None].expand_as(v_ours)
    report["velocity_cfg3"] = diff(v_ours[valid], v_up[valid])

    # --- full sampler --------------------------------------------------------
    mask = torch.zeros(B, 3, L, 22, device=device)
    for b, k in enumerate((1, 2, 5, 30)):
        frames = torch.randperm(int(lengths[b]), device=device, generator=g)[:k]
        mask[b, :, frames, 0] = 1.0
    y = torch.randn(B, 3, L, 22, device=device, generator=g) * mask
    noise0 = torch.randn(B, 3, L, 22, device=device, generator=g)
    model_kwargs = dict(conds=torch.cat([cond, torch.zeros_like(cond)]), attention_mask=attn.repeat(2, 1, 1, 1),
                        cfg=3.0, A=mask.repeat(2, 1, 1, 1), y=y.repeat(2, 1, 1, 1))

    torch.manual_seed(1234)
    with torch.no_grad():
        x_up = up.gen_diffusion.sample_projflow(num_steps=args.steps)(
            torch.cat([noise0, noise0]), up.forward_with_CFG, **model_kwargs)[-1][:B]

    torch.manual_seed(1234)
    with torch.no_grad():  # upstream draws ε for both CFG halves; take the conditional half
        x_ours = projflow_sample(lambda xx, tt: ours.guided_velocity(xx, tt, cond, attn, 3.0), noise0, mask, y,
                                 ProjFlowConfig(num_steps=args.steps),
                                 noise=lambda tmpl: torch.randn(2 * B, *tmpl.shape[1:], device=device)[:B])
    report["sampler"] = diff(x_ours[valid], x_up[valid])
    hard = mask > 0.5
    report["sampler_keyframe_err_ours"] = float((x_ours[hard] - y[hard]).abs().max())
    report["sampler_keyframe_err_upstream"] = float((x_up[hard] - y[hard]).abs().max())

    # --- joints from canonical features --------------------------------------
    for golden in map(Path, args.golden_new_joints):
        vecs = torch.from_numpy(np.load(Path(args.canonical_dir) / golden.name)).float()
        report[f"joints_{golden.stem}"] = diff(recover_joints_from_ric(vecs), torch.from_numpy(np.load(golden)))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
