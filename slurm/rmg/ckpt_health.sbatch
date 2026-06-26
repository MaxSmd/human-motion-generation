#!/bin/bash
#SBATCH --job-name=rmg-ckpt-health
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/rmg-ckpt-health-%j.out
#SBATCH --error=slurm/logs/rmg-ckpt-health-%j.err
#
# Diagnostic: report whether a checkpoint's tensors are finite. Splits the
# question of "did the model diverge" (model state_dict NaN) from "is sampling
# the bug" (model finite but samples NaN). CPU-only work, runs in seconds.
#
#   CKPT=/path/to/latest.pt sbatch slurm/rmg/ckpt_health.sbatch
#
# Reads model state_dict, EMA shadow, and (cheaply) optimizer exp_avg tensors,
# printing per-section tensor counts, how many are non-finite, and a few example
# keys + the magnitude range of the worst offenders.

set -euo pipefail
mkdir -p slurm/logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(dirname "${SCRIPT_DIR}")}"
fi
IMAGE="${IMAGE:-${HOME}/rmg.sqsh}"
CKPT="${CKPT:-}"

if [ -z "$CKPT" ]; then
    echo "ERROR: set CKPT=/path/to/latest.pt"; exit 1
fi

enroot start \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --env "PYTHONPATH=${REPO}/src" \
    --env "CKPT=${CKPT}" \
    "${IMAGE}" \
    bash -lc '
        set -euo pipefail
        cd '"${REPO}"'
        python -u - <<"PY"
import os, torch

ckpt = os.environ["CKPT"]
print(f"[ckpt-health] loading {ckpt}", flush=True)
p = torch.load(ckpt, map_location="cpu", weights_only=False)
print(f"[ckpt-health] step = {p.get(\"step\")}", flush=True)

def report(name, d):
    if not d:
        print(f"[ckpt-health] {name}: (absent)", flush=True)
        return
    floats = {k: v for k, v in d.items()
              if torch.is_tensor(v) and torch.is_floating_point(v)}
    bad = [k for k, v in floats.items() if not torch.isfinite(v).all()]
    print(f"[ckpt-health] {name}: {len(floats)} float tensors, "
          f"{len(bad)} non-finite", flush=True)
    for k in bad[:8]:
        v = floats[k]
        nbad = int((~torch.isfinite(v)).sum())
        print(f"    NONFINITE {k}: {tuple(v.shape)} "
              f"({nbad}/{v.numel()} bad)", flush=True)
    # Show the magnitude range of a healthy-looking tensor for sanity.
    finite_keys = [k for k in floats if k not in set(bad)]
    if finite_keys:
        k0 = finite_keys[0]
        v0 = floats[k0]
        print(f"    sample finite tensor {k0}: "
              f"min={v0.min().item():.3e} max={v0.max().item():.3e}", flush=True)

report("model", p.get("model"))
ema = p.get("ema")
report("ema.shadow", ema.get("shadow") if ema else None)

# Optimizer Adam moment buffers — NaN here means a NaN gradient was applied,
# which pinpoints divergence even if a later step partially recovered params.
opt = p.get("optimizer")
if opt and "state" in opt:
    exp_avg = {f"state[{i}].exp_avg": s["exp_avg"]
               for i, s in opt["state"].items() if "exp_avg" in s}
    report("optimizer.exp_avg", exp_avg)
PY
    '
