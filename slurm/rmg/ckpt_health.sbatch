#!/bin/bash
#SBATCH --job-name=rmg-ckpt-health
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:30:00
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
import os, glob, torch

# CKPT may be: a single file, a comma-separated list, a glob, or a directory
# (scans ckpt-*.pt inside, step-sorted). Multiple checkpoints -> one compact
# summary line each so you can bisect to the last healthy step in one job.
raw = os.environ["CKPT"]
paths = []
for entry in raw.split(","):
    entry = entry.strip()
    if not entry:
        continue
    if os.path.isdir(entry):
        paths.extend(sorted(glob.glob(os.path.join(entry, "ckpt-*.pt"))))
    else:
        paths.extend(sorted(glob.glob(entry)) or [entry])
if not paths:
    raise SystemExit(f"[ckpt-health] no checkpoints matched: {raw}")

def count_bad(d):
    if not d:
        return 0, 0
    floats = [v for v in d.values()
              if torch.is_tensor(v) and torch.is_floating_point(v)]
    bad = sum(1 for v in floats if not torch.isfinite(v).all())
    return bad, len(floats)

def magnitudes(d):
    # Max |w| over all params, and the name of the largest-magnitude tensor.
    # Growing max|w| across steps == the model is diverging (weights exploding).
    if not d:
        return 0.0, 0.0, "-"
    mx, mx_name, ssum, scount = 0.0, "-", 0.0, 0
    for k, v in d.items():
        if not (torch.is_tensor(v) and torch.is_floating_point(v)):
            continue
        finite = v[torch.isfinite(v)]
        if finite.numel() == 0:
            return float("inf"), float("inf"), k
        m = finite.abs().max().item()
        if m > mx:
            mx, mx_name = m, k
        ssum += finite.abs().sum().item()
        scount += finite.numel()
    return mx, (ssum / max(scount, 1)), mx_name

print(f"[ckpt-health] scanning {len(paths)} checkpoint(s)", flush=True)
for path in paths:
    p = torch.load(path, map_location="cpu", weights_only=False)
    step = p.get("step")
    mb, mt = count_bad(p.get("model"))
    ema = p.get("ema")
    eb, et = count_bad(ema.get("shadow") if ema else None)
    opt = p.get("optimizer")
    ob = ot = 0
    if opt and "state" in opt:
        exp = {i: s["exp_avg"] for i, s in opt["state"].items() if "exp_avg" in s}
        ob, ot = count_bad(exp)
    mx, mean_abs, mx_name = magnitudes(p.get("model"))
    tag = "OK " if (mb == 0 and eb == 0) else "BAD"
    print(f"[ckpt-health] {tag} step={step} {os.path.basename(path)} "
          f"model={mb}/{mt} ema={eb}/{et} opt_exp_avg={ob}/{ot} "
          f"max|w|={mx:.3g} mean|w|={mean_abs:.3g} @{mx_name}", flush=True)
    del p
PY
    '
