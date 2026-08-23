#!/bin/bash
#SBATCH --job-name=momask-mtrans
#SBATCH --partition=12g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1,ccc:75
#SBATCH --nodes=1
#SBATCH --output=slurm/logs/momask-mtrans-paperstyle-%j.out
#SBATCH --error=slurm/logs/momask-mtrans-paperstyle-%j.err
#
# Paper-faithful M-Transformer training on canonical HumanML3D motions up to
# 196 frames. The compute-capability filter avoids the 12g partition's unsupported
# compute-capability 5.2/6.1 cards.

set -euo pipefail

export TOKEN_STAGE=masked
export RUN_NAME="${RUN_NAME:-momask-canonical-mtrans-vq150k-maskfix196-clip500e}"

bash slurm/momask/train_momask_tokens_paperstyle.sbatch
