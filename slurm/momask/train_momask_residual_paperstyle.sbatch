#!/bin/bash
#SBATCH --job-name=momask-rtrans
#SBATCH --partition=12g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1,ccc:75
#SBATCH --nodes=1
#SBATCH --output=slurm/logs/momask-rtrans-paperstyle-%j.out
#SBATCH --error=slurm/logs/momask-rtrans-paperstyle-%j.err
#
# Paper-faithful R-Transformer training on canonical HumanML3D motions up to
# 196 frames. The compute-capability filter avoids the 12g partition's unsupported
# compute-capability 5.2/6.1 cards.

set -euo pipefail

export TOKEN_STAGE=residual
export RUN_NAME="${RUN_NAME:-momask-canonical-rtrans-releaseparity196-clip500e}"

bash slurm/momask/train_momask_tokens_paperstyle.sbatch
