#!/bin/bash
#SBATCH --job-name=projflow-setup
#SBATCH --partition=12g
#SBATCH --nodelist=frankfurt,hamburg,kassel,leipzig
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/projflow-setup-%j.out
#SBATCH --error=slurm/logs/projflow-setup-%j.err
#
# Stage A (faithful upstream reproduction) — one-time asset setup.
#
# Upstream ProjFlow resolves checkpoints/, glove/, datasets/ and utils/ relative
# to the working directory. To keep the external/ProjFlow submodule pristine we
# build a separate work dir that symlinks every upstream top-level entry and
# holds the downloaded assets next to them:
#
#   ${WORK}/
#     evaluation_ProjFlow.py, models/, utils/, ...   -> external/ProjFlow/*
#     checkpoints/t2m/ACMDM_Raw_Flow_S_PatchSize22/  generator (HF: Akihisa-Watanabe/ProjFlow)
#     checkpoints/t2m/text_mot_match{,_clip}/        67-D evaluators (HF: cr8br0ze/evaluators_humanml3d)
#     glove/our_vab_*                                GloVe (ACMDM's gdrive archive)
#     datasets/HumanML3D/{texts,*.txt}               from external/HumanML3D/HumanML3D
#     datasets/HumanML3D/new_joints/                 rebuilt from canonical new_joint_vecs
#                                                    (the share keeps 1 original file; it is
#                                                    used as the golden check)
#     pyuser/                                        isolated pip --user site (clip, timm, ...)
#
# timm is pinned to upstream's 1.0.9: newer timm passes device= to the
# norm_layer, which upstream's LlamaRMSNorm rejects.
# The container has no wget/unzip; downloads go through huggingface_hub and
# archives through python -m zipfile.
# Idempotent: each step skips when its output is already in place.
#
# Usage:
#     sbatch slurm/projflow/setup_upstream.sbatch

set -euo pipefail
mkdir -p slurm/logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    REPO="${REPO:-$(dirname "$(dirname "${SCRIPT_DIR}")")}"
fi
IMAGE="${IMAGE:-${HOME}/rmg.sqsh}"
HF_CACHE="${HF_CACHE:-${HOME}/hf-cache}"
WORK="${PROJFLOW_WORK:-${HOME}/projflow-work}"
H3D_SRC="${H3D_SRC:-${REPO}/external/HumanML3D/HumanML3D}"
NEW_JOINT_VECS="${NEW_JOINT_VECS:-${HOME}/rmg-runs/h3d-canonical/new_joint_vecs}"
GEN_NAME="ACMDM_Raw_Flow_S_PatchSize22"
GLOVE_GDRIVE_ID="1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n"   # ACMDM README glove link

cd "${REPO}"
test -f external/ProjFlow/evaluation_ProjFlow.py || git submodule update --init -- external/ProjFlow

enroot start \
    --rw \
    --mount /mnt:mnt \
    --mount /tmp:tmp \
    --env "HF_HOME=${HF_CACHE}" \
    --env "PYTHONUSERBASE=${WORK}/pyuser" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        mkdir -p '${WORK}' '${WORK}/downloads'
        cd '${WORK}'

        echo '[setup] linking upstream code'
        for entry in '${REPO}'/external/ProjFlow/*; do
            ln -sfn \"\${entry}\" \"\$(basename \"\${entry}\")\"
        done

        echo '[setup] python deps -> ${WORK}/pyuser'
        pip install --user --quiet ftfy regex imageio gdown
        pip install --user --quiet --no-deps timm==1.0.9 git+https://github.com/openai/CLIP.git

        echo '[setup] generator checkpoint'
        if [ ! -f checkpoints/t2m/${GEN_NAME}/model/latest.tar ]; then
            python -c \"from huggingface_hub import snapshot_download; snapshot_download('Akihisa-Watanabe/ProjFlow', allow_patterns=['checkpoints/t2m/${GEN_NAME}/**'], local_dir='.')\"
        fi

        echo '[setup] 67-D evaluators'
        if [ ! -f checkpoints/t2m/text_mot_match/model/finest.tar ] || [ ! -f checkpoints/t2m/text_mot_match_clip/model/finest.tar ]; then
            [ -f downloads/evaluators.zip ] && python -m zipfile -t downloads/evaluators.zip >/dev/null 2>&1 \
                || python -c \"from huggingface_hub import hf_hub_download; import shutil; shutil.copy(hf_hub_download('cr8br0ze/evaluators_humanml3d', 'evaluators_humanml3d.zip'), 'downloads/evaluators.zip')\"
            rm -rf downloads/evaluators && python -m zipfile -e downloads/evaluators.zip downloads/evaluators
            mkdir -p checkpoints/t2m
            for name in text_mot_match text_mot_match_clip; do
                src=\$(find downloads/evaluators -type d -name \"\${name}\" | head -1)
                [ -n \"\${src}\" ] || { echo \"[setup] ERROR: \${name} not in evaluators zip\"; find downloads/evaluators -maxdepth 4; exit 1; }
                rm -rf checkpoints/t2m/\${name} && cp -r \"\${src}\" checkpoints/t2m/\${name}
            done
        fi

        echo '[setup] GloVe'
        if [ ! -f glove/our_vab_data.npy ]; then
            [ -f downloads/glove.zip ] && python -m zipfile -t downloads/glove.zip >/dev/null 2>&1 \
                || python -m gdown '${GLOVE_GDRIVE_ID}' -O downloads/glove.zip
            rm -rf downloads/glove && python -m zipfile -e downloads/glove.zip downloads/glove
            src=\$(dirname \"\$(find downloads/glove -name our_vab_data.npy | head -1)\")
            mkdir -p glove && cp \"\${src}\"/our_vab_* glove/
        fi

        echo '[setup] HumanML3D view'
        mkdir -p datasets/HumanML3D
        for entry in train.txt val.txt test.txt train_val.txt all.txt; do
            ln -sfn '${H3D_SRC}'/\${entry} datasets/HumanML3D/\${entry}
        done
        [ -L datasets/HumanML3D/new_joints ] && rm datasets/HumanML3D/new_joints
        PYTHONPATH='${REPO}/src:${REPO}/slurm/projflow/compat' python -u -m projflow.scripts.build_new_joints \
            --vecs-dir '${NEW_JOINT_VECS}' \
            --out-dir datasets/HumanML3D/new_joints \
            --upstream '${REPO}/external/ProjFlow' \
            --golden '${H3D_SRC}'/new_joints/*.npy
        if [ ! -d datasets/HumanML3D/texts ]; then
            rm -rf downloads/texts && python -m zipfile -e '${H3D_SRC}'/texts.zip downloads/texts
            src=\$(dirname \"\$(find downloads/texts -name '000000.txt' | head -1)\")
            mv \"\${src}\" datasets/HumanML3D/texts
        fi

        echo '[setup] verifying'
        python - <<'PY'
import os, numpy as np
root = 'datasets/HumanML3D'
test = [l.strip() for l in open(f'{root}/test.txt') if l.strip()]
miss_j = [n for n in test if not os.path.exists(f'{root}/new_joints/{n}.npy')]
miss_t = [n for n in test if not os.path.exists(f'{root}/texts/{n}.txt')]
shape = np.load(f'{root}/new_joints/{test[0]}.npy').shape
print(f'test clips: {len(test)}  missing joints: {len(miss_j)}  missing texts: {len(miss_t)}  sample shape: {shape}')
for p in ['checkpoints/t2m/${GEN_NAME}/model/latest.tar',
          'checkpoints/t2m/text_mot_match/model/finest.tar',
          'checkpoints/t2m/text_mot_match_clip/model/finest.tar',
          'glove/our_vab_data.npy', 'glove/our_vab_idx.pkl', 'glove/our_vab_words.pkl']:
    print(('OK      ' if os.path.exists(p) else 'MISSING ') + p)
assert not miss_j and not miss_t and shape[1:] == (22, 3)
PY
        python -c 'import clip, timm; print(\"[setup] clip + timm import OK\")'
        echo '[setup] DONE'
    "
