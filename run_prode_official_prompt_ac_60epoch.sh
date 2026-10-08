#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON=/home/yishouzhuo/miniconda3/envs/sfa/bin/python
CONDA=/home/yishouzhuo/miniconda3/bin/conda
RUN_ROOT=${RUN_ROOT:-./output/prode_official_prompt_ac_60epoch}
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export PYTHONHASHSEED=2020
export CUBLAS_WORKSPACE_CONFIG=:4096:8

ARGS=(
    image_target_of_oh_vs_new.py --cfg cfgs/office-home/ProDe.yaml
    MODEL.METHOD ProDe_old
    SETTING.S 0 SETTING.T 1 SETTING.OUTPUT_SRC fraction_full SETTING.SEED 2020
    TEST.MAX_EPOCH 60 TEST.INTERVAL 30
    CUDNN.BENCHMARK False CUDNN.DETERMINISTIC True
    SAVE_DIR "$RUN_ROOT"
)

mkdir -p "$RUN_ROOT/.completed"
exec > >(tee -a "$RUN_ROOT/runner.log") 2>&1

cp cfgs/office-home/ProDe.yaml "$RUN_ROOT/ProDe.yaml"
"$CONDA" list -p /home/yishouzhuo/miniconda3/envs/sfa --explicit > "$RUN_ROOT/conda-explicit.txt"
"$PYTHON" -m pip freeze > "$RUN_ROOT/pip-freeze.txt"
cut -d' ' -f1 data/office-home/Art_list.txt data/office-home/Clipart_list.txt \
    | sort -u | xargs sha256sum > "$RUN_ROOT/dataset-images.sha256"

{
    echo "created_at=$(date -Is)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "git_describe=$(git describe --tags --always)"
    echo "git_tracked_status=$(git status --porcelain --untracked-files=no | tr '\n' ';')"
    echo "local_baseline_tag=prode-old-local-before-official-20261008"
    echo "local_baseline_commit=$(git rev-parse prode-old-local-before-official-20261008^{})"
    echo "upstream_repository=https://github.com/tntek/source-free-domain-adaptation"
    echo "upstream_commit=5f7fe578c64ad84a9126157c3aab3aa4ba825bc6"
    echo "upstream_ProDe_sha256=5fcf85be796689986463cb843e84f04b72b0cfb6f3db5666b1756788c232805e"
    echo "python=$PYTHON"
    echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
    echo "pythonhashseed=$PYTHONHASHSEED"
    echo "cublas_workspace_config=$CUBLAS_WORKSPACE_CONFIG"
    printf 'command='
    printf '%q ' "$PYTHON" "${ARGS[@]}"
    printf '\n'
    uname -a
    "$PYTHON" -c 'import torch, torchvision; print("torch=" + torch.__version__); print("cuda=" + str(torch.version.cuda)); print("torchvision=" + torchvision.__version__)'
    nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
    sha256sum \
        src/methods/oh/ProDe_old.py \
        src/utils/IID_losses_official.py \
        cfgs/office-home/ProDe.yaml \
        data/office-home/Art_list.txt \
        data/office-home/Clipart_list.txt \
        data/office-home/classname.txt \
        Models/fraction_full/uda/office-home/A/source_F.pt \
        Models/fraction_full/uda/office-home/A/source_B.pt \
        Models/fraction_full/uda/office-home/A/source_C.pt \
        "$RUN_ROOT/dataset-images.sha256"
} > "$RUN_ROOT/manifest.txt"

cat "$RUN_ROOT/manifest.txt"

if [[ ! -f "$RUN_ROOT/.completed/ProDe_official" ]]; then
    "$PYTHON" "${ARGS[@]}"
    touch "$RUN_ROOT/.completed/ProDe_official"
fi

echo "[$(date -Is)] complete method=ProDe_official task=AC epochs=60"
