#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON=/home/yishouzhuo/miniconda3/envs/sfa/bin/python
CONDA=/home/yishouzhuo/miniconda3/bin/conda
RUN_ROOT=${RUN_ROOT:-./output/prode_adaptive_prompt_12tasks_60epoch}
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONHASHSEED=2020
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "$RUN_ROOT/.completed"
exec > >(tee -a "$RUN_ROOT/runner.log") 2>&1

cp cfgs/office-home/ProDe.yaml "$RUN_ROOT/ProDe.yaml"
"$CONDA" list -p /home/yishouzhuo/miniconda3/envs/sfa --explicit > "$RUN_ROOT/conda-explicit.txt"
"$PYTHON" -m pip freeze > "$RUN_ROOT/pip-freeze.txt"
cut -d' ' -f1 data/office-home/Art_list.txt data/office-home/Clipart_list.txt \
    data/office-home/Product_list.txt data/office-home/RealWorld_list.txt \
    | sort -u | xargs sha256sum > "$RUN_ROOT/dataset-images.sha256"

{
    echo "created_at=$(date -Is)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "git_describe=$(git describe --tags --always)"
    echo "git_tracked_status=$(git status --porcelain --untracked-files=no | tr '\n' ';')"
    echo "python=$PYTHON"
    echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
    echo "pythonhashseed=$PYTHONHASHSEED"
    echo "cublas_workspace_config=$CUBLAS_WORKSPACE_CONFIG"
    echo "method=ProDe adaptive alpha with original soft prompt"
    echo "tasks=AC,AP,AR,CA,CP,CR,PA,PC,PR,RA,RC,RP"
    echo "epochs=60"
    echo "seed=2020"
    uname -a
    "$PYTHON" -c 'import torch, torchvision; print("torch=" + torch.__version__); print("cuda=" + str(torch.version.cuda)); print("torchvision=" + torchvision.__version__)'
    nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
    sha256sum \
        src/methods/oh/ProDe.py \
        src/utils/adaptive_alpha.py \
        clip/custom_clip.py \
        cfgs/office-home/ProDe.yaml \
        data/office-home/Art_list.txt \
        data/office-home/Clipart_list.txt \
        data/office-home/Product_list.txt \
        data/office-home/RealWorld_list.txt \
        data/office-home/classname.txt \
        Models/fraction_full/uda/office-home/A/source_F.pt \
        Models/fraction_full/uda/office-home/A/source_B.pt \
        Models/fraction_full/uda/office-home/A/source_C.pt \
        Models/fraction_full/uda/office-home/C/source_F.pt \
        Models/fraction_full/uda/office-home/C/source_B.pt \
        Models/fraction_full/uda/office-home/C/source_C.pt \
        Models/fraction_full/uda/office-home/P/source_F.pt \
        Models/fraction_full/uda/office-home/P/source_B.pt \
        Models/fraction_full/uda/office-home/P/source_C.pt \
        Models/fraction_full/uda/office-home/R/source_F.pt \
        Models/fraction_full/uda/office-home/R/source_B.pt \
        Models/fraction_full/uda/office-home/R/source_C.pt \
        "$RUN_ROOT/dataset-images.sha256"
} > "$RUN_ROOT/manifest.txt"

cat "$RUN_ROOT/manifest.txt"

domain_abbr=(A C P R)
for s in 0 1 2 3; do
    for t in 0 1 2 3; do
        if [[ "$s" == "$t" ]]; then
            continue
        fi

        task="${domain_abbr[$s]}${domain_abbr[$t]}"
        marker="$RUN_ROOT/.completed/$task"
        if [[ -f "$marker" ]]; then
            echo "[$(date -Is)] skip completed task=$task"
            continue
        fi

        echo "[$(date -Is)] begin task=$task source=$s target=$t"
        "$PYTHON" image_target_of_oh_vs_new.py --cfg cfgs/office-home/ProDe.yaml \
            MODEL.METHOD ProDe \
            SETTING.S "$s" SETTING.T "$t" SETTING.OUTPUT_SRC fraction_full SETTING.SEED 2020 \
            TEST.MAX_EPOCH 60 TEST.INTERVAL 30 \
            ProDe.TRAIN_FRACTION 1.0 ProDe.TUNE_TEXT_FEATURES 0 \
            ProDe.USE_ADAPTIVE_ALPHA 1 ProDe.MERGE_FIXED_ALPHA -1.0 \
            ProDe.MERGE_USE_CONFIDENCE 1 ProDe.MERGE_USE_STABILITY 1 \
            ProDe.MERGE_USE_PRIOR_TERM 1 ProDe.MERGE_USE_LOSS_GAP 1 \
            CUDNN.BENCHMARK False CUDNN.DETERMINISTIC True \
            SAVE_DIR "$RUN_ROOT"
        touch "$marker"
        echo "[$(date -Is)] complete task=$task"
    done
done

echo "[$(date -Is)] complete all tasks"
