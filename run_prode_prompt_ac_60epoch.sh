#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON=/home/yishouzhuo/miniconda3/envs/sfa/bin/python
RUN_ROOT=${RUN_ROOT:-./output/prode_prompt_ac_60epoch}
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

mkdir -p "$RUN_ROOT/.completed"
exec > >(tee -a "$RUN_ROOT/runner.log") 2>&1

echo "[$(date -Is)] start commit=$(git rev-parse --short HEAD) gpu=$CUDA_VISIBLE_DEVICES"

if [[ ! -f "$RUN_ROOT/.completed/ProDe_old" ]]; then
    echo "[$(date -Is)] begin method=ProDe_old task=AC epochs=60"
    "$PYTHON" image_target_of_oh_vs_new.py --cfg cfgs/office-home/ProDe.yaml \
        MODEL.METHOD ProDe_old \
        SETTING.S 0 SETTING.T 1 SETTING.OUTPUT_SRC fraction_full \
        TEST.MAX_EPOCH 60 TEST.INTERVAL 30 \
        ProDe.TRAIN_FRACTION 1.0 \
        SAVE_DIR "$RUN_ROOT/prode_old"
    touch "$RUN_ROOT/.completed/ProDe_old"
    echo "[$(date -Is)] complete method=ProDe_old"
fi

if [[ ! -f "$RUN_ROOT/.completed/ProDe_adaptive" ]]; then
    echo "[$(date -Is)] begin method=ProDe_adaptive task=AC epochs=60"
    "$PYTHON" image_target_of_oh_vs_new.py --cfg cfgs/office-home/ProDe.yaml \
        MODEL.METHOD ProDe \
        SETTING.S 0 SETTING.T 1 SETTING.OUTPUT_SRC fraction_full \
        TEST.MAX_EPOCH 60 TEST.INTERVAL 30 \
        ProDe.TRAIN_FRACTION 1.0 ProDe.TUNE_TEXT_FEATURES 0 \
        ProDe.USE_ADAPTIVE_ALPHA 1 ProDe.MERGE_FIXED_ALPHA -1.0 \
        ProDe.MERGE_USE_CONFIDENCE 1 ProDe.MERGE_USE_STABILITY 1 \
        ProDe.MERGE_USE_PRIOR_TERM 1 ProDe.MERGE_USE_LOSS_GAP 1 \
        SAVE_DIR "$RUN_ROOT/prode_adaptive"
    touch "$RUN_ROOT/.completed/ProDe_adaptive"
    echo "[$(date -Is)] complete method=ProDe_adaptive"
fi

echo "[$(date -Is)] complete all experiments"
