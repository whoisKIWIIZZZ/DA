#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON=/home/yishouzhuo/miniconda3/envs/sfa/bin/python
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

"$PYTHON" image_target_of_oh_vs_new.py --cfg cfgs/office-home/ProDe.yaml \
    SETTING.S 0 SETTING.T 1 SETTING.OUTPUT_SRC fraction_full \
    ProDe.TRAIN_FRACTION 1.0 ProDe.USE_ADAPTIVE_ALPHA 1 \
    ProDe.MERGE_FIXED_ALPHA -1.0 \
    ProDe.MERGE_USE_CONFIDENCE 1 ProDe.MERGE_USE_STABILITY 1 \
    ProDe.MERGE_USE_PRIOR_TERM 1 ProDe.MERGE_USE_LOSS_GAP 1 \
    ProDe.TUNE_TEXT_FEATURES 1 \
    ProDe.TEXT_FEATURE_LR_MULT 0.1 \
    ProDe.TEXT_FEATURE_ANCHOR_WEIGHT 0.1 \
    SAVE_DIR ./output/prode_text_feature_ac
