#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for target in 2 3; do
    for source_setting in fraction_full fraction_80; do
        "$PYTHON" image_target_of_oh_vs_new.py --cfg cfgs/office-home/difo.yaml \
            SETTING.S 0 SETTING.T "$target" \
            SETTING.OUTPUT_SRC "$source_setting" DIFO.TRAIN_FRACTION 1.0 \
            DIFO.MERGE_FIXED_ALPHA -1.0 \
            DIFO.MERGE_USE_CONFIDENCE 1 DIFO.MERGE_USE_STABILITY 1 \
            DIFO.MERGE_USE_PRIOR_TERM 1 DIFO.MERGE_USE_LOSS_GAP 1 \
            SAVE_DIR "./output_difo_source_fraction_ap_ar/${source_setting}_target_${target}"
    done
done
