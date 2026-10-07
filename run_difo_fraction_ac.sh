#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for method in difo difo_old; do
    for fraction in 0.5 0.75 1.0; do
        "$PYTHON" image_target_of_oh_vs_new.py --cfg cfgs/office-home/difo.yaml \
            MODEL.METHOD "$method" SETTING.S 0 SETTING.T 1 \
            SETTING.OUTPUT_SRC fraction_full DIFO.TRAIN_FRACTION "$fraction" \
            DIFO.MERGE_FIXED_ALPHA -1.0 \
            DIFO.MERGE_USE_CONFIDENCE 1 DIFO.MERGE_USE_STABILITY 1 \
            DIFO.MERGE_USE_PRIOR_TERM 1 DIFO.MERGE_USE_LOSS_GAP 1 \
            SAVE_DIR "./output_difo_fraction_ac/${method}_full_source_target_${fraction}"
    done

    "$PYTHON" image_target_of_oh_vs_new.py --cfg cfgs/office-home/difo.yaml \
        MODEL.METHOD "$method" SETTING.S 0 SETTING.T 1 \
        SETTING.OUTPUT_SRC fraction_80 DIFO.TRAIN_FRACTION 1.0 \
        DIFO.MERGE_FIXED_ALPHA -1.0 \
        DIFO.MERGE_USE_CONFIDENCE 1 DIFO.MERGE_USE_STABILITY 1 \
        DIFO.MERGE_USE_PRIOR_TERM 1 DIFO.MERGE_USE_LOSS_GAP 1 \
        SAVE_DIR "./output_difo_fraction_ac/${method}_source_80_target_full"
done
