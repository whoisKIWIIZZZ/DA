#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

"$PYTHON" image_target_of_oh_vs_new.py --cfg cfgs/office-home/source.yaml \
    SETTING.S 0 SETTING.T 1 SOURCE.TRTE train SOURCE.TRAIN_FRACTION 1.0 \
    SETTING.OUTPUT_SRC fraction_full SAVE_DIR ./output_fraction_ac/source_full

"$PYTHON" image_target_of_oh_vs_new.py --cfg cfgs/office-home/source.yaml \
    SETTING.S 0 SETTING.T 1 SOURCE.TRTE train SOURCE.TRAIN_FRACTION 0.8 \
    SETTING.OUTPUT_SRC fraction_80 SAVE_DIR ./output_fraction_ac/source_80

for fraction in 0.5 0.75 1.0; do
    "$PYTHON" image_target_of_oh_vs_new.py --cfg cfgs/office-home/ProDe.yaml \
        SETTING.S 0 SETTING.T 1 SETTING.OUTPUT_SRC fraction_full \
        ProDe.TRAIN_FRACTION "$fraction" ProDe.USE_ADAPTIVE_ALPHA 1 \
        ProDe.MERGE_FIXED_ALPHA -1.0 \
        ProDe.MERGE_USE_CONFIDENCE 1 ProDe.MERGE_USE_STABILITY 1 \
        ProDe.MERGE_USE_PRIOR_TERM 1 ProDe.MERGE_USE_LOSS_GAP 1 \
        SAVE_DIR "./output_fraction_ac/full_source_target_${fraction}"
done

"$PYTHON" image_target_of_oh_vs_new.py --cfg cfgs/office-home/ProDe.yaml \
    SETTING.S 0 SETTING.T 1 SETTING.OUTPUT_SRC fraction_80 \
    ProDe.TRAIN_FRACTION 1.0 ProDe.USE_ADAPTIVE_ALPHA 1 \
    ProDe.MERGE_FIXED_ALPHA -1.0 \
    ProDe.MERGE_USE_CONFIDENCE 1 ProDe.MERGE_USE_STABILITY 1 \
    ProDe.MERGE_USE_PRIOR_TERM 1 ProDe.MERGE_USE_LOSS_GAP 1 \
    SAVE_DIR ./output_fraction_ac/source_80_target_full
