#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for fraction in 0.5 0.75 1.0; do
    "$PYTHON" image_target_of_oh_vs_new.py --cfg cfgs/office-home/ProDe.yaml \
        MODEL.METHOD ProDe_old SETTING.S 0 SETTING.T 1 \
        SETTING.OUTPUT_SRC fraction_full ProDe.TRAIN_FRACTION "$fraction" \
        SAVE_DIR "./output_prode_old_fraction_ac/full_source_target_${fraction}"
done

"$PYTHON" image_target_of_oh_vs_new.py --cfg cfgs/office-home/ProDe.yaml \
    MODEL.METHOD ProDe_old SETTING.S 0 SETTING.T 1 \
    SETTING.OUTPUT_SRC fraction_80 ProDe.TRAIN_FRACTION 1.0 \
    SAVE_DIR ./output_prode_old_fraction_ac/source_80_target_full
