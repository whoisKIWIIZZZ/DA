#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for seed in 2021 2022 2023; do
    for source_setting in fraction_full fraction_80; do
        "$PYTHON" image_target_of_oh_vs_new.py --cfg cfgs/office-home/difo.yaml \
            SETTING.S 0 SETTING.T 1 SETTING.SEED "$seed" \
            SETTING.OUTPUT_SRC "$source_setting" DIFO.TRAIN_FRACTION 1.0 \
            DIFO.MERGE_FIXED_ALPHA -1.0 \
            DIFO.MERGE_USE_CONFIDENCE 1 DIFO.MERGE_USE_STABILITY 1 \
            DIFO.MERGE_USE_PRIOR_TERM 1 DIFO.MERGE_USE_LOSS_GAP 1 \
            SAVE_DIR "./output_difo_ac_repeats/seed_${seed}_${source_setting}"
    done
done
