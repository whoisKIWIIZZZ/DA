#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for source in 0 1 2 3; do
    for target in 0 1 2 3; do
        [[ "$source" == "$target" ]] && continue

        # A→C was already completed for both source settings.
        [[ "$source" == 0 && "$target" == 1 ]] && continue

        for source_setting in fraction_full fraction_80; do
            "$PYTHON" image_target_of_oh_vs_new.py --cfg cfgs/office-home/difo.yaml \
                MODEL.METHOD difo_old \
                SETTING.S "$source" SETTING.T "$target" SETTING.SEED 2020 \
                SETTING.OUTPUT_SRC "$source_setting" DIFO.TRAIN_FRACTION 1.0 \
                SAVE_DIR "./output_difo_old_all_cases/${source_setting}_s${source}_t${target}"
        done
    done
done
