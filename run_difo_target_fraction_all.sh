#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for source in 0 1 2 3; do
    for target in 0 1 2 3; do
        [[ "$source" == "$target" ]] && continue

        # A→C target fractions 0.5 and 0.75 were already completed.
        [[ "$source" == 0 && "$target" == 1 ]] && continue

        for fraction in 0.5 0.75; do
            "$PYTHON" image_target_of_oh_vs_new.py --cfg cfgs/office-home/difo.yaml \
                SETTING.S "$source" SETTING.T "$target" SETTING.SEED 2020 \
                SETTING.OUTPUT_SRC fraction_full DIFO.TRAIN_FRACTION "$fraction" \
                DIFO.MERGE_FIXED_ALPHA -1.0 \
                DIFO.MERGE_USE_CONFIDENCE 1 DIFO.MERGE_USE_STABILITY 1 \
                DIFO.MERGE_USE_PRIOR_TERM 1 DIFO.MERGE_USE_LOSS_GAP 1 \
                SAVE_DIR "./output_difo_target_fraction_all/fraction_${fraction}_s${source}_t${target}"
        done
    done
done
