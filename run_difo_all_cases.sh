#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
domain_abbr=(A C P R)

for source in 1 2 3; do
    for source_setting in fraction_full fraction_80; do
        if [[ "$source_setting" == fraction_full ]]; then
            fraction=1.0
        else
            fraction=0.8
        fi

        if [[ ! -f "Models/${source_setting}/uda/office-home/${domain_abbr[$source]}/source_F.pt" ]]; then
            "$PYTHON" image_target_of_oh_vs_new.py --cfg cfgs/office-home/source.yaml \
                SETTING.S "$source" SETTING.SEED 2020 \
                SOURCE.TRTE train SOURCE.TRAIN_FRACTION "$fraction" \
                SETTING.OUTPUT_SRC "$source_setting" \
                SAVE_DIR "./output_difo_all_cases/source_${source_setting}_s${source}"
        fi
    done
done

for source in 0 1 2 3; do
    for target in 0 1 2 3; do
        [[ "$source" == "$target" ]] && continue

        # A→C/P/R was already completed for both source settings.
        [[ "$source" == 0 ]] && continue

        for source_setting in fraction_full fraction_80; do
            "$PYTHON" image_target_of_oh_vs_new.py --cfg cfgs/office-home/difo.yaml \
                SETTING.S "$source" SETTING.T "$target" SETTING.SEED 2020 \
                SETTING.OUTPUT_SRC "$source_setting" DIFO.TRAIN_FRACTION 1.0 \
                DIFO.MERGE_FIXED_ALPHA -1.0 \
                DIFO.MERGE_USE_CONFIDENCE 1 DIFO.MERGE_USE_STABILITY 1 \
                DIFO.MERGE_USE_PRIOR_TERM 1 DIFO.MERGE_USE_LOSS_GAP 1 \
                SAVE_DIR "./output_difo_all_cases/${source_setting}_s${source}_t${target}"
        done
    done
done
