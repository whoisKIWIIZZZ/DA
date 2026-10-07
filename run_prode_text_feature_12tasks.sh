#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON=/home/yishouzhuo/miniconda3/envs/sfa/bin/python
RUN_ROOT=${RUN_ROOT:-./output/prode_text_feature_reliable_12tasks}
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

mkdir -p "$RUN_ROOT/.completed"
exec > >(tee -a "$RUN_ROOT/runner.log") 2>&1

domain_abbr=(A C P R)
echo "[$(date -Is)] start commit=$(git rev-parse --short HEAD) gpu=$CUDA_VISIBLE_DEVICES"

for s in 0 1 2 3; do
    for t in 0 1 2 3; do
        if [[ "$s" == "$t" ]]; then
            continue
        fi

        task="${domain_abbr[$s]}${domain_abbr[$t]}"
        marker="$RUN_ROOT/.completed/$task"
        if [[ -f "$marker" ]]; then
            echo "[$(date -Is)] skip completed task=$task"
            continue
        fi

        echo "[$(date -Is)] begin task=$task source=$s target=$t"
        "$PYTHON" image_target_of_oh_vs_new.py --cfg cfgs/office-home/ProDe.yaml \
            SETTING.S "$s" SETTING.T "$t" SETTING.OUTPUT_SRC source \
            ProDe.TRAIN_FRACTION 1.0 ProDe.USE_ADAPTIVE_ALPHA 1 \
            ProDe.MERGE_FIXED_ALPHA -1.0 \
            ProDe.MERGE_USE_CONFIDENCE 1 ProDe.MERGE_USE_STABILITY 1 \
            ProDe.MERGE_USE_PRIOR_TERM 1 ProDe.MERGE_USE_LOSS_GAP 1 \
            ProDe.TUNE_TEXT_FEATURES 1 \
            ProDe.TEXT_FEATURE_LR_MULT 0.1 \
            ProDe.TEXT_FEATURE_ANCHOR_WEIGHT 0.1 \
            ProDe.TEXT_FEATURE_IIC_WEIGHT 1.0 \
            ProDe.TEXT_FEATURE_DISTILL_WEIGHT 1.0 \
            ProDe.TEXT_FEATURE_RELIABILITY_TEMPERATURE 0.1 \
            ProDe.TEXT_FEATURE_REQUIRE_AGREEMENT 1 \
            SAVE_DIR "$RUN_ROOT"
        touch "$marker"
        echo "[$(date -Is)] complete task=$task"
    done
done

echo "[$(date -Is)] complete all tasks"
