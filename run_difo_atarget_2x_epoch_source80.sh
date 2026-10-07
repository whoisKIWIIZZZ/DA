#!/usr/bin/env bash
set -euo pipefail

# 补 run_difo_atarget_2x_epoch.sh 缺的 80% 源那一格：
# 目标域为 Art 的三个任务（C→A / P→A / R→A），源模型用 fraction_80，epoch 翻倍（15 → 30），
# 目标域用完整训练集。其余设置与 run_difo_atarget_2x_epoch.sh 一致。

cd "$(dirname "$0")"
PYTHON=${PYTHON:-/home/yishouzhuo/miniconda3/envs/sfa/bin/python}
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for source in 1 2 3; do
    "$PYTHON" image_target_of_oh_vs_new.py --cfg cfgs/office-home/difo.yaml \
        SETTING.S "$source" SETTING.T 0 SETTING.SEED 2020 \
        SETTING.OUTPUT_SRC fraction_80 DIFO.TRAIN_FRACTION 1.0 \
        TEST.MAX_EPOCH 30 \
        DIFO.MERGE_FIXED_ALPHA -1.0 \
        DIFO.MERGE_USE_CONFIDENCE 1 DIFO.MERGE_USE_STABILITY 1 \
        DIFO.MERGE_USE_PRIOR_TERM 1 DIFO.MERGE_USE_LOSS_GAP 1 \
        SAVE_DIR "./output_difo_atarget_2x_epoch_source80/source_80_s${source}_t0"
done
