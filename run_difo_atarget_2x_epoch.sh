#!/usr/bin/env bash
set -euo pipefail

# 目标域为 Art 的三个任务（C→A / P→A / R→A）单独把训练 epoch 翻倍（15 → 30）。
# 动机：Art 只有 2427 张图，15 epoch 下只有 570 步，而其余任务有 1035~1050 步，
# 三者在总优化步数上不可比。翻倍后 CA/PA/RA 为 1140 步。
# 其余设置与 run_difo_target_fraction_all.sh 完全一致（源模型 fraction_full、seed 2020、自适应 alpha 参数）。
# 只跑 difo（当前自适应版本），difo_old 保持 1x epoch。

cd "$(dirname "$0")"
PYTHON=${PYTHON:-/home/yishouzhuo/miniconda3/envs/sfa/bin/python}
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for source in 1 2 3; do
    target=0   # A
    for fraction in 0.5 0.75 1.0; do
        "$PYTHON" image_target_of_oh_vs_new.py --cfg cfgs/office-home/difo.yaml \
            SETTING.S "$source" SETTING.T "$target" SETTING.SEED 2020 \
            SETTING.OUTPUT_SRC fraction_full DIFO.TRAIN_FRACTION "$fraction" \
            TEST.MAX_EPOCH 30 \
            DIFO.MERGE_FIXED_ALPHA -1.0 \
            DIFO.MERGE_USE_CONFIDENCE 1 DIFO.MERGE_USE_STABILITY 1 \
            DIFO.MERGE_USE_PRIOR_TERM 1 DIFO.MERGE_USE_LOSS_GAP 1 \
            SAVE_DIR "./output_difo_atarget_2x_epoch/fraction_${fraction}_s${source}_t${target}"
    done
done
