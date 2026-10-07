#!/usr/bin/env bash
set -euo pipefail

# difo_old 的目标域数据量消融，与 run_difo_target_fraction_all.sh 一一对应。
# A→C 已在 output_difo_fraction_ac 里跑过（0.5/0.75/1.0），这里跳过。
# 其余 11 个任务 × 2 个占比 = 22 次；源模型固定 fraction_full，epoch 保持 15。

cd "$(dirname "$0")"
PYTHON=${PYTHON:-/home/yishouzhuo/miniconda3/envs/sfa/bin/python}
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for source in 0 1 2 3; do
    for target in 0 1 2 3; do
        [[ "$source" == "$target" ]] && continue

        # A→C target fractions 0.5 and 0.75 were already completed.
        [[ "$source" == 0 && "$target" == 1 ]] && continue

        for fraction in 0.5 0.75; do
            "$PYTHON" image_target_of_oh_vs_new.py --cfg cfgs/office-home/difo.yaml \
                MODEL.METHOD difo_old \
                SETTING.S "$source" SETTING.T "$target" SETTING.SEED 2020 \
                SETTING.OUTPUT_SRC fraction_full DIFO.TRAIN_FRACTION "$fraction" \
                DIFO.MERGE_FIXED_ALPHA -1.0 \
                DIFO.MERGE_USE_CONFIDENCE 1 DIFO.MERGE_USE_STABILITY 1 \
                DIFO.MERGE_USE_PRIOR_TERM 1 DIFO.MERGE_USE_LOSS_GAP 1 \
                SAVE_DIR "./output_difo_old_target_fraction_all/fraction_${fraction}_s${source}_t${target}"
        done
    done
done
