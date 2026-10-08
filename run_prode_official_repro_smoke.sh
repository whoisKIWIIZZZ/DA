#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON=/home/yishouzhuo/miniconda3/envs/sfa/bin/python
RUN_ROOT=${RUN_ROOT:-./output/prode_official_repro_smoke}
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export PYTHONHASHSEED=2020
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "$RUN_ROOT/.completed"
exec > >(tee -a "$RUN_ROOT/runner.log") 2>&1

for run in 1 2; do
    if [[ -f "$RUN_ROOT/.completed/run${run}" ]]; then
        continue
    fi
    "$PYTHON" image_target_of_oh_vs_new.py --cfg cfgs/office-home/ProDe.yaml \
        MODEL.METHOD ProDe_old \
        SETTING.S 0 SETTING.T 1 SETTING.OUTPUT_SRC fraction_full SETTING.SEED 2020 \
        TEST.MAX_EPOCH 1 TEST.INTERVAL 1 \
        CUDNN.BENCHMARK False CUDNN.DETERMINISTIC True \
        SAVE_DIR "$RUN_ROOT/run${run}"
    touch "$RUN_ROOT/.completed/run${run}"
done

: > "$RUN_ROOT/reproducibility.txt"
for component in F B C; do
    "$PYTHON" tools/compare_state_dicts.py \
        "$RUN_ROOT/run1/uda/office-home/AC/ProDe_old/target_${component}_ProDe_old.pt" \
        "$RUN_ROOT/run2/uda/office-home/AC/ProDe_old/target_${component}_ProDe_old.pt" \
        | tee -a "$RUN_ROOT/reproducibility.txt"
done

echo "[$(date -Is)] deterministic smoke complete" | tee -a "$RUN_ROOT/reproducibility.txt"
