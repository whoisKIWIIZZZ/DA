#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON=/home/yishouzhuo/miniconda3/envs/sfa/bin/python
RUN_ROOT=${RUN_ROOT:-./output/prode_official_prompt_ac_60epoch}
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

mkdir -p "$RUN_ROOT/.completed"
exec > >(tee -a "$RUN_ROOT/runner.log") 2>&1

echo "[$(date -Is)] start commit=$(git rev-parse --short HEAD) upstream=5f7fe578c64ad84a9126157c3aab3aa4ba825bc6 gpu=$CUDA_VISIBLE_DEVICES"

if [[ ! -f "$RUN_ROOT/.completed/ProDe_official" ]]; then
    "$PYTHON" image_target_of_oh_vs_new.py --cfg cfgs/office-home/ProDe.yaml \
        MODEL.METHOD ProDe_old \
        SETTING.S 0 SETTING.T 1 SETTING.OUTPUT_SRC fraction_full \
        TEST.MAX_EPOCH 60 TEST.INTERVAL 30 \
        SAVE_DIR "$RUN_ROOT"
    touch "$RUN_ROOT/.completed/ProDe_official"
fi

echo "[$(date -Is)] complete method=ProDe_official task=AC epochs=60"
