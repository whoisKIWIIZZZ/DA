#!/usr/bin/env bash
set -euo pipefail

PYTHON="/home/yishouzhuo/miniconda3/envs/sfa/bin/python"
CONFIG="cfgs/office-home/source_open.yaml"
export MPLCONFIGDIR="/tmp/sfda-matplotlib"
CUDA_COMPAT_DIR="/tmp/sfda-cuda-lib"
mkdir -p "${CUDA_COMPAT_DIR}"
ln -sfn /lib/x86_64-linux-gnu/libcuda.so.1 "${CUDA_COMPAT_DIR}/libcuda.so"
export LD_LIBRARY_PATH="${CUDA_COMPAT_DIR}:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
mkdir -p logs

for source_domain in 0 1 2 3; do
    echo "[$(date '+%F %T')] Pretraining open-set source domain ${source_domain} on GPU 0"
    CUDA_VISIBLE_DEVICES=0 "${PYTHON}" image_target_of_oh_vs_new.py \
        --cfg "${CONFIG}" \
        SETTING.S "${source_domain}" \
        2>&1 | tee "logs/source_open_s${source_domain}.log"
done
