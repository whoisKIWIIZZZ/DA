#!/bin/bash
# 并行跑 office-home grid 配置，每张 GPU 保持固定数量的并发任务。
# 用法: bash grid_all_oh.sh [difo|ProDe] [GPU范围] [每卡并发数] [日志目录]
# 示例: bash grid_all_oh.sh difo 0-1 2
#       bash grid_all_oh.sh ProDe 0,2,3 2

set -u

METHOD="${1:-difo}"
GPU_SPEC="${2:-0-1}"
JOBS_PER_GPU="${3:-2}"

case "$METHOD" in
    difo|DIFO)
        METHOD="difo"
        PATTERN="cfgs/office/ablations/grid/difo_*.yaml"
        ;;
    ProDe|prode|PRODE)
        METHOD="ProDe"
        PATTERN="cfgs/office/ablations/grid/ProDe_*.yaml"
        ;;
    *)
        echo "用法: bash $0 {difo|ProDe} {GPU范围，如 0-1 或 0,1} [每卡并发数]" >&2
        exit 1
        ;;
esac

LOG_DIR="${4:-logs/office-home/grid/$METHOD}"

if ! [[ "$JOBS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
    echo "错误: 每卡并发数必须是正整数，当前值: $JOBS_PER_GPU" >&2
    exit 1
fi

parse_gpus() {
    local spec="$1"
    local start end gpu

    if [[ "$spec" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        start="${BASH_REMATCH[1]}"
        end="${BASH_REMATCH[2]}"
        if (( start > end )); then
            echo "错误: GPU 范围起点不能大于终点: $spec" >&2
            return 1
        fi
        for ((gpu = start; gpu <= end; gpu++)); do
            GPUS+=("$gpu")
        done
    elif [[ "$spec" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        IFS=',' read -r -a GPUS <<< "$spec"
    else
        echo "错误: GPU 范围格式无效: $spec（支持 0-1 或 0,1）" >&2
        return 1
    fi
}

GPUS=()
parse_gpus "$GPU_SPEC" || exit 1

mapfile -t CFGS < <(compgen -G "$PATTERN" | sort)

if (( ${#CFGS[@]} == 0 )); then
    echo "错误: 没有找到配置文件: $PATTERN" >&2
    exit 1
fi

export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
PYTHON="/home/yishouzhuo/miniconda3/envs/sfa/bin/python"
DA="uda"
G="False"
DOMAIN_ABBR=(A C P R)
S_LIST=(0 1 2)
T_LIST=(0 1 2)
mkdir -p "$LOG_DIR"

run_cfg() {
    local cfg="$1"
    local gpu="$2"
    local cfg_name log_file
    local source target

    cfg_name="$(basename "$cfg" .yaml)"
    log_file="$LOG_DIR/${cfg_name}.log"
    echo "[GPU $gpu] START: $cfg -> $log_file"

    {
        echo "[GPU $gpu] CFG: $cfg | DA: $DA"
        for source in "${S_LIST[@]}"; do
            for target in "${T_LIST[@]}"; do
                [[ "$source" == "$target" ]] && continue
                echo "[GPU $gpu] ===== ${DOMAIN_ABBR[$source]} -> ${DOMAIN_ABBR[$target]} | $(basename "$cfg") ====="
                if ! CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" image_target_of_oh_vs_new.py --cfg "$cfg" \
                    SETTING.S "$source" SETTING.T "$target" DA "$DA" GSFDA "$G"; then
                    echo "[GPU $gpu] FAILED: $cfg S=$source T=$target" >&2
                fi
            done
        done
    } > "$log_file" 2>&1

    echo "[GPU $gpu] DONE: $cfg -> $log_file"
}

run_gpu_queue() {
    local gpu_index="$1"
    local gpu="$2"
    local gpu_count="${#GPUS[@]}"
    local batch_start offset index pid
    local pids=()

    for ((batch_start = gpu_index; batch_start < ${#CFGS[@]}; batch_start += gpu_count * JOBS_PER_GPU)); do
        pids=()

        for ((offset = 0; offset < JOBS_PER_GPU; offset++)); do
            index=$((batch_start + offset * gpu_count))
            (( index >= ${#CFGS[@]} )) && break
            run_cfg "${CFGS[$index]}" "$gpu" &
            pids+=("$!")
        done

        # 同一 GPU 上本批任务全部结束后，才允许启动下一批。
        for pid in "${pids[@]}"; do
            wait "$pid"
        done
    done
}

echo "===== METHOD: $METHOD | GPUS: ${GPUS[*]} | JOBS/GPU: $JOBS_PER_GPU | CFGS: ${#CFGS[@]} | LOGS: $LOG_DIR ====="

for gpu_index in "${!GPUS[@]}"; do
    run_gpu_queue "$gpu_index" "${GPUS[$gpu_index]}" &
done

wait
echo "ALL GRID DONE: $METHOD"
