#!/bin/bash
# VISDA-C 上的 adaptive-alpha 消融: base(full) + noloss + noprior。
# 用法: bash ablation_visda.sh [GROUP] [GPU_ID]
#   GROUP: all(默认, base+noloss+noprior) | difo_ab(只跑两个消融) | difo_full(只跑base)
#
# VISDA-C 只有一个 domain pair: S=0(train) -> T=1(validation), 即 cfg.name = "TV",
# 所以每个配置只跑一次。日志里的结果行形如:
#   Task: TV, Iter:N/N; Accuracy = xx.xx%;...
#
# 建议后台跑, 日志落到 analysis/ 便于之后统一解析:
#   nohup bash ablation_visda.sh all 1 > analysis/ablation_difo_visda.log 2>&1 &
# 解析结果:
#   python3 analysis/parse_ablation.py analysis/ablation_difo_visda.log
GROUP="${1:-all}"
GPU="${2:-1}"

BASE_CFG="cfgs/visda/difo.yaml"
ABL_CFGS=(cfgs/visda/ablations/difo_noloss.yaml cfgs/visda/ablations/difo_noprior.yaml)

case "$GROUP" in
    all)       CFGS=("$BASE_CFG" "${ABL_CFGS[@]}") ;;
    difo_ab)   CFGS=("${ABL_CFGS[@]}") ;;
    difo_full) CFGS=("$BASE_CFG") ;;
    *)
        echo "用法: bash ablation_visda.sh {all|difo_ab|difo_full} [GPU_ID]"
        exit 1
        ;;
esac

DA="uda"
G=False
PYTHON=/home/yishouzhuo/miniconda3/envs/sfa/bin/python
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# 预检: difo 会从 output_dir_src 读 source_{F,B,C}.pt, 缺了会白跑
SRC_CKPT="Models/source/${DA}/VISDA-C/T/source_F.pt"
if [[ ! -f "$SRC_CKPT" ]]; then
    echo "ERROR: 缺少 source 权重 $SRC_CKPT, 请先跑 source 预训练"
    exit 1
fi

echo "===== DATASET: visda | GROUP: $GROUP | GPU: $GPU ====="

s_list=(0)
t_list=(1)

for cfg in "${CFGS[@]}"; do
    echo "CFG: $cfg | GPU: $GPU | DA: $DA"

    for s in "${s_list[@]}"; do
        for t in "${t_list[@]}"; do
            [[ $s == $t ]] && continue
            echo "===== ${s} -> ${t} ====="
            CUDA_VISIBLE_DEVICES=$GPU $PYTHON image_target_of_oh_vs_new.py --cfg "$cfg" \
                SETTING.S "$s" SETTING.T "$t" DA $DA GSFDA $G \
                || echo "FAILED: $cfg S=$s T=$t"
        done
    done
done

echo "ALL ABLATION DONE: visda ($GROUP)"
