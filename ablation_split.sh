#!/bin/bash
# 把 ablations 拆成四组跑(组内顺序执行,内部复用 ablation_run.sh):
#   bash ablation_split.sh difo_ab   [GPU_ID]   # difo 组件消融+固定alpha (ab2~ab8)
#   bash ablation_split.sh difo_hp   [GPU_ID]   # difo 超参数敏感性 (hp_*)
#   bash ablation_split.sh ProDe_ab  [GPU_ID]   # ProDe 组件消融+固定alpha (ab0,ab2~ab8)
#   bash ablation_split.sh ProDe_hp  [GPU_ID]   # ProDe 超参数敏感性 (hp_*)
#
# ab1(full) 是 base 配置本身(cfgs/office-home/<METHOD>.yaml),不在此脚本内,单独跑:
#   bash ablation_run.sh cfgs/office-home/difo.yaml 1
# 建议后台: nohup bash ablation_split.sh difo_ab 1 > logs/ablation_difo_ab.log 2>&1 &
GROUP="${1:-ProDe_ab}"
GPU="${2:-1}"

case "$GROUP" in
    difo_ab)   PATTERN="cfgs/office-home/ablations/difo_ab*.yaml" ;;
    difo_hp)   PATTERN="cfgs/office-home/ablations/difo_hp*.yaml" ;;
    ProDe_ab)  PATTERN="cfgs/office-home/ablations/ProDe_ab*.yaml" ;;
    ProDe_hp)  PATTERN="cfgs/office-home/ablations/ProDe_hp*.yaml" ;;
    *)
        echo "用法: bash ablation_split.sh {difo_ab|difo_hp|ProDe_ab|ProDe_hp} [GPU_ID]"
        exit 1
        ;;
esac

echo "===== GROUP: $GROUP | GPU: $GPU ====="
for cfg in $PATTERN; do
    bash ablation_run.sh "$cfg" $GPU
done
echo "GROUP DONE: $GROUP"
