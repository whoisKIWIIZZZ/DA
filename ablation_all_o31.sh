#!/bin/bash
# 顺序跑完某方法全部 adaptive-alpha 消融实验: base(full) + 所有变体。
# 用法: bash ablation_all_oh.sh [ProDe|difo] [GPU_ID]
# 建议后台: nohup bash ablation_all_oh.sh ProDe 1 > logs/ablation_ProDe.log 2>&1 &
#
# 每个配置跑 12 个 domain pair,单卡约 30min/配置量级,全量 ~10h,
# 可按需注释掉 CFG 列表中的条目只跑子集。
GROUP="${1:-ProDe_ab}"
GPU="${2:-1}"

case "$GROUP" in
    difo_ab)   PATTERN="cfgs/office/ablations/difo_*.yaml" ;;
    difo_hp)   PATTERN="cfgs/office-home/ablations/difo_hp*.yaml" ;;
    ProDe_ab)  PATTERN="cfgs/office-home/ablations/ProDe_ab*.yaml" ;;
    ProDe_hp)  PATTERN="cfgs/office-home/ablations/ProDe_hp*.yaml" ;;
    *)
        echo "用法: bash ablation_split.sh {difo_ab|difo_hp|ProDe_ab|ProDe_hp} [GPU_ID]"
        exit 1
        ;;
esac

echo "===== METHOD: $METHOD | GPU: $GPU ====="

# ab1 = full(base 配置本身)
# bash ablation_run.sh "$BASE" $GPU

for cfg in $PATTERN; do
    DATASET="office-home"
    DA="uda"         
    G=False

    export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
    PYTHON=/home/yishouzhuo/miniconda3/envs/sfa/bin/python

    echo "CFG: $cfg | GPU: $GPU | DA: $DA"
    domain_abbr=(A C P R)
    s_list=(0 1 2)
    t_list=(0 1 2)

    for s in ${s_list[*]}; do
        for t in ${t_list[*]}; do
            [[ $s == $t ]] && continue
            echo "===== ${domain_abbr[$s]} -> ${domain_abbr[$t]} ====="
            CUDA_VISIBLE_DEVICES=$GPU $PYTHON image_target_of_oh_vs_new.py --cfg "$cfg" \
                SETTING.S "$s" SETTING.T "$t" DA $DA GSFDA $G \
                || echo "FAILED: $cfg S=$s T=$t"
        done
    done
done

echo "ALL ABLATION DONE: $METHOD"
