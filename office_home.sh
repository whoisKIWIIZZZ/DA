#!/bin/bash
DATASET="office-home" # imagenet_c domainnet126 office-home
METHOD="diff_align"          # shot nrc plue difo source-neg source_clip
BRIDGE_MODE=false             # true=启用bridge实验checkpoint保存


echo DATASET: $DATASET
echo METHOD: $METHOD
echo BRIDGE_MODE: $BRIDGE_MODE

# 域索引→缩写映射 (Art/Clipart/Product/RealWorld → A/C/P/R)
domain_abbr=(A C P R)
PYTHON=/data02/home/scv8780/.conda/envs/sfa/bin/python

# source_free_adaptation() {
# if [ "$DATASET" == "office-home" ];then
s_list=(0)
t_list=(1)
# fji
for s in ${s_list[*]}; do
    for t in ${t_list[*]}; do
    if [[ $s = $t ]]
        then
        continue
    fi
        S_LETTER=${domain_abbr[$s]}
        T_LETTER=${domain_abbr[$t]}

        if $BRIDGE_MODE; then
            CUDA_VISIBLE_DEVICES=0 $PYTHON image_target_of_oh_vs_new.py --cfg "cfgs/${DATASET}/${METHOD}.yaml" \
                SETTING.S "$s" SETTING.T "$t" \
                BRIDGE.ENABLE True &
        else
            CUDA_VISIBLE_DEVICES=0 $PYTHON image_target_of_oh_vs_new.py --cfg "cfgs/${DATASET}/${METHOD}.yaml" \
                SETTING.S "$s" SETTING.T "$t" &
        fi
        wait

        # 训练结束 → 自动运行 bridge 分析
        if $BRIDGE_MODE; then
            BRIDGE_DIR="output_Models/uda/${DATASET}/${S_LETTER}${T_LETTER}/${METHOD}/bridge_ckpts"
            echo "========== Bridge 分析: ${S_LETTER}→${T_LETTER} =========="

            echo "--- Layer 0-1: 几何恒等式 + 模态分解 ---"
            $PYTHON tools/analyze_crossover.py "$BRIDGE_DIR"

            echo "--- Layer 2: NTK 共识算子 + 信任半径 ---"
            $PYTHON tools/analyze_ntk.py "$BRIDGE_DIR"

            echo "========== 分析完成 =========="
        fi
    done
done
