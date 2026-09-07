#!/bin/bash
DATASET="office-home" # imagenet_c domainnet126 office-home
METHOD="ProDe"          # shot nrc plue difo source-neg source_clip
BRIDGE_MODE=false             # true=启用bridge实验checkpoint保存
DA="pda"
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
echo DATASET: $DATASET
echo METHOD: $METHOD
echo BRIDGE_MODE: $BRIDGE_MODE

# 域索引→缩写映射 (Art/Clipart/Product/RealWorld → A/C/P/R)
domain_abbr=(A C P R)
PYTHON=/home/yishouzhuo/miniconda3/envs/sfa/bin/python

# source_free_adaptation() {
# if [ "$DATASET" == "office-home" ];then
s_list=(0 1 2 3)
t_list=(0 1 2 3)
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
                BRIDGE.ENABLE True 
        else
            CUDA_VISIBLE_DEVICES=1 $PYTHON image_target_of_oh_vs_new.py --cfg "cfgs/${DATASET}/${METHOD}.yaml" \
                SETTING.S "$s" SETTING.T "$t"  DA $DA
        fi
    done
done
