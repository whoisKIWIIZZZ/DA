#!/bin/bash
DATASET="office-home" # imagenet_c domainnet126 office-home
METHOD="diff_align"          # shot nrc plue difo source-neg source_clip
BRIDGE_MODE=true             # true=启用bridge实验checkpoint保存

echo DATASET: $DATASET
echo METHOD: $METHOD
echo BRIDGE_MODE: $BRIDGE_MODE


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
        if $BRIDGE_MODE; then
            CUDA_VISIBLE_DEVICES=0 /data/home/scv8780/.conda/envs/sfa/bin/python image_target_of_oh_vs_new.py --cfg "cfgs/${DATASET}/${METHOD}.yaml" \
                SETTING.S "$s" SETTING.T "$t" \
                BRIDGE.ENABLE True &
        else
            CUDA_VISIBLE_DEVICES=0 /data/home/scv8780/.conda/envs/sfa/bin/python image_target_of_oh_vs_new.py --cfg "cfgs/${DATASET}/${METHOD}.yaml" \
                SETTING.S "$s" SETTING.T "$t" &
        fi
        wait
    done
done
