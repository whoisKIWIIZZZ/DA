#!/bin/sh
DATASET="visda" # imagenet_c domainnet126 officehome
METHOD="diff_align"          # shot nrc plue difo
PYTHON=/home/yishouzhuo/miniconda3/envs/sfa/bin/python
echo DATASET: $DATASET
echo METHOD: $METHOD
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# source_free_adaptation() {
# if [ "$DATASET" == "office-home" ];then
s_list=(0)
t_list=(1)
# fi
for s in ${s_list[*]}; do
    for t in ${t_list[*]}; do
    if [[ $s = $t ]]
        then
        continue
    fi
        CUDA_VISIBLE_DEVICES=3 $PYTHON image_target_of_oh_vs_new.py --cfg "cfgs/${DATASET}/${METHOD}.yaml" \
            SETTING.S "$s" SETTING.T "$t" &
        wait
    done
done 

