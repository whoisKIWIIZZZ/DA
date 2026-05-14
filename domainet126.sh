#!/bin/sh
DATASET="domainnet126" # imagenet_c domainnet126 officehome
METHOD="diff_align"          # shot nrc plue difo ProDe diff_align_plus

echo DATASET: $DATASET
echo METHOD: $METHOD


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
        CUDA_VISIBLE_DEVICES=0 ~/.conda/envs/sfa/bin/python image_target_in_126.py --cfg "cfgs/${DATASET}/${METHOD}.yaml" \
            SETTING.S "$s" SETTING.T "$t" &
        wait
    done
done 

