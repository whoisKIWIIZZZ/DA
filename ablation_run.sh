#!/bin/bash
# 单个消融配置跑完 office-home 全部 12 个 domain pair(顺序执行)。
# 用法: bash ablation_run.sh <cfg文件> [GPU_ID]
# 例:   bash ablation_run.sh cfgs/office-home/ablations/ProDe_ab4_nogap.yaml 1
CFG="$1"
GPU="${2:-1}"
DATASET="office-home"
DA="uda"          # pda=partial DA,同 office_home.sh
G=False

export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
PYTHON=/home/yishouzhuo/miniconda3/envs/sfa/bin/python

echo "CFG: $CFG | GPU: $GPU | DA: $DA"
domain_abbr=(A C P R)
s_list=(0 1 2 3)
t_list=(0 1 2 3)

for s in ${s_list[*]}; do
    for t in ${t_list[*]}; do
        [[ $s == $t ]] && continue
        echo "===== ${domain_abbr[$s]} -> ${domain_abbr[$t]} ====="
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON image_target_of_oh_vs_new.py --cfg "$CFG" \
            SETTING.S "$s" SETTING.T "$t" DA $DA GSFDA $G \
            || echo "FAILED: $CFG S=$s T=$t"
    done
done
echo "ALL DONE: $CFG"
