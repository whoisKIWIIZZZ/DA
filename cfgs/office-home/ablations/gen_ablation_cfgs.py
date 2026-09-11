#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 adaptive-alpha 消融实验的 yaml 配置(从 base cfg 派生)。

用法(在仓库根目录):
    python cfgs/office/ablations/gen_ablation_cfgs.py

对 METHOD=ProDe 和 METHOD=difo 各生成一套变体到 cfgs/office/ablations/。
ab1(full)= base 配置本身(cfgs/office/<METHOD>.yaml),不重复生成。

实验清单:
  ab0_noadaptive    (仅 ProDe)  USE_ADAPTIVE_ALPHA=0,旧式 out+clip-0.1*bank 融合
  ab2_noconf        MERGE_USE_CONFIDENCE=0       可靠性只用稳定性
  ab3_nostab        MERGE_USE_STABILITY=0        可靠性只用置信度
  ab4_nogap         MERGE_USE_LOSS_GAP=0         alpha=sigma(logit(prior)/T)
  ab4b_nogap_direct MERGE_USE_LOSS_GAP=0+PRIOR_DIRECT=1  alpha=prior(不做温度收缩)
  ab5_noprior       MERGE_USE_PRIOR_TERM=0        alpha=sigma(loss_gap/T)
  ab6_fixed05       MERGE_FIXED_ALPHA=0.5         等权融合(不含bank项,见下)
  ab7_target_only   MERGE_FIXED_ALPHA=1.0         只用target模型输出
  ab8_clip_only     MERGE_FIXED_ALPHA=0.0         只用CLIP输出
  hp_robustT1/10    MERGE_ROBUST_TEMPERATURE=1.0/10.0
  hp_cs_0505/0802   CONFIDENCE/STABILITY 权重 0.5/0.5、0.8/0.2
  hp_minw_010/000   MERGE_MIN_WEIGHT=0.1/0.0
  hp_gapclamp_2     MERGE_LOSS_GAP_CLAMP=2.0
  hp_obsprior       MERGE_USE_OBSERVATION_PRIOR=1 (观测次数先验)
  hp_clipstab       MERGE_CLIP_USE_STABILITY=1    clip侧也加入stability

二维网格(单独放到 ablations/grid/):
  hp_cs_XXYY_T*      X+Y=10,alpha=X/10 (CONFIDENCE_WEIGHT),
                     STABILITY_WEIGHT=Y/10,ROBUST_TEMPERATURE=*
                     alpha∈{0.1..0.9},tau∈{1..9} → 每种方法 81 个

注意: ProDe 的融合公式为 new_clip = alpha*out + (1-alpha)*clip - 0.1*bank,
ab6/ab7/ab8 中 bank 惩罚项保持不变;difo 为概率空间 alpha*q_target+(1-alpha)*q_clip。
"""
import os

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT_DIR = os.path.join(REPO, 'cfgs', 'office', 'ablations')
GRID_DIR = os.path.join(OUT_DIR, 'grid')

# 每个实验: (描述, section 键覆盖)
EXPERIMENTS = [
    ('ab0_noadaptive', '关掉自适应 gate,退回旧式 out+clip-0.1*bank 融合',
     {'USE_ADAPTIVE_ALPHA': 0}),
    ('ab2_noconf', '可靠性只用稳定性(去掉置信度项)',
     {'MERGE_USE_CONFIDENCE': 0}),
    ('ab3_nostab', '可靠性只用置信度(去掉稳定性项)',
     {'MERGE_USE_STABILITY': 0}),
    ('ab4_nogap', '去掉 loss gap,alpha=sigma(logit(prior)/T)',
     {'MERGE_USE_LOSS_GAP': 0}),
    ('ab4b_nogap_direct', '去掉 loss gap 且不做温度收缩,alpha=prior',
     {'MERGE_USE_LOSS_GAP': 0, 'MERGE_PRIOR_DIRECT': 1}),
    ('ab5_noprior', '去掉可靠性先验,alpha=sigma(loss_gap/T)',
     {'MERGE_USE_PRIOR_TERM': 0}),
    ('ab6_fixed05', '固定 alpha=0.5 等权融合',
     {'MERGE_FIXED_ALPHA': 0.5}),
    ('ab7_target_only', '固定 alpha=1.0,只用 target 模型输出',
     {'MERGE_FIXED_ALPHA': 1.0}),
    ('ab8_clip_only', '固定 alpha=0.0,只用 CLIP 输出',
     {'MERGE_FIXED_ALPHA': 0.0}),
    ('hp_robustT1', 'MERGE_ROBUST_TEMPERATURE=1.0(更激进)',
     {'MERGE_ROBUST_TEMPERATURE': 1.0}),
    ('hp_robustT10', 'MERGE_ROBUST_TEMPERATURE=10.0(更保守)',
     {'MERGE_ROBUST_TEMPERATURE': 10.0}),
    ('hp_cs_0505', '置信度/稳定性权重 0.5/0.5',
     {'MERGE_CONFIDENCE_WEIGHT': 0.5, 'MERGE_STABILITY_WEIGHT': 0.5}),
    ('hp_cs_0802', '置信度/稳定性权重 0.8/0.2',
     {'MERGE_CONFIDENCE_WEIGHT': 0.8, 'MERGE_STABILITY_WEIGHT': 0.2}),
    ('hp_minw_010', 'MERGE_MIN_WEIGHT=0.1(更强截断)',
     {'MERGE_MIN_WEIGHT': 0.1}),
    ('hp_minw_000', 'MERGE_MIN_WEIGHT=0.0(不截断)',
     {'MERGE_MIN_WEIGHT': 0.0}),
    ('hp_gapclamp_2', 'loss gap 截断宽度 2.0',
     {'MERGE_LOSS_GAP_CLAMP': 2.0}),
    ('hp_obsprior', '启用观测次数先验(MERGE_USE_OBSERVATION_PRIOR=1)',
     {'MERGE_USE_OBSERVATION_PRIOR': 1}),
    ('hp_clipstab', 'clip 侧可靠性也加入 clip_stability',
     {'MERGE_CLIP_USE_STABILITY': 1}),
]

# 方法 -> (base cfg, yaml section 名)
METHODS = {
    'ProDe': ('ProDe.yaml', 'ProDe'),
    'difo': ('difo.yaml', 'DIFO'),
}

# 二维网格参数: alpha = x/10 ∈ {0.1..0.9},tau ∈ {1..9}
GRID_ALPHAS = [round(i / 10, 1) for i in range(1, 10)]
GRID_TAUS = list(range(1, 10))


def gen_method(method, section_name):
    base_path = os.path.join(REPO, 'cfgs', 'office', METHODS[method][0])
    with open(base_path) as f:
        base = yaml.safe_load(f)
    for name, desc, overrides in EXPERIMENTS:
        if name == 'ab0_noadaptive' and method != 'ProDe':
            continue  # ab0 仅 ProDe 有 USE_ADAPTIVE_ALPHA 开关
        variant = yaml.safe_load(yaml.safe_dump(base))
        variant[section_name].update(overrides)
        header = (
            '# 自动生成,勿手改;重新生成: python cfgs/office/ablations/gen_ablation_cfgs.py\n'
            '# 实验: {}  |  {}\n'
            '# 改动: {}\n'
        ).format(name, desc, overrides)
        out_path = os.path.join(OUT_DIR, '{}_{}.yaml'.format(method, name))
        with open(out_path, 'w') as f:
            f.write(header)
            yaml.safe_dump(variant, f, sort_keys=False, default_flow_style=False)
        print('wrote', os.path.relpath(out_path, REPO))


def gen_grid(method, section_name):
    """生成 alpha(CONFIDENCE_WEIGHT) x tau(ROBUST_TEMPERATURE) 的二维网格配置。"""
    base_path = os.path.join(REPO, 'cfgs', 'office', METHODS[method][0])
    with open(base_path) as f:
        base = yaml.safe_load(f)

    for alpha in GRID_ALPHAS:
        x = int(round(alpha * 10))       # 置信度权重分子
        y = 10 - x                        # 稳定性权重分子
        stab = round(y / 10, 1)
        for tau in GRID_TAUS:
            overrides = {
                'MERGE_CONFIDENCE_WEIGHT': alpha,
                'MERGE_STABILITY_WEIGHT': stab,
                'MERGE_ROBUST_TEMPERATURE': float(tau),
            }
            variant = yaml.safe_load(yaml.safe_dump(base))
            variant[section_name].update(overrides)

            name = 'hp_cs_{:02d}{:02d}_T{}'.format(x, y, tau)
            header = (
                '# 自动生成,勿手改;重新生成: python cfgs/office/ablations/gen_ablation_cfgs.py\n'
                '# 二维网格: alpha(CONFIDENCE_WEIGHT)={alpha}, '
                'STABILITY_WEIGHT={stab}, ROBUST_TEMPERATURE={tau}\n'
                '# 改动: {ov}\n'
            ).format(alpha=alpha, stab=stab, tau=tau, ov=overrides)

            out_path = os.path.join(GRID_DIR, '{}_{}.yaml'.format(method, name))
            with open(out_path, 'w') as f:
                f.write(header)
                yaml.safe_dump(variant, f, sort_keys=False, default_flow_style=False)

    print('wrote {} grid cfgs into {}'.format(
        len(GRID_ALPHAS) * len(GRID_TAUS), os.path.relpath(GRID_DIR, REPO)))


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(GRID_DIR, exist_ok=True)
    for method, (_, section_name) in METHODS.items():
        gen_method(method, section_name)
    for method, (_, section_name) in METHODS.items():
        gen_grid(method, section_name)
    print('done')