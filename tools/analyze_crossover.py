#!/usr/bin/env python3
"""
Bridge 实验事后分析脚本 —— 定理 A 机制验证

基于 bridge 文档 (thmA_real_experiment_bridge.md)，读取训练时保存的 checkpoint，
在预测函数空间中验证反超 (crossover) 的几何机制。

用法:
    python tools/analyze_crossover.py <bridge_ckpt_dir> [--output <output_dir>]

输出:
    - layer0_brier_identity.png    第0层: Brier 恒等式验证
    - layer05_trust_radius.png     第0.5层: 信任半径
    - layer1_modal_decomp.png      第1层: 模态分解
    - report.json                  数值报告
"""

import os, glob, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict


# ==============================================================================
# 核心: 预测空间量定义 (bridge 文档 §1)
# ==============================================================================

def compute_bridge_quantities(P_c, P_tgt, labels, class_num):
    """
    给定 checkpoint 预测，计算所有 bridge 变量。

    输入:
        P_c:    [N, C] base model softmax
        P_tgt:  [N, C] CLIP softmax
        labels: [N]  ground truth (仅用于事后分析)
    返回 dict:
        G:      [N, C] prediction gap = P_tgt - P_c
        U:      [N, C] residual = Y - P_c
        Z:      [N, C] source gap = Y - P_tgt(0)  (初始 CLIP 残差)
        brier_c, brier_tgt, norm_G_sq, norm_U_sq
    """
    N, C = P_c.shape
    Y = torch.zeros(N, C).scatter(1, labels.long().unsqueeze(1), 1)

    G = P_tgt - P_c           # prediction gap
    U = Y - P_c               # residual
    # 行和为零约束: G 和 U 自动满足(softmax 行和为 1, one-hot 行和为 1)

    norm_G_sq = (G ** 2).sum() / N
    norm_U_sq = (U ** 2).sum() / N
    inner_UG = (U * G).sum() / N
    brier_c = ((Y - P_c) ** 2).sum() / N
    brier_tgt = ((Y - P_tgt) ** 2).sum() / N

    return {
        'G': G, 'U': U, 'Z': None,  # Z 需要 P_s (初始 P_tgt)
        'brier_c': brier_c.item(),
        'brier_tgt': brier_tgt.item(),
        'norm_G_sq': norm_G_sq.item(),
        'norm_U_sq': norm_U_sq.item(),
        'inner_UG': inner_UG.item(),
    }


def set_final_Z(all_ckpts):
    """设置所有 checkpoint 的 Z = Y - P_tgt(0)，即初始 CLIP 残差"""
    if len(all_ckpts) == 0:
        return
    P_s = all_ckpts[0]['P_tgt']  # 初始 CLIP 预测
    labels = all_ckpts[0]['labels']
    N, C = P_s.shape
    Y = torch.zeros(N, C).scatter(1, labels.long().unsqueeze(1), 1)
    Z = Y - P_s
    for ckpt in all_ckpts:
        ckpt['Z'] = Z
        ckpt['norm_Z_sq'] = (Z ** 2).sum().item() / N


# ==============================================================================
# 加载
# ==============================================================================

def load_all_checkpoints(bridge_dir):
    """加载所有 bridge_*.pt 文件，按 iter 排序；无文件返回 None"""
    if not os.path.isdir(bridge_dir):
        print(f"警告: 目录不存在: {bridge_dir}，跳过分析")
        return None
    files = sorted(glob.glob(os.path.join(bridge_dir, 'bridge_*.pt')))
    if not files:
        print(f"警告: {bridge_dir} 中未找到 bridge checkpoint，跳过分析")
        return None

    ckpts = []
    for f in files:
        ckpt = torch.load(f, map_location='cpu', weights_only=False)
        ckpt['_file'] = os.path.basename(f)
        ckpts.append(ckpt)

    ckpts.sort(key=lambda x: x['iter'])
    print(f"加载了 {len(ckpts)} 个 checkpoint (iter {ckpts[0]['iter']} → {ckpts[-1]['iter']})")

    # 补充 bridge 量
    class_num = ckpts[0]['P_c'].shape[1]
    for ckpt in ckpts:
        q = compute_bridge_quantities(
            ckpt['P_c'], ckpt['P_tgt'], ckpt['labels'], class_num)
        ckpt.update({k: v for k, v in q.items() if k not in ('Z',)})

    set_final_Z(ckpts)
    return ckpts


# ==============================================================================
# 第 0 层: Brier 恒等式 (§2)
# ==============================================================================

def layer0_verify(ckpts, output_dir):
    """
    ΔL_Brier(k) = ||G_k||²/N - 2<U_k, G_k>/N  (严格恒等式)

    画三条曲线: Q_k(↑gap cost), R_k(↓alignment benefit), Δ_k(=gap)
    """
    iters = [c['iter'] for c in ckpts]
    Q = [c['norm_G_sq'] for c in ckpts]          # gap cost
    R = [2 * c['inner_UG'] for c in ckpts]        # alignment benefit
    delta_direct = [c['brier_tgt'] - c['brier_c'] for c in ckpts]  # 直接 Brier gap
    delta_from_id = [q - r for q, r in zip(Q, R)]                  # 恒等式计算

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Q_k + R_k + Δ_k
    ax = axes[0]
    ax.plot(iters, Q, 'r-', label=r'$Q_k = \|G_k\|^2/N$ (gap cost)')
    ax.plot(iters, R, 'b-', label=r'$R_k = 2\langle U_k,G_k\rangle/N$ (alignment)')
    ax.plot(iters, delta_direct, 'k--', label=r'$\Delta_k$ (Brier gap)')
    ax.axhline(0, color='gray', linestyle=':')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.set_title('Layer 0: Brier Identity Components')
    ax.legend(fontsize=8)

    # (b) 反超判据: R_k > Q_k ?
    ax = axes[1]
    crossover_signal = [r - q for q, r in zip(Q, R)]
    colors = ['green' if s > 0 else 'red' for s in crossover_signal]
    ax.bar(range(len(iters)), crossover_signal, color=colors, alpha=0.6)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel('Checkpoint index')
    ax.set_ylabel(r'$R_k - Q_k$')
    ax.set_title('Crossover Criterion: $R_k > Q_k$ → Crossover')
    ax.set_xticks(range(0, len(iters), max(1, len(iters)//10)))
    ax.set_xticklabels([str(iters[i]) for i in range(0, len(iters), max(1, len(iters)//10))],
                       rotation=45, fontsize=7)

    # (c) Sanity check: Δ 恒等式误差
    ax = axes[2]
    errors = [abs(d - f) for d, f in zip(delta_direct, delta_from_id)]
    ax.semilogy(iters, errors, 'ko-', markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('|Δ_direct - Δ_identity|')
    ax.set_title('Sanity Check: Identity Residual (should be ~0)')

    plt.tight_layout()
    path = os.path.join(output_dir, 'layer0_brier_identity.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Layer 0] 保存: {path}")

    # 报告
    report = {
        'crossover_detected': any(r > q for q, r in zip(Q, R)),
        'crossover_checkpoints': [
            int(iters[i]) for i in range(len(iters)) if R[i] > Q[i]
        ],
        'max_identity_error': float(max(errors)),
    }
    return report


# ==============================================================================
# 第 0.5 层: 信任半径 (§3)
# ==============================================================================

def layer05_trust_radius(ckpts, output_dir):
    """
    检验:
      - ||Z|| = sqrt(Brier(P_s))  (可观测)
      - 初始领先判据: ||Z|| < ||G₀||/2 → ΔL(0) < 0
    """
    if len(ckpts) < 1:
        return {}

    c0 = ckpts[0]
    norm_Z = np.sqrt(c0['norm_Z_sq'])
    norm_G0 = np.sqrt(c0['norm_G_sq'])
    threshold = norm_G0 / 2
    pred_initial_lead = norm_Z < threshold
    actual_initial_gap = c0['brier_tgt'] - c0['brier_c']

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_data = [norm_Z, norm_G0, threshold]
    bar_labels = [r'$\|Z\|$ (source gap)', r'$\|G_0\|$ (initial gap)', r'$\|G_0\|/2$ (threshold)']
    colors = ['orange', 'steelblue', 'gray']
    bars = ax.bar(bar_labels, bar_data, color=colors, alpha=0.7)
    ax.bar_label(bars, fmt='%.4f')
    ax.set_ylabel('Frobenius norm / sqrt(N)')
    ax.set_title(
        f'Layer 0.5: Trust Radius\n'
        f'Prediction: {"LEAD" if pred_initial_lead else "NO LEAD"} '
        f'(||Z|| {"<" if pred_initial_lead else "≥"} ||G₀||/2)\n'
        f'Actual ΔL(0) = {actual_initial_gap:.4f} '
        f'({"Negative→lead" if actual_initial_gap < 0 else "Positive→no lead"})'
    )

    plt.tight_layout()
    path = os.path.join(output_dir, 'layer05_trust_radius.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Layer 0.5] 保存: {path}")

    return {
        'norm_Z': float(norm_Z),
        'norm_G0': float(norm_G0),
        'threshold': float(threshold),
        'pred_initial_lead': bool(pred_initial_lead),
        'actual_initial_delta': float(actual_initial_gap),
        'prediction_correct': bool(pred_initial_lead == (actual_initial_gap < 0)),
    }


# ==============================================================================
# 第 1 层: 模态分解 (§4-5, 方法 A: 轨迹 PCA)
# ==============================================================================

def layer1_modal_decomp(ckpts, output_dir, top_k=6):
    """
    方法 A (轨迹驱动模态):
      1. 对 G_k 做 PCA → 模态基 {e_i}
      2. 投影: g_i(k) = <G_k, e_i>_D
      3. 逐模贡献: f_i(k) = g_i(k)² - 2 u_i(k) g_i(k)
    """
    if len(ckpts) < 3:
        print("[Layer 1] checkpoint 太少 (<3)，跳过模态分解")
        return {}

    iters = np.array([c['iter'] for c in ckpts])

    # 收集 G_k 向量
    N, C = ckpts[0]['P_c'].shape
    G_stack = torch.stack([c['G'].flatten() for c in ckpts]).numpy()  # [T, N*C]
    T = G_stack.shape[0]

    # PCA
    G_mean = G_stack.mean(axis=0, keepdims=True)
    G_centered = G_stack - G_mean
    # 使用 compact SVD
    U_svd, S, Vt = np.linalg.svd(G_centered, full_matrices=False)
    # Vt: [r, N*C], 每行是一个主方向 (在展平空间)
    r_eff = min(top_k, len(S))
    eigs = (S ** 2) / T  # 方差解释

    # 模态基: 每个 e_i 是 [N, C] 矩阵
    modal_bases = [Vt[i].reshape(N, C) for i in range(r_eff)]

    # 逐模投影
    g_proj = np.zeros((r_eff, T))  # [mode, time]
    u_proj = np.zeros((r_eff, T))
    for i in range(r_eff):
        e_i = modal_bases[i]
        for t in range(T):
            g_proj[i, t] = (ckpts[t]['G'].numpy() * e_i).sum() / N
            u_proj[i, t] = (ckpts[t]['U'].numpy() * e_i).sum() / N

    # 逐模贡献 f_i(k) = g_i² - 2 u_i g_i
    f_modal = g_proj ** 2 - 2 * u_proj * g_proj  # [mode, time]

    # === 画图 ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # (a) 特征值谱
    ax = axes[0, 0]
    ax.bar(range(1, r_eff + 1), eigs[:r_eff] / eigs.sum() * 100, alpha=0.7)
    ax.set_xlabel('Mode index')
    ax.set_ylabel('Explained variance (%)')
    ax.set_title('Gap Trajectory PCA Spectrum')

    # (b) 主模态投影 g_i(k)
    ax = axes[0, 1]
    for i in range(min(3, r_eff)):
        ax.plot(iters, g_proj[i], 'o-', markersize=3, label=f'Mode {i+1}')
    ax.axhline(0, color='gray', linestyle=':')
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$g_i(k) = \langle G_k, e_i \rangle$')
    ax.set_title('Top Modal Projections of Gap')
    ax.legend(fontsize=8)

    # (c) 逐模贡献 f_i(k)
    ax = axes[1, 0]
    bottom_pos = np.zeros(T)
    bottom_neg = np.zeros(T)
    for i in range(r_eff):
        pos = np.maximum(f_modal[i], 0)
        neg = np.minimum(f_modal[i], 0)
        ax.bar(iters, pos, bottom=bottom_pos, alpha=0.7,
               label=f'Mode {i+1} ($a_i$={g_proj[i,0]*u_proj[i,0]:.4f})')
        bottom_pos += pos
        # 负贡献单独处理
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$f_i(k)$ contribution')
    ax.set_title('Modal Contribution Decomposition')
    ax.legend(fontsize=7)

    # (d) 正/负贡献堆叠面积
    ax = axes[1, 1]
    total_pos = np.sum(np.maximum(f_modal, 0), axis=0)
    total_neg = np.sum(np.minimum(f_modal, 0), axis=0)
    ax.fill_between(iters, 0, total_pos, alpha=0.5, color='green', label='Positive modes')
    ax.fill_between(iters, total_neg, 0, alpha=0.5, color='red', label='Negative modes')
    ax.plot(iters, total_pos + total_neg, 'k-', linewidth=1, label='Total $\Delta L$')
    ax.axhline(0, color='gray', linestyle=':')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Contribution to Brier gap')
    ax.set_title('Stacked Modal Contributions')
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, 'layer1_modal_decomp.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Layer 1] 保存: {path}")

    # a_i 符号报告
    a_i = g_proj[:, 0] * u_proj[:, 0]
    modal_report = []
    for i in range(r_eff):
        dominates = a_i[i] > 0 and np.any(f_modal[i] < 0)
        modal_report.append({
            'mode': i + 1,
            'a_i': float(a_i[i]),
            'a_i_sign': 'positive' if a_i[i] > 0 else 'negative',
            'explained_var_pct': float(eigs[i] / eigs.sum() * 100),
            'dominates_crossover': bool(dominates),
            'max_neg_contribution': float(np.min(f_modal[i])),
        })

    return {
        'n_modes': r_eff,
        'top_explained_var_pct': float(eigs[:r_eff].sum() / eigs.sum() * 100),
        'modes': modal_report,
    }


# ==============================================================================
# 主入口
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Bridge 实验事后分析')
    parser.add_argument('bridge_dir', help='bridge checkpoint 目录路径')
    parser.add_argument('--output', '-o', default=None,
                        help='输出目录 (默认: bridge_dir/analysis)')
    args = parser.parse_args()

    bridge_dir = args.bridge_dir
    output_dir = args.output or os.path.join(bridge_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)

    print(f"=" * 60)
    print(f"Bridge 实验分析")
    print(f"  Checkpoint 目录: {bridge_dir}")
    print(f"  输出目录:        {output_dir}")
    print(f"=" * 60)

    # 加载
    ckpts = load_all_checkpoints(bridge_dir)
    if ckpts is None:
        return  # 目录不存在或无checkpoint，静默退出

    # 报告容器
    report = {
        'bridge_dir': bridge_dir,
        'n_checkpoints': len(ckpts),
        'iter_range': [int(ckpts[0]['iter']), int(ckpts[-1]['iter'])],
        'class_num': int(ckpts[0]['P_c'].shape[1]),
        'num_samples': int(ckpts[0]['labels'].shape[0]),
    }

    # 第 0 层
    print("\n" + "=" * 40)
    print("Layer 0: Brier Identity Verification (§2)")
    print("=" * 40)
    report['layer0'] = layer0_verify(ckpts, output_dir)
    print(f"  反超检测: {report['layer0']['crossover_detected']}")
    print(f"  恒等式最大误差: {report['layer0']['max_identity_error']:.2e}")

    # 第 0.5 层
    print("\n" + "=" * 40)
    print("Layer 0.5: Trust Radius (§3.6)")
    print("=" * 40)
    report['layer05'] = layer05_trust_radius(ckpts, output_dir)
    if report['layer05']:
        print(f"  ||Z|| = {report['layer05']['norm_Z']:.4f}")
        print(f"  ||G₀||/2 = {report['layer05']['threshold']:.4f}")
        print(f"  预测正确: {report['layer05']['prediction_correct']}")

    # 第 1 层
    print("\n" + "=" * 40)
    print("Layer 1: Modal Decomposition (§4-5)")
    print("=" * 40)
    report['layer1'] = layer1_modal_decomp(ckpts, output_dir)
    if report['layer1']:
        print(f"  模态数: {report['layer1']['n_modes']}")
        print(f"  前 {report['layer1']['n_modes']} 模态解释方差: {report['layer1']['top_explained_var_pct']:.1f}%")
        for m in report['layer1']['modes']:
            print(f"    Mode {m['mode']}: a_i={m['a_i']:.4f} ({m['a_i_sign']}), "
                  f"var={m['explained_var_pct']:.1f}%, "
                  f"dominates={m['dominates_crossover']}")

    # 保存报告
    report_path = os.path.join(output_dir, 'report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告保存: {report_path}")
    print("Done.")


if __name__ == '__main__':
    main()
