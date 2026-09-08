#!/usr/bin/env python3
"""Plot RC-task gate diagnostics from rc_data.csv into two PDF figures.

fig1_rc_selection.pdf
    Three stacked panels over training epochs:
      (a) target test accuracy,
      (b) accuracy on samples where the two branches disagree: selected
          (mixed) accuracy is the bold blue line, the gray band spans the two
          single-branch accuracies, light-blue area marks selection gain,
      (c) disagreement rate.

fig2_rc_alpha.pdf
    Learned mixing weight alpha: mean line with p10-p90 band, and the 0.5
    decision threshold (>= 0.5 -> target branch, < 0.5 -> CLIP branch).
    Additionally, two new methods are overlaid:
      - DIFO: exponential(0.1) sampled 64 times per step (p10-p90 band + mean)
      - ProDe: constant p10/mean/p90 = 0.5

Usage: python plot_rc.py [CSV_PATH]
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "rc_data.csv"

# ----------------------------------------------------------------------
# 科研风格配色：低饱和、色盲友好、黑白打印可区分
# ----------------------------------------------------------------------
BLUE = "#0072B2"          # 主蓝色：selected / DSS mean
AQUA = "#009E73"          # 青绿色：target accuracy
ORANGE = "#D55E00"        # 橙色：disagreement rate / DIFO
BLUE_FILL = "#cde3f8"     # 浅蓝填充：selection gain / DSS p10-p90
NEUTRAL_FILL = "#e6e6e6"  # 灰色填充：分支区间
EDGE = "#999999"          # 灰色线：单分支 / p10-p90 边界
INK = "#111111"           # 主文字
INK_2 = "#333333"         # 轴标签
INK_MUTED = "#666666"     # 刻度、辅助文字
GRID = "#e0e0e0"          # 网格
AXIS = "#999999"          # 轴线

DIFO_COLOR = "#D55E00"    # DIFO 橙色
DIFO_FILL = "#fcd9b8"     # DIFO 浅橙填充
PRODE_COLOR = "#333333"   # ProDe 深灰

plt.rcParams.update({
    # Times New Roman 兼容设置
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",

    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,

    "axes.linewidth": 0.6,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,

    "lines.solid_capstyle": "round",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",

    # PDF 字体嵌入，保证跨设备显示一致
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    names = [n for n in rows[0] if n != "task"]
    return {name: np.array([float(r[name]) for r in rows]) for name in names}


def add_epoch_column(d, total_epochs=30):
    """将 iter 列转换为 epoch 列，假设总 epoch 数为 total_epochs。"""
    max_iter = d["iter"].max()
    steps_per_epoch = max_iter / total_epochs
    d["epoch"] = d["iter"] / steps_per_epoch
    return d


def style(ax, x_max):
    """统一坐标轴样式：白底、无上/右边框、细网格、动态 x 范围（epoch）。"""
    ax.set_facecolor("white")
    ax.set_xlim(0, x_max * 1.08)          # 右侧留白，防止标注被截断
    ax.set_xticks(np.arange(0, x_max + 1, 5))  # 每5个epoch一个刻度
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Epoch")


def title(ax, text):
    """图内小标题：左对齐、加粗，符合科研图习惯。"""
    ax.set_title(text, loc="left", fontsize=9, color=INK,
                 pad=4, fontweight="bold")


def fig1(d, out_pdf):
    x = d["epoch"]
    x_max = float(x[-1])

    fig, axes = plt.subplots(
        3, 1, sharex=True, figsize=(3.5, 5.0),
        gridspec_kw={"hspace": 0.30}
    )

    # -------------------------
    # (a) target test accuracy
    # -------------------------
    ax = axes[0]
    style(ax, x_max)
    title(ax, "(a) Target test accuracy")
    ax.plot(x, d["accuracy"], color=AQUA, lw=1.4, zorder=3)
    ax.set_ylim(60, 80)
    ax.set_yticks([60, 65, 70, 75, 80])
    ax.set_ylabel("Accuracy (%)")
    # 最终值标注
    ax.text(
        x[-1] + 0.3, d["accuracy"][-1],
        f"{d['accuracy'][-1]:.1f}%",
        fontsize=8, color=INK, va="center", zorder=5
    )

    # ------------------------------------------------
    # (b) accuracy on disagreed samples
    # ------------------------------------------------
    ax = axes[1]
    style(ax, x_max)
    title(ax, "(b) Accuracy on disagreed samples")

    # 灰色带：两个单分支之间的范围
    ax.fill_between(
        x, d["disacc_target"], d["disacc_clip"],
        color=NEUTRAL_FILL, zorder=1
    )

    # 蓝色区域：selection 相对于较好分支的增益
    better = np.maximum(d["disacc_target"], d["disacc_clip"])
    gains = d["disacc_selected"] - better
    ax.fill_between(
        x, better, d["disacc_selected"],
        where=gains >= 0,
        color=BLUE_FILL, zorder=2
    )

    # 单分支曲线：Target 实线，CLIP 虚线
    ax.plot(x, d["disacc_target"], color=EDGE, lw=0.9, ls="-", zorder=3)
    ax.plot(x, d["disacc_clip"], color=EDGE, lw=0.9, ls=(0, (3, 1)), zorder=3)

    # Selected 曲线
    ax.plot(x, d["disacc_selected"], color=BLUE, lw=1.6, zorder=4)

    # 纵轴拉大，保证所有元素可见
    ax.set_ylim(0, 60)
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
    ax.set_ylabel("Accuracy (%)")

    # 图例：标准图例
    handles = [
        Line2D([0], [0], color=BLUE, lw=1.6, label="Selected"),
        Line2D([0], [0], color=EDGE, lw=0.9, ls="-", label="Target"),
        Line2D([0], [0], color=EDGE, lw=0.9, ls=(0, (3, 1)), label="CLIP"),
        Patch(facecolor=NEUTRAL_FILL, edgecolor="none", label="Branch range"),
        Patch(facecolor=BLUE_FILL, edgecolor="none", label="Selection gain"),
    ]
    ax.legend(
        handles=handles, loc="lower right", fontsize=7,
        frameon=False, ncol=2, handlelength=2.0,
        columnspacing=0.8, labelspacing=0.5
    )

    # 最大增益箭头与标注（位置根据 epoch 轴调整）
    k = int(np.argmax(gains))
    ax.annotate(
        "", xy=(x[k], d["disacc_selected"][k]),
        xytext=(x[k], better[k]),
        arrowprops=dict(arrowstyle="-", color=INK, lw=0.9)
    )
    ax.text(
        x[k] + 0.3, 0.5 * (better[k] + d["disacc_selected"][k]),
        f"+{gains[k]:.1f} pp",
        fontsize=7.5, color=INK, va="center",
        bbox=dict(fc="white", ec="none", pad=0.8), zorder=6
    )

    # -------------------------
    # (c) disagreement rate
    # -------------------------
    ax = axes[2]
    style(ax, x_max)
    title(ax, "(c) Disagreement rate")
    ax.plot(x, d["disagree"], color=ORANGE, lw=1.4, zorder=3)
    ax.set_ylim(15, 50)
    ax.set_yticks([20, 30, 40, 50])
    ax.set_ylabel("Disagree (%)")
    ax.set_xlabel("Epoch")
    ax.text(
        x[-1] + 0.3, d["disagree"][-1],
        f"{d['disagree'][-1]:.1f}%",
        fontsize=8, color=INK, va="center", zorder=5
    )

    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_pdf)


def fig2(d, out_pdf):
    x = d["epoch"]
    x_max = float(x[-1])

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    style(ax, x_max)

    # 决策阈值：虚线
    ax.axhline(0.5, color=INK_MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)

    # ---------------- DSS: learned alpha ----------------
    ax.fill_between(
        x, d["alpha_p10"], d["alpha_p90"],
        color=BLUE_FILL, zorder=2
    )
    ax.plot(x, d["alpha_p10"], color=EDGE, lw=0.8, zorder=3)
    ax.plot(x, d["alpha_p90"], color=EDGE, lw=0.8, zorder=3)
    ax.plot(x, d["alpha_mean"], color=BLUE, lw=1.6, zorder=4)

    # ---------------- DIFO: exponential(0.1) ----------------
    np.random.seed(42)  # 保持可重复性
    difo_p10, difo_mean, difo_p90 = [], [], []
    for _ in x:
        samples = np.random.exponential(scale=0.1, size=64)
        difo_p10.append(np.percentile(samples, 10))
        difo_mean.append(np.mean(samples))
        difo_p90.append(np.percentile(samples, 90))

    ax.fill_between(x, difo_p10, difo_p90, color=DIFO_FILL, zorder=2)
    ax.plot(x, difo_p10, color=DIFO_COLOR, lw=0.8, zorder=3, alpha=0.7)
    ax.plot(x, difo_p90, color=DIFO_COLOR, lw=0.8, zorder=3, alpha=0.7)
    ax.plot(x, difo_mean, color=DIFO_COLOR, lw=1.6, zorder=4)

    # ---------------- ProDe: constant 0.5 ----------------
    ax.axhline(0.5, color=PRODE_COLOR, lw=2.0, zorder=5)

    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel(r"$\alpha$")
    ax.set_xlabel("Epoch")

    # ---------------- 图例 ----------------
    handles = [
        Line2D([0], [0], color=BLUE, lw=1.6, label="DSS mean"),
        Patch(facecolor=BLUE_FILL, edgecolor="none", label="DSS p10–p90"),
        Line2D([0], [0], color=DIFO_COLOR, lw=1.6, label="DIFO mean"),
        Patch(facecolor=DIFO_FILL, edgecolor="none", label="DIFO p10–p90"),
        Line2D([0], [0], color=PRODE_COLOR, lw=2.0, label="ProDe"),
        Line2D([0], [0], color=INK_MUTED, lw=0.9,
               ls=(0, (4, 3)), label="Decision threshold"),
    ]
    ax.legend(
        handles=handles, loc="lower right", fontsize=7,
        frameon=False, handlelength=2.0,
        columnspacing=0.8, labelspacing=0.5
    )

    # 终值标注：只保留 DSS mean 最终值
    ax.text(
        x[-1] + 0.3, d["alpha_mean"][-1],
        f"{d['alpha_mean'][-1]:.3f}",
        fontsize=8, color=INK, va="center", zorder=5
    )

    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_pdf)


def main(csv_path):
    d = load(csv_path)
    d = add_epoch_column(d, total_epochs=30)
    fig1(d, HERE / "fig1_rc_selection.pdf")
    fig2(d, HERE / "fig2_rc_alpha.pdf")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV)