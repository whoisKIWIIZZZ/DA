#!/usr/bin/env python3
"""Plot target and CLIP accuracy curves from DIFO evaluation logs."""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


HERE = Path(__file__).resolve().parent
DIFO_LOG = HERE / "AC_difo_260907_221308.txt"
DSS_LOG = HERE / "AC_difo_260908_100058.txt"
OUTPUT = HERE / "AC_difo_accuracy_comparison.pdf"

ACCURACY_PATTERN = re.compile(
    r"Accuracy\s*=\s*([0-9]+(?:\.[0-9]+)?)%\s*->\s*"
    r"CLIP_Accuracy\s*=\s*([0-9]+(?:\.[0-9]+)?)%"
)

TRAIN_END_EPOCH = 15
GRAY = "#808080"

LEGEND_KW = dict(
    loc="lower right",
    frameon=True,
    framealpha=0.96,
    edgecolor="#BFBFBF",
    fancybox=False,
    fontsize=8,
    handlelength=2.8,
    borderpad=0.6,
)


def parse_accuracies(path):
    matches = ACCURACY_PATTERN.findall(path.read_text(encoding="utf-8"))
    if not matches:
        raise ValueError(f"No paired accuracy records found in {path}")
    target, clip = zip(*((float(a), float(b)) for a, b in matches))
    return list(target), list(clip)


def find_cjk_font():
    """Return the path of an installed CJK font, or None."""
    for family in (
        "Times New Roman",
        "SimHei",
    ):
        try:
            return font_manager.findfont(
                font_manager.FontProperties(family=family), fallback_to_default=False
            )
        except ValueError:
            continue
    return None


def plot_segment(ax, values, color, linestyle, marker, start=0, stop=None, label=None):
    """Plot values[start:stop] (epochs start+1 .. stop)."""
    epochs = range(1, len(values) + 1)
    if stop is None:
        stop = len(values)
    ax.plot(
        epochs[start:stop],
        values[start:stop],
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=1.55,
        marker=marker,
        markersize=3.2,
        markerfacecolor="white",
        markeredgewidth=0.8,
        markevery=2,
        zorder=3,
    )


def style_axis(ax):
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.7)


def main():
    difo_target, difo_clip = parse_accuracies(DIFO_LOG)
    dss_target, dss_clip = parse_accuracies(DSS_LOG)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(4.8, 6.0))
    outer = fig.add_gridspec(
        2, 1, height_ratios=[1.45, 2.0], hspace=0.15,
        left=0.14, right=0.97, top=0.97, bottom=0.09,
    )
    ax_a = fig.add_subplot(outer[0])
    ax_b = fig.add_subplot(outer[1])

    blue = "#0072B2"
    vermillion = "#D55E00"

    # ---- (a) DIFO Target and CLIP ----
    plot_segment(ax_a, difo_target, blue, "-", "o", label="DIFO Target")
    plot_segment(ax_a, difo_clip, blue, (0, (4, 2)), "s", label="DIFO CLIP")
    style_axis(ax_a)
    ax_a.set_ylabel("Accuracy (%)")
    ax_a.legend(**LEGEND_KW)
    ax_a.set_title("(a) DIFO Target and CLIP Accuracy", loc="left", fontsize=10)
    ax_a.set_ylim(42, 73)
    ax_a.set_yticks(range(45, 75, 5))

    # ---- (b) Target Accuracy Comparison (no broken axis) ----
    # 绘制完整曲线（训练期与训练后颜色不同，但不断轴）
    plot_segment(ax_b, difo_target, blue, "-", "o",
                 start=0, stop=TRAIN_END_EPOCH, label="DIFO Target")
    plot_segment(ax_b, difo_target, GRAY, "-", "o",
                 start=TRAIN_END_EPOCH - 1, stop=len(difo_target))
    plot_segment(ax_b, dss_target, vermillion, "-", "s",
                 start=0, stop=TRAIN_END_EPOCH, label="DIFO w/ DSS Target")
    plot_segment(ax_b, dss_target, GRAY, "-", "s",
                 start=TRAIN_END_EPOCH - 1, stop=len(dss_target))

    # 训练终点竖线
    ax_b.axvline(TRAIN_END_EPOCH, color=GRAY, linestyle="--", linewidth=1.0, zorder=2)
    cjk_path = find_cjk_font()
    ax_b.text(
        TRAIN_END_EPOCH - 0.3,
        73,  # 直接使用实际坐标
        "Training",
        va="top",
        ha="right",
        color=GRAY,
        fontsize=8,
        fontproperties=font_manager.FontProperties(fname=cjk_path) if cjk_path else None,
        zorder=5,
    )

    style_axis(ax_b)
    ax_b.set_ylabel("Accuracy (%)")
    ax_b.legend(**LEGEND_KW)
    ax_b.set_title("(b) Target Accuracy Comparison", loc="left", fontsize=10)
    ax_b.set_ylim(40, 76)
    ax_b.set_yticks([40, 45, 50, 55, 60, 65, 70, 75])
    ax_b.set_xlim(1, 30)
    ax_b.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax_b.set_xlabel("Epoch")

    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)
    print(f"DIFO points: {len(difo_target)}; DIFO w/ DSS points: {len(dss_target)}")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()