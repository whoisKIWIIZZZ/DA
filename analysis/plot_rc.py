#!/usr/bin/env python3
"""Plot RC-task gate diagnostics from rc_data.csv into two PDF figures.

fig1_rc_selection.pdf
    Three stacked panels over training iterations:
      (a) target test accuracy,
      (b) accuracy on samples where the two branches disagree: the selected
          (mixed) accuracy is the bold blue line, the gray band spans the two
          single-branch accuracies (target vs CLIP), and the light-blue area
          marks how much selection wins over the better single branch,
      (c) disagreement rate.

fig2_rc_alpha.pdf
    The learned mixing weight alpha: mean line with the p10-p90 band and the
    0.5 decision threshold (>= 0.5 -> target branch, < 0.5 -> CLIP branch).
    Additionally, two new methods are overlaid:
      - DIFO: exponential(0.1) sampled 64 times per step (p10‑p90 band + mean)
      - ProDe: constant p10/mean/p90 = 0.5

Usage: python plot_rc.py [CSV_PATH]
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "rc_data.csv"

# Light-mode palette
BLUE = "#2a78d6"
AQUA = "#1baf7a"
ORANGE = "#eb6834"
BLUE_FILL = "#cde2fb"
NEUTRAL_FILL = "#efeeea"
EDGE = "#b6b5ae"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Colors for new methods
DIFO_COLOR = "#d95f0e"      # orange-red
DIFO_FILL = "#fdd49e"       # light orange
PRODE_COLOR = "#4d4d4d"     # dark gray

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.linewidth": 0.8,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.solid_capstyle": "round",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

XMAX = 2420  # right margin reserved for end-of-line labels


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    names = [n for n in rows[0] if n != "task"]
    return {name: np.array([float(r[name]) for r in rows]) for name in names}


def style(ax):
    ax.set_facecolor("white")
    ax.set_xlim(0, XMAX)
    ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def title(ax, text):
    ax.set_title(text, loc="left", fontsize=8.5, color=INK, pad=4)


def swatch(ax, x, y, length, color, lw=2.6):
    ax.plot([x, x + length], [y, y], color=color, lw=lw, zorder=6,
            clip_on=False)


def fig1(d, out_pdf):
    x = d["iter"]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(3.5, 4.8),
                             gridspec_kw={"hspace": 0.30})

    # (a) overall target test accuracy
    ax = axes[0]
    style(ax)
    title(ax, "(a) Target test accuracy")
    ax.plot(x, d["accuracy"], color=AQUA, lw=1.4, zorder=3)
    ax.set_ylim(60, 80)
    ax.set_yticks([60, 65, 70, 75, 80])
    ax.set_ylabel("Accuracy (%)")
    ax.text(x[-1] + 8, d["accuracy"][-1], "%.1f%%" % d["accuracy"][-1],
            fontsize=8, color=INK, va="center", zorder=5)

    # (b) accuracy on disagreed samples: single-side band + selected + gain
    ax = axes[1]
    style(ax)
    title(ax, "(b) Accuracy on disagreed samples")
    ax.fill_between(x, d["disacc_target"], d["disacc_clip"],
                    color=NEUTRAL_FILL, zorder=1)
    better = np.maximum(d["disacc_target"], d["disacc_clip"])
    ax.fill_between(x, better, d["disacc_selected"],
                    where=d["disacc_selected"] >= better,
                    color=BLUE_FILL, zorder=2)
    for col in ("disacc_target", "disacc_clip"):
        ax.plot(x, d[col], color=EDGE, lw=0.9, zorder=3)
    ax.plot(x, d["disacc_selected"], color=BLUE, lw=1.6, zorder=4)
    ax.set_ylim(10, 56)
    ax.set_yticks([10, 20, 30, 40, 50])
    ax.set_ylabel("Accuracy (%)")
    # final per-branch values, top-left
    vals = [("Selected", d["disacc_selected"][-1], BLUE),
            ("Target", d["disacc_target"][-1], EDGE),
            ("CLIP", d["disacc_clip"][-1], EDGE)]
    for i, (name, v, c) in enumerate(vals):
        y = 53.5 - 4.5 * i
        swatch(ax, 60, y, 65, c)
        ax.text(135, y, "%s  %.1f%%" % (name, v), fontsize=7.5, color=INK,
                va="center", zorder=6)
    # biggest gain of selection over the better single branch
    gains = d["disacc_selected"] - better
    k = int(np.argmax(gains))
    ax.annotate("", xy=(x[k], d["disacc_selected"][k]),
                xytext=(x[k], better[k]),
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.9))
    ax.text(x[k] + 28, 0.5 * (better[k] + d["disacc_selected"][k]),
            "+%.1f pp" % gains[k], fontsize=7.5, color=INK, va="center",
            bbox=dict(fc="white", ec="none", pad=0.8), zorder=6)

    # (c) disagreement rate
    ax = axes[2]
    style(ax)
    title(ax, "(c) Disagreement rate")
    ax.plot(x, d["disagree"], color=ORANGE, lw=1.4, zorder=3)
    ax.set_ylim(15, 50)
    ax.set_yticks([20, 30, 40, 50])
    ax.set_ylabel("Disagree (%)")
    ax.set_xlabel("Iteration")
    ax.text(x[-1] + 8, d["disagree"][-1], "%.1f%%" % d["disagree"][-1],
            fontsize=8, color=INK, va="center", zorder=5)

    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_pdf)


def fig2(d, out_pdf):
    x = d["iter"]
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    style(ax)

    # --- decision threshold (dashed) ---
    ax.axhline(0.5, color=INK_MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)

    # --- DSS: learned alpha (original) ---
    ax.fill_between(x, d["alpha_p10"], d["alpha_p90"],
                    color=BLUE_FILL, zorder=2)
    ax.plot(x, d["alpha_p10"], color=EDGE, lw=0.8, zorder=3)
    ax.plot(x, d["alpha_p90"], color=EDGE, lw=0.8, zorder=3)
    ax.plot(x, d["alpha_mean"], color=BLUE, lw=1.6, zorder=4)

    # --- DIFO: exponential(0.1) sampled 64 times per step ---
    np.random.seed(42)  # for reproducibility
    difo_p10 = []
    difo_mean = []
    difo_p90 = []
    for xi in x:
        samples = np.random.exponential(scale=0.1, size=64)
        difo_p10.append(np.percentile(samples, 10))
        difo_mean.append(np.mean(samples))
        difo_p90.append(np.percentile(samples, 90))

    ax.fill_between(x, difo_p10, difo_p90, color=DIFO_FILL, zorder=2)
    ax.plot(x, difo_p10, color=DIFO_COLOR, lw=0.8, zorder=3, alpha=0.7)
    ax.plot(x, difo_p90, color=DIFO_COLOR, lw=0.8, zorder=3, alpha=0.7)
    ax.plot(x, difo_mean, color=DIFO_COLOR, lw=1.6, zorder=4)

    # --- ProDe: constant p10/mean/p90 = 0.5 ---
    ax.axhline(0.5, color=PRODE_COLOR, lw=2.0, zorder=5, linestyle='-')

    # --- y-axis range and ticks ---
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel(r"$\alpha$")
    ax.set_xlabel("Iteration")

    # --- Annotations (swatches + text) ---

    # Threshold text (unchanged)
    ax.text(40, 0.505, r"decision threshold  $\alpha$ = 0.5", fontsize=7,
            color=INK_MUTED, va="bottom", zorder=5,
            bbox=dict(fc="white", ec="none", pad=0.6))

    # DSS annotation (original)
    swatch(ax, 60, 0.77, 65, BLUE)
    ax.text(135, 0.77, "DSS mean", fontsize=7.5, color=INK, va="center", zorder=6)
    ax.add_patch(plt.Rectangle((60, 0.72), 65, 0.03,
                               facecolor=BLUE_FILL, edgecolor='none',
                               zorder=6, clip_on=False))
    ax.text(135, 0.735, "p10–p90", fontsize=7.5, color=INK, va="center", zorder=6)

    # DIFO annotation (new)
    last_p10 = difo_p10[-1]
    last_mean = difo_mean[-1]
    last_p90 = difo_p90[-1]
    # swatch for mean
    swatch(ax, 60, last_mean, 65, DIFO_COLOR)
    ax.text(135, last_mean, "DIFO mean", fontsize=7.5, color=INK, va="center", zorder=6)
    # rectangle for p10–p90
    ax.add_patch(plt.Rectangle((60, last_p10), 65, last_p90 - last_p10,
                               facecolor=DIFO_FILL, edgecolor='none',
                               zorder=6, clip_on=False))
    ax.text(135, (last_p10 + last_p90) / 2, "p10–p90", fontsize=7.5,
            color=INK, va="center", zorder=6)

    # ProDe annotation (new)
    swatch(ax, 60, 0.5, 65, PRODE_COLOR)
    ax.text(135, 0.5, "ProDe", fontsize=7.5, color=INK, va="center", zorder=6)

    # End-of-line value for DSS (kept as in original)
    ax.text(x[-1] + 8, d["alpha_mean"][-1], "%.3f" % d["alpha_mean"][-1],
            fontsize=8, color=INK, va="center", zorder=5)

    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_pdf)


def main(csv_path):
    d = load(csv_path)
    fig1(d, HERE / "fig1_rc_selection.pdf")
    fig2(d, HERE / "fig2_rc_alpha.pdf")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV)