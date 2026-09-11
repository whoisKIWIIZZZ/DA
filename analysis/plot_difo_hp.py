#!/usr/bin/env python3
"""Parse DIFO hyperparameter logs and draw publication-ready ablation plots."""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


HERE = Path(__file__).resolve().parent
DEFAULT_LOG = HERE / "difo_hp.log"

CFG_PATTERN = re.compile(r"^CFG:\s+.*?/(?P<name>[^/\s]+)\.yaml\b")
FINAL_ACCURACY_PATTERN = re.compile(
    r"Task:\s*AC,\s*Iter:(?P<current>\d+)/(?P<total>\d+);"
    r"\s*Accuracy\s*=\s*(?P<accuracy>\d+(?:\.\d+)?)%"
)
CS_PATTERN = re.compile(r"^difo_hp_cs_(?P<x>\d{2})(?P<y>\d{2})$")
TAU_PATTERN = re.compile(r"^difo_hp_robustT(?P<tau>\d+(?:\.\d+)?)$")

BLUE = "#0072B2"
ORANGE = "#D55E00"
GRID = "#D9D9D9"
AXIS = "#4D4D4D"


def parse_results(log_path):
    """Return sorted (hyperparameter, accuracy) pairs for alpha and tau."""
    alpha_results = {}
    tau_results = {}
    current_config = None

    for line in log_path.read_text(encoding="utf-8").splitlines():
        config_match = CFG_PATTERN.match(line)
        if config_match:
            current_config = config_match.group("name")
            continue

        accuracy_match = FINAL_ACCURACY_PATTERN.search(line)
        if not accuracy_match or current_config is None:
            continue
        if accuracy_match.group("current") != accuracy_match.group("total"):
            continue

        accuracy = float(accuracy_match.group("accuracy"))
        cs_match = CS_PATTERN.match(current_config)
        tau_match = TAU_PATTERN.match(current_config)

        if cs_match:
            x = int(cs_match.group("x"))
            y = int(cs_match.group("y"))
            if x + y != 10:
                raise ValueError(
                    f"Invalid coefficient pair in {current_config}: {x} + {y} != 10"
                )
            alpha_results[x / 10] = accuracy
        elif tau_match:
            tau_results[float(tau_match.group("tau"))] = accuracy

    if not alpha_results:
        raise ValueError(f"No difo_hp_cs_* final AC results found in {log_path}")
    if not tau_results:
        raise ValueError(f"No difo_hp_robustT* final AC results found in {log_path}")

    return sorted(alpha_results.items()), sorted(tau_results.items())


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.7,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": AXIS,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def plot_ablation(points, xlabel, color, output_stem):
    x_values, accuracies = zip(*points)
    fig, ax = plt.subplots(figsize=(3.35, 2.45))

    ax.plot(
        x_values,
        accuracies,
        color=color,
        linewidth=1.7,
        marker="o",
        markersize=5.2,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.25,
        zorder=3,
    )

    best_index = max(range(len(accuracies)), key=accuracies.__getitem__)
    ax.scatter(
        [x_values[best_index]],
        [accuracies[best_index]],
        s=24,
        color=color,
        zorder=4,
    )
    ax.annotate(
        f"{accuracies[best_index]:.2f}",
        xy=(x_values[best_index], accuracies[best_index]),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8,
        color=AXIS,
    )

    spread = max(accuracies) - min(accuracies)
    padding = max(0.25, spread * 0.28)
    ax.set_ylim(min(accuracies) - padding, max(accuracies) + padding * 1.65)
    ax.set_xticks(x_values)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy (%)")
    ax.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.55)

    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def format_points(name, points):
    return f"{name}: " + ", ".join(f"{value:g} -> {accuracy:.2f}%" for value, accuracy in points)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="?", type=Path, default=DEFAULT_LOG)
    parser.add_argument(
        "--output-dir", type=Path, default=HERE, help="directory for PDF and PNG files"
    )
    args = parser.parse_args()

    alpha_points, tau_points = parse_results(args.log)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    outputs = []
    outputs.extend(
        plot_ablation(
            alpha_points, r"Hyperparameter $\alpha$", BLUE, args.output_dir / "difo_hp_alpha"
        )
    )
    outputs.extend(
        plot_ablation(
            tau_points, r"Hyperparameter $\tau$", ORANGE, args.output_dir / "difo_hp_tau"
        )
    )

    print(format_points("alpha", alpha_points))
    print(format_points("tau", tau_points))
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
