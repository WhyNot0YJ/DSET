#!/usr/bin/env python3
"""
Efficiency vs. Accuracy Pareto Frontier Plot for CVPR/ICCV Paper (Section 4.2.2)

Generates a publication-ready figure demonstrating DSET's superior Pareto frontier
compared to RT-DETR and YOLO baselines.

Dependencies:
    pip install matplotlib seaborn

Usage:
    python plot_efficiency_pareto.py
    python plot_efficiency_pareto.py --output_dir ../figures

Output:
    figures/efficiency_pareto.pdf (vector format for LaTeX)
    figures/efficiency_pareto.png (high DPI for presentations)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import seaborn as sns


# =============================================================================
# Hardcoded Data Points (from benchmarking)
# =============================================================================

# DSET-R18 (Ours) - Red, Star marker
DSET = {
    "points": [{"name": "DSET-R18", "gflops": 68.3, "mAP": 0.680}],
    "color": "#C41E3A",  # Academic red
    "marker": "*",
    "markersize": 16,
    "label": "DSET (Ours)",
}

# RT-DETR-R18 (Baseline) - Blue, Circle marker
RTDETR = {
    "points": [{"name": "RT-DETR-R18", "gflops": 67.6, "mAP": 0.649}],
    "color": "#1E3A8A",  # Deep blue
    "marker": "o",
    "markersize": 10,
    "label": "RT-DETR",
}

# YOLO-S only (YOLOv8-S, YOLOv10-S) - Grey, Square/Triangle, Scatter only
YOLO = {
    "points": [
        {"name": "YOLOv10-S", "gflops": 99.2, "mAP": 0.707},
        {"name": "YOLOv8-S", "gflops": 114.7, "mAP": 0.689},
    ],
    "color": "#6B7280",  # Neutral grey
    "markers": ["s", "^"],  # Square for v10, Triangle for v8
    "markersize": 10,
    "label": "YOLO",
}


def setup_publication_style():
    """Configure matplotlib for publication-quality output."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.dpi": 150,
    })


def plot_pareto_frontier(output_dir: Path):
    """Generate the efficiency vs. accuracy Pareto frontier plot."""
    setup_publication_style()
    sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.alpha": 0.6})

    fig, ax = plt.subplots(figsize=(7, 5))

    # Data coordinates for annotations
    dset_x, dset_y = 68.3, 0.680
    rtdetr_x, rtdetr_y = 67.6, 0.649
    yolo8_x, yolo8_y = 114.7, 0.689

    # --- Plot Data (Scatter only, NO connecting lines) ---

    # DSET-R18: Star marker
    for p in DSET["points"]:
        ax.scatter(
            p["gflops"], p["mAP"],
            c=DSET["color"],
            marker=DSET["marker"],
            s=DSET["markersize"] ** 2,
            label=DSET["label"],
            zorder=5,
            edgecolors="white",
            linewidths=0.5,
        )

    # RT-DETR-R18: Circle marker
    for p in RTDETR["points"]:
        ax.scatter(
            p["gflops"], p["mAP"],
            c=RTDETR["color"],
            marker=RTDETR["marker"],
            s=RTDETR["markersize"] ** 2,
            label=RTDETR["label"],
            zorder=4,
            edgecolors="white",
            linewidths=0.5,
        )

    # YOLO-S: Scatter only (YOLOv8-S, YOLOv10-S)
    for i, p in enumerate(YOLO["points"]):
        ax.scatter(
            p["gflops"],
            p["mAP"],
            c=YOLO["color"],
            marker=YOLO["markers"][i],
            s=YOLO["markersize"] ** 2,
            label=p["name"],
            zorder=3,
            edgecolors="white",
            linewidths=0.5,
        )

    # --- Vertical Arrow: Accuracy Gain (RT-DETR-R18 -> DSET-R18) ---
    arrow_vert = FancyArrowPatch(
        (rtdetr_x, rtdetr_y),
        (dset_x, dset_y),
        arrowstyle="->",
        color="#15803D",
        linewidth=2,
        mutation_scale=16,
        zorder=6,
    )
    ax.add_patch(arrow_vert)
    ax.text(
        dset_x + 6,
        (rtdetr_y + dset_y) / 2,
        "+3.1% mAP",
        fontsize=11,
        fontweight="bold",
        color="#15803D",
        va="center",
        ha="left",
        zorder=6,
    )

    # --- Horizontal Arrow: Efficiency Gain (YOLOv8-S -> DSET-R18) ---
    arrow_horiz = FancyArrowPatch(
        (yolo8_x, yolo8_y),
        (dset_x + 5, yolo8_y),
        arrowstyle="->",
        color="#15803D",
        linewidth=2,
        mutation_scale=16,
        zorder=6,
    )
    ax.add_patch(arrow_horiz)
    ax.text(
        (yolo8_x + dset_x) / 2,
        yolo8_y - 0.012,
        "~40% Less GFLOPs",
        fontsize=11,
        fontweight="bold",
        color="#15803D",
        va="top",
        ha="center",
        zorder=6,
    )

    # --- Axis setup ---
    ax.set_xlabel("Computational Cost (GFLOPs)")
    ax.set_ylabel("COCO mAP (50-95)")
    ax.set_xlim(50, 130)
    ax.set_ylim(0.64, 0.74)
    ax.set_aspect("auto")

    # Legend: lower right
    ax.legend(loc="lower right", framealpha=0.95)

    plt.tight_layout()

    # --- Save outputs ---
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "efficiency_pareto.pdf"
    png_path = output_dir / "efficiency_pareto.png"

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate Efficiency vs. Accuracy Pareto frontier plot for paper."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for figures (default: figures/ relative to script)",
    )
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = Path(__file__).parent / "figures"

    plot_pareto_frontier(output_dir)


if __name__ == "__main__":
    main()
