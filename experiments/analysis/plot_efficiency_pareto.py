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
import matplotlib.patches as mpatches
import seaborn as sns


# =============================================================================
# Hardcoded Data Points (from benchmarking)
# =============================================================================

# DSET (Ours) - Red, Star marker, Solid line
DSET = {
    "points": [
        {"name": "DSET-R18", "gflops": 68.3, "mAP": 0.680},
        {"name": "DSET-R34", "gflops": 104.5, "mAP": 0.702},
    ],
    "color": "#C41E3A",  # Academic red
    "marker": "*",
    "linestyle": "-",
    "linewidth": 2.5,
    "markersize": 16,
    "label": "DSET (Ours)",
}

# RT-DETR (Baseline) - Blue, Circle marker, Dashed line
RTDETR = {
    "points": [
        {"name": "RT-DETR-R18", "gflops": 67.6, "mAP": 0.649},
        {"name": "RT-DETR-R34", "gflops": 103.7, "mAP": 0.692},
    ],
    "color": "#1E3A8A",  # Deep blue
    "marker": "o",
    "linestyle": "--",
    "linewidth": 2,
    "markersize": 10,
    "label": "RT-DETR",
}

# YOLO Series - Grey, Square/Triangle, Scatter only (no lines)
YOLO = {
    "points": [
        {"name": "YOLOv10-S", "gflops": 99.2, "mAP": 0.707},
        {"name": "YOLOv10-M", "gflops": 256.1, "mAP": 0.729},
        {"name": "YOLOv8-S", "gflops": 114.7, "mAP": 0.689},
        {"name": "YOLOv8-M", "gflops": 316.4, "mAP": 0.726},
    ],
    "color": "#6B7280",  # Neutral grey
    "markers": ["s", "^", "s", "^"],  # Square for S, Triangle for M
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

    # --- Pareto Frontier Highlight: shaded region for DSET-R18 advantage ---
    # DSET-R18: (68.3, 0.680) - Best low-cost point ("High Precision @ Low Cost")
    dset_r18_x, dset_r18_y = 68.3, 0.680

    # Light shaded region: "High Precision @ Low Cost" zone
    rect = mpatches.Rectangle(
        (50, 0.670),
        width=dset_r18_x - 50 + 5,
        height=0.02,
        linewidth=0,
        facecolor=DSET["color"],
        alpha=0.15,
        zorder=0,
    )
    ax.add_patch(rect)

    # --- Plot Data ---

    # DSET: Solid line + stars
    dset_x = [p["gflops"] for p in DSET["points"]]
    dset_y = [p["mAP"] for p in DSET["points"]]
    ax.plot(
        dset_x, dset_y,
        color=DSET["color"],
        marker=DSET["marker"],
        linestyle=DSET["linestyle"],
        linewidth=DSET["linewidth"],
        markersize=DSET["markersize"],
        label=DSET["label"],
        zorder=5,
    )

    # RT-DETR: Dashed line + circles
    rtdetr_x = [p["gflops"] for p in RTDETR["points"]]
    rtdetr_y = [p["mAP"] for p in RTDETR["points"]]
    ax.plot(
        rtdetr_x, rtdetr_y,
        color=RTDETR["color"],
        marker=RTDETR["marker"],
        linestyle=RTDETR["linestyle"],
        linewidth=RTDETR["linewidth"],
        markersize=RTDETR["markersize"],
        label=RTDETR["label"],
        zorder=4,
    )

    # YOLO: Scatter only (no lines)
    for i, p in enumerate(YOLO["points"]):
        ax.scatter(
            p["gflops"],
            p["mAP"],
            c=YOLO["color"],
            marker=YOLO["markers"][i],
            s=YOLO["markersize"] ** 2,
            label=YOLO["label"] if i == 0 else None,
            zorder=3,
            edgecolors="white",
            linewidths=0.5,
        )

    # --- Annotate DSET-R18: "High Precision @ Low Cost" ---
    ax.annotate(
        "High Precision @ Low Cost",
        xy=(dset_r18_x, dset_r18_y),
        xytext=(dset_r18_x + 25, dset_r18_y - 0.008),
        fontsize=10,
        fontweight="bold",
        color=DSET["color"],
        ha="left",
        arrowprops=dict(
            arrowstyle="->",
            color=DSET["color"],
            lw=1.2,
            connectionstyle="arc3,rad=0.1",
        ),
        zorder=6,
    )

    # --- Axis setup ---
    ax.set_xlabel("Computational Cost (GFLOPs)")
    ax.set_ylabel("COCO mAP (50-95)")
    ax.set_xlim(50, 350)
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
