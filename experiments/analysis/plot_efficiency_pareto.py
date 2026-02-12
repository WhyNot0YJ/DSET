#!/usr/bin/env python3
"""
Edge-Focus Efficiency vs. Accuracy Plot for CVPR/ICCV Paper (Section 4.2.2)

Zoomed-in scatter plot comparing DSET-R18 vs RT-DETR-R18, YOLOv8-S, YOLOv10-S.
Highlights the "Sweet Spot" (High Accuracy, Low GFLOPs) in the top-left.

Dependencies:
    pip install matplotlib seaborn

Usage:
    python plot_efficiency_pareto.py
    python plot_efficiency_pareto.py --output_dir ../figures

Output:
    figures/efficiency_edge_focus.pdf (vector format for LaTeX)
    figures/efficiency_edge_focus.png (high DPI)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import seaborn as sns


# =============================================================================
# Hardcoded Data Points - Edge-Focus (R18, S only)
# =============================================================================

# DSET-R18 (Ours) - Red Star, Size 250
DSET = {
    "points": [{"name": "DSET-R18", "gflops": 68.3, "mAP": 0.680}],
    "color": "#C41E3A",
    "marker": "*",
    "size": 250,
    "label": "DSET (Ours)",
}

# RT-DETR-R18 (Base) - Blue Circle, Size 180
RTDETR = {
    "points": [{"name": "RT-DETR-R18", "gflops": 67.6, "mAP": 0.649}],
    "color": "#1E3A8A",
    "marker": "o",
    "size": 180,
    "label": "RT-DETR",
}

# YOLOv10-S - Grey Triangle, Size 120
# YOLOv8-S - Grey Square, Size 120
YOLO = {
    "points": [
        {"name": "YOLOv10-S", "gflops": 99.2, "mAP": 0.707},
        {"name": "YOLOv8-S", "gflops": 114.7, "mAP": 0.689},
    ],
    "color": "#6B7280",
    "markers": ["^", "s"],  # Triangle for v10, Square for v8
    "size": 120,
    "label": "YOLO",
}


def setup_publication_style():
    """Configure matplotlib for publication-quality output (Times New Roman, 14pt)."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "figure.dpi": 150,
    })


def plot_pareto_frontier(output_dir: Path):
    """Generate Edge-Focus zoomed-in scatter plot."""
    setup_publication_style()
    sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.alpha": 0.6})

    fig, ax = plt.subplots(figsize=(7, 5))

    # Data coordinates
    dset_x, dset_y = 68.3, 0.680
    rtdetr_x, rtdetr_y = 67.6, 0.649
    yolo10_x, yolo10_y = 99.2, 0.707

    # --- Target Zone: Light red/pink box (top-left = High Accuracy, Low FLOPs) ---
    rect = Rectangle(
        (60, 0.67),
        width=25,
        height=0.05,
        linewidth=0,
        facecolor="#FECACA",  # Very light red/pink
        alpha=0.4,
        zorder=0,
    )
    ax.add_patch(rect)

    # --- Plot Data (Scatter only) ---

    # DSET-R18: Red Star, Size 250
    for p in DSET["points"]:
        ax.scatter(
            p["gflops"], p["mAP"],
            c=DSET["color"],
            marker=DSET["marker"],
            s=DSET["size"],
            label=DSET["label"],
            zorder=5,
            edgecolors="white",
            linewidths=1,
        )

    # RT-DETR-R18: Blue Circle, Size 180
    for p in RTDETR["points"]:
        ax.scatter(
            p["gflops"], p["mAP"],
            c=RTDETR["color"],
            marker=RTDETR["marker"],
            s=RTDETR["size"],
            label=RTDETR["label"],
            zorder=4,
            edgecolors="white",
            linewidths=1,
        )

    # YOLO-S: Grey Triangle (v10), Square (v8), Size 120
    for i, p in enumerate(YOLO["points"]):
        ax.scatter(
            p["gflops"], p["mAP"],
            c=YOLO["color"],
            marker=YOLO["markers"][i],
            s=YOLO["size"],
            label=p["name"],
            zorder=3,
            edgecolors="white",
            linewidths=0.5,
        )

    # --- Arrow 1: Accuracy Gain (RT-DETR -> DSET, upward) ---
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
        dset_x + 4,
        (rtdetr_y + dset_y) / 2,
        "+3.1% mAP",
        fontsize=12,
        fontweight="bold",
        color="#15803D",
        va="center",
        ha="left",
        zorder=6,
    )

    # --- Arrow 2: Efficiency Gain (YOLOv10-S -> DSET, leftward) ---
    arrow_horiz = FancyArrowPatch(
        (yolo10_x, yolo10_y),
        (dset_x + 5, yolo10_y),
        arrowstyle="->",
        color="#15803D",
        linewidth=2,
        mutation_scale=16,
        zorder=6,
    )
    ax.add_patch(arrow_horiz)
    ax.text(
        (yolo10_x + dset_x) / 2,
        yolo10_y + 0.008,
        "-30% GFLOPs",
        fontsize=12,
        fontweight="bold",
        color="#15803D",
        va="bottom",
        ha="center",
        zorder=6,
    )

    # --- Axis setup (Zoomed: tight limits) ---
    ax.set_xlabel("Computational Cost (GFLOPs)")
    ax.set_ylabel("COCO mAP (50-95)")
    ax.set_xlim(60, 120)
    ax.set_ylim(0.64, 0.72)
    ax.set_aspect("auto")

    # Legend: Top left
    ax.legend(loc="upper left", framealpha=0.95)

    plt.tight_layout()

    # --- Save outputs ---
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "efficiency_edge_focus.pdf"
    png_path = output_dir / "efficiency_edge_focus.png"

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
