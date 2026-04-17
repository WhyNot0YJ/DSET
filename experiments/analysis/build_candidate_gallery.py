#!/usr/bin/env python3
"""
Candidate gallery builder for CaS-DETR qualitative figure selection.

Given a COCO-style test-set JSON and ours/baseline checkpoints, this script
samples N images from the test set, runs both models on each, and writes a
multi-page PDF where every row shows:

    [Original | Importance Map S5 | Ours mask+pred | Baseline pred]

Each row is annotated with image_id and file_name so you can quickly mark
the good cases and copy their file paths back into the final 4x4 figure
script.

Designed for Figure-5 style selection on DAIR-V2X and UA-DETRAC test splits.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

# Repo root and CaS-DETR on path
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
_cas_detr_root = _project_root / "experiments" / "CaS-DETR"
for p in (_project_root, _cas_detr_root):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Reuse helpers from visualize_dual_aperture_cas_detr.py
_vis_path = _script_dir / "visualize_dual_aperture_cas_detr.py"
_spec = importlib.util.spec_from_file_location("_vis_mod", _vis_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load module from {_vis_path}")
_vis_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vis_mod)

load_model_and_post = _vis_mod.load_model_and_post
class_names_from_yaml = _vis_mod.class_names_from_yaml
colors_for_classes = _vis_mod.colors_for_classes
process_single_scenario = _vis_mod.process_single_scenario
build_baseline_overlay = _vis_mod.build_baseline_overlay


def load_coco_images(ann_json: str) -> List[Dict[str, Any]]:
    with open(ann_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    imgs = data.get("images", [])
    if not imgs:
        raise RuntimeError(f"No images found in {ann_json}")
    return imgs


def sample_image_entries(
    images: Sequence[Dict[str, Any]],
    count: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    if count >= len(images):
        return list(images)
    return rng.sample(list(images), count)


def resolve_image_path(
    entry: Dict[str, Any],
    image_root: str,
    strip_prefix: Optional[str],
) -> Path:
    fn = entry["file_name"]
    if strip_prefix and fn.startswith(strip_prefix):
        fn = fn[len(strip_prefix):]
    return Path(image_root) / fn


def render_gallery_pdf(
    rows_panels: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]],
    output_pdf: str,
    rows_per_page: int,
    fig_width: float,
    row_height: float,
    dpi: int,
) -> None:
    """Render all rows into a multi-page PDF.

    Each entry in rows_panels is (original, heatmap, ours_pred, baseline_pred,
    row_label, stat_text).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    matplotlib.rcParams["pdf.compression"] = 9
    matplotlib.rcParams["font.family"] = "serif"

    col_titles = [
        "Original",
        "Importance Map S5",
        "Ours (mask + pred)",
        "Baseline pred",
    ]

    out_path = Path(output_pdf)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(rows_panels)
    with PdfPages(str(out_path)) as pdf:
        page_idx = 0
        for start in range(0, total, rows_per_page):
            chunk = rows_panels[start:start + rows_per_page]
            n_rows = len(chunk)
            fig, axes = plt.subplots(
                nrows=n_rows,
                ncols=4,
                figsize=(fig_width, row_height * n_rows),
            )
            if n_rows == 1:
                axes = np.array([axes])
            plt.subplots_adjust(wspace=0.02, hspace=0.25, left=0.08, right=0.99, top=0.97, bottom=0.02)

            for ri, (o1, o2, o3, o4, label, stat_text) in enumerate(chunk):
                imgs = [o1, o2, o3, o4]
                for ci, (ax, img) in enumerate(zip(axes[ri], imgs)):
                    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    if ri == 0:
                        ax.set_title(col_titles[ci], fontsize=8, fontweight="bold")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.axis("off")
                    if ci == 2 and stat_text:
                        ax.text(
                            0.03,
                            0.97,
                            stat_text,
                            transform=ax.transAxes,
                            ha="left",
                            va="top",
                            color="white",
                            fontsize=6,
                            bbox={
                                "facecolor": "black",
                                "edgecolor": "none",
                                "boxstyle": "round,pad=0.15",
                                "alpha": 0.8,
                            },
                        )
                # Row label on the left of the leftmost axis
                axes[ri][0].text(
                    -0.02,
                    0.5,
                    label,
                    transform=axes[ri][0].transAxes,
                    ha="right",
                    va="center",
                    fontsize=7,
                    rotation=0,
                )

            page_idx += 1
            pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"  Page {page_idx}: wrote {n_rows} rows (rows {start + 1}..{start + n_rows}/{total})")


def main():
    parser = argparse.ArgumentParser(description="Build candidate gallery PDF for Figure 5 case selection")
    # Ours model (CaS-DETR full)
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True)
    # Baseline model (all-off / DEIM-S)
    parser.add_argument("--baseline_config", type=str, required=True)
    parser.add_argument("--baseline_resume", type=str, required=True)
    # Test set
    parser.add_argument("--ann_json", type=str, required=True, help="COCO-style test annotation json")
    parser.add_argument("--image_root", type=str, required=True, help="Root dir containing the file_name paths")
    parser.add_argument(
        "--strip_prefix",
        type=str,
        default=None,
        help="Optional prefix to strip from file_name before joining with image_root (e.g. 'image/')",
    )
    # Sampling
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    # Inference
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_epoch", type=int, default=5)
    parser.add_argument("--baseline_eval_epoch", type=int, default=5)
    parser.add_argument("--conf_threshold", type=float, default=0.3)
    # Output
    parser.add_argument("--output", type=str, required=True, help="Output PDF path")
    parser.add_argument("--rows_per_page", type=int, default=8)
    parser.add_argument("--fig_width", type=float, default=14.0)
    parser.add_argument("--row_height", type=float, default=1.9)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument(
        "--index_json",
        type=str,
        default=None,
        help="Optional path to dump the sampled [(page, row, image_id, file_name)] index as JSON",
    )
    args = parser.parse_args()

    # Load test images list
    images = load_coco_images(args.ann_json)
    print(f"Loaded {len(images)} images from {args.ann_json}")
    sampled = sample_image_entries(images, args.num_samples, args.seed)
    print(f"Sampling {len(sampled)} images (seed={args.seed})")

    # Load models once
    print(f"Loading OURS model: {args.resume}")
    model, postprocessor, cfg = load_model_and_post(args.config, args.resume, args.device)
    class_names = class_names_from_yaml(cfg.yaml_cfg)
    colors = colors_for_classes(len(class_names))

    print(f"Loading BASELINE model: {args.baseline_resume}")
    b_model, b_postprocessor, b_cfg = load_model_and_post(args.baseline_config, args.baseline_resume, args.device)
    b_class_names = class_names_from_yaml(b_cfg.yaml_cfg)
    b_colors = colors_for_classes(len(b_class_names))

    rows_panels: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]] = []
    index_records: List[Dict[str, Any]] = []

    for i, entry in enumerate(sampled):
        img_path = resolve_image_path(entry, args.image_root, args.strip_prefix)
        image_id = entry.get("id", -1)
        file_name = entry.get("file_name", "")
        if not img_path.exists():
            print(f"  [{i + 1}/{len(sampled)}] SKIP missing: {img_path}")
            continue
        try:
            o1, o2, o3, stat_text = process_single_scenario(
                model,
                postprocessor,
                img_path,
                args.device,
                int(args.eval_epoch),
                float(args.conf_threshold),
                class_names,
                colors,
                verbose=False,
            )
            o4 = build_baseline_overlay(
                b_model,
                b_postprocessor,
                img_path,
                args.device,
                int(args.baseline_eval_epoch),
                float(args.conf_threshold),
                b_class_names,
                b_colors,
            )
        except Exception as e:
            print(f"  [{i + 1}/{len(sampled)}] ERROR on {file_name}: {e}")
            continue

        label = f"#{i + 1}\nid={image_id}\n{file_name}"
        rows_panels.append((o1, o2, o3, o4, label, stat_text))
        index_records.append({
            "rank": i + 1,
            "image_id": image_id,
            "file_name": file_name,
            "abs_path": str(img_path),
        })
        print(f"  [{i + 1}/{len(sampled)}] OK  id={image_id}  {file_name}  {stat_text}")

    if not rows_panels:
        print("No rows produced. Nothing to save.")
        return

    print(f"Rendering {len(rows_panels)} rows into PDF: {args.output}")
    render_gallery_pdf(
        rows_panels,
        args.output,
        rows_per_page=int(args.rows_per_page),
        fig_width=float(args.fig_width),
        row_height=float(args.row_height),
        dpi=int(args.dpi),
    )

    if args.index_json:
        idx_path = Path(args.index_json)
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(index_records, f, indent=2, ensure_ascii=False)
        print(f"Saved index: {idx_path}")

    print(f"Saved gallery: {args.output}")


if __name__ == "__main__":
    main()
