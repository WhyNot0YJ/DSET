#!/usr/bin/env python3
"""
Dual-Sparse Aperture Visualization (Figure 5)

Generates a single 4x4 composite figure for qualitative analysis:
- 4 rows: diverse scenarios (e.g., Daytime Sparse, Nighttime, Different Viewpoint, Medium Flow)
- 4 columns: Original Image + Predicted Boxes | S5 Coarse Heatmap | S4 Fine Heatmap | Combined Dual-Sparse + Predicted Boxes

Default: 4 images from ./image/, output figure5_qualitative_final.pdf (Vector, 300 DPI).
Uses align_map_to_image for physical alignment (S4 stride=16, S5 stride=32).
"""

import sys
import argparse
import json
import cv2
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

# Setup paths
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from experiments.dset.batch_inference import (
        load_model,
        preprocess_image,
        postprocess_outputs,
        draw_boxes,
        CLASS_NAMES,
        COLORS,
    )
    from experiments.dset.visualize_ground_truth import draw_gt_boxes
except ImportError:
    sys.path.insert(0, str(_project_root))
    from experiments.dset.batch_inference import (
        load_model,
        preprocess_image,
        postprocess_outputs,
        draw_boxes,
        CLASS_NAMES,
        COLORS,
    )
    from experiments.dset.visualize_ground_truth import draw_gt_boxes

def align_map_to_image(map_2d, h_feat, w_feat, H_tensor, W_tensor, orig_h, orig_w, normalize_before_resize=False):
    """
    Align feature map to original image using physical space calibration.
    (Reused from visualize_sparsity.py - critical for Dual-Sparse alignment.)

    Handles crop padding to avoid offset when S4 and S5 have different scales.
    """
    valid_h_feat = int(round(orig_h * (h_feat / H_tensor)))
    valid_w_feat = int(round(orig_w * (w_feat / W_tensor)))
    map_valid = map_2d[:valid_h_feat, :valid_w_feat]

    if normalize_before_resize:
        map_min = map_valid.min()
        map_max = map_valid.max()
        if map_max - map_min > 1e-8:
            map_valid = (map_valid - map_min) / (map_max - map_min + 1e-8)
        else:
            map_valid = np.zeros_like(map_valid)

    map_final = cv2.resize(map_valid, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return map_final


def kept_indices_to_level_mask(kept_indices, level_start, level_size, spatial_shape):
    """Convert global kept_indices to per-level binary mask."""
    h, w = spatial_shape
    mask_flat = np.zeros(level_size, dtype=np.float32)
    if kept_indices is None:
        mask_flat[:] = 1.0
    else:
        indices = kept_indices[0].cpu().numpy()
        indices = indices[indices >= 0]
        for idx in indices:
            local_idx = int(idx) - level_start
            if 0 <= local_idx < level_size:
                mask_flat[local_idx] = 1.0
    return mask_flat.reshape(h, w)


def _resolve_gt_annotation_path(image_path, gt_path=None, annotations_dir=None, data_root=None):
    """Resolve GT annotation path (YOLO .txt only)."""
    path = Path(image_path)
    stem = path.stem

    if gt_path:
        p = Path(gt_path)
        if p.is_file():
            return str(p)
        if p.is_dir():
            cand = p / f"{stem}.txt"
            if cand.exists():
                return str(cand)

    if annotations_dir:
        cand = Path(annotations_dir) / f"{stem}.txt"
        if cand.exists():
            return str(cand)

    if data_root:
        dr = Path(data_root)
        for cand in [dr / "labels" / "train" / f"{stem}.txt", dr / "labels" / "val" / f"{stem}.txt"]:
            if cand.exists():
                return str(cand)

    return None


def _load_yolo_annotations(label_path: Path, img_h: int, img_w: int) -> list:
    """Load YOLO format .txt (class_id cx cy w h normalized) and convert to draw_gt_boxes format."""
    if not label_path.exists():
        return []
    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    out = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except (ValueError, IndexError):
            continue
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        if x2 <= x1 or y2 <= y1:
            continue
        if cid < 0 or cid >= len(CLASS_NAMES):
            continue
        out.append({
            "class_id": cid,
            "class_name": CLASS_NAMES[cid],
            "bbox": [x1, y1, x2, y2],
        })
    return out


def process_single_scenario(model, postprocessor, image_path, device, target_size, conf_threshold=0.3,
                            gt_path=None, annotations_dir=None, data_root=None, draw_gt=False, verbose=True):
    """
    Run inference on one image, capture S4/S5 data.
    Returns: (orig_with_gt_boxes, s5_overlay, s4_overlay, combined_with_pred_boxes)
    Column 1: Original + GT; Column 4: Combined Dual-Sparse + Predictions.
    """
    orig_image = cv2.imread(str(image_path))
    if orig_image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    img_tensor, _, meta = preprocess_image(str(image_path), target_size=target_size)
    img_tensor = img_tensor.to(device)
    orig_h, orig_w = meta["orig_size"][0].tolist()
    padded_h, padded_w = meta["padded_h"], meta["padded_w"]

    kept_indices_captured = [None]
    encoder_info_captured = [None]

    def pruner_hook(module, inputs, outputs):
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            _, kept_indices, _ = outputs
            kept_indices_captured[0] = kept_indices

    def encoder_hook(module, inputs, outputs):
        if isinstance(outputs, tuple) and len(outputs) == 2:
            _, info = outputs
            encoder_info_captured[0] = info
        elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            first = outputs[0]
            if hasattr(first, "encoder_info"):
                encoder_info_captured[0] = first.encoder_info

    hook_pruner = None
    hook_encoder = None
    if hasattr(model, "encoder"):
        if hasattr(model.encoder, "shared_token_pruner") and model.encoder.shared_token_pruner is not None:
            hook_pruner = model.encoder.shared_token_pruner.register_forward_hook(pruner_hook)
        hook_encoder = model.encoder.register_forward_hook(encoder_hook)

    with torch.no_grad():
        outputs = model(img_tensor)

    if hook_pruner:
        hook_pruner.remove()
    if hook_encoder:
        hook_encoder.remove()

    # Predictions
    labels, boxes, scores = postprocess_outputs(
        outputs, postprocessor, meta, conf_threshold, target_size, device, verbose=False
    )

    encoder_info = encoder_info_captured[0]
    if encoder_info is None and isinstance(outputs, dict) and "encoder_info" in outputs:
        encoder_info = outputs["encoder_info"]
    if encoder_info is None:
        raise RuntimeError("Could not extract encoder_info from model output.")

    spatial_shapes = encoder_info.get("spatial_shapes", [])
    level_sizes = [h * w for h, w in spatial_shapes]
    level_start_index = encoder_info.get("level_start_index")
    if level_start_index is not None:
        level_start_index = level_start_index.cpu().numpy().tolist()
    else:
        level_start_index = [0] + list(np.cumsum(level_sizes[:-1]))

    layer_wise_heatmaps = encoder_info.get("layer_wise_heatmaps", [])
    if len(layer_wise_heatmaps) < 2:
        raise RuntimeError("Need at least 2 levels (S4, S5).")

    kept_indices = kept_indices_captured[0]
    s4_idx, s5_idx = 0, 1
    s4_shape = spatial_shapes[s4_idx]
    s5_shape = spatial_shapes[s5_idx]

    s4_scores_sigmoid = torch.sigmoid(layer_wise_heatmaps[s4_idx][0, 0]).cpu().numpy()
    s5_scores_sigmoid = torch.sigmoid(layer_wise_heatmaps[s5_idx][0, 0]).cpu().numpy()

    s4_mask = kept_indices_to_level_mask(
        kept_indices, level_start_index[s4_idx], level_sizes[s4_idx], s4_shape
    )
    s5_mask = kept_indices_to_level_mask(
        kept_indices, level_start_index[s5_idx], level_sizes[s5_idx], s5_shape
    )

    h_s4, w_s4 = s4_shape
    h_s5, w_s5 = s5_shape

    s5_heatmap_aligned = align_map_to_image(
        s5_scores_sigmoid, h_s5, w_s5, padded_h, padded_w, orig_h, orig_w, normalize_before_resize=True
    )
    s4_heatmap_aligned = align_map_to_image(
        s4_scores_sigmoid, h_s4, w_s4, padded_h, padded_w, orig_h, orig_w, normalize_before_resize=True
    )
    s5_mask_aligned = align_map_to_image(s5_mask, h_s5, w_s5, padded_h, padded_w, orig_h, orig_w)
    s4_mask_aligned = align_map_to_image(s4_mask, h_s4, w_s4, padded_h, padded_w, orig_h, orig_w)

    combined_mask = np.maximum(s4_mask_aligned, s5_mask_aligned)
    combined_image = orig_image.copy()
    combined_image[combined_mask < 0.5] = 0

    s5_uint8 = (s5_heatmap_aligned * 255).astype(np.uint8)
    s5_colormap = cv2.applyColorMap(s5_uint8, cv2.COLORMAP_JET)
    s5_overlay = cv2.addWeighted(orig_image.copy(), 0.4, s5_colormap, 0.6, 0)

    s4_uint8 = (s4_heatmap_aligned * 255).astype(np.uint8)
    s4_colormap = cv2.applyColorMap(s4_uint8, cv2.COLORMAP_JET)
    s4_overlay = cv2.addWeighted(orig_image.copy(), 0.4, s4_colormap, 0.6, 0)

    # Column 1: Original Image + Ground Truth (optional)
    orig_with_boxes = orig_image.copy()
    if draw_gt:
        gt_path_resolved = _resolve_gt_annotation_path(image_path, gt_path, annotations_dir, data_root)
        if gt_path_resolved and Path(gt_path_resolved).exists():
            ann_path = Path(gt_path_resolved)
            try:
                annotations = _load_yolo_annotations(ann_path, orig_h, orig_w)
                if annotations:
                    orig_with_boxes = draw_gt_boxes(
                        orig_image.copy(), annotations, show_labels=True, line_thickness=1, colors=COLORS
                    )
                    if verbose:
                        print(f"  ✓ GT: {len(annotations)} boxes from {ann_path.name}")
                elif verbose:
                    print(f"  ⚠ GT file empty: {ann_path}")
            except Exception as e:
                if verbose:
                    print(f"  ⚠ GT load failed: {e}")
        elif verbose:
            stem = Path(image_path).stem
            if data_root:
                tried = [
                    str(Path(data_root) / "labels" / "train" / f"{stem}.txt"),
                    str(Path(data_root) / "labels" / "val" / f"{stem}.txt"),
                ]
                print(f"  ⚠ No GT for {Path(image_path).name}. Tried: {tried}. Ensure images match dataset stems.")
            else:
                tried = gt_path_resolved or "(auto-detect failed)"
                print(f"  ⚠ No GT for {Path(image_path).name}. Tried: {tried}. Use --annotations_dir or --data_root")

    # Column 4: Combined Dual-Sparse + Predicted boxes
    combined_with_boxes = draw_boxes(combined_image.copy(), labels, boxes, scores, CLASS_NAMES, COLORS)

    return orig_with_boxes, s5_overlay, s4_overlay, combined_with_boxes


def run_qualitative_4x4_grid(
    model,
    postprocessor,
    image_paths,
    device="cuda",
    output_path="figure5_qualitative_final.pdf",
    target_size=1280,
    conf_threshold=0.3,
    gt_path=None,
    annotations_dir=None,
    data_root=None,
    draw_gt=False,
):
    """Build single 4x4 grid figure (4 scenarios x 4 columns)."""
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 10))
    plt.subplots_adjust(wspace=0.01, hspace=0.05)

    col_titles = ["Original Image", r"$S_5$ Coarse Heatmap", r"$S_4$ Fine Heatmap", "Fused Token Mask"]

    for row, image_path in enumerate(image_paths):
        print(f"Processing scenario {row + 1}/4: {image_path}")
        orig_with_boxes, s5_overlay, s4_overlay, combined_with_boxes = process_single_scenario(
            model, postprocessor, image_path, device, target_size, conf_threshold,
            gt_path=gt_path, annotations_dir=annotations_dir, data_root=data_root, draw_gt=draw_gt,
        )
        imgs = [orig_with_boxes, s5_overlay, s4_overlay, combined_with_boxes]
        for col, (ax, img) in enumerate(zip(axes[row], imgs)):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if row == 0:
                ax.set_title(col_titles[col], fontweight="bold", fontfamily="serif")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=300, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def run_dual_aperture_visualization(
    model,
    image_path,
    device="cuda",
    output_path="figure5_dual_aperture.pdf",
    target_size=1280,
    gt_path=None,
    zoom_region=None,
    class_names=None,
    draw_gt=False,
):
    """
    Main routine: run inference, capture S4/S5 data, build 4-column figure.
    """
    # 1. Load and preprocess
    print(f"Loading image: {image_path}")
    orig_image = cv2.imread(str(image_path))
    if orig_image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    img_tensor, _, meta = preprocess_image(str(image_path), target_size=target_size)
    img_tensor = img_tensor.to(device)
    orig_h, orig_w = meta["orig_size"][0].tolist()
    padded_h, padded_w = meta["padded_h"], meta["padded_w"]

    # 2. Hooks: capture kept_indices (pruner) and encoder_info (encoder)
    kept_indices_captured = [None]
    encoder_info_captured = [None]

    def pruner_hook(module, inputs, outputs):
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            _, kept_indices, _ = outputs
            kept_indices_captured[0] = kept_indices

    def encoder_hook(module, inputs, outputs):
        # Encoder returns (outs, encoder_info) when return_encoder_info=True
        if isinstance(outputs, tuple) and len(outputs) == 2:
            _, info = outputs
            encoder_info_captured[0] = info
        elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            first = outputs[0]
            if hasattr(first, "encoder_info"):
                encoder_info_captured[0] = first.encoder_info

    hook_pruner = None
    hook_encoder = None
    if hasattr(model, "encoder"):
        if hasattr(model.encoder, "shared_token_pruner") and model.encoder.shared_token_pruner is not None:
            hook_pruner = model.encoder.shared_token_pruner.register_forward_hook(pruner_hook)
        hook_encoder = model.encoder.register_forward_hook(encoder_hook)

    # 3. Run inference
    print("Running inference...")
    model.eval()
    if hasattr(model, "encoder") and hasattr(model.encoder, "set_epoch"):
        model.encoder.set_epoch(0)

    with torch.no_grad():
        outputs = model(img_tensor)

    if hook_pruner:
        hook_pruner.remove()
    if hook_encoder:
        hook_encoder.remove()

    # 4. Extract encoder_info (from hook or model output)
    encoder_info = encoder_info_captured[0]
    if encoder_info is None and isinstance(outputs, dict) and "encoder_info" in outputs:
        encoder_info = outputs["encoder_info"]

    if encoder_info is None:
        raise RuntimeError("Could not extract encoder_info from model output. Is the model DSET with dual-scale encoder?")

    # 5. Parse S4 (index 1) and S5 (index 2) from encoder
    use_encoder_idx = getattr(model.encoder, "use_encoder_idx", [1, 2])
    spatial_shapes = encoder_info.get("spatial_shapes", [])
    level_sizes = [h * w for h, w in spatial_shapes]
    level_start_index = encoder_info.get("level_start_index")
    if level_start_index is not None:
        level_start_index = level_start_index.cpu().numpy().tolist()
    else:
        level_start_index = [0] + list(np.cumsum(level_sizes[:-1]))

    layer_wise_heatmaps = encoder_info.get("layer_wise_heatmaps", [])
    if len(layer_wise_heatmaps) < 2:
        raise RuntimeError("Need at least 2 levels (S4, S5). Check use_encoder_idx.")

    kept_indices = kept_indices_captured[0]

    # 6. Build per-level scores and masks
    # Index 0 = S4 (stride 16), Index 1 = S5 (stride 32) when use_encoder_idx=[1,2]
    s4_idx, s5_idx = 0, 1
    s4_shape = spatial_shapes[s4_idx]
    s5_shape = spatial_shapes[s5_idx]

    s4_scores = layer_wise_heatmaps[s4_idx][0, 0].detach().cpu().numpy()
    s5_scores = layer_wise_heatmaps[s5_idx][0, 0].detach().cpu().numpy()

    s4_scores_sigmoid = torch.sigmoid(torch.from_numpy(s4_scores)).numpy()
    s5_scores_sigmoid = torch.sigmoid(torch.from_numpy(s5_scores)).numpy()

    s4_mask = kept_indices_to_level_mask(
        kept_indices, level_start_index[s4_idx], level_sizes[s4_idx], s4_shape
    )
    s5_mask = kept_indices_to_level_mask(
        kept_indices, level_start_index[s5_idx], level_sizes[s5_idx], s5_shape
    )

    # 7. Align to image (critical: different strides for S4=16, S5=32)
    h_s4, w_s4 = s4_shape
    h_s5, w_s5 = s5_shape

    s5_heatmap_aligned = align_map_to_image(
        s5_scores_sigmoid, h_s5, w_s5, padded_h, padded_w, orig_h, orig_w, normalize_before_resize=True
    )
    s4_heatmap_aligned = align_map_to_image(
        s4_scores_sigmoid, h_s4, w_s4, padded_h, padded_w, orig_h, orig_w, normalize_before_resize=True
    )
    s5_mask_aligned = align_map_to_image(
        s5_mask, h_s5, w_s5, padded_h, padded_w, orig_h, orig_w
    )
    s4_mask_aligned = align_map_to_image(
        s4_mask, h_s4, w_s4, padded_h, padded_w, orig_h, orig_w
    )

    # 8. Combined mask (Logical OR): pruned = black
    combined_mask = np.maximum(s4_mask_aligned, s5_mask_aligned)
    combined_image = orig_image.copy()
    combined_image[combined_mask < 0.5] = 0

    # 9. Heatmap overlays (JET colormap)
    s5_uint8 = (s5_heatmap_aligned * 255).astype(np.uint8)
    s5_colormap = cv2.applyColorMap(s5_uint8, cv2.COLORMAP_JET)
    s5_overlay = cv2.addWeighted(orig_image.copy(), 0.4, s5_colormap, 0.6, 0)

    s4_uint8 = (s4_heatmap_aligned * 255).astype(np.uint8)
    s4_colormap = cv2.applyColorMap(s4_uint8, cv2.COLORMAP_JET)
    s4_overlay = cv2.addWeighted(orig_image.copy(), 0.4, s4_colormap, 0.6, 0)

    # 10. Draw GT boxes on original image (optional)
    gt_boxes_loaded = []
    if draw_gt and gt_path and Path(gt_path).exists():
        try:
            class_names = class_names or [
                "Car", "Truck", "Van", "Bus", "Pedestrian",
                "Cyclist", "Motorcyclist", "Trafficcone"
            ]
            gt_labels, gt_boxes, gt_scores = load_gt_boxes(gt_path, class_names)
            gt_boxes_loaded = gt_boxes
            for box in gt_boxes:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(orig_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        except Exception as e:
            print(f"Warning: Could not load GT: {e}")

    # Auto zoom region: smallest GT box if available, else center 25% of image
    if zoom_region is None:
        if len(gt_boxes_loaded) > 0:
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in gt_boxes_loaded]
            idx = np.argmin(areas)
            x1, y1, x2, y2 = gt_boxes_loaded[idx]
            zoom_region = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        else:
            w, h = orig_w, orig_h
            zw, zh = int(w * 0.25), int(h * 0.25)
            zx = (w - zw) // 2
            zy = (h - zh) // 2
            zoom_region = (zx, zy, zw, zh)

    # 11. Build 4-column matplotlib figure
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    # Column 1: Original Image
    axes[0].imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image", fontweight="bold", fontfamily="serif")

    # Column 2: S5 Coarse Heatmap
    axes[1].imshow(cv2.cvtColor(s5_overlay, cv2.COLOR_BGR2RGB))
    axes[1].set_title(r"$S_5$ Coarse Heatmap", fontweight="bold", fontfamily="serif")

    # Column 3: S4 Fine Heatmap (with optional zoom-in)
    axes[2].imshow(cv2.cvtColor(s4_overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title(r"$S_4$ Fine Heatmap", fontweight="bold", fontfamily="serif")

    # Zoom-in on small object (Column 3) - highlights "Contextual Halo" detail
    zx, zy, zw, zh = zoom_region
    if zw > 0 and zh > 0:
        rect = Rectangle((zx, zy), zw, zh, linewidth=2, edgecolor="red", facecolor="none")
        axes[2].add_patch(rect)
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(axes[2], width="35%", height="35%", loc="upper right", borderpad=1)
        s4_crop = s4_overlay[zy : zy + zh, zx : zx + zw]
        if s4_crop.size > 0:
            axins.imshow(cv2.cvtColor(s4_crop, cv2.COLOR_BGR2RGB))
        axins.set_xticks([])
        axins.set_yticks([])
        axins.set_title("Zoom (Halo)", fontsize=8)

    # Column 4: Combined Dual-Sparse
    axes[3].imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    axes[3].set_title("Fused Token Mask", fontweight="bold", fontfamily="serif")

    plt.tight_layout()
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=300, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def _load_default_config():
    """Load default config/checkpoint from benchmark_pruning_curve_example.json"""
    cfg_path = _script_dir / "benchmark_pruning_curve_example.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for model_cfg in data.values():
            if "config" in model_cfg and "checkpoint" in model_cfg:
                return model_cfg["config"], model_cfg["checkpoint"]
    return None, None


def _get_default_images(image_dir=None, data_root=None, max_count=4):
    """Get up to max_count images. Tries image_dir/image/, then data_root/images/train|val/."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    candidates = []
    base = Path(image_dir or ".") / "image"
    if base.exists():
        candidates = sorted([p for p in base.iterdir() if p.suffix.lower() in exts], key=lambda p: str(p))
    if not candidates and data_root:
        dr = Path(data_root)
        for sub in ["images/train", "images/val", "image"]:
            folder = dr / sub
            if folder.exists():
                candidates = sorted([p for p in folder.iterdir() if p.suffix.lower() in exts], key=lambda p: str(p))
                break
    return [str(p) for p in candidates[:max_count]]


def main():
    default_config, default_checkpoint = _load_default_config()

    parser = argparse.ArgumentParser(description="Dual-Sparse Aperture Visualization (Figure 5)")
    parser.add_argument("--image", type=str, default=None, help="Input image path (default: 4 images from ./image/)")
    parser.add_argument("--config", type=str, default=default_config, help="Model config YAML")
    parser.add_argument("--checkpoint", type=str, default=default_checkpoint, help="Model checkpoint")
    parser.add_argument("--output", type=str, default="figure5_qualitative_final.pdf", help="Output PDF path")
    parser.add_argument("--gt_path", type=str, default=None,
                        help="GT .txt path (single file) or directory with {stem}.txt")
    parser.add_argument("--annotations_dir", type=str, default=None,
                        help="Directory with GT .txt: {annotations_dir}/{stem}.txt")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Data root: labels/train|val/{stem}.txt; images from images/train|val if ./image/ empty")
    parser.add_argument("--zoom", type=float, nargs=4, default=None,
                        help="Zoom region: x y w h (e.g., 200 100 150 120)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target_size", type=int, default=1280)
    parser.add_argument("--image_dir", type=str, default=".", help="Base dir for image folder (default: cwd)")
    parser.add_argument("--conf_threshold", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--draw_gt", action="store_true", help="Draw Ground Truth boxes on column 1 (default: off)")
    parser.add_argument("--mode", type=str, default="grid", choices=["grid", "single"],
                        help="grid: 4x4 composite figure (default); single: one 4-column figure per image")
    args = parser.parse_args()

    # Resolve images: --image or 4 from image/ or data_root/images/train|val
    if args.image:
        image_paths = [args.image]
    else:
        image_paths = _get_default_images(args.image_dir, args.data_root, max_count=4)
        if not image_paths:
            print("Error: No --image provided and no images in ./image/ or --data_root/images/train|val")
            return

    if not args.config or not args.checkpoint:
        print("Error: --config and --checkpoint required (or set defaults in benchmark_pruning_curve_example.json)")
        return

    model, postprocessor = load_model(args.config, args.checkpoint, args.device)
    zoom_region = tuple(map(int, args.zoom)) if args.zoom else None

    if args.mode == "grid" and len(image_paths) >= 4:
        # 4x4 composite figure (4 scenarios x 4 columns)
        run_qualitative_4x4_grid(
            model,
            postprocessor,
            image_paths[:4],
            device=args.device,
            output_path=args.output,
            target_size=args.target_size,
            conf_threshold=args.conf_threshold,
            gt_path=args.gt_path,
            annotations_dir=args.annotations_dir,
            data_root=args.data_root,
            draw_gt=args.draw_gt,
        )
    elif args.mode == "grid" and len(image_paths) < 4:
        print(f"Warning: grid mode requires 4 images, found {len(image_paths)}. Falling back to single mode.")
        for i, image_path in enumerate(image_paths):
            run_dual_aperture_visualization(
                model, image_path, device=args.device,
                output_path=args.output if len(image_paths) == 1 else f"{Path(args.output).stem}_{i}.pdf",
                target_size=args.target_size, gt_path=args.gt_path, zoom_region=zoom_region, draw_gt=args.draw_gt,
            )
    else:
        for i, image_path in enumerate(image_paths):
            out_path = args.output
            if len(image_paths) > 1:
                out_path = str(Path(args.output).parent / f"{Path(args.output).stem}_{i}.pdf")
            run_dual_aperture_visualization(
                model, image_path, device=args.device, output_path=out_path,
                target_size=args.target_size, gt_path=args.gt_path, zoom_region=zoom_region, draw_gt=args.draw_gt,
            )


if __name__ == "__main__":
    main()
