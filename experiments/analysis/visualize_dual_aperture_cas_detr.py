#!/usr/bin/env python3
"""
4x4 paper-style qualitative figure for CaS-DETR checkpoints (YAMLConfig + .pth).

Rows: four input images. Columns: original image, S5 coarse heatmap, S5 token mask,
prediction + GT overlay. S4 is not visualized. Input to the network is fixed 640x640,
same as tools/inference/torch_inf.py stretch resize.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# Repo root and CaS-DETR on path
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
_cas_detr_root = _project_root / "experiments" / "CaS-DETR"
for p in (_project_root, _cas_detr_root):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from engine.core import YAMLConfig  # noqa: E402

# Load train_end_inference_vis by file path so we do not import experiments.common
# package, whose __init__ pulls in optional `common.*` modules.
_train_end_vis_path = _project_root / "experiments" / "common" / "train_end_inference_vis.py"
_spec = importlib.util.spec_from_file_location("_train_end_inference_vis", _train_end_vis_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load module from {_train_end_vis_path}")
_train_end_vis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train_end_vis)
DEFAULT_COLORS_BGR = _train_end_vis.DEFAULT_COLORS_BGR
draw_boxes_bgr = _train_end_vis.draw_boxes_bgr

INPUT_SIZE = 640


def class_names_from_yaml(yaml_cfg: Dict[str, Any]) -> List[str]:
    names = yaml_cfg.get("vis_class_names")
    if names:
        return list(names)
    nested = yaml_cfg.get("train_end_vis")
    if isinstance(nested, dict) and nested.get("class_names"):
        return list(nested["class_names"])
    ann = None
    vd = yaml_cfg.get("val_dataloader")
    if isinstance(vd, dict):
        ds = vd.get("dataset")
        if isinstance(ds, dict):
            ann = ds.get("ann_file")
    if ann:
        p = Path(ann)
        if p.is_file():
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            cats = data.get("categories", [])
            if cats:
                return [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
    n = int(yaml_cfg.get("num_classes", 80))
    return [f"class_{i}" for i in range(n)]


def colors_for_classes(n: int) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    for i in range(n):
        out.append(DEFAULT_COLORS_BGR[i % len(DEFAULT_COLORS_BGR)])
    return out


def local_kept_indices_to_mask(
    kept_indices: Optional[torch.Tensor],
    spatial_shape: Tuple[int, int],
    batch_idx: int = 0,
) -> np.ndarray:
    h, w = spatial_shape
    n = h * w
    mask_flat = np.zeros(n, dtype=np.float32)
    if kept_indices is None:
        mask_flat[:] = 1.0
    else:
        idx = kept_indices[batch_idx].detach().cpu().numpy().reshape(-1)
        idx = idx[idx >= 0]
        for i in idx:
            ii = int(i)
            if 0 <= ii < n:
                mask_flat[ii] = 1.0
    return mask_flat.reshape(h, w)


def scores_hw_to_orig(
    scores_hw: np.ndarray,
    orig_w: int,
    orig_h: int,
    normalize: bool,
) -> np.ndarray:
    u = cv2.resize(
        scores_hw.astype(np.float32),
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    u = cv2.resize(
        u,
        (orig_w, orig_h),
        interpolation=cv2.INTER_LINEAR,
    )
    if normalize:
        u_min = float(u.min())
        u_max = float(u.max())
        if u_max - u_min > 1e-8:
            u = (u - u_min) / (u_max - u_min + 1e-8)
        else:
            u = np.zeros_like(u, dtype=np.float32)
    return u


def mask_hw_to_orig(mask_hw: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
    u = cv2.resize(
        mask_hw.astype(np.float32),
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )
    return cv2.resize(
        u,
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,
    )


def load_model_and_post(cfg_path: str, resume: str, device: str):
    cfg = YAMLConfig(cfg_path, resume=resume)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    ckpt = torch.load(resume, map_location="cpu")
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    cfg.model.load_state_dict(state)

    model = cfg.model.eval().to(device)
    postprocessor = cfg.postprocessor.eval().to(device)
    return model, postprocessor, cfg


def preprocess_image_640(path: str, device: str):
    im_pil = Image.open(path).convert("RGB")
    w, h = im_pil.size
    transforms = T.Compose(
        [
            T.Resize((INPUT_SIZE, INPUT_SIZE)),
            T.ToTensor(),
        ]
    )
    tensor = transforms(im_pil).unsqueeze(0).to(device)
    orig_target_sizes = torch.tensor([[w, h]], device=device, dtype=torch.float32)
    return tensor, orig_target_sizes


def postprocess_to_drawable(
    results: List[Dict[str, Any]],
    conf_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r0 = results[0]
    labels = r0["labels"].detach().cpu().numpy()
    boxes = r0["boxes"].detach().cpu().numpy()
    scores = r0["scores"].detach().cpu().numpy()
    keep = scores > conf_threshold
    return labels[keep], boxes[keep], scores[keep]


def _resolve_gt_annotation_path(
    image_path: Path,
    gt_path: Optional[str],
    annotations_dir: Optional[str],
    data_root: Optional[str],
) -> Optional[str]:
    stem = image_path.stem
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
        for cand in (
            dr / "labels" / "train" / f"{stem}.txt",
            dr / "labels" / "val" / f"{stem}.txt",
        ):
            if cand.exists():
                return str(cand)
    return None


def _load_yolo_annotations(
    label_path: Path,
    img_h: int,
    img_w: int,
    class_names: Sequence[str],
) -> List[Dict[str, Any]]:
    if not label_path.exists():
        return []
    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    out: List[Dict[str, Any]] = []
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
        if cid < 0 or cid >= len(class_names):
            continue
        out.append(
            {
                "class_id": cid,
                "class_name": class_names[cid],
                "bbox": [x1, y1, x2, y2],
            }
        )
    return out


def draw_gt_yolo(
    bgr: np.ndarray,
    annotations: List[Dict[str, Any]],
    class_names: Sequence[str],
    colors: Sequence[Tuple[int, int, int]],
) -> np.ndarray:
    img = bgr.copy()
    for ann in annotations:
        cid = int(ann["class_id"])
        x1, y1, x2, y2 = [int(round(v)) for v in ann["bbox"]]
        color = colors[cid % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        name = class_names[cid] if cid < len(class_names) else str(cid)
        cv2.putText(
            img,
            name,
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return img


def add_panel_text(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    if not text:
        return out
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(out, (8, 8), (20 + text_w, 18 + text_h), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (14, 14 + text_h),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def draw_prediction_gt_overlay(
    bgr: np.ndarray,
    labels: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_names: Sequence[str],
    colors: Sequence[Tuple[int, int, int]],
    gt_annotations: Sequence[Dict[str, Any]],
) -> np.ndarray:
    overlay = bgr.copy()
    if gt_annotations:
        for ann in gt_annotations:
            cid = int(ann["class_id"])
            x1, y1, x2, y2 = [int(round(v)) for v in ann["bbox"]]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)
            name = class_names[cid] if cid < len(class_names) else str(cid)
            cv2.putText(
                overlay,
                f"GT {name}",
                (x1, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
    overlay = draw_boxes_bgr(
        overlay,
        labels,
        boxes,
        scores,
        list(class_names),
        colors,
    )
    return overlay


def process_single_scenario(
    model: torch.nn.Module,
    postprocessor: torch.nn.Module,
    image_path: Path,
    device: str,
    conf_threshold: float,
    class_names: Sequence[str],
    colors: Sequence[Tuple[int, int, int]],
    gt_path: Optional[str],
    annotations_dir: Optional[str],
    data_root: Optional[str],
    draw_gt: bool,
    verbose: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    orig_bgr = cv2.imread(str(image_path))
    if orig_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    orig_h, orig_w = orig_bgr.shape[:2]
    img_tensor, orig_target_sizes = preprocess_image_640(str(image_path), device)

    kept_per_level: List[Optional[torch.Tensor]] = []

    def pruner_hook(module, inputs, outputs):
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            _, kept_indices, _ = outputs
            kept_per_level.append(kept_indices)

    hook_pruner = None
    enc = getattr(model, "encoder", None)
    pruner = getattr(enc, "shared_token_pruner", None) if enc is not None else None
    if pruner is not None:
        hook_pruner = pruner.register_forward_hook(pruner_hook)

    if enc is not None and hasattr(enc, "set_epoch"):
        enc.set_epoch(0)

    with torch.no_grad():
        outputs = model(img_tensor)

    if hook_pruner:
        hook_pruner.remove()

    if not isinstance(outputs, dict):
        raise RuntimeError("Model output must be a dict with pred_logits and encoder_info.")

    encoder_info = outputs.get("encoder_info")
    if encoder_info is None:
        raise RuntimeError("Missing encoder_info. Use a CaS-DETR DEIM model with dual-scale encoder.")

    importance_scores_list = encoder_info.get("importance_scores_list") or []
    feat_shapes_list = encoder_info.get("feat_shapes_list") or []
    if len(importance_scores_list) < 2 or len(feat_shapes_list) < 2:
        raise RuntimeError(
            "Need at least two encoder levels in importance_scores_list; "
            "S5 uses index 1. Check hybrid_encoder use_encoder_idx."
        )

    s5_idx = 1
    logits_s5 = importance_scores_list[s5_idx]
    shape_s5 = feat_shapes_list[s5_idx]

    if logits_s5 is None:
        raise RuntimeError("importance_scores_list[1] is None for S5.")

    h5, w5 = shape_s5
    s5_scores = torch.sigmoid(logits_s5[0]).reshape(h5, w5).detach().cpu().numpy()

    if len(kept_per_level) >= 2:
        s5_mask = local_kept_indices_to_mask(kept_per_level[s5_idx], shape_s5)
    else:
        s5_mask = np.ones(shape_s5, dtype=np.float32)
        if verbose:
            if pruner is None:
                print("  Note: no shared_token_pruner; fused mask shows full image.")
            else:
                print(
                    "  Warning: expected two pruner forwards; S5 mask fallback to full."
                )

    s5_aligned = scores_hw_to_orig(s5_scores, orig_w, orig_h, normalize=True)
    s5_mask_o = mask_hw_to_orig(s5_mask, orig_w, orig_h)

    masked_image = orig_bgr.copy()
    masked_image[s5_mask_o < 0.5] = 0

    s5_u8 = (np.clip(s5_aligned, 0.0, 1.0) * 255).astype(np.uint8)
    s5_color = cv2.applyColorMap(s5_u8, cv2.COLORMAP_JET)
    s5_overlay = cv2.addWeighted(orig_bgr.copy(), 0.4, s5_color, 0.6, 0)

    results = postprocessor(outputs, orig_target_sizes)
    labels, boxes, scores = postprocess_to_drawable(results, conf_threshold)

    gt_annotations: List[Dict[str, Any]] = []
    if draw_gt:
        resolved = _resolve_gt_annotation_path(image_path, gt_path, annotations_dir, data_root)
        if resolved and Path(resolved).exists():
            ann_path = Path(resolved)
            try:
                gt_annotations = _load_yolo_annotations(ann_path, orig_h, orig_w, class_names)
                if verbose:
                    if gt_annotations:
                        print(f"  GT: {len(gt_annotations)} boxes from {ann_path.name}")
                    else:
                        print(f"  GT file empty: {ann_path}")
            except OSError as e:
                if verbose:
                    print(f"  GT load failed: {e}")
        elif verbose:
            print(f"  No GT label for {image_path.name}")

    keep_ratio = None
    dynamic_keep_ratio = encoder_info.get("dynamic_keep_ratio")
    if isinstance(dynamic_keep_ratio, torch.Tensor) and dynamic_keep_ratio.numel() > 0:
        keep_ratio = float(dynamic_keep_ratio.detach().flatten()[0].cpu().item())

    prune_ratio = None
    token_pruning_ratios = encoder_info.get("token_pruning_ratios") or []
    if len(token_pruning_ratios) > s5_idx:
        prune_ratio = float(token_pruning_ratios[s5_idx])
        if keep_ratio is None:
            keep_ratio = 1.0 - prune_ratio

    stat_parts: List[str] = []
    if keep_ratio is not None:
        stat_parts.append(f"keep={keep_ratio:.2f}")
    if prune_ratio is not None:
        stat_parts.append(f"prune={prune_ratio:.2f}")
    stat_text = "  ".join(stat_parts)

    masked_panel = add_panel_text(masked_image, stat_text)
    pred_gt_overlay = draw_prediction_gt_overlay(
        orig_bgr.copy(),
        labels,
        boxes,
        scores,
        class_names,
        colors,
        gt_annotations,
    )
    if stat_text:
        pred_gt_overlay = add_panel_text(pred_gt_overlay, stat_text)

    return orig_bgr.copy(), s5_overlay, masked_panel, pred_gt_overlay


def run_qualitative_4x4_grid(
    model: torch.nn.Module,
    postprocessor: torch.nn.Module,
    image_paths: Sequence[str],
    device: str,
    output_path: str,
    conf_threshold: float,
    class_names: Sequence[str],
    colors: Sequence[Tuple[int, int, int]],
    gt_path: Optional[str],
    annotations_dir: Optional[str],
    data_root: Optional[str],
    draw_gt: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 10))
    plt.subplots_adjust(wspace=0.01, hspace=0.05)

    col_titles = [
        "Original Image",
        r"$S_5$ Coarse Heatmap",
        r"$S_5$ Token Mask",
        "Prediction + GT",
    ]

    for row, image_path_str in enumerate(image_paths):
        p = Path(image_path_str)
        print(f"Processing row {row + 1}/4: {p}")
        o1, o2, o3, o4 = process_single_scenario(
            model,
            postprocessor,
            p,
            device,
            conf_threshold,
            class_names,
            colors,
            gt_path,
            annotations_dir,
            data_root,
            draw_gt,
            verbose=True,
        )
        imgs = [o1, o2, o3, o4]
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


def _get_default_images(image_dir: Optional[str], data_root: Optional[str], max_count: int = 4) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    candidates: List[Path] = []
    base = Path(image_dir or ".") / "image"
    if base.exists():
        candidates = sorted(
            [p for p in base.iterdir() if p.suffix.lower() in exts],
            key=lambda x: str(x),
        )
    if not candidates and data_root:
        dr = Path(data_root)
        for sub in ("images/train", "images/val", "image"):
            folder = dr / sub
            if folder.exists():
                candidates = sorted(
                    [p for p in folder.iterdir() if p.suffix.lower() in exts],
                    key=lambda x: str(x),
                )
                break
    return [str(p) for p in candidates[:max_count]]


def main():
    parser = argparse.ArgumentParser(description="CaS-DETR 4x4 aperture figure, S5 heatmap only")
    parser.add_argument("-c", "--config", type=str, required=True, help="CaS-DETR YAML config")
    parser.add_argument("-r", "--resume", type=str, required=True, help="Checkpoint .pth")
    parser.add_argument("--output", type=str, default="figure5_qualitative_cas_detr.pdf")
    parser.add_argument("--image", type=str, default=None, help="Single image; default 4 from ./image/")
    parser.add_argument("--image_dir", type=str, default=".", help="Base directory for ./image/")
    parser.add_argument("--data_root", type=str, default=None, help="Fallback image search like val script")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conf_threshold", type=float, default=0.3)
    parser.add_argument("--gt_path", type=str, default=None)
    parser.add_argument("--annotations_dir", type=str, default=None)
    parser.add_argument("--draw_gt", action="store_true")
    args = parser.parse_args()

    if args.image:
        paths = [args.image]
    else:
        paths = _get_default_images(args.image_dir, args.data_root, max_count=4)
        if len(paths) < 4:
            print(
                f"Error: need 4 images for 4x4 grid, found {len(paths)}. "
                "Use ./image/ with 4 images or --data_root with images/train or images/val."
            )
            return

    model, postprocessor, cfg = load_model_and_post(args.config, args.resume, args.device)
    yaml_cfg = cfg.yaml_cfg
    class_names = class_names_from_yaml(yaml_cfg)
    colors = colors_for_classes(len(class_names))

    run_qualitative_4x4_grid(
        model,
        postprocessor,
        paths[:4],
        args.device,
        args.output,
        args.conf_threshold,
        class_names,
        colors,
        args.gt_path,
        args.annotations_dir,
        args.data_root,
        args.draw_gt,
    )


if __name__ == "__main__":
    main()
