#!/usr/bin/env python3
"""
4x4 paper-style qualitative figure for CaS-DETR checkpoints (YAMLConfig + .pth).

Rows: four input images. Columns: original image, S5 coarse heatmap, S5 token mask,
and prediction on the pruned image. S4 is not visualized. Heatmap and mask use the
last HybridEncoder stage; with a single stage in the config, that stage is used.
Input to the network is fixed 640x640, same as tools/inference/torch_inf.py stretch resize.
Supports one or two checkpoints: rows [0..split_index-1] use model A, remaining rows use model B.
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


def draw_prediction_overlay(
    bgr: np.ndarray,
    labels: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_names: Sequence[str],
    colors: Sequence[Tuple[int, int, int]],
) -> np.ndarray:
    overlay = bgr.copy()
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
    eval_epoch: int,
    conf_threshold: float,
    class_names: Sequence[str],
    colors: Sequence[Tuple[int, int, int]],
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
        enc.set_epoch(int(eval_epoch))

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
    if not importance_scores_list or not feat_shapes_list:
        raise RuntimeError("importance_scores_list or feat_shapes_list is empty.")
    if len(importance_scores_list) != len(feat_shapes_list):
        raise RuntimeError(
            "feat_shapes_list length does not match importance_scores_list."
        )

    # Last entry is the last HybridEncoder stage, usually the coarsest map.
    # With num_encoder_layers==1 and use_encoder_idx: [2], there is only one stage.
    s_idx = len(importance_scores_list) - 1
    logits_s5 = importance_scores_list[s_idx]
    shape_s5 = feat_shapes_list[s_idx]

    if logits_s5 is None:
        raise RuntimeError(f"importance_scores_list[{s_idx}] is None.")

    h5, w5 = shape_s5
    s5_scores = torch.sigmoid(logits_s5[0]).reshape(h5, w5).detach().cpu().numpy()

    if len(kept_per_level) > s_idx:
        s5_mask = local_kept_indices_to_mask(kept_per_level[s_idx], shape_s5)
    else:
        s5_mask = np.ones(shape_s5, dtype=np.float32)
        if verbose:
            if pruner is None:
                print("  Note: no shared_token_pruner; fused mask shows full image.")
            else:
                print(
                    "  Warning: pruner hook count below encoder stage index; mask fallback to full."
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

    keep_ratio = None
    dynamic_keep_ratio = encoder_info.get("dynamic_keep_ratio")
    if isinstance(dynamic_keep_ratio, torch.Tensor) and dynamic_keep_ratio.numel() > 0:
        keep_ratio = float(dynamic_keep_ratio.detach().flatten()[0].cpu().item())

    prune_ratio = None
    token_pruning_ratios = encoder_info.get("token_pruning_ratios") or []
    if len(token_pruning_ratios) > s_idx:
        prune_ratio = float(token_pruning_ratios[s_idx])
        if keep_ratio is None:
            keep_ratio = 1.0 - prune_ratio

    stat_parts: List[str] = []
    if keep_ratio is not None:
        stat_parts.append(f"keep={keep_ratio:.2f}")
    if prune_ratio is not None:
        stat_parts.append(f"prune={prune_ratio:.2f}")
    stat_text = "  ".join(stat_parts)

    masked_panel = add_panel_text(masked_image, stat_text)
    pred_overlay = draw_prediction_overlay(
        masked_image.copy(),
        labels,
        boxes,
        scores,
        class_names,
        colors,
    )
    if stat_text:
        pred_overlay = add_panel_text(pred_overlay, stat_text)

    return orig_bgr.copy(), s5_overlay, masked_panel, pred_overlay


def run_qualitative_4x4_grid(
    image_paths: Sequence[str],
    row_model_bundle: Sequence[Tuple[torch.nn.Module, torch.nn.Module, Sequence[str], Sequence[Tuple[int, int, int]], int]],
    device: str,
    output_path: str,
    conf_threshold: float,
    save_dpi: int,
    fig_width: float,
    fig_height: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["pdf.compression"] = 9
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(fig_width, fig_height))
    plt.subplots_adjust(wspace=0.01, hspace=0.05)

    col_titles = [
        "Original Image",
        r"$S_5$ Coarse Heatmap",
        r"$S_5$ Token Mask",
        r"$S_5$ Token Mask + Prediction",
    ]

    for row, image_path_str in enumerate(image_paths):
        p = Path(image_path_str)
        print(f"Processing row {row + 1}/4: {p}")
        model, postprocessor, class_names, colors, eval_epoch = row_model_bundle[row]
        o1, o2, o3, o4 = process_single_scenario(
            model,
            postprocessor,
            p,
            device,
            eval_epoch,
            conf_threshold,
            class_names,
            colors,
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
    save_kwargs: Dict[str, Any] = {
        "dpi": int(save_dpi),
        "bbox_inches": "tight",
    }
    suffix = out_path.suffix.lower()
    if suffix in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        save_kwargs["format"] = suffix.lstrip(".")
    plt.savefig(str(out_path), **save_kwargs)
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


def _validate_images(images: Sequence[str]) -> List[str]:
    if len(images) != 4:
        raise ValueError(f"Need exactly 4 images for 4x4 grid, got {len(images)}")
    checked: List[str] = []
    for p in images:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        checked.append(str(path))
    return checked


def main():
    parser = argparse.ArgumentParser(description="CaS-DETR 4x4 aperture figure, S5 heatmap only")
    parser.add_argument("-c", "--config", type=str, required=True, help="CaS-DETR YAML config")
    parser.add_argument("-r", "--resume", type=str, required=True, help="Checkpoint .pth")
    parser.add_argument("--config_b", type=str, default=None, help="Optional second YAML config for rows after split_index")
    parser.add_argument("--resume_b", type=str, default=None, help="Optional second checkpoint for rows after split_index")
    parser.add_argument("--split_index", type=int, default=2, help="Rows [0:split_index] use model A, remaining use model B")
    parser.add_argument("--output", type=str, default="figure5_qualitative_cas_detr.pdf")
    parser.add_argument("--images", type=str, nargs="+", default=None, help="Exactly 4 image paths in row order")
    parser.add_argument("--image", type=str, default=None, help="Single image path (backward compatibility)")
    parser.add_argument("--image_dir", type=str, default=".", help="Base directory for ./image/")
    parser.add_argument("--data_root", type=str, default=None, help="Fallback image search like val script")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_epoch", type=int, default=5, help="Encoder epoch for CAIP/CASS warmup logic")
    parser.add_argument("--eval_epoch_b", type=int, default=None, help="Optional eval epoch for second model")
    parser.add_argument("--conf_threshold", type=float, default=0.3)
    parser.add_argument("--dpi", type=int, default=200, help="Save DPI, lower is smaller file size")
    parser.add_argument("--fig_width", type=float, default=12.0, help="Figure width in inches")
    parser.add_argument("--fig_height", type=float, default=7.5, help="Figure height in inches")
    args = parser.parse_args()

    if args.images:
        paths = _validate_images(args.images)
    elif args.image:
        paths = _validate_images([args.image] * 4)
    else:
        paths = _get_default_images(args.image_dir, args.data_root, max_count=4)
        if len(paths) != 4:
            print(
                f"Error: need 4 images for 4x4 grid, found {len(paths)}. "
                "Use ./image/ with 4 images or --data_root with images/train or images/val."
            )
            return

    model_a, postprocessor_a, cfg_a = load_model_and_post(args.config, args.resume, args.device)
    class_names_a = class_names_from_yaml(cfg_a.yaml_cfg)
    colors_a = colors_for_classes(len(class_names_a))
    eval_epoch_a = int(args.eval_epoch)

    model_b, postprocessor_b, class_names_b, colors_b = model_a, postprocessor_a, class_names_a, colors_a
    eval_epoch_b = eval_epoch_a if args.eval_epoch_b is None else int(args.eval_epoch_b)
    if args.resume_b:
        config_b = args.config_b or args.config
        model_b, postprocessor_b, cfg_b = load_model_and_post(config_b, args.resume_b, args.device)
        class_names_b = class_names_from_yaml(cfg_b.yaml_cfg)
        colors_b = colors_for_classes(len(class_names_b))

    split_index = max(0, min(4, int(args.split_index)))
    row_model_bundle: List[Tuple[torch.nn.Module, torch.nn.Module, Sequence[str], Sequence[Tuple[int, int, int]], int]] = []
    for i in range(4):
        if i < split_index:
            row_model_bundle.append((model_a, postprocessor_a, class_names_a, colors_a, eval_epoch_a))
        else:
            row_model_bundle.append((model_b, postprocessor_b, class_names_b, colors_b, eval_epoch_b))

    run_qualitative_4x4_grid(
        paths[:4],
        row_model_bundle,
        args.device,
        args.output,
        args.conf_threshold,
        args.dpi,
        args.fig_width,
        args.fig_height,
    )


if __name__ == "__main__":
    main()
