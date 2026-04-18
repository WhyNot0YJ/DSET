#!/usr/bin/env python3
"""
4x4 paper-style qualitative figure for CaS-DETR checkpoints (YAMLConfig + .pth).

Rows: four input images. Columns: original image, S5 coarse heatmap,
S5 token mask + prediction, and baseline prediction. The baseline column can
reuse the current model or use a separate checkpoint. Heatmap and mask use the
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
DEFAULT_COLORS_BGR = _train_end_vis.DEFAULT_COLORS_BGR
DEFAULT_COLORS_BGR = _train_end_vis.DEFAULT_COLORS_BGR
DEFAULT_COLORS_BGR = _train_end_vis.DEFAULT_COLORS_BGR
DEFAULT_COLORS_BGR = _train_end_vis.DEFAULT_COLORS_BGR
draw_boxes_bgr_default = _train_end_vis.draw_boxes_bgr

INPUT_SIZE = 640

# Row i uses image_paths[i]. For each row: pick k CaS boxes of this class with smallest y1, then draw them on baseline column.
_BASELINE_FN_FROM_CAS_SPECS: Sequence[Tuple[str, int]] = (
    ("Van", 1),
    ("Motorcyclist", 1),
    ("car", 1),
    ("car", 2),
)


def _norm_class_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def pick_topmost_boxes_by_class(
    labels: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_names: Sequence[str],
    target_class: str,
    k: int,
) -> np.ndarray:
    if k <= 0 or len(labels) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    want = _norm_class_name(target_class)
    cand: List[Tuple[float, float, np.ndarray]] = []
    for label, box, score in zip(labels, boxes, scores):
        li = int(label)
        if li < 0 or li >= len(class_names):
            continue
        if _norm_class_name(class_names[li]) != want:
            continue
        b = np.asarray(box, dtype=np.float32).reshape(4)
        cand.append((float(b[1]), float(score), b.copy()))
    if not cand:
        return np.zeros((0, 4), dtype=np.float32)
    cand.sort(key=lambda t: (t[0], -t[1]))
    picked = [t[2] for t in cand[:k]]
    return np.stack(picked, axis=0)


def draw_baseline_failure_highlight(bgr: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
    """Thick white outline plus red box on BGR image for baseline missed-region emphasis."""
    if len(boxes_xyxy) == 0:
        return bgr
    out = bgr.copy()
    if not out.flags["C_CONTIGUOUS"]:
        out = np.ascontiguousarray(out)
    h, w = out.shape[:2]
    short = min(h, w)
    th = max(4, int(round(short / 140.0)))
    white_th = th + max(3, th // 2)
    red_bgr = (0, 0, 255)
    white_bgr = (255, 255, 255)
    font_scale = max(0.65, short / 800.0)
    txt_th = max(2, int(round(short / 500.0)))
    for box in boxes_xyxy:
        x1, y1, x2, y2 = [int(round(float(t))) for t in box]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        cv2.rectangle(out, (x1, y1), (x2, y2), white_bgr, white_th, lineType=cv2.LINE_AA)
        cv2.rectangle(out, (x1, y1), (x2, y2), red_bgr, th, lineType=cv2.LINE_AA)
        tag = "FN"
        (tw, tht), _bl = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_th)
        pad = max(3, txt_th)
        tx1, ty1 = x1, max(0, y1 - tht - 2 * pad)
        ty2 = ty1 + tht + 2 * pad
        tx2 = min(w - 1, tx1 + tw + 2 * pad)
        cv2.rectangle(out, (tx1, ty1), (tx2, ty2), red_bgr, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            out,
            tag,
            (tx1 + pad, ty2 - pad),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            txt_th,
            lineType=cv2.LINE_AA,
        )
    return out


def draw_boxes_bgr_hd(
    image: np.ndarray,
    labels: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_names: Sequence[str],
    colors: Sequence[Tuple[int, int, int]],
) -> np.ndarray:
    """High-resolution box drawing with line width and font size that scale
    with image short side so boxes remain visible after downscaling to PDF.
    """
    if not image.flags["C_CONTIGUOUS"]:
        image = np.ascontiguousarray(image)
    if len(labels) == 0:
        return image

    h, w = image.shape[:2]
    short_side = min(h, w)
    # ~4 px at 1080p, ~2 px at 540p, minimum 2.
    line_thickness = max(2, int(round(short_side / 270.0)))
    font_scale = max(0.5, short_side / 1080.0 * 0.9)
    text_thickness = max(1, int(round(short_side / 720.0)))

    n_cls = len(class_names)
    n_col = len(colors)
    for label, box, score in zip(labels, boxes, scores):
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        li = int(label)
        if li < 0 or li >= n_cls:
            continue
        color = colors[li % n_col]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness, lineType=cv2.LINE_AA)

        class_name = class_names[li]
        label_text = f"{class_name}: {float(score):.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )
        pad = max(2, int(round(short_side / 540.0)))
        text_y = max(text_h + 2 * pad, y1)
        cv2.rectangle(
            image,
            (x1, text_y - text_h - 2 * pad),
            (x1 + text_w + 2 * pad, text_y),
            color,
            -1,
        )
        cv2.putText(
            image,
            label_text,
            (x1 + pad, text_y - pad),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            text_thickness,
            lineType=cv2.LINE_AA,
        )
    return image


# Use HD variant for grid figure.
draw_boxes_bgr = draw_boxes_bgr_hd


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


def run_model_inference(
    model: torch.nn.Module,
    image_path: Path,
    device: str,
    eval_epoch: int,
    capture_pruner: bool = False,
):
    orig_bgr = cv2.imread(str(image_path))
    if orig_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    img_tensor, orig_target_sizes = preprocess_image_640(str(image_path), device)
    kept_per_level: List[Optional[torch.Tensor]] = []

    def pruner_hook(module, inputs, outputs):
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            _, kept_indices, _ = outputs
            kept_per_level.append(kept_indices)

    hook_pruner = None
    enc = getattr(model, "encoder", None)
    pruner = getattr(enc, "shared_token_pruner", None) if enc is not None else None
    if capture_pruner and pruner is not None:
        hook_pruner = pruner.register_forward_hook(pruner_hook)

    if enc is not None and hasattr(enc, "set_epoch"):
        enc.set_epoch(int(eval_epoch))

    with torch.no_grad():
        outputs = model(img_tensor)

    if hook_pruner is not None:
        hook_pruner.remove()

    return orig_bgr, orig_target_sizes, outputs, kept_per_level, pruner


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


def add_panel_badge(ax, text: str) -> None:
    """Draw a resolution-independent badge in axes coordinates."""
    if not text:
        return
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=6,
        fontfamily="serif",
        bbox={
            "facecolor": "black",
            "edgecolor": "none",
            "boxstyle": "round,pad=0.15",
            "alpha": 0.8,
        },
    )


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, np.ndarray, np.ndarray, np.ndarray]:
    orig_bgr, orig_target_sizes, outputs, kept_per_level, pruner = run_model_inference(
        model,
        image_path,
        device,
        eval_epoch,
        capture_pruner=True,
    )
    orig_h, orig_w = orig_bgr.shape[:2]

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

    stat_text = f"keep={keep_ratio:.2f}" if keep_ratio is not None else ""

    pred_overlay = draw_prediction_overlay(
        masked_image.copy(),
        labels,
        boxes,
        scores,
        class_names,
        colors,
    )

    return orig_bgr.copy(), s5_overlay, pred_overlay, stat_text, labels, boxes, scores


def build_baseline_overlay(
    model: torch.nn.Module,
    postprocessor: torch.nn.Module,
    image_path: Path,
    device: str,
    eval_epoch: int,
    conf_threshold: float,
    class_names: Sequence[str],
    colors: Sequence[Tuple[int, int, int]],
) -> np.ndarray:
    orig_bgr, orig_target_sizes, outputs, _, _ = run_model_inference(
        model,
        image_path,
        device,
        eval_epoch,
        capture_pruner=False,
    )
    results = postprocessor(outputs, orig_target_sizes)
    labels, boxes, scores = postprocess_to_drawable(results, conf_threshold)
    return draw_prediction_overlay(
        orig_bgr.copy(),
        labels,
        boxes,
        scores,
        class_names,
        colors,
    )


def run_qualitative_4x4_grid(
    image_paths: Sequence[str],
    row_model_bundle: Sequence[Tuple[torch.nn.Module, torch.nn.Module, Sequence[str], Sequence[Tuple[int, int, int]], int]],
    baseline_row_model_bundle: Sequence[Tuple[torch.nn.Module, torch.nn.Module, Sequence[str], Sequence[Tuple[int, int, int]], int]],
    device: str,
    output_path: str,
    conf_threshold: float,
    save_dpi: int,
    fig_width: float,
    fig_height: float,
    jpeg_quality: int = 85,
    png_compress_level: int = 6,
    pdf_slim_fonts: bool = False,
    mark_baseline_failure_from_cas: bool = True,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["pdf.compression"] = 9
    out_suffix = Path(output_path).suffix.lower()
    if out_suffix == ".pdf" and pdf_slim_fonts:
        # Fewer font bytes in the PDF; panel images still use PNG Flate inside the file.
        matplotlib.rcParams["pdf.use14corefonts"] = True
        matplotlib.rcParams["pdf.fonttype"] = 42
    else:
        matplotlib.rcParams["pdf.use14corefonts"] = False
        matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType so text stays editable
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]
    # interpolation "none" triggers unsampled PDF embedding of the full numpy array, ignoring savefig dpi
    # and inflating files to tens of MB. "nearest" resamples to the figure dpi and keeps hard box edges.
    matplotlib.rcParams["image.interpolation"] = "nearest"
    matplotlib.rcParams["image.resample"] = True

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(fig_width, fig_height))
    plt.subplots_adjust(wspace=0.01, hspace=0.05)
    if mark_baseline_failure_from_cas:
        print(
            "Baseline FN highlights: ON — red or white rings and FN tags use CaS box positions; "
            "see row warnings if a class has no detection above --conf_threshold."
        )

    col_titles = [
        "Original Image",
        "Importance Map $S_5$",
        r"$S_5$ Token Mask + Prediction",
        "Baseline Prediction",
    ]

    for row, image_path_str in enumerate(image_paths):
        p = Path(image_path_str)
        print(f"Processing row {row + 1}/4: {p}")
        model, postprocessor, class_names, colors, eval_epoch = row_model_bundle[row]
        baseline_model, baseline_postprocessor, baseline_class_names, baseline_colors, baseline_eval_epoch = baseline_row_model_bundle[row]
        o1, o2, o3, stat_text, cas_labels, cas_boxes, cas_scores = process_single_scenario(
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
        o4 = build_baseline_overlay(
            baseline_model,
            baseline_postprocessor,
            p,
            device,
            baseline_eval_epoch,
            conf_threshold,
            baseline_class_names,
            baseline_colors,
        )
        if mark_baseline_failure_from_cas and row < len(_BASELINE_FN_FROM_CAS_SPECS):
            tcls, tk = _BASELINE_FN_FROM_CAS_SPECS[row]
            fn_boxes = pick_topmost_boxes_by_class(
                cas_labels,
                cas_boxes,
                cas_scores,
                class_names,
                tcls,
                tk,
            )
            if len(fn_boxes) == 0:
                print(
                    f"  Warning: row {row} no CaS prediction for class '{tcls}'; "
                    "baseline column left without failure highlight."
                )
            else:
                if len(fn_boxes) < tk:
                    print(
                        f"  Warning: row {row} wanted {tk} '{tcls}' box(es); "
                        f"CaS only has {len(fn_boxes)} above conf threshold."
                    )
                o4 = draw_baseline_failure_highlight(o4, fn_boxes)
                print(
                    f"  Baseline column: drew {len(fn_boxes)} failure highlight(s) from CaS '{tcls}' boxes."
                )
        imgs = [o1, o2, o3, o4]
        for col, (ax, img) in enumerate(zip(axes[row], imgs)):
            ax.imshow(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                interpolation="nearest",
                resample=True,
            )
            if row == 0:
                ax.set_title(col_titles[col], fontweight="bold", fontfamily="serif")
            if stat_text and col == 2:
                add_panel_badge(ax, stat_text)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: Dict[str, Any] = {
        "dpi": int(save_dpi),
        "bbox_inches": "tight",
        "pad_inches": 0.02,
    }
    suffix = out_path.suffix.lower()
    if suffix in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        save_kwargs["format"] = suffix.lstrip(".")
    jq = max(1, min(95, int(jpeg_quality)))
    pcl = max(0, min(9, int(png_compress_level)))
    if suffix in {".jpg", ".jpeg"}:
        save_kwargs["pil_kwargs"] = {"quality": jq, "optimize": True}
    elif suffix == ".png":
        save_kwargs["pil_kwargs"] = {"compress_level": pcl}
    # PDF keeps vector text; panel photos are rasterized at save_dpi, so lower dpi or inches for smaller PDF.
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
    parser = argparse.ArgumentParser(description="CaS-DETR 4x4 aperture figure with baseline prediction")
    parser.add_argument("-c", "--config", type=str, required=True, help="CaS-DETR YAML config")
    parser.add_argument("-r", "--resume", type=str, required=True, help="Checkpoint .pth")
    parser.add_argument("--config_b", type=str, default=None, help="Optional second YAML config for rows after split_index")
    parser.add_argument("--resume_b", type=str, default=None, help="Optional second checkpoint for rows after split_index")
    parser.add_argument("--baseline_config", type=str, default=None, help="Optional baseline YAML config")
    parser.add_argument("--baseline_resume", type=str, default=None, help="Optional baseline checkpoint for all rows")
    parser.add_argument("--baseline_config_b", type=str, default=None, help="Optional second baseline config for rows after split_index")
    parser.add_argument("--baseline_resume_b", type=str, default=None, help="Optional second baseline checkpoint for rows after split_index")
    parser.add_argument("--split_index", type=int, default=2, help="Rows [0:split_index] use model A, remaining use model B")
    parser.add_argument(
        "--output",
        type=str,
        default="figure5_qualitative_cas_detr.pdf",
        help="Path to save the figure. Default is PDF; use --compact for a smaller PDF.",
    )
    parser.add_argument("--images", type=str, nargs="+", default=None, help="Exactly 4 image paths in row order")
    parser.add_argument("--image", type=str, default=None, help="Single image path (backward compatibility)")
    parser.add_argument("--image_dir", type=str, default=".", help="Base directory for ./image/")
    parser.add_argument("--data_root", type=str, default=None, help="Fallback image search like val script")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_epoch", type=int, default=5, help="Encoder epoch for CAIP/CASS warmup logic")
    parser.add_argument("--eval_epoch_b", type=int, default=None, help="Optional eval epoch for second model")
    parser.add_argument("--baseline_eval_epoch", type=int, default=None, help="Optional eval epoch for baseline model")
    parser.add_argument("--baseline_eval_epoch_b", type=int, default=None, help="Optional eval epoch for second baseline model")
    parser.add_argument("--conf_threshold", type=float, default=0.3)
    parser.add_argument(
        "--dpi",
        type=int,
        default=240,
        help="Save DPI. Lower values shrink raster bytes in PDF or PNG.",
    )
    parser.add_argument("--fig_width", type=float, default=14.0, help="Figure width in inches")
    parser.add_argument("--fig_height", type=float, default=8.8, help="Figure height in inches")
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        metavar="Q",
        help="JPEG quality 1 to 95 when --output ends with .jpg or .jpeg. Lower is smaller.",
    )
    parser.add_argument(
        "--png-compress-level",
        type=int,
        default=6,
        metavar="L",
        help="PNG zlib level 0 to 9 when --output ends with .png. 9 gives smallest PNG.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Smaller PDF or PNG: dpi 240, fig 14×8.8 in, PNG zlib 9. Overrides --dpi and figure size. For .jpg only, caps jpeg quality at 80.",
    )
    parser.add_argument(
        "--pdf-slim-fonts",
        action="store_true",
        help="PDF only: use standard PDF core fonts to shrink file size; title or math may look slightly different.",
    )
    parser.add_argument(
        "--mark-baseline-failure-from-cas",
        dest="mark_baseline_failure_from_cas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Draw red FN markers on the baseline column from CaS box positions in column 3. "
            "Default on; use --no-mark-baseline-failure-from-cas to disable. "
            "Row or class list: _BASELINE_FN_FROM_CAS_SPECS."
        ),
    )
    args = parser.parse_args()

    if args.compact:
        args.dpi = 240
        args.fig_width = 14.0
        args.fig_height = 8.8
        args.png_compress_level = max(int(args.png_compress_level), 9)
        out_sfx = Path(args.output).suffix.lower()
        if out_sfx in {".jpg", ".jpeg"}:
            args.jpeg_quality = min(int(args.jpeg_quality), 80)
        print(
            f"Compact preset: dpi=240, fig 14×8.8 in, png_compress_level={args.png_compress_level}"
            + (f", jpeg_quality={args.jpeg_quality}" if out_sfx in {'.jpg', '.jpeg'} else "")
            + ("; PDF output" if out_sfx == ".pdf" else "")
            + f" → {args.output}"
        )

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

    baseline_model_a = model_a
    baseline_postprocessor_a = postprocessor_a
    baseline_class_names_a = class_names_a
    baseline_colors_a = colors_a
    baseline_eval_epoch_a = eval_epoch_a if args.baseline_eval_epoch is None else int(args.baseline_eval_epoch)
    if args.baseline_resume:
        baseline_config_a = args.baseline_config or args.config
        baseline_model_a, baseline_postprocessor_a, baseline_cfg_a = load_model_and_post(
            baseline_config_a,
            args.baseline_resume,
            args.device,
        )
        baseline_class_names_a = class_names_from_yaml(baseline_cfg_a.yaml_cfg)
        baseline_colors_a = colors_for_classes(len(baseline_class_names_a))

    baseline_model_b = baseline_model_a if args.baseline_resume else model_b
    baseline_postprocessor_b = baseline_postprocessor_a if args.baseline_resume else postprocessor_b
    baseline_class_names_b = baseline_class_names_a if args.baseline_resume else class_names_b
    baseline_colors_b = baseline_colors_a if args.baseline_resume else colors_b
    default_baseline_eval_epoch_b = eval_epoch_b if not args.baseline_resume else baseline_eval_epoch_a
    baseline_eval_epoch_b = (
        default_baseline_eval_epoch_b
        if args.baseline_eval_epoch_b is None
        else int(args.baseline_eval_epoch_b)
    )
    if args.baseline_resume_b:
        baseline_config_b = args.baseline_config_b or args.baseline_config or args.config_b or args.config
        baseline_model_b, baseline_postprocessor_b, baseline_cfg_b = load_model_and_post(
            baseline_config_b,
            args.baseline_resume_b,
            args.device,
        )
        baseline_class_names_b = class_names_from_yaml(baseline_cfg_b.yaml_cfg)
        baseline_colors_b = colors_for_classes(len(baseline_class_names_b))

    split_index = max(0, min(4, int(args.split_index)))
    row_model_bundle: List[Tuple[torch.nn.Module, torch.nn.Module, Sequence[str], Sequence[Tuple[int, int, int]], int]] = []
    baseline_row_model_bundle: List[Tuple[torch.nn.Module, torch.nn.Module, Sequence[str], Sequence[Tuple[int, int, int]], int]] = []
    for i in range(4):
        if i < split_index:
            row_model_bundle.append((model_a, postprocessor_a, class_names_a, colors_a, eval_epoch_a))
            baseline_row_model_bundle.append(
                (
                    baseline_model_a,
                    baseline_postprocessor_a,
                    baseline_class_names_a,
                    baseline_colors_a,
                    baseline_eval_epoch_a,
                )
            )
        else:
            row_model_bundle.append((model_b, postprocessor_b, class_names_b, colors_b, eval_epoch_b))
            baseline_row_model_bundle.append(
                (
                    baseline_model_b,
                    baseline_postprocessor_b,
                    baseline_class_names_b,
                    baseline_colors_b,
                    baseline_eval_epoch_b,
                )
            )

    run_qualitative_4x4_grid(
        paths[:4],
        row_model_bundle,
        baseline_row_model_bundle,
        args.device,
        args.output,
        args.conf_threshold,
        args.dpi,
        args.fig_width,
        args.fig_height,
        args.jpeg_quality,
        args.png_compress_level,
        args.pdf_slim_fonts,
        args.mark_baseline_failure_from_cas,
    )


if __name__ == "__main__":
    main()
