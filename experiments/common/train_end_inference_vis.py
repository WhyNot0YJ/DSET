"""
训练结束后在验证集上导出「预测 | GT」对比图，使用统一的 BGR 可视化样式
（BGR、OpenCV、可选 DAIR camera JSON GT 第三列）。

在实验 yaml 中设置 ``vis_after_train: true`` 或 ``train_end_vis: { enable: true }`` 开启。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torchvision

# --- BGR 调色板：默认按 8 类顺序定义，更多类则循环使用 ---
DEFAULT_COLORS_BGR: List[Tuple[int, int, int]] = [
    (255, 0, 0),
    (0, 255, 0),
    (255, 128, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 128, 128),
]


def draw_boxes_bgr(
    image: np.ndarray,
    labels: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_names: Sequence[str],
    colors: Sequence[Tuple[int, int, int]],
) -> np.ndarray:
    """与 ``cas_detr.batch_inference.draw_boxes`` 行为一致。"""
    if not image.flags["C_CONTIGUOUS"]:
        image = np.ascontiguousarray(image)

    if len(labels) == 0:
        return image

    n_cls = len(class_names)
    n_col = len(colors)
    for label, box, score in zip(labels, boxes, scores):
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, min(x1, image.shape[1] - 1))
        y1 = max(0, min(y1, image.shape[0] - 1))
        x2 = max(0, min(x2, image.shape[1] - 1))
        y2 = max(0, min(y2, image.shape[0] - 1))

        li = int(label)
        if li < 0 or li >= n_cls:
            continue
        color = colors[li % n_col]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        class_name = class_names[li]
        label_text = f"{class_name}: {float(score):.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_y = max(text_h + 4, y1)
        cv2.rectangle(image, (x1, text_y - text_h - 4), (x1 + text_w, text_y), color, -1)
        cv2.putText(
            image,
            label_text,
            (x1, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    return image


def _unwrap_dataset(ds: Any) -> Any:
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    return ds


def _resolve_image_path(dataset: Any, target: Dict[str, Any], image_id: int) -> str:
    import os

    p = target.get("image_path")
    if isinstance(p, str) and p:
        return p
    ds = _unwrap_dataset(dataset)
    info = ds.coco.loadImgs(image_id)[0]
    fn = info["file_name"]
    root = getattr(ds, "img_folder", None) or getattr(ds, "root", "") or ""
    return os.path.join(root, fn)


def _gt_boxes_xyxy_pixel_from_target(target: Dict[str, Any]) -> np.ndarray:
    boxes = target["boxes"]
    if hasattr(boxes, "as_tensor"):
        t = boxes.as_tensor()
    else:
        t = boxes
    t = t.detach().cpu().float()
    orig = target["orig_size"].detach().cpu().float()
    W, H = orig[0].item(), orig[1].item()
    xyxy = torchvision.ops.box_convert(t, "cxcywh", "xyxy")
    xyxy[:, 0] *= W
    xyxy[:, 2] *= W
    xyxy[:, 1] *= H
    xyxy[:, 3] *= H
    return xyxy.numpy()


def _class_names(yaml_cfg: Dict[str, Any], dataset: Any) -> List[str]:
    names = yaml_cfg.get("vis_class_names")
    if names:
        return list(names)
    nested = yaml_cfg.get("train_end_vis")
    if isinstance(nested, dict) and nested.get("class_names"):
        return list(nested["class_names"])
    ds = _unwrap_dataset(dataset)
    if hasattr(ds, "categories"):
        cats = sorted(ds.categories, key=lambda c: c["id"])
        return [c["name"] for c in cats]
    n = int(yaml_cfg.get("num_classes", 80))
    return [f"class_{i}" for i in range(n)]


def _colors(n: int) -> List[Tuple[int, int, int]]:
    out = []
    for i in range(n):
        out.append(DEFAULT_COLORS_BGR[i % len(DEFAULT_COLORS_BGR)])
    return out


def _get_gt_annotation_path(image_path: Path) -> Optional[Path]:
    """Infer DAIR-style camera annotation path from image path."""
    try:
        if image_path.parent.name == "image":
            data_root = image_path.parent.parent
            for cand in (
                data_root / "annotations" / "camera" / f"{image_path.stem}.json",
                data_root / "infrastructure-side" / "annotations" / "camera" / f"{image_path.stem}.json",
            ):
                if cand.exists():
                    return cand
        json_path = image_path.with_suffix(".json")
        if json_path.exists():
            return json_path
    except Exception:
        return None
    return None


def _load_dair_gt_boxes(json_path: Path, class_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load DAIR-V2X camera JSON boxes into draw-ready arrays."""
    with json_path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    name_to_id = {name: i for i, name in enumerate(class_names)}
    merge_map = {"Barrowlist": "Cyclist"}
    ignore_classes = {
        "PedestrianIgnore",
        "CarIgnore",
        "OtherIgnore",
        "Unknown_movable",
        "Unknown_unmovable",
    }

    labels: List[int] = []
    boxes: List[List[float]] = []
    scores: List[float] = []
    for ann in data:
        if "type" not in ann or "2d_box" not in ann:
            continue
        cat_name = merge_map.get(ann["type"], ann["type"])
        if cat_name in ignore_classes or cat_name not in name_to_id:
            continue
        box_2d = ann["2d_box"]
        x1 = float(box_2d.get("xmin", 0.0))
        y1 = float(box_2d.get("ymin", 0.0))
        x2 = float(box_2d.get("xmax", 0.0))
        y2 = float(box_2d.get("ymax", 0.0))
        if x2 <= x1 or y2 <= y1:
            continue
        labels.append(name_to_id[cat_name])
        boxes.append([x1, y1, x2, y2])
        scores.append(1.0)

    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(boxes, dtype=np.float32),
        np.asarray(scores, dtype=np.float32),
    )


def _vis_params(yaml_cfg: Dict[str, Any]) -> Tuple[bool, int, float]:
    nested = yaml_cfg.get("train_end_vis")
    if isinstance(nested, dict):
        enable = bool(nested.get("enable", False))
        num = int(nested.get("num_images", nested.get("num", 8)))
        thr = float(nested.get("score_thr", 0.35))
        if enable:
            return True, num, thr
    enable = bool(yaml_cfg.get("vis_after_train", False))
    num = int(yaml_cfg.get("vis_num_images", 8))
    thr = float(yaml_cfg.get("vis_score_thr", 0.35))
    return enable, num, thr


def _try_dair_json_panel(
    image_path: Path,
    class_names: Sequence[str],
    bgr: np.ndarray,
) -> Optional[np.ndarray]:
    jp = _get_gt_annotation_path(Path(image_path))
    if jp is None:
        return None
    gt_labels, gt_boxes, gt_scores = _load_dair_gt_boxes(jp, list(class_names))
    if len(gt_labels) == 0:
        return None
    panel = draw_boxes_bgr(
        bgr.copy(),
        gt_labels,
        gt_boxes,
        gt_scores,
        list(class_names),
        _colors(len(class_names)),
    )
    cv2.putText(
        panel,
        "DAIR JSON GT",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 128, 0),
        2,
    )
    return panel


def run_train_end_inference_vis(
    module: torch.nn.Module,
    postprocessor: torch.nn.Module,
    val_dataloader: Any,
    device: torch.device,
    output_dir: Path,
    yaml_cfg: Dict[str, Any],
) -> None:
    enable, num_images, score_thr = _vis_params(yaml_cfg)
    if not enable or num_images <= 0:
        return

    out_dir = Path(output_dir) / "vis_train_end"
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = _class_names(yaml_cfg, val_dataloader.dataset)
    colors = _colors(len(class_names))

    m = module.module if hasattr(module, "module") else module
    m.eval()
    postprocessor.eval()

    saved = 0
    for samples, targets in val_dataloader:
        samples = samples.to(device)
        orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)

        with torch.no_grad():
            outputs = m(samples)
        results = postprocessor(outputs, orig_sizes)

        bs = samples.shape[0]
        for i in range(bs):
            if saved >= num_images:
                break
            res = results[i]
            labels = res["labels"].detach().cpu().numpy()
            boxes = res["boxes"].detach().cpu().numpy()
            scores = res["scores"].detach().cpu().numpy()
            mask = scores >= score_thr
            labels, boxes, scores = labels[mask], boxes[mask], scores[mask]

            tgt = targets[i]
            image_id = int(tgt["image_id"].item())
            path_str = _resolve_image_path(val_dataloader.dataset, tgt, image_id)
            bgr = cv2.imread(path_str)
            if bgr is None:
                continue

            pred_vis = draw_boxes_bgr(
                bgr.copy(), labels, boxes, scores, class_names, colors
            )
            cv2.putText(
                pred_vis,
                "Prediction",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )

            gt_l = tgt["labels"].detach().cpu().numpy()
            gt_xyxy = _gt_boxes_xyxy_pixel_from_target(tgt)
            gt_scores = np.ones(len(gt_l), dtype=np.float32)
            gt_vis = draw_boxes_bgr(
                bgr.copy(), gt_l, gt_xyxy, gt_scores, class_names, colors
            )
            cv2.putText(
                gt_vis,
                "Ground Truth",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            panels = [pred_vis, gt_vis]
            dair = _try_dair_json_panel(Path(path_str), class_names, bgr)
            if dair is not None:
                panels.append(dair)

            final = cv2.hconcat(panels)
            cv2.imwrite(str(out_dir / f"sample_{saved:03d}_id{image_id}.jpg"), final)
            saved += 1

        if saved >= num_images:
            break

    print(f"[train_end_vis] wrote {saved} image(s) under {out_dir}")


def cfg_dict_for_vis(cfg: Any) -> Dict[str, Any]:
    """Merge ``yaml_cfg`` with top-level runtime keys so CLI 覆盖的 vis_* 生效。"""
    d: Dict[str, Any] = {}
    if hasattr(cfg, "yaml_cfg") and cfg.yaml_cfg:
        d.update(cfg.yaml_cfg)
    for k in (
        "vis_after_train",
        "vis_num_images",
        "vis_score_thr",
        "vis_class_names",
        "num_classes",
        "train_end_vis",
    ):
        if hasattr(cfg, k):
            v = getattr(cfg, k, None)
            if v is not None:
                d[k] = v
    return d


def maybe_run_train_end_vis(
    dist_is_main: bool,
    module: torch.nn.Module,
    postprocessor: torch.nn.Module,
    val_dataloader: Any,
    device: torch.device,
    output_dir: Any,
    yaml_cfg: Dict[str, Any],
) -> None:
    if not dist_is_main:
        return
    enable, _, _ = _vis_params(yaml_cfg)
    if not enable:
        return
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("[train_end_vis] skipped: opencv-python not installed")
        return
    if output_dir is None:
        return
    try:
        run_train_end_inference_vis(
            module,
            postprocessor,
            val_dataloader,
            device,
            Path(output_dir),
            yaml_cfg,
        )
    except Exception as e:
        print(f"[train_end_vis] failed: {e}")
