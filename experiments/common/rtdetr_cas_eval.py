"""
RT-DETR（``rtdetrv2_pytorch``）训练结束后，输出与 CaS_DETR 一致的 val/test 指标（mAP、E/M/H、S/M/L）与 CSV。

使用前将 ``experiments`` 目录加入 ``sys.path``（``train_adapter.py`` 已处理）。
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from common.cas_style_map_metrics import compute_cas_style_map_metrics
from common.detr_eval_utils import (
    log_detr_eval_summary,
    run_detr_benchmark,
    write_detr_eval_csv,
)

logger = logging.getLogger(__name__)


def cxcywh_to_xywh_orig(
    boxes: torch.Tensor,
    img_w: int,
    img_h: int,
    orig_w: float,
    orig_h: float,
    letterbox_pad: Optional[torch.Tensor] = None,
    letterbox_scale: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Convert normalized ``cxcywh`` GT boxes back to original-image ``xywh``."""
    if boxes.numel() == 0:
        return np.zeros((0, 4), dtype=np.float64)

    device = boxes.device
    ow = float(orig_w)
    oh = float(orig_h)
    normalized = bool((boxes.max() <= 1.01).item())
    if normalized:
        cx = boxes[:, 0] * img_w
        cy = boxes[:, 1] * img_h
        bw = boxes[:, 2] * img_w
        bh = boxes[:, 3] * img_h
    else:
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    if letterbox_pad is not None and letterbox_scale is not None:
        pad = letterbox_pad.to(device=device, dtype=cx.dtype)
        scale = letterbox_scale.to(device=device, dtype=cx.dtype).reshape(-1)[0]
        pad_left, pad_top = pad[0], pad[1]
        x1 = ((cx - bw / 2) - pad_left) / scale
        y1 = ((cy - bh / 2) - pad_top) / scale
        x2 = ((cx + bw / 2) - pad_left) / scale
        y2 = ((cy + bh / 2) - pad_top) / scale
        x1 = x1.clamp(0, ow)
        y1 = y1.clamp(0, oh)
        x2 = x2.clamp(0, ow)
        y2 = y2.clamp(0, oh)
        w = (x2 - x1).clamp(min=1.0)
        h = (y2 - y1).clamp(min=1.0)
        return torch.stack([x1, y1, w, h], dim=1).cpu().numpy()

    sx = img_w / ow
    sy = img_h / oh
    x1 = ((cx - bw / 2) / sx).clamp(0, ow)
    y1 = ((cy - bh / 2) / sy).clamp(0, oh)
    w = (bw / sx).clamp(min=1.0, max=ow)
    h = (bh / sy).clamp(min=1.0, max=oh)
    return torch.stack([x1, y1, w, h], dim=1).cpu().numpy()


def _gt_xyxy_to_xywh_orig(
    boxes_xyxy: torch.Tensor,
    current_w: int,
    current_h: int,
    orig_w: float,
    orig_h: float,
) -> np.ndarray:
    """验证集未做 ``ConvertBoxes(cxcywh)`` 时，GT 为 resize 后图像上的 xyxy 像素坐标。"""
    if boxes_xyxy.numel() == 0:
        return np.zeros((0, 4), dtype=np.float64)
    ow, oh = float(orig_w), float(orig_h)
    cw, ch = float(current_w), float(current_h)
    sx, sy = ow / cw, oh / ch
    x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
    x1o = (x1 * sx).clamp(0, ow)
    y1o = (y1 * sy).clamp(0, oh)
    x2o = (x2 * sx).clamp(0, ow)
    y2o = (y2 * sy).clamp(0, oh)
    w = (x2o - x1o).clamp(min=1.0)
    h = (y2o - y1o).clamp(min=1.0)
    return torch.stack([x1o, y1o, w, h], dim=-1).cpu().numpy()


def _gt_to_xywh_numpy(
    true_boxes: torch.Tensor,
    current_w: int,
    current_h: int,
    orig_w: float,
    orig_h: float,
    letterbox_pad: Optional[torch.Tensor],
    letterbox_scale: Optional[torch.Tensor],
) -> np.ndarray:
    if true_boxes.numel() == 0:
        return np.zeros((0, 4), dtype=np.float64)
    # 归一化 cxcywh（与训练 ``ConvertBoxes(cxcywh, normalize=True)`` 一致）
    if float(true_boxes.max()) <= 1.01 + 1e-5:
        return cxcywh_to_xywh_orig(
            true_boxes, current_w, current_h, orig_w, orig_h,
            letterbox_pad, letterbox_scale,
        )
    return _gt_xyxy_to_xywh_orig(true_boxes, current_w, current_h, orig_w, orig_h)


def _pred_cat_id_from_label(lab: int, dataset) -> int:
    """模型输出 label（0..C-1）→ COCO ``category_id``。"""
    l2c = getattr(dataset, "label2category", None)
    if isinstance(l2c, dict) and lab in l2c:
        return int(l2c[lab])
    return int(lab) + 1


def _gt_cat_id_from_tensor(lab: int, dataset) -> int:
    """
    将 GT ``labels`` 张量中的类别下标 0..C-1 映射为 COCO ``category_id``。

    ``rtdetrv2_pytorch`` 的 ``CocoDetection`` 在 ``remap_mscoco_category=False`` 时仍用
    ``category2label`` 把标注写成连续下标，与预测侧 ``_pred_cat_id_from_label`` 须一致。
    """
    return _pred_cat_id_from_label(int(lab), dataset)


def collect_rtdetr_predictions_and_targets(
    model: nn.Module,
    postprocessor: nn.Module,
    *,
    dataloader,
    device: torch.device,
    dataset,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[int, Tuple[int, int]],
    int,
    int,
]:
    """
    与 CaS_DETR ``_run_ema_eval_on_dataloader`` 一致：``image_id`` 为 loader 顺序下从 0 开始的编号。
    """
    all_predictions: List[Dict[str, Any]] = []
    all_targets: List[Dict[str, Any]] = []
    image_id_to_size: Dict[int, Tuple[int, int]] = {}
    current_h, current_w = 640, 640
    sample_offset = 0

    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            batch_size = int(images.shape[0])
            _, _, current_h, current_w = images.shape
            images = images.to(device, non_blocking=True)
            targets = [
                {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                for t in targets
            ]

            for i, _target in enumerate(targets):
                image_id = sample_offset + i
                osz = _target.get("orig_size")
                if osz is not None and isinstance(osz, torch.Tensor) and osz.numel() >= 2:
                    # RT-DETR: ``orig_size`` = [W, H]
                    ow, oh = int(osz[0].item()), int(osz[1].item())
                    image_id_to_size[image_id] = (ow, oh)
                else:
                    image_id_to_size[image_id] = (current_w, current_h)

            outputs = model(images)
            orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessor(outputs, orig_sizes)

            for i in range(batch_size):
                image_id = sample_offset + i
                tgt = targets[i]
                orig_w, orig_h = tgt["orig_size"].tolist()
                lb_pad = tgt.get("letterbox_pad")
                lb_scale = tgt.get("letterbox_scale")

                res = results[i]
                if isinstance(res, dict) and "labels" in res:
                    labs = res["labels"]
                    boxes_xyxy = res["boxes"]
                    scores = res["scores"]
                    for j in range(len(labs)):
                        lab = int(labs[j].item())
                        if lab < 0:
                            continue
                        cat_id = _pred_cat_id_from_label(lab, dataset)
                        x1, y1, x2, y2 = boxes_xyxy[j].tolist()
                        w = max(1.0, x2 - x1)
                        h = max(1.0, y2 - y1)
                        all_predictions.append(
                            {
                                "image_id": image_id,
                                "category_id": cat_id,
                                "bbox": [float(x1), float(y1), float(w), float(h)],
                                "score": float(scores[j].item()),
                            }
                        )

                if "labels" not in tgt or "boxes" not in tgt:
                    continue
                true_labels = tgt["labels"]
                true_boxes = tgt["boxes"]
                if len(true_labels) == 0:
                    continue

                gt_boxes_np = _gt_to_xywh_numpy(
                    true_boxes, current_w, current_h, orig_w, orig_h, lb_pad, lb_scale
                )
                gt_cls_np = true_labels.cpu().numpy()
                gt_area = gt_boxes_np[:, 2] * gt_boxes_np[:, 3]
                gt_h = gt_boxes_np[:, 3]

                has_iscrowd = "iscrowd" in tgt
                has_occ = "occluded_state" in tgt
                has_trunc = "truncated_state" in tgt
                iscrowd_np = tgt["iscrowd"].cpu().numpy() if has_iscrowd else None
                occ_np = tgt["occluded_state"].cpu().numpy() if has_occ else None
                trunc_np = tgt["truncated_state"].cpu().numpy() if has_trunc else None

                for j in range(len(gt_cls_np)):
                    lab_idx = int(gt_cls_np[j])
                    cat_id = _gt_cat_id_from_tensor(lab_idx, dataset)
                    ann = {
                        "image_id": image_id,
                        "category_id": cat_id,
                        "bbox": gt_boxes_np[j].tolist(),
                        "area": float(gt_area[j]),
                        "bbox_height": float(gt_h[j]),
                    }
                    if iscrowd_np is not None:
                        ann["iscrowd"] = int(iscrowd_np[j])
                    if occ_np is not None:
                        ann["occluded_state"] = float(occ_np[j])
                    if trunc_np is not None:
                        ann["truncated_state"] = float(trunc_np[j])
                    all_targets.append(ann)

            sample_offset += batch_size

    return all_predictions, all_targets, image_id_to_size, current_h, current_w


def categories_from_dataset(dataset) -> List[Dict[str, Any]]:
    return list(dataset.categories)


def infer_dair_categorical_trunc(dataset) -> bool:
    name = getattr(dataset, "__class__", type(dataset)).__name__
    if name == "DAIRV2XDetection":
        return True
    root = str(getattr(dataset, "data_root", "")).lower()
    return "dair" in root and "v2x" in root


def config_stub_for_csv(yaml_cfg: Dict[str, Any], base_config_path: Path) -> Dict[str, Any]:
    ds = yaml_cfg.get("train_dataloader", {}).get("dataset", {})
    return {
        "data": {
            "data_root": str(ds.get("data_root", "")),
            "dataset_class": str(ds.get("type", "")),
        },
        "model": {},
        "_config_path": str(base_config_path),
    }


def load_best_checkpoint_for_eval(solver, ck_path: Path, device: torch.device) -> None:
    """将 ``best.pth`` / ``last.pth`` 载入 EMA 或 ``model``（与验证时一致）。"""
    from src.misc import dist_utils

    state = torch.load(ck_path, map_location=device, weights_only=False)
    ck_resolved = str(Path(ck_path).resolve())
    if solver.ema is not None and state.get("ema"):
        solver.ema.load_state_dict(state["ema"])
        logger.info("CaS 评估已加载 checkpoint: %s（ema）", ck_resolved)
    elif "model" in state:
        m = dist_utils.de_parallel(solver.model)
        m.load_state_dict(state["model"])
        logger.info("CaS 评估已加载 checkpoint: %s（model）", ck_resolved)
        if solver.ema is not None:
            solver.ema.module.load_state_dict(m.state_dict())
            logger.info(
                "checkpoint 无 ema 键，已将 model 权重同步到 EMA；推理与训练期 val 一致使用 ema.module"
            )
    else:
        raise RuntimeError(f"无法从 {ck_path} 解析 model/ema 权重")


def build_test_dataloader_if_available(cfg) -> Optional[Any]:
    """
    若 ``<data_root>/annotations/instances_test.json`` 存在，则按 ``val_dataloader`` 复制一份
    ``split: test`` 的 DataLoader。
    """
    from src.misc import dist_utils

    y0 = cfg.yaml_cfg
    ds_cfg = y0.get("val_dataloader", {}).get("dataset", {})
    root = Path(str(ds_cfg.get("data_root", "")))
    ann = root / "annotations" / "instances_test.json"
    if not ann.is_file():
        return None

    yc = copy.deepcopy(y0)
    test_cfg = copy.deepcopy(yc["val_dataloader"])
    test_cfg["dataset"] = copy.deepcopy(test_cfg["dataset"])
    test_cfg["dataset"]["split"] = "test"
    yc["test_dataloader"] = test_cfg

    orig = cfg.yaml_cfg
    cfg.yaml_cfg = yc
    try:
        loader = cfg.build_dataloader("test_dataloader")
        return dist_utils.warp_loader(loader, shuffle=yc["test_dataloader"].get("shuffle", False))
    finally:
        cfg.yaml_cfg = orig


def run_rtdetr_cas_style_eval_after_fit(
    solver,
    cfg,
    base_config_path: Path,
    *,
    experiment_name: str = "rtdetr",
    checkpoint_path: Optional[Path] = None,
) -> None:
    """
    在已有权重上跑 CaS 风格 val，可选 test，并写 ``eval_metrics.csv``。

    训练结束或 ``--test-only --cas-eval`` 时调用。未指定 ``checkpoint_path`` 时从
    ``output_dir`` 下取 ``best.pth``，否则 ``last.pth``。
    """
    from src.misc import dist_utils

    if not dist_utils.is_main_process():
        return

    out = Path(cfg.output_dir)
    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        ck = Path(checkpoint_path)
    else:
        ck = out / "best.pth" if (out / "best.pth").is_file() else out / "last.pth"
    if not ck.is_file():
        logger.warning(
            "未找到权重：请设置 checkpoint_path 或在 output_dir 放置 best.pth / last.pth，跳过 CaS 风格评估"
        )
        return

    device = solver.device
    load_best_checkpoint_for_eval(solver, ck, device)

    module = solver.ema.module if solver.ema else solver.model
    if hasattr(module, "eval"):
        module.eval()

    yaml_cfg = cfg.yaml_cfg
    cfg_stub = config_stub_for_csv(yaml_cfg, base_config_path)

    bench_dict = None
    try:
        bench_dict = run_detr_benchmark(
            module, cfg_stub, experiment_name, device, logger,
        )
    except Exception as exc:
        logger.warning("Model benchmark 失败（不影响评估）: %s", exc)

    val_loader = solver.val_dataloader
    val_ds = val_loader.dataset
    dair_cat = infer_dair_categorical_trunc(val_ds)
    cats = categories_from_dataset(val_ds)

    preds, gts, id2sz, ih, iw = collect_rtdetr_predictions_and_targets(
        module,
        solver.postprocessor,
        dataloader=val_loader,
        device=device,
        dataset=val_ds,
    )

    class_names = [str(c["name"]) for c in cats]

    if len(gts) == 0:
        logger.warning("val 上无 GT，跳过 CaS 风格评估")
        return

    metrics = compute_cas_style_map_metrics(
        preds,
        gts,
        cats,
        image_id_to_size=id2sz,
        img_h=ih,
        img_w=iw,
        print_per_category=True,
        compute_difficulty=True,
        dair_categorical_trunc=dair_cat,
    )

    log_detr_eval_summary(logger, "val", metrics, bench_dict)
    csv_path = write_detr_eval_csv(
        out,
        cfg_stub,
        experiment_name,
        "val",
        metrics,
        class_names,
        bench_dict,
        aggregate_at_parent=False,
    )
    logger.info("✓ best_model [val] 评估完成 → %s", csv_path)

    test_loader = build_test_dataloader_if_available(cfg)
    if test_loader is None:
        logger.info("无 instances_test.json 或未构建 test loader，跳过 test 评估。")
        return

    test_ds = test_loader.dataset
    preds_t, gts_t, id2sz_t, ih_t, iw_t = collect_rtdetr_predictions_and_targets(
        module,
        solver.postprocessor,
        dataloader=test_loader,
        device=device,
        dataset=test_ds,
    )
    if len(gts_t) == 0:
        logger.info("test 划分无 GT，跳过 test 评估。")
        return

    metrics_t = compute_cas_style_map_metrics(
        preds_t,
        gts_t,
        cats,
        image_id_to_size=id2sz_t,
        img_h=ih_t,
        img_w=iw_t,
        print_per_category=True,
        compute_difficulty=True,
        dair_categorical_trunc=dair_cat,
    )
    log_detr_eval_summary(logger, "test", metrics_t, bench_dict)
    csv_path_t = write_detr_eval_csv(
        out,
        cfg_stub,
        experiment_name,
        "test",
        metrics_t,
        class_names,
        bench_dict,
        aggregate_at_parent=False,
    )
    logger.info("✓ best_model [test] 评估完成 → %s", csv_path_t)
