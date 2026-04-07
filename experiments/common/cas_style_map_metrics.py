"""
Provide shared COCO-style mAP and KITTI difficulty metrics for DETR-family trainers.
"""

from __future__ import annotations

import os
import sys
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from common.det_eval_metrics import (
    coco_ap_at_iou50_all,
    coco_area_ap_at_iou50,
    extract_per_category_ap_from_coco_eval,
)


def _run_coco_eval(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    categories: List[Dict[str, Any]],
    img_h: int,
    img_w: int,
    image_id_to_size: Optional[Dict[int, Tuple[int, int]]] = None,
    print_summary: bool = False,
) -> Optional[COCOeval]:
    if len(targets) == 0:
        return None

    coco_gt: Dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": categories,
        "info": {"description": "COCO eval", "version": "1.0", "year": 2024},
    }

    image_ids = {int(target["image_id"]) for target in targets}
    for img_id in image_ids:
        if image_id_to_size and img_id in image_id_to_size:
            w, h = image_id_to_size[img_id]
        else:
            w, h = img_w, img_h
        coco_gt["images"].append({"id": img_id, "width": w, "height": h})

    for i, target in enumerate(targets):
        ann = dict(target)
        ann["id"] = i + 1
        coco_gt["annotations"].append(ann)

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        coco_gt_obj = COCO()
        coco_gt_obj.dataset = coco_gt
        coco_gt_obj.createIndex()
        coco_dt = coco_gt_obj.loadRes(predictions)
        coco_eval = COCOeval(coco_gt_obj, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
    finally:
        sys.stdout = old_stdout

    if print_summary:
        coco_eval.summarize()
    else:
        sys.stdout = StringIO()
        try:
            coco_eval.summarize()
        finally:
            sys.stdout = old_stdout

    return coco_eval


def compute_cas_style_map_metrics(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    categories: List[Dict[str, Any]],
    *,
    image_id_to_size: Optional[Dict[int, Tuple[int, int]]] = None,
    img_h: int = 640,
    img_w: int = 640,
    print_per_category: bool = False,
) -> Dict[str, Any]:
    """
    计算共享的 COCO-style mAP 指标（含全局与 COCO 面积档 small/medium/large 的 @0.5 与 @0.5:0.95）。
    """
    if len(predictions) == 0 or len(targets) == 0:
        return {
            "mAP_0.5": 0.0,
            "mAP_0.75": 0.0,
            "mAP_0.5_0.95": 0.0,
            "mAP_s": 0.0,
            "mAP_m": 0.0,
            "mAP_l": 0.0,
            "AP_small": 0.0,
            "AP_medium": 0.0,
            "AP_large": 0.0,
            "AP_small_50": 0.0,
            "AP_medium_50": 0.0,
            "AP_large_50": 0.0,
            "AR_small": 0.0,
            "AR_medium": 0.0,
            "AR_large": 0.0,
            "AR_100": 0.0,
        }

    per_cat_50: Dict[str, float] = {}
    per_cat_5095: Dict[str, float] = {}

    try:
        coco_eval = _run_coco_eval(
            predictions, targets, categories, img_h, img_w, image_id_to_size=image_id_to_size,
            print_summary=print_per_category or bool(os.getenv("CAS_DEBUG_COCO_SUMMARY")),
        )
        if coco_eval is None:
            raise RuntimeError("COCOeval failed")

        s50, m50, l50 = coco_area_ap_at_iou50(coco_eval)

        if print_per_category:
            per_cat_50, per_cat_5095 = extract_per_category_ap_from_coco_eval(
                coco_eval, categories
            )

        _s = coco_eval.stats
        _n = len(_s)
        result: Dict[str, Any] = {
            "mAP_0.5": float(_s[1]),
            "mAP_0.75": float(_s[2]),
            "mAP_0.5_0.95": float(_s[0]),
            "mAP_s": float(_s[3]) if _n > 3 else 0.0,
            "mAP_m": float(_s[4]) if _n > 4 else 0.0,
            "mAP_l": float(_s[5]) if _n > 5 else 0.0,
            "AP_small": float(_s[3]) if _n > 3 else 0.0,
            "AP_medium": float(_s[4]) if _n > 4 else 0.0,
            "AP_large": float(_s[5]) if _n > 5 else 0.0,
            "AP_small_50": float(s50),
            "AP_medium_50": float(m50),
            "AP_large_50": float(l50),
            "AR_small": float(_s[8]) if _n > 8 else 0.0,
            "AR_medium": float(_s[9]) if _n > 9 else 0.0,
            "AR_large": float(_s[10]) if _n > 10 else 0.0,
            "AR_100": float(_s[7]) if _n > 7 else 0.0,
        }

        for cat_name in per_cat_5095.keys():
            result[f"AP50_{cat_name}"] = per_cat_50.get(cat_name, 0.0)
            result[f"AP5095_{cat_name}"] = per_cat_5095[cat_name]

        return result
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("cas_style_map_metrics 失败: %s", exc)
        return {
            "mAP_0.5": 0.0,
            "mAP_0.75": 0.0,
            "mAP_0.5_0.95": 0.0,
            "mAP_s": 0.0,
            "mAP_m": 0.0,
            "mAP_l": 0.0,
            "AP_small": 0.0,
            "AP_medium": 0.0,
            "AP_large": 0.0,
            "AP_small_50": 0.0,
            "AP_medium_50": 0.0,
            "AP_large_50": 0.0,
            "AR_small": 0.0,
            "AR_medium": 0.0,
            "AR_large": 0.0,
            "AR_100": 0.0,
        }
