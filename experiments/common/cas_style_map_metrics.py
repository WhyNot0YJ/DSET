"""
与 ``cas_detr/train.py`` 中 ``_compute_map_metrics`` / ``_compute_difficulty_aps`` 等价的纯函数实现，
供 RT-DETR 等独立训练入口复用，避免复制 CaS_DETR 训练器类。
"""

from __future__ import annotations

import copy
import os
import sys
from collections import Counter
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from common.det_eval_metrics import (
    coco_ap_at_iou50_all,
    coco_area_ap_at_iou50,
    extract_per_category_ap_from_coco_eval,
    kitti_difficulty_from_coco_ann,
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


def _compute_difficulty_aps(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    categories: List[Dict[str, Any]],
    img_h: int,
    img_w: int,
    image_id_to_size: Optional[Dict[int, Tuple[int, int]]],
    dair_categorical_trunc: bool,
) -> Dict[str, float]:
    easy_targets: List[Dict[str, Any]] = []
    moderate_targets: List[Dict[str, Any]] = []
    hard_targets: List[Dict[str, Any]] = []

    for target in targets:
        level = kitti_difficulty_from_coco_ann(
            target, dair_categorical_trunc=dair_categorical_trunc
        )
        t_easy = copy.deepcopy(target)
        if level != "easy":
            t_easy["iscrowd"] = 1
        easy_targets.append(t_easy)

        t_mod = copy.deepcopy(target)
        if level != "moderate":
            t_mod["iscrowd"] = 1
        moderate_targets.append(t_mod)

        t_hard = copy.deepcopy(target)
        if level != "hard":
            t_hard["iscrowd"] = 1
        hard_targets.append(t_hard)

    easy_eval = _run_coco_eval(
        predictions, easy_targets, categories, img_h, img_w, image_id_to_size=image_id_to_size
    )
    moderate_eval = _run_coco_eval(
        predictions, moderate_targets, categories, img_h, img_w, image_id_to_size=image_id_to_size
    )
    hard_eval = _run_coco_eval(
        predictions, hard_targets, categories, img_h, img_w, image_id_to_size=image_id_to_size
    )

    return {
        "AP_easy": coco_ap_at_iou50_all(easy_eval),
        "AP_moderate": coco_ap_at_iou50_all(moderate_eval),
        "AP_hard": coco_ap_at_iou50_all(hard_eval),
    }


def compute_cas_style_map_metrics(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    categories: List[Dict[str, Any]],
    *,
    image_id_to_size: Optional[Dict[int, Tuple[int, int]]] = None,
    img_h: int = 640,
    img_w: int = 640,
    print_per_category: bool = False,
    compute_difficulty: bool = False,
    dair_categorical_trunc: bool = False,
) -> Dict[str, Any]:
    """
    与 CaS_DETR ``_compute_map_metrics`` 返回字段一致（含 E/M/H、S/M/L @0.5 与 @0.5:0.95）。
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
            "AP_easy": 0.0,
            "AP_moderate": 0.0,
            "AP_hard": 0.0,
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

        difficulty_metrics = {
            "AP_easy": 0.0,
            "AP_moderate": 0.0,
            "AP_hard": 0.0,
        }
        if compute_difficulty:
            difficulty_metrics = _compute_difficulty_aps(
                predictions,
                targets,
                categories,
                img_h,
                img_w,
                image_id_to_size,
                dair_categorical_trunc,
            )

        gt_boxes_easy = gt_boxes_moderate = gt_boxes_hard = gt_boxes_ignore = 0
        if compute_difficulty:
            dc = Counter()
            for t in targets:
                lev = kitti_difficulty_from_coco_ann(
                    t, dair_categorical_trunc=dair_categorical_trunc
                )
                dc[lev] += 1
            gt_boxes_easy = int(dc.get("easy", 0))
            gt_boxes_moderate = int(dc.get("moderate", 0))
            gt_boxes_hard = int(dc.get("hard", 0))
            gt_boxes_ignore = int(dc.get("ignore", 0))

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
            "AP_easy": float(difficulty_metrics["AP_easy"]),
            "AP_moderate": float(difficulty_metrics["AP_moderate"]),
            "AP_hard": float(difficulty_metrics["AP_hard"]),
        }
        if compute_difficulty:
            result["gt_boxes_easy"] = gt_boxes_easy
            result["gt_boxes_moderate"] = gt_boxes_moderate
            result["gt_boxes_hard"] = gt_boxes_hard
            result["gt_boxes_ignore"] = gt_boxes_ignore

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
            "AP_easy": 0.0,
            "AP_moderate": 0.0,
            "AP_hard": 0.0,
        }
