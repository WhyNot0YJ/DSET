"""
DETR / YOLO 共用的 KITTI 难度与 COCO 评估辅助。

- 难度分档：与 ``yolo_validator_utils.MultiScaleMetricsCalculator.categorize_by_kitti_difficulty``
  及 ``BaseYOLOTrainer._normalize_truncation`` 一致，便于与 YOLO 训练后 KITTI 评估对比。
- AP_easy / moderate / hard：DETR 侧应取 COCOeval.stats[1]（AP@IoU=0.50），与 YOLO 的 mAP@0.5 口径一致；
  stats[0] 为 AP@0.5:0.95，不宜与 YOLO 的 KITTI 难度表直接对比。
- AP_small/medium/large：COCO 默认 stats[3:6] 为 0.5:0.95 按面积均值；YOLO 训练后多尺度为 IoU=0.50 单阈值。
  本模块提供 ``coco_area_ap_at_iou50`` 供「与 YOLO 公平对比」。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 与 yolo_validator_utils.MultiScaleMetricsCalculator 一致
SMALL_AREA_THRESHOLD = 32 * 32
MEDIUM_AREA_THRESHOLD = 96 * 96
MIN_HEIGHT = 25


def normalize_truncation(tr: float, *, dair_categorical: bool = False) -> float:
    """
    将数据集里的截断字段映射为 KITTI 风格比例 [0, 1]，供难度阈值比较。

    DAIR-V2X：truncated_state ∈ {0,1,2} 为类别，非连续比例。
    UA-DETRAC 等：连续比例，裁剪到 [0, 1]。
    """
    tr = float(tr)
    if dair_categorical:
        k = int(round(tr))
        if k == 0:
            return 0.0
        if k == 1:
            return 0.20
        if k == 2:
            return 0.40
        return max(0.0, min(1.0, tr))
    return max(0.0, min(1.0, tr))


def _normalize_occlusion_level(occlusion: float) -> int:
    try:
        value = float(occlusion)
    except (TypeError, ValueError):
        return 3
    if value < 0:
        return 3
    if value >= 1.0:
        return int(round(value))
    if value <= 0.15:
        return 0
    if value <= 0.50:
        return 1
    if value <= 0.80:
        return 2
    return 3


def kitti_difficulty_label(
    height_px: float,
    occluded_raw: float,
    truncated_raw: float,
    *,
    dair_categorical_trunc: bool,
) -> str:
    """
    返回 'easy' | 'moderate' | 'hard' | 'ignore'（与 YOLO 侧 KITTI 分档一致）。
    """
    if height_px < MIN_HEIGHT:
        return "ignore"

    tr = normalize_truncation(truncated_raw, dair_categorical=dair_categorical_trunc)
    occ_level = _normalize_occlusion_level(float(occluded_raw))

    if height_px >= 40 and occ_level == 0 and tr <= 0.15:
        return "easy"
    if height_px >= 25 and occ_level <= 1 and tr <= 0.30:
        return "moderate"
    if height_px >= 25 and occ_level <= 2 and tr <= 0.50:
        return "hard"
    return "ignore"


def kitti_difficulty_from_coco_ann(
    ann: Dict[str, Any],
    *,
    dair_categorical_trunc: bool,
) -> str:
    """从 COCO 风格 ann dict（含 bbox xywh、occluded_state、truncated_state）计算难度。"""
    bbox = ann.get("bbox", [0, 0, 0, 0])
    h = float(ann.get("bbox_height", bbox[3] if len(bbox) > 3 else 0.0))
    occ = float(ann.get("occluded_state", 0))
    tr = float(ann.get("truncated_state", 0))
    return kitti_difficulty_label(h, occ, tr, dair_categorical_trunc=dair_categorical_trunc)


def coco_ap_at_iou50_all(coco_eval) -> float:
    """
    主 AP@IoU=0.50（COCOeval.stats[1]）。

    COCOeval 约定：当评估集中无正样本 GT 时，stats[1] 返回 -1。
    此处统一 clamp 为 0.0，与 YOLO 侧 ``_compute_ap_for_class`` 在
    ``n_pos == 0`` 时返回 0.0 的行为保持一致，避免 CSV 中出现 -1。
    """
    if coco_eval is None or not hasattr(coco_eval, "stats") or len(coco_eval.stats) < 2:
        return 0.0
    return max(0.0, float(coco_eval.stats[1]))


def coco_area_ap_at_iou50(coco_eval) -> Tuple[float, float, float]:
    """
    按 COCO 面积划分（small/medium/large）在 IoU=0.50 下的 AP 均值（与 YOLO 多尺度 mAP@0.5 口径对齐）。

    从 precision[T,R,K,A,M] 中 T=0 对应 IoU=0.5；A=1,2,3 为 small, medium, large。
    """
    if coco_eval is None or coco_eval.eval is None or "precision" not in coco_eval.eval:
        return 0.0, 0.0, 0.0
    precision = coco_eval.eval["precision"]
    # [T, R, K, A, M]
    if precision.size == 0:
        return 0.0, 0.0, 0.0
    iou_idx = 0
    max_det_idx = precision.shape[4] - 1
    out = []
    for area_idx in (1, 2, 3):
        p = precision[iou_idx, :, :, area_idx, max_det_idx]
        p = p[p > -1]
        out.append(float(np.mean(p)) if p.size > 0 else 0.0)
    return out[0], out[1], out[2]


# ── CSV 指标输出（DETR / YOLO 共用）──────────────────────────────────────

EVAL_CSV_FIELDS: List[str] = [
    "model",
    "dataset",
    "eval_split",
    "mAP_50",
    "mAP_75",
    "mAP_5095",
    "AP_easy",
    "AP_moderate",
    "AP_hard",
    "AP_small_50",
    "AP_medium_50",
    "AP_large_50",
    "AP_small_5095",
    "AP_medium_5095",
    "AP_large_5095",
]

BENCHMARK_CSV_FIELDS: List[str] = [
    "Params_M",
    "Active_Params_M",
    "GFLOPs",
    "FPS",
    "Latency_ms",
]


def write_eval_csv(
    path: Path,
    model: str,
    dataset: str,
    eval_split: str,
    metrics: Dict[str, float],
    *,
    class_names: Optional[List[str]] = None,
    append: bool = False,
    benchmark: Optional[Dict[str, float]] = None,
) -> None:
    """
    将一行评估指标写入 CSV。

    ``class_names`` 非空时，额外写入 ``AP50_<cls>``（每类 AP@0.5）与 ``AP5095_<cls>``（每类 AP@0.5:0.95）列；
    对应键名必须为 ``AP50_<cls>``、``AP5095_<cls>``（不再使用 ``mAP_<cls>`` 表示每类指标）。

    ``benchmark`` 非空时写入 GFLOPs / Params_M / FPS / Latency_ms 列
    （由 ``common.model_benchmark.benchmark_to_dict`` 生成）。
    """
    key_map = {
        "mAP_50": "mAP_0.5",
        "mAP_75": "mAP_0.75",
        "mAP_5095": "mAP_0.5_0.95",
        "AP_easy": "AP_easy",
        "AP_moderate": "AP_moderate",
        "AP_hard": "AP_hard",
        "AP_small_50": "AP_small_50",
        "AP_medium_50": "AP_medium_50",
        "AP_large_50": "AP_large_50",
        "AP_small_5095": "AP_small",
        "AP_medium_5095": "AP_medium",
        "AP_large_5095": "AP_large",
    }
    fieldnames = list(EVAL_CSV_FIELDS)
    if class_names:
        for name in class_names:
            fieldnames.append(f"AP50_{name}")
            fieldnames.append(f"AP5095_{name}")
    fieldnames.extend(BENCHMARK_CSV_FIELDS)

    row: Dict[str, str] = {
        "model": model,
        "dataset": dataset,
        "eval_split": eval_split,
    }
    for csv_col, metric_key in key_map.items():
        v = metrics.get(metric_key, 0.0)
        row[csv_col] = f"{float(v):.6f}" if isinstance(v, (int, float)) else str(v)

    if class_names:
        for name in class_names:
            v50 = metrics.get(f"AP50_{name}", 0.0)
            v5095 = metrics.get(f"AP5095_{name}", 0.0)
            row[f"AP50_{name}"] = f"{float(v50):.6f}" if isinstance(v50, (int, float)) else str(v50)
            row[f"AP5095_{name}"] = f"{float(v5095):.6f}" if isinstance(v5095, (int, float)) else str(v5095)

    if benchmark:
        for bk in BENCHMARK_CSV_FIELDS:
            bv = benchmark.get(bk, "")
            row[bk] = f"{float(bv):.2f}" if isinstance(bv, (int, float)) else str(bv)

    mode = "a" if append else "w"
    write_header = not append or not path.exists() or path.stat().st_size == 0

    with path.open(mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def dataset_display_name(config: Dict) -> str:
    """从 DETR 配置中推断对外展示的数据集名。"""
    dc = config.get("data", {})
    ds_cls = dc.get("dataset_class", "")
    root = str(dc.get("data_root", "")).lower()
    if ds_cls == "DAIRV2XDetection" or "dair-v2x" in root or "dairv2x" in root:
        return "DAIR-V2X"
    if ds_cls == "CocoFolderDetection" or "uadetrac" in root or "ua-detrac" in root:
        return "UA-DETRAC"
    return Path(dc.get("data_root", "unknown")).stem


def dataset_dir_name(config: Dict) -> str:
    """用于日志子目录的短名（小写无连字符）。"""
    display = dataset_display_name(config)
    return {"DAIR-V2X": "dairv2x", "UA-DETRAC": "uadetrac"}.get(display, display.lower())


def model_display_name(config: Dict, fallback_experiment_name: str = "") -> str:
    """从 DETR 配置或 experiment_name 推断模型短名。"""
    backbone = config.get("model", {}).get("backbone", "")
    backbone_short = backbone.replace("presnet", "r").replace("pres", "r") if "presnet" in backbone else backbone

    cfg_path = config.get("_config_path", "")
    stem = Path(cfg_path).stem if cfg_path else ""

    if stem:
        return stem
    if fallback_experiment_name:
        return fallback_experiment_name
    if backbone_short:
        return f"detr_{backbone_short}"
    return "unknown"

