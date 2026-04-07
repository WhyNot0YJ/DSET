"""
DETR / YOLO 共用的 KITTI 难度与 COCO 评估辅助。

- AP_small/medium/large @0.5：``coco_area_ap_at_iou50``；@0.5:0.95 可用 stats[3:6]。
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_LOG = logging.getLogger(__name__)

# pycocotools（与 DETR / YOLO COCOeval 路径共用）
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:  # pragma: no cover
    COCO = None  # type: ignore[misc, assignment]
    COCOeval = None  # type: ignore[misc, assignment]

# 供 YOLO/DETR 日志区分「未安装 pycocotools」与「无 GT」
PYCOCOTOOLS_AVAILABLE: bool = COCO is not None

# 与 yolo_validator_utils.MultiScaleMetricsCalculator 一致
SMALL_AREA_THRESHOLD = 32 * 32
MEDIUM_AREA_THRESHOLD = 96 * 96
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


def extract_per_category_ap_from_coco_eval(
    coco_eval: Any,
    categories: List[Dict[str, Any]],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    从 ``COCOeval.accumulate()`` 后的 ``precision[T,R,K,A,M]`` 提取每类 AP@0.5 与 AP@0.5:0.95。

    Extract per-category AP@0.5 and AP@0.5:0.95 from ``COCOeval`` for shared DETR / YOLO reporting.
    """
    per_cat_50 = {str(cat["name"]): 0.0 for cat in categories}
    per_cat_5095 = {str(cat["name"]): 0.0 for cat in categories}
    if coco_eval is None or not hasattr(coco_eval, "eval") or "precision" not in coco_eval.eval:
        return per_cat_50, per_cat_5095

    try:
        precision = coco_eval.eval["precision"]
        area_index = 0
        max_det_index = len(coco_eval.params.maxDets) - 1
        cat_id_to_index = {cat_id: idx for idx, cat_id in enumerate(coco_eval.params.catIds)}

        for cat in categories:
            cat_id = cat["id"]
            cat_name = str(cat["name"])
            if cat_id not in cat_id_to_index:
                continue

            cat_index = cat_id_to_index[cat_id]
            p50 = precision[0, :, cat_index, area_index, max_det_index]
            v50 = p50[p50 > -1]
            per_cat_50[cat_name] = float(np.mean(v50)) if v50.size > 0 else 0.0
            p5095 = precision[:, :, cat_index, area_index, max_det_index]
            v5095 = p5095[p5095 > -1]
            per_cat_5095[cat_name] = float(np.mean(v5095)) if v5095.size > 0 else 0.0
    except Exception:
        pass

    return per_cat_50, per_cat_5095


def run_coco_bbox_eval(
    coco_gt: Dict[str, Any],
    predictions: List[Dict[str, Any]],
) -> Optional[Any]:
    """
    对 COCO 格式的 ``gt`` + 检测 ``predictions`` 跑一次 ``COCOeval``（bbox），抑制 stdout。

    Returns:
        ``coco_eval`` 或失败时 ``None``（例如未安装 pycocotools、无有效 GT）。
    """
    if COCO is None or COCOeval is None:
        return None
    if not coco_gt.get("annotations"):
        return None

    from io import StringIO
    import sys

    try:
        # pycocotools COCO.loadRes 会执行 res.dataset['info'] = copy.deepcopy(self.dataset['info'])，缺省 KeyError
        coco_in = dict(coco_gt)
        coco_in.setdefault(
            "info",
            {"description": "CaS_DETR_eval", "version": "1.0", "year": 2024},
        )

        coco_gt_obj = COCO()
        coco_gt_obj.dataset = coco_in
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            coco_gt_obj.createIndex()
        finally:
            sys.stdout = old_stdout

        sys.stdout = StringIO()
        try:
            coco_dt = coco_gt_obj.loadRes(predictions)
        finally:
            sys.stdout = old_stdout

        coco_eval = COCOeval(coco_gt_obj, coco_dt, "bbox")
        sys.stdout = StringIO()
        try:
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        finally:
            sys.stdout = old_stdout

        return coco_eval
    except Exception as exc:
        _LOG.warning("COCOeval 异常（将跳过该次评估）: %s", exc, exc_info=True)
        return None


def coco_area_bucket_name(area: float) -> str:
    """按 COCO 面积阈值返回 ``small`` / ``medium`` / ``large``。"""
    area = float(area)
    if area < SMALL_AREA_THRESHOLD:
        return "small"
    if area < MEDIUM_AREA_THRESHOLD:
        return "medium"
    return "large"


def coco_area_bucket_counts_from_xywh_annotations(
    annotations: List[Dict[str, Any]],
) -> Dict[str, int]:
    """
    统计 COCO ``bbox=[x,y,w,h]`` 标注列表在 small / medium / large 上的数量。
    """
    counts = {"small": 0, "medium": 0, "large": 0}
    for ann in annotations:
        bbox = ann.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4:
            continue
        w = float(bbox[2])
        h = float(bbox[3])
        if w <= 0 or h <= 0:
            continue
        counts[coco_area_bucket_name(w * h)] += 1
    return counts


def format_area_bucket_counts(prefix: str, counts: Dict[str, int]) -> str:
    """将面积桶计数格式化成日志短句。"""
    total = int(counts.get("small", 0) + counts.get("medium", 0) + counts.get("large", 0))
    return (
        f"{prefix} total={total}  "
        f"small={int(counts.get('small', 0))}  "
        f"medium={int(counts.get('medium', 0))}  "
        f"large={int(counts.get('large', 0))}"
    )


# ── CSV 指标输出（DETR / YOLO 共用）──────────────────────────────────────

EVAL_CSV_FIELDS: List[str] = [
    "model",
    "dataset",
    "eval_split",
    "mAP_50",
    "mAP_75",
    "mAP_5095",
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

