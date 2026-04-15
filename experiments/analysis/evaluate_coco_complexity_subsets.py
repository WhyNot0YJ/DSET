#!/usr/bin/env python3
"""
按图像真实目标数量切分 COCO 测试集，并比较不同模型在各复杂度子集上的指标与保留率。

脚本功能概述：
1. 使用 CaS-DETR 的 ``YAMLConfig``、``DetSolver`` 与 ``config + checkpoint`` 在测试集上推理，
   将输出转为 COCO 检测结果后再评估。
2. 按每张图像 GT 数量排序后，按图像数严格三等分切成 Simple / Medium / Complex。
3. 对每个模型、每个子集分别运行 COCOeval，提取 mAP^{50:95}、AP50、AP_S^{50:95}、AP_S^{50}。
4. 统计对应子集上的平均 keep ratio，并以 Markdown 表格打印结果。

默认路径假设：
- GT 为 DAIR-V2X COCO 测试标注。
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import json
import math
import os
import sys
import warnings as py_warnings
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
# ``.../<repo>/experiments/analysis`` -> 仓库根 ``.../<repo>``
REPO_ROOT = SCRIPT_DIR.parents[1]
CAS_ROOT = REPO_ROOT / "experiments" / "CaS-DETR"
DEFAULT_GT_PATH = Path("/root/autodl-fs/datasets/DAIR-V2X/annotations/instances_test.json")


def resolve_repo_path(path: Path) -> Path:
    """相对路径按仓库根拼接后再 resolve，避免 ``chdir(CAS_ROOT)`` 后重复多出一段 ``experiments/CaS-DETR``。"""
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    return (REPO_ROOT / p).resolve()


def import_coco_api() -> Tuple[Any, Any]:
    """延迟导入 pycocotools，保证 `--help` 在缺依赖时也能正常工作。"""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:  # pragma: no cover - 依赖缺失时只做提示
        raise SystemExit("未找到 pycocotools，请先安装：pip install pycocotools") from exc
    return COCO, COCOeval


@dataclass(frozen=True)
class OnlineModelSpec:
    """单个模型的 config、checkpoint 与 keep-ratio 语义。"""

    name: str
    config_path: Path
    resume_path: Path
    fixed_keep_ratio: Optional[float]
    is_dynamic: bool
    yaml_updates: Tuple[str, ...]


@dataclass(frozen=True)
class SubsetSplit:
    """保存三个复杂度子集及 GT 数量映射。"""

    simple_ids: List[int]
    medium_ids: List[int]
    complex_ids: List[int]
    gt_count_by_image: Dict[int, int]


def parse_args() -> argparse.Namespace:
    """解析命令行参数，并提供适合本仓库的默认值。"""
    parser = argparse.ArgumentParser(
        description="按 GT 数量分组，评估 Fixed / Dynamic 模型在不同复杂度子集上的 AP_S 与 keep ratio。"
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=DEFAULT_GT_PATH,
        help="COCO Ground Truth 标注文件路径，默认使用 DAIR-V2X 测试集。",
    )
    parser.add_argument(
        "--test-img-folder",
        type=Path,
        default=None,
        help="覆盖 val_dataloader.dataset.img_folder，默认与 GT 同级的数据集根，如 .../DAIR-V2X。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="传入 YAMLConfig 的 device，例如 cuda:0；默认自动选择。",
    )
    parser.add_argument(
        "--online-update",
        nargs="+",
        default=None,
        help="传给 YAMLConfig 的 -u 更新项，例如 HybridEncoder.token_keep_ratio=0.3。",
    )
    parser.add_argument(
        "--encoder-epoch",
        type=int,
        default=-1,
        help="同步 HybridEncoder 内部 epoch，用于 CAIP 动态裁剪的 warmup 判断；"
        "-1 表示使用 checkpoint 中的 last_epoch。",
    )
    parser.add_argument(
        "--online-fixed-03-config",
        type=Path,
        default=None,
        help="Fixed 0.3 的 YAML 配置路径。",
    )
    parser.add_argument(
        "--online-fixed-03-resume",
        type=Path,
        default=None,
        help="Fixed 0.3 的 checkpoint 路径。",
    )
    parser.add_argument(
        "--online-fixed-03-update",
        nargs="+",
        default=None,
        help="仅作用于 Fixed 0.3 的额外 YAML 更新项。",
    )
    parser.add_argument(
        "--online-fixed-07-config",
        type=Path,
        default=None,
        help="Fixed 0.7 的 YAML 配置路径。",
    )
    parser.add_argument(
        "--online-fixed-07-resume",
        type=Path,
        default=None,
        help="Fixed 0.7 的 checkpoint 路径。",
    )
    parser.add_argument(
        "--online-fixed-07-update",
        nargs="+",
        default=None,
        help="仅作用于 Fixed 0.7 的额外 YAML 更新项。",
    )
    parser.add_argument(
        "--online-fixed-10-config",
        type=Path,
        default=None,
        help="Fixed 1.0 的 YAML 配置路径。",
    )
    parser.add_argument(
        "--online-fixed-10-resume",
        type=Path,
        default=None,
        help="Fixed 1.0 的 checkpoint 路径。",
    )
    parser.add_argument(
        "--online-fixed-10-update",
        nargs="+",
        default=None,
        help="仅作用于 Fixed 1.0 的额外 YAML 更新项。",
    )
    parser.add_argument(
        "--online-dynamic-config",
        type=Path,
        default=None,
        help="Dynamic CaS-DETR 的 YAML 配置路径。",
    )
    parser.add_argument(
        "--online-dynamic-resume",
        type=Path,
        default=None,
        help="Dynamic CaS-DETR 的 checkpoint 路径。",
    )
    parser.add_argument(
        "--online-dynamic-update",
        nargs="+",
        default=None,
        help="仅作用于 Dynamic 模型的额外 YAML 更新项。",
    )
    parser.add_argument(
        "--dynamic-keep-fallback-json",
        type=Path,
        default=None,
        help="若前向中拿不到 encoder_info.dynamic_keep_ratio，则回退读取该 JSON。",
    )
    parser.add_argument(
        "--fixed-keep-03",
        type=float,
        default=0.3,
        help="Fixed 0.3 模型对应的 keep ratio 常数。",
    )
    parser.add_argument(
        "--fixed-keep-07",
        type=float,
        default=0.7,
        help="Fixed 0.7 模型对应的 keep ratio 常数。",
    )
    parser.add_argument(
        "--fixed-keep-10",
        type=float,
        default=1.0,
        help="Fixed 1.0 模型对应的 keep ratio 常数。",
    )
    return parser.parse_args()


def _resolve_tuning_checkpoint(cfg: Any) -> None:
    """与 ``CaS-DETR/train.py`` 相同：解析 tuning 权重路径，缺文件则清空 tuning。"""
    t = getattr(cfg, "tuning", None)
    if not t:
        return
    root = CAS_ROOT.resolve()
    p = Path(t)
    if not p.is_absolute():
        p = (root / p).resolve()
    p = str(p)
    if os.path.isfile(p):
        cfg.tuning = p
        if getattr(cfg, "yaml_cfg", None) is not None:
            cfg.yaml_cfg["tuning"] = p
    else:
        print(f"[WARN] tuning checkpoint not found: {p}, use HGNet stage1 only.")
        cfg.tuning = None
        if getattr(cfg, "yaml_cfg", None) is not None:
            cfg.yaml_cfg.pop("tuning", None)


def _prepend_sys_path_for_cas() -> None:
    """让 ``from engine...`` 解析到 ``experiments/CaS-DETR/engine``。"""
    cas_dir = str(CAS_ROOT.resolve())
    if cas_dir not in sys.path:
        sys.path.insert(0, cas_dir)


def _merge_model_yaml_updates(global_updates: Optional[Sequence[str]], local_updates: Optional[Sequence[str]]) -> Tuple[str, ...]:
    merged: List[str] = []
    if global_updates:
        merged.extend([str(x) for x in global_updates])
    if local_updates:
        merged.extend([str(x) for x in local_updates])
    return tuple(merged)


def build_online_model_specs(args: argparse.Namespace) -> List[OnlineModelSpec]:
    """由命令行组装 4 个模型描述；缺任一路径则报错提示。"""
    specs: List[OnlineModelSpec] = [
        OnlineModelSpec(
            name="Fixed 0.3",
            config_path=args.online_fixed_03_config,
            resume_path=args.online_fixed_03_resume,
            fixed_keep_ratio=args.fixed_keep_03,
            is_dynamic=False,
            yaml_updates=_merge_model_yaml_updates(args.online_update, args.online_fixed_03_update),
        ),
        OnlineModelSpec(
            name="Fixed 0.7",
            config_path=args.online_fixed_07_config,
            resume_path=args.online_fixed_07_resume,
            fixed_keep_ratio=args.fixed_keep_07,
            is_dynamic=False,
            yaml_updates=_merge_model_yaml_updates(args.online_update, args.online_fixed_07_update),
        ),
        OnlineModelSpec(
            name="Fixed 1.0",
            config_path=args.online_fixed_10_config,
            resume_path=args.online_fixed_10_resume,
            fixed_keep_ratio=args.fixed_keep_10,
            is_dynamic=False,
            yaml_updates=_merge_model_yaml_updates(args.online_update, args.online_fixed_10_update),
        ),
        OnlineModelSpec(
            name="Dynamic CaS-DETR",
            config_path=args.online_dynamic_config,
            resume_path=args.online_dynamic_resume,
            fixed_keep_ratio=None,
            is_dynamic=True,
            yaml_updates=_merge_model_yaml_updates(args.online_update, args.online_dynamic_update),
        ),
    ]
    for spec in specs:
        if spec.config_path is None or spec.resume_path is None:
            raise ValueError(
                f"需要为 {spec.name} 提供 --online-*-config 与 --online-*-resume，"
                f"当前 config={spec.config_path!r}, resume={spec.resume_path!r}"
            )
        ensure_file_exists(resolve_repo_path(Path(spec.config_path)), f"{spec.name} 配置文件")
        ensure_file_exists(resolve_repo_path(Path(spec.resume_path)), f"{spec.name} checkpoint")
    return specs


def ensure_file_exists(path: Path, description: str) -> None:
    """在真正运行前检查关键文件是否存在，避免评估中途失败。"""
    if not path.exists():
        raise FileNotFoundError(f"{description}不存在: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{description}不是有效文件: {path}")


def load_dynamic_keep_ratios(path: Path) -> Dict[int, float]:
    """读取动态 keep-ratio 日志，并将 image_id 统一转成 int。"""
    ensure_file_exists(path, "动态 keep-ratio 日志")
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"动态 keep-ratio JSON 顶层必须是 dict: {path}")

    normalized: Dict[int, float] = {}
    for image_id, keep_ratio in raw.items():
        try:
            normalized[int(image_id)] = float(keep_ratio)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"动态 keep-ratio 中存在无法解析的键值: image_id={image_id!r}, keep_ratio={keep_ratio!r}"
            ) from exc
    return normalized


def _sync_hybrid_encoder_epoch(model: Any, epoch: int) -> None:
    """将 HybridEncoder._epoch 与训练侧 epoch 对齐，避免 CAIP warmup 在 test-only 下恒为 0。"""
    module = model.module if hasattr(model, "module") else model
    enc = getattr(module, "encoder", None)
    if enc is not None and hasattr(enc, "set_epoch"):
        enc.set_epoch(int(epoch))


def _dataset_label2category_map(data_loader: Any) -> Optional[Mapping[int, int]]:
    """沿 dataset wrapper 向内查找 ``CocoDetection.label2category``。"""
    ds = getattr(data_loader, "dataset", None)
    for _ in range(8):
        if ds is None:
            break
        mapping = getattr(ds, "label2category", None)
        if mapping is not None:
            return mapping
        ds = getattr(ds, "dataset", None)
    return None


def detection_dict_to_coco_results(
    image_id: int,
    detection: Mapping[str, Any],
    *,
    remap_mscoco_category: bool,
    label2category: Optional[Mapping[int, int]],
) -> List[Dict[str, Any]]:
    """将 PostProcessor 输出的单张图结果转为 COCO bbox 结果列表。"""
    import torch

    if not detection:
        return []
    boxes = detection.get("boxes")
    scores = detection.get("scores")
    labels = detection.get("labels")
    if boxes is None or scores is None or labels is None:
        return []

    if len(labels) == 0:
        return []

    xmin, ymin, xmax, ymax = boxes.unbind(1)
    boxes_xywh = torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1).tolist()
    scores_list = scores.tolist()
    labels_list = labels.tolist()

    results: List[Dict[str, Any]] = []
    for k in range(len(boxes_xywh)):
        label_id = int(labels_list[k])
        if remap_mscoco_category:
            category_id = label_id
        elif label2category is not None:
            category_id = int(label2category[label_id])
        else:
            category_id = label_id

        results.append(
            {
                "image_id": int(image_id),
                "category_id": category_id,
                "bbox": boxes_xywh[k],
                "score": float(scores_list[k]),
            }
        )
    return results


def collect_predictions_online(
    spec: OnlineModelSpec,
    *,
    gt_ann_path: Path,
    img_folder: Path,
    device: Optional[str],
    encoder_epoch: int,
) -> Tuple[List[Dict[str, Any]], Dict[int, float]]:
    """加载 CaS-DETR 模型并在 val_dataloader 上推理，返回 COCO 结果列表与逐图 keep-ratio。"""
    import torch

    _prepend_sys_path_for_cas()
    from engine.core import YAMLConfig, yaml_utils
    from engine.solver import TASKS

    config_abs = resolve_repo_path(Path(spec.config_path))
    resume_abs = resolve_repo_path(Path(spec.resume_path))

    update_dict = dict(yaml_utils.parse_cli(list(spec.yaml_updates)))
    update_dict.update(
        {
            k: v
            for k, v in {
                "resume": str(resume_abs),
                "device": device,
            }.items()
            if v is not None
        }
    )
    # 与 train.py 一致：``resume`` 为整模 checkpoint 时不能与 YAML 里的 HGNet ``tuning`` 并存。
    update_dict["tuning"] = None

    prev_cwd = Path.cwd()
    try:
        os.chdir(CAS_ROOT.resolve())
        cfg = YAMLConfig(str(config_abs), **update_dict)
        _resolve_tuning_checkpoint(cfg)

        if cfg.resume and getattr(cfg, "tuning", None):
            raise RuntimeError("Use either resume or tuning, not both.")

        if cfg.resume or getattr(cfg, "tuning", None):
            if "HGNetv2" in cfg.yaml_cfg:
                cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        cfg.yaml_cfg.setdefault("val_dataloader", {})
        cfg.yaml_cfg["val_dataloader"].setdefault("dataset", {})
        cfg.yaml_cfg["val_dataloader"]["dataset"]["ann_file"] = str(Path(gt_ann_path).resolve())
        cfg.yaml_cfg["val_dataloader"]["dataset"]["img_folder"] = str(Path(img_folder).resolve())

        solver_cls = TASKS[cfg.yaml_cfg["task"]]
        solver = solver_cls(cfg)
        solver.eval()
        remap_mscoco_category = bool(cfg.yaml_cfg.get("remap_mscoco_category", False))
        label2category = _dataset_label2category_map(solver.val_dataloader)

        if encoder_epoch < 0:
            enc_epoch = int(solver.last_epoch)
        else:
            enc_epoch = int(encoder_epoch)
        _sync_hybrid_encoder_epoch(solver.model, enc_epoch)
        if solver.ema is not None:
            _sync_hybrid_encoder_epoch(solver.ema.module, enc_epoch)

        keep_by_image: Dict[int, float] = {}
        coco_results: List[Dict[str, Any]] = []

        module = solver.ema.module if solver.ema else solver.model
        module.eval()
        solver.criterion.eval()

        for samples, targets in solver.val_dataloader:
            samples = samples.to(solver.device)
            targets = [{k: v.to(solver.device) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                outputs = module(samples)

            if not isinstance(outputs, dict):
                raise TypeError(f"{spec.name} 前向输出应为 dict，实际为 {type(outputs)!r}")

            encoder_info = outputs.get("encoder_info") if isinstance(outputs, dict) else None
            dyn = None
            if isinstance(encoder_info, dict):
                dyn = encoder_info.get("dynamic_keep_ratio")

            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = solver.postprocessor(outputs, orig_target_sizes)

            if dyn is not None:
                import torch as _torch

                if isinstance(dyn, _torch.Tensor):
                    if dyn.dim() == 0:
                        dyn_list = [float(dyn.item())] * len(targets)
                    else:
                        dyn_list = [float(x) for x in dyn.detach().float().cpu().reshape(-1).tolist()]
                else:
                    dyn_list = [float(dyn)] * len(targets)
            else:
                dyn_list = [None] * len(targets)

            if len(dyn_list) != len(targets):
                raise RuntimeError(
                    f"{spec.name} dynamic_keep_ratio 长度与 batch 不一致: "
                    f"len(ratios)={len(dyn_list)}, batch={len(targets)}"
                )

            for idx, target in enumerate(targets):
                image_id = int(target["image_id"].item())
                if dyn_list[idx] is not None:
                    keep_by_image[image_id] = float(dyn_list[idx])
                det = results[idx] if idx < len(results) else {}
                coco_results.extend(
                    detection_dict_to_coco_results(
                        image_id,
                        det,
                        remap_mscoco_category=remap_mscoco_category,
                        label2category=label2category,
                    )
                )

        return coco_results, keep_by_image
    finally:
        os.chdir(prev_cwd)


def split_image_ids_by_gt_density(coco_gt: Any) -> SubsetSplit:
    """统计每张图的 GT 数量，并按图像数严格三等分切分三个子集。"""
    image_ids = sorted(coco_gt.getImgIds())
    gt_count_by_image: Dict[int, int] = {}

    for image_id in image_ids:
        ann_ids = coco_gt.getAnnIds(imgIds=[image_id], iscrowd=None)
        gt_count_by_image[int(image_id)] = len(ann_ids)

    counts = np.array(list(gt_count_by_image.values()), dtype=np.float64)
    if counts.size == 0:
        raise ValueError("GT 中没有任何图像，无法进行子集划分。")

    simple_ids, medium_ids, complex_ids = split_image_ids_strict_terciles(gt_count_by_image)

    return SubsetSplit(
        simple_ids=simple_ids,
        medium_ids=medium_ids,
        complex_ids=complex_ids,
        gt_count_by_image=gt_count_by_image,
    )


def split_image_ids_strict_terciles(gt_count_by_image: Mapping[int, int]) -> Tuple[List[int], List[int], List[int]]:
    """按 ``GT object count`` 升序排序后，按图像数最均匀地切成三组。"""
    sorted_items = sorted(
        ((int(image_id), int(count)) for image_id, count in gt_count_by_image.items()),
        key=lambda x: (x[1], x[0]),
    )
    total = len(sorted_items)
    if total == 0:
        return [], [], []

    base = total // 3
    remainder = total % 3
    sizes = [base + (1 if i < remainder else 0) for i in range(3)]
    boundaries = [0, sizes[0], sizes[0] + sizes[1], total]

    simple_ids = [image_id for image_id, _ in sorted_items[boundaries[0] : boundaries[1]]]
    medium_ids = [image_id for image_id, _ in sorted_items[boundaries[1] : boundaries[2]]]
    complex_ids = [image_id for image_id, _ in sorted_items[boundaries[2] : boundaries[3]]]
    return simple_ids, medium_ids, complex_ids


def build_empty_coco_dt(coco_gt: Any) -> Any:
    """构造一个空预测结果的 COCO 对象。

    当某个预测文件完全为空时，`coco_gt.loadRes([])` 会报错，因此这里手动创建空结果集。
    这样即使模型没有任何输出，也仍然可以在指定子集上安全评估。
    """
    COCO, _ = import_coco_api()
    coco_dt = COCO()
    coco_dt.dataset = {
        "info": copy.deepcopy(coco_gt.dataset.get("info", {})),
        "licenses": copy.deepcopy(coco_gt.dataset.get("licenses", [])),
        "images": copy.deepcopy(coco_gt.dataset.get("images", [])),
        "categories": copy.deepcopy(coco_gt.dataset.get("categories", [])),
        "annotations": [],
    }
    coco_dt.createIndex()
    return coco_dt


def ensure_coco_dataset_metadata(coco_gt: Any) -> None:
    """补齐 ``pycocotools.COCO.loadRes`` 依赖的顶层元信息字段。"""
    if not hasattr(coco_gt, "dataset") or not isinstance(coco_gt.dataset, dict):
        raise ValueError("coco_gt.dataset 不存在或格式非法，无法构造 COCO results。")
    coco_gt.dataset.setdefault("info", {})
    coco_gt.dataset.setdefault("licenses", [])


def load_coco_dt(coco_gt: Any, predictions: Sequence[Mapping]) -> Any:
    """根据预测列表构造 COCO results 对象。"""
    if not predictions:
        return build_empty_coco_dt(coco_gt)
    ensure_coco_dataset_metadata(coco_gt)
    return coco_gt.loadRes(list(predictions))


def safe_stat(value: float) -> str:
    """统一格式化指标，缺失时打印 nan。"""
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def extract_aps50_from_coco_eval(coco_eval: Any) -> float:
    """从 ``COCOeval.eval["precision"]`` 中提取 ``AP_S^{50}``。"""
    precision = coco_eval.eval.get("precision") if isinstance(coco_eval.eval, dict) else None
    if precision is None or not hasattr(precision, "shape"):
        return float("nan")

    iou_thrs = np.asarray(coco_eval.params.iouThrs, dtype=np.float64)
    area_labels = list(coco_eval.params.areaRngLbl)
    max_dets = list(coco_eval.params.maxDets)

    if iou_thrs.size == 0:
        return float("nan")
    iou_idx = int(np.argmin(np.abs(iou_thrs - 0.50)))
    if abs(float(iou_thrs[iou_idx]) - 0.50) > 1e-6:
        return float("nan")

    if "small" not in area_labels or 100 not in max_dets:
        return float("nan")
    area_idx = int(area_labels.index("small"))
    max_det_idx = int(max_dets.index(100))

    # precision 形状: [TxRxKxAxM]
    slice_pr = precision[iou_idx, :, :, area_idx, max_det_idx]
    valid = slice_pr[slice_pr > -1]
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid))


def evaluate_subset(coco_gt: Any, coco_dt: Any, subset_img_ids: Sequence[int]) -> Dict[str, float]:
    """在指定图像子集上运行 COCOeval，并返回 4 个核心指标。"""
    if not subset_img_ids:
        return {
            "map5095": float("nan"),
            "ap50": float("nan"),
            "aps5095": float("nan"),
            "aps50": float("nan"),
        }

    _, COCOeval = import_coco_api()
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = list(subset_img_ids)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "map5095": float(coco_eval.stats[0]),
        "ap50": float(coco_eval.stats[1]),
        "aps5095": float(coco_eval.stats[3]),
        "aps50": extract_aps50_from_coco_eval(coco_eval),
    }


def compute_average_keep_ratio(
    subset_img_ids: Sequence[int],
    *,
    fixed_keep_ratio: Optional[float] = None,
    dynamic_keep_ratios: Optional[Mapping[int, float]] = None,
) -> Tuple[float, List[int]]:
    """计算给定子集的平均 keep ratio。

    返回值包含：
    - 平均 keep ratio
    - 缺失 keep-ratio 的 image_id 列表
    """
    if not subset_img_ids:
        return float("nan"), []

    if fixed_keep_ratio is not None:
        return float(fixed_keep_ratio), []

    if dynamic_keep_ratios is None:
        raise ValueError("动态模型计算平均 keep ratio 时必须提供动态 keep-ratio 字典。")

    values: List[float] = []
    missing_ids: List[int] = []
    for image_id in subset_img_ids:
        if int(image_id) not in dynamic_keep_ratios:
            missing_ids.append(int(image_id))
            continue
        values.append(float(dynamic_keep_ratios[int(image_id)]))

    if not values:
        return float("nan"), missing_ids

    return float(mean(values)), missing_ids


def summarize_subset_counts(subset_name: str, subset_ids: Sequence[int], gt_count_by_image: Mapping[int, int]) -> str:
    """生成人类可读的子集统计信息。"""
    if not subset_ids:
        return f"- {subset_name}: num_images=0, mean_gt_count=nan, min_gt_count=nan, max_gt_count=nan"
    counts = [gt_count_by_image[int(image_id)] for image_id in subset_ids]
    return (
        f"- {subset_name}: num_images={len(subset_ids)}, "
        f"mean_gt_count={mean(counts):.2f}, min_gt_count={min(counts)}, max_gt_count={max(counts)}"
    )


def markdown_main_table(rows: Sequence[Mapping[str, str]]) -> str:
    """生成主结果表，展示 keep ratio 与 AP_S^{50}。"""
    header = (
        "| Model | Simple Keep-Ratio | Simple AP_S^{50} | "
        "Medium Keep-Ratio | Medium AP_S^{50} | Complex Keep-Ratio | Complex AP_S^{50} |"
    )
    sep = "| --- | --- | --- | --- | --- | --- | --- |"
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| {model} | {simple_keep} | {simple_aps50} | {medium_keep} | "
            "{medium_aps50} | {complex_keep} | {complex_aps50} |".format(**row)
        )
    return "\n".join(lines)


def markdown_map_table(rows: Sequence[Mapping[str, str]]) -> str:
    """生成 mAP^{50:95} 子集对比表。"""
    header = "| Model | Simple mAP^{50:95} | Medium mAP^{50:95} | Complex mAP^{50:95} |"
    sep = "| --- | --- | --- | --- |"
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| {model} | {simple_map5095} | {medium_map5095} | {complex_map5095} |".format(**row)
        )
    return "\n".join(lines)


def markdown_ap50_table(rows: Sequence[Mapping[str, str]]) -> str:
    """生成补充 AP50 表，方便分析不同复杂度子集上的召回/定位变化。"""
    header = "| Model | Simple AP50 | Medium AP50 | Complex AP50 |"
    sep = "| --- | --- | --- | --- |"
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| {model} | {simple_ap50} | {medium_ap50} | {complex_ap50} |".format(**row)
        )
    return "\n".join(lines)


def print_warning_lines(warnings: Iterable[str]) -> None:
    """将告警信息集中打印，避免核心表格被大量提示打断。"""
    warning_list = [w for w in warnings if w]
    if not warning_list:
        return
    print("\n## Warnings")
    for warning in warning_list:
        print(f"- {warning}")


def _default_img_folder_from_gt(gt_ann_path: Path) -> Path:
    """由 ``.../annotations/instances_test.json`` 推断数据集根目录 ``.../DAIR-V2X``。"""
    return gt_ann_path.resolve().parent.parent


def _nan_to_none(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def save_scene_complexity_results(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """将逐模型逐子集结果同时保存为 CSV 与 JSON。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "scene_complexity_results.csv"
    json_path = output_dir / "scene_complexity_results.json"

    fieldnames = [
        "model",
        "subset_name",
        "num_images",
        "mean_gt_count",
        "mean_keep_ratio",
        "mAP5095",
        "AP50",
        "APS5095",
        "APS50",
        "min_gt_count",
        "max_gt_count",
        "missing_dynamic_keep_count",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))

    json_rows = [{k: _nan_to_none(v) for k, v in row.items()} for row in rows]
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_rows, f, ensure_ascii=False, indent=2)

    return csv_path, json_path


def run_online(args: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("需要 PyTorch，请先安装 torch。") from exc

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    COCO, _ = import_coco_api()
    gt_path = resolve_repo_path(Path(args.gt))
    ensure_file_exists(gt_path, "GT 标注文件")
    coco_gt = COCO(str(gt_path))
    subset_split = split_image_ids_by_gt_density(coco_gt)

    img_folder = (
        resolve_repo_path(Path(args.test_img_folder))
        if args.test_img_folder is not None
        else _default_img_folder_from_gt(gt_path)
    )

    fallback_keep: Optional[Dict[int, float]] = None
    if args.dynamic_keep_fallback_json is not None:
        fallback_keep = load_dynamic_keep_ratios(resolve_repo_path(Path(args.dynamic_keep_fallback_json)))

    online_specs = build_online_model_specs(args)

    subset_map = {
        "simple": subset_split.simple_ids,
        "medium": subset_split.medium_ids,
        "complex": subset_split.complex_ids,
    }

    print("# Complexity Split Summary")
    print(f"- GT path: `{gt_path}`")
    print(f"- Image folder: `{img_folder}`")
    print(f"- Device: `{args.device}`")
    print("- Current analysis split: test (default), overridable via --gt.")
    print("- Subset policy: strict terciles by image count after sorting GT object count.")
    print(summarize_subset_counts("Simple", subset_split.simple_ids, subset_split.gt_count_by_image))
    print(summarize_subset_counts("Medium", subset_split.medium_ids, subset_split.gt_count_by_image))
    print(summarize_subset_counts("Complex", subset_split.complex_ids, subset_split.gt_count_by_image))

    main_rows: List[Dict[str, str]] = []
    map_rows: List[Dict[str, str]] = []
    ap50_rows: List[Dict[str, str]] = []
    result_records: List[Dict[str, Any]] = []
    warning_notes: List[str] = []

    for spec in online_specs:
        print(f"\n# Inference: {spec.name}")
        print(f"- config: `{spec.config_path}`")
        print(f"- resume: `{spec.resume_path}`")

        with py_warnings.catch_warnings():
            py_warnings.simplefilter("ignore")
            predictions, keep_online = collect_predictions_online(
                spec,
                gt_ann_path=gt_path,
                img_folder=img_folder,
                device=args.device,
                encoder_epoch=int(args.encoder_epoch),
            )

        coco_dt = load_coco_dt(coco_gt, predictions)
        print(f"- coco detection lines: {len(predictions)}")
        print(f"- keep-ratio records: {len(keep_online)}")

        keep_map: Optional[Dict[int, float]] = None
        if spec.is_dynamic:
            keep_map = dict(keep_online)
            if fallback_keep:
                for image_id, ratio in fallback_keep.items():
                    keep_map.setdefault(int(image_id), float(ratio))
            if not keep_map:
                warning_notes.append(
                    f"{spec.name} 未得到任何逐图 keep-ratio，后续 keep-ratio 相关统计将为 NaN。"
                )
        else:
            keep_map = None

        row: MutableMapping[str, str] = {"model": spec.name}
        dynamic_subset_keeps: List[float] = []

        for subset_name, subset_img_ids in subset_map.items():
            print(f"\n## {spec.name} / {subset_name.capitalize()}")
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    metrics = evaluate_subset(coco_gt, coco_dt, subset_img_ids)

            avg_keep_ratio, missing_ids = compute_average_keep_ratio(
                subset_img_ids,
                fixed_keep_ratio=spec.fixed_keep_ratio if not spec.is_dynamic else None,
                dynamic_keep_ratios=keep_map if spec.is_dynamic else None,
            )

            if spec.is_dynamic and missing_ids:
                preview = ", ".join(str(image_id) for image_id in missing_ids[:10])
                suffix = " ..." if len(missing_ids) > 10 else ""
                warning_notes.append(
                    f"{spec.name} 在 {subset_name} 子集缺少 {len(missing_ids)} 个动态 keep-ratio 记录，"
                    f"这些图像未参与该子集 keep-ratio 均值计算: {preview}{suffix}"
                )

            gt_counts = [subset_split.gt_count_by_image[int(image_id)] for image_id in subset_img_ids]
            mean_gt_count = float(mean(gt_counts)) if gt_counts else float("nan")
            min_gt_count = int(min(gt_counts)) if gt_counts else 0
            max_gt_count = int(max(gt_counts)) if gt_counts else 0

            if spec.is_dynamic:
                dynamic_subset_keeps.append(avg_keep_ratio)

            row[f"{subset_name}_keep"] = safe_stat(avg_keep_ratio)
            row[f"{subset_name}_aps50"] = safe_stat(metrics["aps50"])
            row[f"{subset_name}_map5095"] = safe_stat(metrics["map5095"])
            row[f"{subset_name}_ap50"] = safe_stat(metrics["ap50"])
            result_records.append(
                {
                    "model": spec.name,
                    "subset_name": subset_name.capitalize(),
                    "num_images": len(subset_img_ids),
                    "mean_gt_count": mean_gt_count,
                    "mean_keep_ratio": avg_keep_ratio,
                    "mAP5095": metrics["map5095"],
                    "AP50": metrics["ap50"],
                    "APS5095": metrics["aps5095"],
                    "APS50": metrics["aps50"],
                    "min_gt_count": min_gt_count,
                    "max_gt_count": max_gt_count,
                    "missing_dynamic_keep_count": len(missing_ids) if spec.is_dynamic else 0,
                }
            )

        if spec.is_dynamic and len(dynamic_subset_keeps) == 3:
            print(
                "- Dynamic mean keep-ratio by subset: "
                f"Simple={safe_stat(dynamic_subset_keeps[0])}, "
                f"Medium={safe_stat(dynamic_subset_keeps[1])}, "
                f"Complex={safe_stat(dynamic_subset_keeps[2])}"
            )
            if all(not math.isnan(v) for v in dynamic_subset_keeps):
                is_monotonic = dynamic_subset_keeps[0] <= dynamic_subset_keeps[1] <= dynamic_subset_keeps[2]
                print(
                    "- Dynamic mean keep-ratio monotonic check (Simple <= Medium <= Complex): "
                    f"{is_monotonic}"
                )
                if not is_monotonic:
                    warning_notes.append(
                        f"{spec.name} 的平均 keep-ratio 未随复杂度单调上升: "
                        f"{dynamic_subset_keeps[0]:.4f} -> {dynamic_subset_keeps[1]:.4f} -> {dynamic_subset_keeps[2]:.4f}"
                    )
            else:
                warning_notes.append(f"{spec.name} 至少一个子集的平均 keep-ratio 为 NaN，无法进行单调性检查。")

        main_rows.append(dict(row))
        map_rows.append(dict(row))
        ap50_rows.append(dict(row))

    print("\n# Main Results")
    print(markdown_main_table(main_rows))

    print("\n# mAP^{50:95} Results")
    print(markdown_map_table(map_rows))

    print("\n# AP50 Results")
    print(markdown_ap50_table(ap50_rows))

    csv_path, json_path = save_scene_complexity_results(result_records, Path.cwd())
    print("\n# Saved Files")
    print(f"- CSV: `{csv_path}`")
    print(f"- JSON: `{json_path}`")

    print_warning_lines(warning_notes)
    return 0


def main() -> int:
    args = parse_args()
    return run_online(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
