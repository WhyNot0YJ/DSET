#!/usr/bin/env python3
"""
按图像真实目标数量切分 COCO 测试集，并比较不同模型在各复杂度子集上的 AP_S 与保留率。

脚本功能概述：
1. 使用 CaS-DETR 的 ``YAMLConfig``、``DetSolver`` 与 ``config + checkpoint`` 在测试集上推理，
   将输出转为 COCO 检测结果后再评估。
2. 按每张图像的 GT 数量计算 33.3% / 66.6% 分位数，并切成 Sparse / Normal / Crowded 三组。
3. 对每个模型、每个子集分别运行 COCOeval，提取 AP_small 与 AP50。
4. 统计对应子集上的平均 keep ratio，并以 Markdown 表格打印结果。

默认路径假设：
- GT 为 DAIR-V2X COCO 测试标注。
"""

from __future__ import annotations

import argparse
import contextlib
import copy
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
REPO_ROOT = SCRIPT_DIR.parents[1]
CAS_ROOT = REPO_ROOT / "experiments" / "CaS-DETR"
DEFAULT_GT_PATH = Path("/root/autodl-fs/datasets/DAIR-V2X/annotations/instances_test.json")


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
    """保存三个复杂度子集及分位点信息。"""

    sparse_ids: List[int]
    normal_ids: List[int]
    crowded_ids: List[int]
    gt_count_by_image: Dict[int, int]
    q33: float
    q66: float


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
    experiments_dir = str(REPO_ROOT / "experiments")
    if experiments_dir not in sys.path:
        sys.path.insert(0, experiments_dir)


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
        ensure_file_exists(Path(spec.config_path), f"{spec.name} 配置文件")
        ensure_file_exists(Path(spec.resume_path), f"{spec.name} checkpoint")
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


def detection_dict_to_coco_results(image_id: int, detection: Mapping[str, Any]) -> List[Dict[str, Any]]:
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

    return [
        {
            "image_id": int(image_id),
            "category_id": int(labels_list[k]),
            "bbox": boxes_xywh[k],
            "score": float(scores_list[k]),
        }
        for k in range(len(boxes_xywh))
    ]


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

    update_dict = dict(yaml_utils.parse_cli(list(spec.yaml_updates)))
    update_dict.update(
        {
            k: v
            for k, v in {
                "resume": str(Path(spec.resume_path).resolve()),
                "device": device,
            }.items()
            if v is not None
        }
    )

    prev_cwd = Path.cwd()
    try:
        os.chdir(CAS_ROOT.resolve())
        cfg = YAMLConfig(str(Path(spec.config_path).resolve()), **update_dict)
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

        if encoder_epoch < 0:
            enc_epoch = int(solver.last_epoch)
        else:
            enc_epoch = int(encoder_epoch)
        _sync_hybrid_encoder_epoch(solver.model, enc_epoch)

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
                coco_results.extend(detection_dict_to_coco_results(image_id, det))

        return coco_results, keep_by_image
    finally:
        os.chdir(prev_cwd)


def split_image_ids_by_gt_density(coco_gt: Any) -> SubsetSplit:
    """统计每张图的 GT 数量，并按 33.3% / 66.6% 分位数切分三个子集。"""
    image_ids = sorted(coco_gt.getImgIds())
    gt_count_by_image: Dict[int, int] = {}

    for image_id in image_ids:
        ann_ids = coco_gt.getAnnIds(imgIds=[image_id], iscrowd=None)
        gt_count_by_image[int(image_id)] = len(ann_ids)

    counts = np.array(list(gt_count_by_image.values()), dtype=np.float64)
    if counts.size == 0:
        raise ValueError("GT 中没有任何图像，无法进行子集划分。")

    q33 = float(np.percentile(counts, 33.3))
    q66 = float(np.percentile(counts, 66.6))

    sparse_ids: List[int] = []
    normal_ids: List[int] = []
    crowded_ids: List[int] = []

    for image_id in image_ids:
        gt_count = gt_count_by_image[int(image_id)]
        if gt_count <= q33:
            sparse_ids.append(int(image_id))
        elif gt_count <= q66:
            normal_ids.append(int(image_id))
        else:
            crowded_ids.append(int(image_id))

    return SubsetSplit(
        sparse_ids=sparse_ids,
        normal_ids=normal_ids,
        crowded_ids=crowded_ids,
        gt_count_by_image=gt_count_by_image,
        q33=q33,
        q66=q66,
    )


def build_empty_coco_dt(coco_gt: Any) -> Any:
    """构造一个空预测结果的 COCO 对象。

    当某个预测文件完全为空时，`coco_gt.loadRes([])` 会报错，因此这里手动创建空结果集。
    这样即使模型没有任何输出，也仍然可以在指定子集上安全评估。
    """
    COCO, _ = import_coco_api()
    coco_dt = COCO()
    coco_dt.dataset = {
        "images": copy.deepcopy(coco_gt.dataset.get("images", [])),
        "categories": copy.deepcopy(coco_gt.dataset.get("categories", [])),
        "annotations": [],
    }
    coco_dt.createIndex()
    return coco_dt


def load_coco_dt(coco_gt: Any, predictions: Sequence[Mapping]) -> Any:
    """根据预测列表构造 COCO results 对象。"""
    if not predictions:
        return build_empty_coco_dt(coco_gt)
    return coco_gt.loadRes(list(predictions))


def safe_stat(value: float) -> str:
    """统一格式化指标，缺失时打印 nan。"""
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def evaluate_subset(coco_gt: Any, coco_dt: Any, subset_img_ids: Sequence[int]) -> Dict[str, float]:
    """在指定图像子集上运行 COCOeval，并返回 AP_small 与 AP50。

    关键点：
    - 必须通过 `coco_eval.params.imgIds = subset_img_ids` 指定子集。
    - `stats[3]` 对应 `AP_small`，即 area=small, maxDets=100。
    - `stats[1]` 对应整体 `AP50`。
    """
    if not subset_img_ids:
        return {"ap_small": float("nan"), "ap50": float("nan")}

    _, COCOeval = import_coco_api()
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = list(subset_img_ids)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "ap_small": float(coco_eval.stats[3]),
        "ap50": float(coco_eval.stats[1]),
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
        return f"- {subset_name}: 0 images"
    counts = [gt_count_by_image[int(image_id)] for image_id in subset_ids]
    return (
        f"- {subset_name}: {len(subset_ids)} images, "
        f"mean GT={mean(counts):.2f}, min GT={min(counts)}, max GT={max(counts)}"
    )


def markdown_main_table(rows: Sequence[Mapping[str, str]]) -> str:
    """生成主结果表，仅展示 keep ratio 与 AP_S。"""
    header = (
        "| Model | Sparse Keep-Ratio | Sparse AP_S | "
        "Normal Keep-Ratio | Normal AP_S | Crowded Keep-Ratio | Crowded AP_S |"
    )
    sep = "| --- | --- | --- | --- | --- | --- | --- |"
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| {model} | {sparse_keep} | {sparse_ap_small} | {normal_keep} | "
            "{normal_ap_small} | {crowded_keep} | {crowded_ap_small} |".format(**row)
        )
    return "\n".join(lines)


def markdown_ap50_table(rows: Sequence[Mapping[str, str]]) -> str:
    """生成补充 AP50 表，方便分析不同复杂度子集上的召回/定位变化。"""
    header = "| Model | Sparse AP50 | Normal AP50 | Crowded AP50 |"
    sep = "| --- | --- | --- | --- |"
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| {model} | {sparse_ap50} | {normal_ap50} | {crowded_ap50} |".format(**row)
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


def run_online(args: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("需要 PyTorch，请先安装 torch。") from exc

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    COCO, _ = import_coco_api()
    ensure_file_exists(args.gt, "GT 标注文件")
    coco_gt = COCO(str(args.gt))
    subset_split = split_image_ids_by_gt_density(coco_gt)

    img_folder = args.test_img_folder or _default_img_folder_from_gt(args.gt)

    fallback_keep: Optional[Dict[int, float]] = None
    if args.dynamic_keep_fallback_json is not None:
        fallback_keep = load_dynamic_keep_ratios(Path(args.dynamic_keep_fallback_json))

    online_specs = build_online_model_specs(args)

    subset_map = {
        "sparse": subset_split.sparse_ids,
        "normal": subset_split.normal_ids,
        "crowded": subset_split.crowded_ids,
    }

    print("# Complexity Split Summary")
    print(f"- GT path: `{args.gt}`")
    print(f"- Image folder: `{img_folder}`")
    print(f"- Device: `{args.device}`")
    print(f"- q33.3 (GT count): {subset_split.q33:.4f}")
    print(f"- q66.6 (GT count): {subset_split.q66:.4f}")
    print(summarize_subset_counts("Sparse", subset_split.sparse_ids, subset_split.gt_count_by_image))
    print(summarize_subset_counts("Normal", subset_split.normal_ids, subset_split.gt_count_by_image))
    print(summarize_subset_counts("Crowded", subset_split.crowded_ids, subset_split.gt_count_by_image))

    main_rows: List[Dict[str, str]] = []
    ap50_rows: List[Dict[str, str]] = []
    warning_notes: List[str] = []

    for spec in online_specs:
        print(f"\n# Inference: {spec.name}")
        print(f"- config: `{spec.config_path}`")
        print(f"- resume: `{spec.resume_path}`")

        with py_warnings.catch_warnings():
            py_warnings.simplefilter("ignore")
            predictions, keep_online = collect_predictions_online(
                spec,
                gt_ann_path=args.gt,
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
                raise RuntimeError(
                    f"{spec.name} 未得到任何逐图 keep-ratio：请检查 HybridEncoder 是否输出 dynamic_keep_ratio，"
                    f"或传入 --dynamic-keep-fallback-json，或通过 --encoder-epoch 关闭 CAIP warmup 分支。"
                )
        else:
            keep_map = None

        row: MutableMapping[str, str] = {"model": spec.name}

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

            row[f"{subset_name}_keep"] = safe_stat(avg_keep_ratio)
            row[f"{subset_name}_ap_small"] = safe_stat(metrics["ap_small"])
            row[f"{subset_name}_ap50"] = safe_stat(metrics["ap50"])

        main_rows.append(dict(row))
        ap50_rows.append(dict(row))

    print("\n# Main Results")
    print(markdown_main_table(main_rows))

    print("\n# AP50 Results")
    print(markdown_ap50_table(ap50_rows))

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
