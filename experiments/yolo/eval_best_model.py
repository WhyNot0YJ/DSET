#!/usr/bin/env python3
"""独立评估：对已有实验目录的权重做 KITTI 与多尺度评估。

- Ultralytics：``weights/best.pt``
- YOLOX：``weights/best_ckpt.pth``（或 ``last_epoch_ckpt.pth``）
- Faster R-CNN：``weights/best.pt``

复用 ``BaseYOLOTrainer._evaluate_kitti_scale_after_training``。
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

_yolo_dir = Path(__file__).resolve().parent
_experiments_root = _yolo_dir.parent
if str(_yolo_dir) not in sys.path:
    sys.path.insert(0, str(_yolo_dir))
if str(_experiments_root) not in sys.path:
    sys.path.insert(0, str(_experiments_root))

from common.dataset_registry import load_dataset_registry, find_dataset_profile_by_data_yaml
from common.model_benchmark import format_benchmark_eval_line


def _resolve_class_names(config: dict, registry_path: Path) -> list:
    """从 dataset registry 解析类名；registry 与 train.py 相同。"""
    data_yaml = config.get("data", {}).get("data_yaml", "")
    try:
        datasets = load_dataset_registry(registry_path)
        profile = find_dataset_profile_by_data_yaml(datasets, data_yaml)
        if profile:
            names = profile.get("class_names", [])
            if isinstance(names, list) and names:
                return [str(n) for n in names]
    except Exception:
        pass
    return []


def _resolve_version(config: dict) -> str:
    """从 config 的 model_name 推断 YOLO 版本。"""
    model_name = config.get("model", {}).get("model_name", "")
    for v in ("12", "11", "10", "8"):
        if f"v{v}" in model_name or f"yolo{v}" in model_name:
            return v
    return "8"


def _is_yolox_experiment(config: dict) -> bool:
    m = config.get("model") or {}
    if m.get("yolox_exp_file"):
        return True
    return "yolox" in str(m.get("model_name", "")).lower()


def main():
    parser = argparse.ArgumentParser(
        description="Ultralytics YOLO、YOLOX 或 Faster R-CNN 实验目录的 KITTI 与多尺度重新评估"
    )
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dataset_registry", type=str, default="configs/datasets.yaml")
    args = parser.parse_args()

    log_dir = Path(args.log_dir).resolve()
    config_yaml = log_dir / "config.yaml"
    if not config_yaml.exists():
        print(f"ERROR: config.yaml 不存在: {config_yaml}")
        sys.exit(1)

    with config_yaml.open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    # CaS_DETR / 本仓库 DETR：checkpoint 非 Ultralytics 格式，勿用本脚本
    _m = config.get("model") or {}
    if isinstance(_m.get("cas_detr"), dict) or str(_m.get("backbone", "")).lower().startswith(
        "presnet"
    ):
        print(
            "ERROR: 该 config 来自 CaS_DETR / DETR 实验（含 model.cas_detr 或 backbone=presnet）。\n"
            "  weights/best.pt 不是 Ultralytics YOLO 权重，eval_best_model.py 无法加载。\n"
            "  请改用 experiments/cas_detr 下的评估；Ultralytics 评估请使用 experiments/yolo 下的实验目录"
            "（config 中含 data.data_yaml，且 weights/best.pt 为 YOLO 训练产出）。"
        )
        sys.exit(1)

    if args.device is not None:
        config.setdefault("misc", {})["device"] = args.device

    class_names = _resolve_class_names(config, Path(args.dataset_registry))
    version = _resolve_version(config)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
        force=True,
    )
    logger = logging.getLogger("eval_best")

    from base_yolo_trainer import BaseYOLOTrainer

    class _EvalTrainer(BaseYOLOTrainer):
        VERSION = version

        def create_model(self):
            pass

        def setup_logging(self):
            self.logger = logging.getLogger("eval_best")
            self.metrics_logger = None

        def _validate_config(self):
            pass

        def _log_initialization_info(self):
            pass

        def _apply_vram_batch_size_rule(self):
            pass

    model_name = (config.get("model") or {}).get("model_name", "")
    is_fasterrcnn = "fasterrcnn" in str(model_name).lower()
    is_yolox = _is_yolox_experiment(config)

    if is_yolox:
        from yolo_yolox_trainer import YOLOXTrainer

        trainer = YOLOXTrainer.__new__(YOLOXTrainer)
        trainer.config = config
        trainer.config_path = str(config_yaml)
        trainer.model_config = config.get("model", {})
        trainer.training_config = config.get("training", {})
        trainer.data_config = config.get("data", {})
        trainer.checkpoint_config = config.get("checkpoint", {})
        trainer.misc_config = config.get("misc", {})
        trainer.log_dir = log_dir
        trainer.logger = logger
        trainer.metrics_logger = None
        trainer.class_names = class_names
        trainer.num_classes = len(class_names) if class_names else 8
        trainer.VERSION = YOLOXTrainer.VERSION
    elif is_fasterrcnn:
        from fasterrcnn_trainer import FasterRCNNTrainer

        trainer = FasterRCNNTrainer.__new__(FasterRCNNTrainer)
        trainer.config = config
        trainer.config_path = str(config_yaml)
        trainer.model_config = config.get("model", {})
        trainer.training_config = config.get("training", {})
        trainer.data_config = config.get("data", {})
        trainer.checkpoint_config = config.get("checkpoint", {})
        trainer.misc_config = config.get("misc", {})
        trainer.log_dir = log_dir
        trainer.logger = logger
        trainer.metrics_logger = None
        trainer.class_names = class_names
        trainer.num_classes = len(class_names) if class_names else 8
        trainer.VERSION = FasterRCNNTrainer.VERSION
    else:
        trainer = _EvalTrainer.__new__(_EvalTrainer)
        trainer.config = config
        trainer.model_config = config.get("model", {})
        trainer.training_config = config.get("training", {})
        trainer.data_config = config.get("data", {})
        trainer.checkpoint_config = config.get("checkpoint", {})
        trainer.misc_config = config.get("misc", {})
        trainer.log_dir = log_dir
        trainer.logger = logger
        trainer.class_names = class_names
        trainer.num_classes = len(class_names)
        trainer.VERSION = version

    logger.info(f"实验目录: {log_dir}")
    logger.info(f"推理设备: {trainer.misc_config.get('device', 'cpu')}")
    if is_yolox:
        logger.info(
            f"后端: YOLOX ({model_name})  |  {trainer.num_classes} 类: "
            f"{', '.join(class_names) or '(未从 registry 解析)'}"
        )
        logger.info(
            "评估权重: weights/best_ckpt.pth；若无则尝试 weights/last_epoch_ckpt.pth、"
            "日志目录下同名文件"
        )
    elif is_fasterrcnn:
        logger.info(
            f"后端: Faster R-CNN ({model_name})  |  {trainer.num_classes} 类: "
            f"{', '.join(class_names) or '(未从 registry 解析)'}"
        )
        logger.info(f"权重: {log_dir / 'weights' / 'best.pt'}")
    else:
        logger.info(
            f"YOLO版本: v{version}  |  共 {len(class_names)} 类: "
            f"{', '.join(class_names) or '从模型读取'}"
        )
        logger.info(f"权重: {log_dir / 'weights' / 'best.pt'}")

    metrics = trainer._evaluate_kitti_scale_after_training(model=None)

    if metrics:
        logger.info("=" * 60)
        logger.info("评估完成")
        logger.info(f"  mAP@0.5 全类:      {metrics.get('mAP_50_all', 0):.4f}")
        logger.info(f"  mAP@0.5:0.95 全类: {metrics.get('mAP_5095_all', 0):.4f}")
        logger.info(
            f"  KITTI E/M/H: {metrics.get('mAP_easy',0):.4f} / "
            f"{metrics.get('mAP_moderate',0):.4f} / {metrics.get('mAP_hard',0):.4f}"
        )
        if 'gt_boxes_easy' in metrics:
            logger.info(
                "  KITTI GT 框数: easy=%d  moderate=%d  hard=%d  ignore=%d",
                int(metrics.get('gt_boxes_easy', 0)),
                int(metrics.get('gt_boxes_moderate', 0)),
                int(metrics.get('gt_boxes_hard', 0)),
                int(metrics.get('gt_boxes_ignore', 0)),
            )
        logger.info(
            f"  COCO 面积档 @0.5 S/M/L:   {metrics.get('mAP_small',0):.4f} / "
            f"{metrics.get('mAP_medium',0):.4f} / {metrics.get('mAP_large',0):.4f}"
        )
        logger.info(
            f"  COCO 面积档 @0.5:0.95 S/M/L: {metrics.get('mAP_small_5095',0):.4f} / "
            f"{metrics.get('mAP_medium_5095',0):.4f} / {metrics.get('mAP_large_5095',0):.4f}"
        )
        if (bm := format_benchmark_eval_line(metrics)):
            logger.info(f"  {bm}")
    else:
        logger.warning("未获得评估结果")


if __name__ == "__main__":
    main()
