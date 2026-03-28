#!/usr/bin/env python3
"""独立评估：对已有实验目录的 best.pt 做 KITTI 与多尺度评估。

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


def main():
    parser = argparse.ArgumentParser(description="YOLO best.pt 重新评估，KITTI 与多尺度")
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
    logger.info(f"best.pt:  {log_dir / 'weights' / 'best.pt'}")
    logger.info(f"推理设备: {trainer.misc_config.get('device', 'cpu')}")
    logger.info(
        f"YOLO版本: v{version}  |  共 {len(class_names)} 类: "
        f"{', '.join(class_names) or '从模型读取'}"
    )

    metrics = trainer._evaluate_kitti_scale_after_training(model=None)

    if metrics:
        logger.info("=" * 60)
        logger.info("评估完成")
        logger.info(f"  mAP@0.5 全类:      {metrics.get('mAP_0.5', 0):.4f}")
        logger.info(f"  mAP@0.5:0.95 全类: {metrics.get('mAP_0.5_0.95', 0):.4f}")
        logger.info(
            f"  KITTI E/M/H: {metrics.get('AP_easy',0):.4f} / "
            f"{metrics.get('AP_moderate',0):.4f} / {metrics.get('AP_hard',0):.4f}"
        )
        if (bm := format_benchmark_eval_line(metrics)):
            logger.info(f"  {bm}")
    else:
        logger.warning("未获得评估结果")


if __name__ == "__main__":
    main()
