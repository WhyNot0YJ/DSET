#!/usr/bin/env python3
"""统一YOLO训练入口（v8/v10/v11/v12）"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import yaml

_experiments_root = Path(__file__).resolve().parent.parent
if str(_experiments_root) not in sys.path:
    sys.path.insert(0, str(_experiments_root))

_yolo_dir = Path(__file__).resolve().parent
_external = _yolo_dir / "external"
if _external.is_dir() and str(_external) not in sys.path:
    sys.path.insert(0, str(_external))
_yolox_repo = _external / "YOLOX"
if _yolox_repo.is_dir() and str(_yolox_repo) not in sys.path:
    sys.path.insert(0, str(_yolox_repo))

from common.dataset_registry import (
    load_dataset_registry,
    resolve_dataset_profile,
    find_dataset_profile_by_data_yaml,
    apply_yolo_dataset_profile as apply_dataset_profile,
)

DEFAULT_CLASS_NAMES: List[str] = []


def normalize_version(version: str) -> str:
    value = version.lower().strip()
    if value.startswith("v"):
        value = value[1:]
    if value not in {"8", "10", "11", "12"}:
        raise ValueError(f"不支持的YOLO版本: {version}，可选: v8/v10/v11/v12")
    return value


def default_config_for_version(version: str) -> Path:
    return Path(f"configs/yolov{version}n_dairv2x.yaml")


def build_trainer(version: str, config: dict, config_path: Optional[str] = None, class_names: Optional[List[str]] = None):
    from base_yolo_trainer import BaseYOLOTrainer

    class UnifiedYOLOTrainer(BaseYOLOTrainer):
        VERSION = "base"

        def __init__(self, trainer_version: str, trainer_config: dict, trainer_config_path: Optional[str] = None):
            self.VERSION = normalize_version(trainer_version)
            super().__init__(trainer_config, trainer_config_path, class_names or DEFAULT_CLASS_NAMES)

        def create_model(self):
            from ultralytics import YOLO

            model_name = self._resolve_model_path()
            self.logger.info(f"✓ 创建YOLO{self.VERSION}模型: {model_name}")
            return YOLO(model_name)

    return UnifiedYOLOTrainer(version, config, config_path)


def find_latest_checkpoint(log_base: str) -> Optional[str]:
    log_dir = Path(log_base)
    if not log_dir.exists():
        return None
    checkpoints = list(log_dir.glob("**/weights/best.pt"))
    if not checkpoints:
        return None
    return str(max(checkpoints, key=lambda path: path.stat().st_mtime))


def main():
    parser = argparse.ArgumentParser(description="统一YOLO训练入口")
    parser.add_argument("--version", type=str, required=True, help="YOLO版本: v8/v10/v11/v12")
    parser.add_argument("--config", type=str, default=None, help="YAML配置文件路径")
    parser.add_argument("--dataset", type=str, default=None, help="数据集键名或别名（在 configs/datasets.yaml 中定义）")
    parser.add_argument("--dataset_registry", type=str, default="configs/datasets.yaml", help="数据集注册表路径")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从指定检查点恢复")
    parser.add_argument("--resume", action="store_true", help="自动从最新检查点恢复")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的epochs")
    args = parser.parse_args()

    version = normalize_version(args.version)
    config_path = Path(args.config) if args.config else default_config_for_version(version)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    selected_class_names = DEFAULT_CLASS_NAMES
    datasets = load_dataset_registry(Path(args.dataset_registry))

    if args.dataset:
        profile = resolve_dataset_profile(datasets, args.dataset)
        config = apply_dataset_profile(config, profile)

        profile_classes = profile.get("class_names", [])
        if isinstance(profile_classes, list) and profile_classes:
            selected_class_names = [str(name) for name in profile_classes]

        print(f"🗂️  使用数据集: {args.dataset} -> {config.get('data', {}).get('data_yaml')}")
    else:
        profile = find_dataset_profile_by_data_yaml(datasets, config.get("data", {}).get("data_yaml", ""))
        if profile:
            profile_classes = profile.get("class_names", [])
            if isinstance(profile_classes, list) and profile_classes:
                selected_class_names = [str(name) for name in profile_classes]

    if args.resume and not args.resume_from_checkpoint:
        log_base = config.get("checkpoint", {}).get("log_dir", "logs")
        latest_checkpoint = find_latest_checkpoint(log_base)
        if latest_checkpoint:
            args.resume_from_checkpoint = latest_checkpoint
            print(f"📦 找到最新检查点: {latest_checkpoint}")

    trainer = build_trainer(
        version=version,
        config=config,
        config_path=str(config_path),
        class_names=selected_class_names,
    )
    trainer.start_training(
        resume_checkpoint=args.resume_from_checkpoint,
        epochs_override=args.epochs,
    )


if __name__ == "__main__":
    main()
