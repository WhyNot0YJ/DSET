#!/usr/bin/env python3
"""
Faster R-CNN (torchvision) 训练入口 — 对照实验。

用法:
    python train_fasterrcnn.py --config configs/fasterrcnn_resnet50_dairv2x.yaml
    python train_fasterrcnn.py --config configs/fasterrcnn_resnet50_uadetrac.yaml
    python train_fasterrcnn.py --config configs/fasterrcnn_resnet50_dairv2x.yaml --dataset dairv2x
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import yaml

_experiments_root = Path(__file__).resolve().parent.parent
if str(_experiments_root) not in sys.path:
    sys.path.insert(0, str(_experiments_root))

_yolo_dir = Path(__file__).resolve().parent
if str(_yolo_dir) not in sys.path:
    sys.path.insert(0, str(_yolo_dir))

from common.dataset_registry import (
    load_dataset_registry,
    resolve_dataset_profile,
    find_dataset_profile_by_data_yaml,
    apply_yolo_dataset_profile as apply_dataset_profile,
)

DEFAULT_CLASS_NAMES: List[str] = []
DEFAULT_CONFIG = Path("configs/fasterrcnn_resnet50_dairv2x.yaml")


def find_latest_checkpoint(log_base: str) -> Optional[str]:
    log_dir = Path(log_base)
    if not log_dir.exists():
        return None
    checkpoints = list(log_dir.glob("**/weights/best.pt"))
    if not checkpoints:
        return None
    return str(max(checkpoints, key=lambda p: p.stat().st_mtime))


def main():
    parser = argparse.ArgumentParser(description="Faster R-CNN 训练入口（对照实验）")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="数据集键名或别名（在 configs/datasets.yaml 中定义）",
    )
    parser.add_argument(
        "--dataset_registry", type=str, default="configs/datasets.yaml",
        help="数据集注册表路径",
    )
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从指定检查点恢复")
    parser.add_argument(
        "--resume_experiment_dir",
        type=str,
        default=None,
        help="已有实验目录（读其中 config.yaml + weights/last.pt，续写同一 logs 目录）",
    )
    parser.add_argument("--resume", action="store_true", help="自动从最新检查点恢复")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的 epochs")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else DEFAULT_CONFIG
    if args.resume_experiment_dir:
        exp = Path(args.resume_experiment_dir).expanduser().resolve()
        last_pt = exp / "weights" / "last.pt"
        cfg_in_exp = exp / "config.yaml"
        if not last_pt.is_file():
            raise FileNotFoundError(f"未找到续训权重: {last_pt}")
        if not cfg_in_exp.is_file():
            raise FileNotFoundError(f"未找到实验配置: {cfg_in_exp}")
        args.resume_from_checkpoint = str(last_pt)
        config_path = cfg_in_exp
        print(f"📂 续训实验目录: {exp}")
        print(f"📦 使用检查点: {last_pt}")

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    selected_class_names = DEFAULT_CLASS_NAMES
    datasets = load_dataset_registry(Path(args.dataset_registry))

    if args.dataset:
        profile = resolve_dataset_profile(datasets, args.dataset)
        config = apply_dataset_profile(config, profile)
        profile_classes = profile.get("class_names", [])
        if isinstance(profile_classes, list) and profile_classes:
            selected_class_names = [str(n) for n in profile_classes]
        print(f"🗂️  使用数据集: {args.dataset} -> {config.get('data', {}).get('data_yaml')}")
    else:
        profile = find_dataset_profile_by_data_yaml(
            datasets, config.get("data", {}).get("data_yaml", "")
        )
        if profile:
            profile_classes = profile.get("class_names", [])
            if isinstance(profile_classes, list) and profile_classes:
                selected_class_names = [str(n) for n in profile_classes]

    if args.resume and not args.resume_from_checkpoint:
        log_base = config.get("checkpoint", {}).get("log_dir", "logs")
        latest = find_latest_checkpoint(log_base)
        if latest:
            args.resume_from_checkpoint = latest
            print(f"📦 找到最新检查点: {latest}")

    from fasterrcnn_trainer import FasterRCNNTrainer

    trainer = FasterRCNNTrainer(
        config=config,
        config_path=str(config_path),
        class_names=selected_class_names,
        resume_checkpoint=args.resume_from_checkpoint,
    )
    trainer.start_training(
        resume_checkpoint=args.resume_from_checkpoint,
        epochs_override=args.epochs,
    )


if __name__ == "__main__":
    main()
