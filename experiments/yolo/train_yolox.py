#!/usr/bin/env python3
"""YOLOX 训练入口（Megvii），与 ``train.py`` 相同的数据集注册与日志/eval 约定。"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import yaml

_experiments_root = Path(__file__).resolve().parent.parent
if str(_experiments_root) not in sys.path:
    sys.path.insert(0, str(_experiments_root))

_yolo_dir = Path(__file__).resolve().parent

from common.dataset_registry import (
    apply_yolo_dataset_profile as apply_dataset_profile,
    find_dataset_profile_by_data_yaml,
    load_dataset_registry,
    resolve_dataset_profile,
)

DEFAULT_CLASS_NAMES: List[str] = []


def default_config() -> Path:
    return Path("configs/yoloxs_dairv2x.yaml")


def merge_coco_root_from_profile(config: dict, profile: Optional[dict]) -> dict:
    if not profile:
        return config
    root = profile.get("coco_data_root")
    if not root:
        return config
    merged = dict(config)
    data = dict(merged.get("data", {}))
    if not data.get("coco_data_root"):
        data["coco_data_root"] = str(root)
    merged["data"] = data
    return merged


def main():
    parser = argparse.ArgumentParser(description="YOLOX 训练（CaS_DETR）")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    parser.add_argument("--dataset", type=str, default=None, help="数据集键名（configs/datasets.yaml）")
    parser.add_argument(
        "--dataset_registry",
        type=str,
        default="configs/datasets.yaml",
        help="数据集注册表路径",
    )
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从指定检查点恢复")
    parser.add_argument("--resume", action="store_true", help="自动从最新检查点恢复")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的 epochs")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else default_config()
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    selected_class_names: List[str] = list(DEFAULT_CLASS_NAMES)
    registry_path = Path(args.dataset_registry)
    if not registry_path.is_absolute():
        registry_path = _yolo_dir / registry_path
    datasets = load_dataset_registry(registry_path)

    profile = None
    if args.dataset:
        profile = resolve_dataset_profile(datasets, args.dataset)
        config = apply_dataset_profile(config, profile)
        config = merge_coco_root_from_profile(config, profile)
        pc = profile.get("class_names", [])
        if isinstance(pc, list) and pc:
            selected_class_names = [str(x) for x in pc]
        print(f"🗂️  使用数据集: {args.dataset} -> {config.get('data', {}).get('data_yaml')}")
    else:
        profile = find_dataset_profile_by_data_yaml(
            datasets, config.get("data", {}).get("data_yaml", "")
        )
        if profile:
            config = merge_coco_root_from_profile(config, profile)
            pc = profile.get("class_names", [])
            if isinstance(pc, list) and pc:
                selected_class_names = [str(x) for x in pc]

    if args.resume and not args.resume_from_checkpoint:
        log_base = config.get("checkpoint", {}).get("log_dir", "logs")
        latest = _find_latest_yolox_ckpt(log_base)
        if latest:
            args.resume_from_checkpoint = latest
            print(f"📦 找到最新检查点: {latest}")

    from yolo_yolox_trainer import YOLOXTrainer

    trainer = YOLOXTrainer(
        config,
        config_path=str(config_path),
        class_names=selected_class_names,
    )
    trainer.start_training(
        resume_checkpoint=args.resume_from_checkpoint,
        epochs_override=args.epochs,
    )


def _find_latest_yolox_ckpt(log_base: str) -> Optional[str]:
    log_dir = Path(log_base)
    if not log_dir.exists():
        return None
    cks = list(log_dir.glob("**/latest_ckpt.pth"))
    if not cks:
        cks = list(log_dir.glob("**/weights/latest_ckpt.pth"))
    if not cks:
        return None
    return str(max(cks, key=lambda p: p.stat().st_mtime))


if __name__ == "__main__":
    main()
