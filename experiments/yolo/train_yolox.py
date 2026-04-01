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
        latest = _find_latest_yolox_ckpt(_yolo_dir, log_base, config)
        if latest:
            args.resume_from_checkpoint = latest
            print(f"📦 找到最新检查点: {latest}")
        else:
            print(
                "⚠️ 未在对应数据集日志子目录下找到 latest_ckpt.pth，"
                "请用 --resume_from_checkpoint 指定路径",
                flush=True,
            )

    from yolo_yolox_trainer import YOLOXTrainer

    trainer = YOLOXTrainer(
        config,
        config_path=str(config_path),
        class_names=selected_class_names,
        resume_checkpoint=args.resume_from_checkpoint,
    )
    trainer.start_training(
        resume_checkpoint=args.resume_from_checkpoint,
        epochs_override=args.epochs,
    )


def _infer_dataset_log_subdir(config: dict) -> Optional[str]:
    """与 BaseYOLOTrainer.setup_logging 一致，只在同一数据集目录下找续训权重。"""
    data_yaml = config.get("data", {}).get("data_yaml", "") or ""
    path_s = str(data_yaml).lower()
    stem = Path(data_yaml).stem.lower() if data_yaml else ""
    if "dair" in path_s or "dairv2x" in stem:
        return "dairv2x"
    if "uadetrac" in path_s or "ua-detrac" in path_s or stem == "data":
        return "uadetrac"
    return None


def _yolox_run_dir_prefix(config: dict) -> str:
    """根据 ``model.model_name`` 得到日志目录下实验文件夹名前缀，例如 yolox_m 对应 ``yolo_yolox_m_``。"""
    ver = "yolox"
    mc = config.get("model") or {}
    mn = mc.get("model_name") or f"yolo{ver}n"
    mn = str(mn)
    if mn.endswith(".pt"):
        mn = mn[:-3]
    inner = mn.replace(f"yolo{ver}", f"v{ver}")
    return f"yolo_{inner}_"


def _ckpt_path_matches_yolox_run_prefix(ckpt_path: Path, run_prefix: str) -> bool:
    return any(seg.startswith(run_prefix) for seg in ckpt_path.parts)


def _find_latest_yolox_ckpt(
    yolo_root: Path, log_base: str, config: dict
) -> Optional[str]:
    """
    在 ``{yolo_root}/{log_base}/{dataset_subdir}/`` 下按修改时间选最新 ``latest_ckpt.pth``，
    避免 DAIR 与 UA-DETRAC 等不同数据集的实验互相误选。
    仅保留实验目录名以当前 ``model.model_name`` 对应前缀开头的 checkpoint，避免 yolox_s 与 yolox_m 混用。
    若能从 ``data_yaml`` 推断子目录则只搜该子目录；否则搜整个 ``log_base``。
    """
    base = (yolo_root / log_base).resolve()
    sub = _infer_dataset_log_subdir(config)
    if sub and (base / sub).is_dir():
        search_root = base / sub
    else:
        search_root = base
    if not search_root.is_dir():
        return None
    cks = list(search_root.glob("**/latest_ckpt.pth"))
    cks.extend(search_root.glob("**/weights/latest_ckpt.pth"))
    if not cks:
        return None
    uniq = {p.resolve(): p for p in cks}
    run_prefix = _yolox_run_dir_prefix(config)
    matched = [
        p for p in uniq.values() if _ckpt_path_matches_yolox_run_prefix(p, run_prefix)
    ]
    if not matched:
        print(
            f"⚠️ 在 {search_root} 下存在 latest_ckpt.pth，但没有与当前 model.model_name "
            f"对应的实验目录前缀 {run_prefix!r}，请用 --resume_from_checkpoint 指定与当前配置同结构的权重。",
            flush=True,
        )
        return None
    best = max(matched, key=lambda p: p.stat().st_mtime)
    return str(best)


if __name__ == "__main__":
    main()
