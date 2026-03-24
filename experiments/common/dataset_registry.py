"""Shared dataset registry for YOLO and DETR (single datasets.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _normalize_key(value: str) -> str:
    return "".join(ch for ch in value.lower().strip() if ch.isalnum())


def load_dataset_registry(registry_path: Path) -> Dict[str, Any]:
    if not registry_path.exists():
        raise FileNotFoundError(f"数据集注册表不存在: {registry_path}")
    with registry_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    datasets = data.get("datasets", {})
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError(f"数据集注册表格式错误或为空: {registry_path}")
    return datasets


def resolve_dataset_profile(datasets: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    target = _normalize_key(dataset_name)
    for key, profile in datasets.items():
        aliases = profile.get("aliases", [])
        candidates = [_normalize_key(str(key))] + [_normalize_key(str(alias)) for alias in aliases]
        if target in candidates:
            return profile

    choices = ", ".join(sorted(datasets.keys()))
    raise ValueError(f"未知数据集: {dataset_name}，可选: {choices}")


def find_dataset_profile_by_data_yaml(datasets: Dict[str, Any], data_yaml: str) -> Optional[Dict[str, Any]]:
    target = str(data_yaml).strip()
    for _, profile in datasets.items():
        if str(profile.get("data_yaml", "")).strip() == target:
            return profile
    return None


def apply_yolo_dataset_profile(config: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(config)
    merged_data = dict(merged.get("data", {}))
    merged_misc = dict(merged.get("misc", {}))

    data_yaml = profile.get("data_yaml")
    if not data_yaml:
        raise ValueError("数据集配置缺少 data_yaml")

    merged_data["data_yaml"] = data_yaml
    merged["data"] = merged_data

    if "num_workers" in profile and "num_workers" not in merged_misc:
        merged_misc["num_workers"] = profile["num_workers"]
    merged["misc"] = merged_misc
    return merged


def apply_detr_dataset_profile(config: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(config)
    merged_data = dict(merged.get("data", {}))
    merged_misc = dict(merged.get("misc", {}))

    root = profile.get("coco_data_root")
    if root:
        merged_data["data_root"] = str(root)

    dclass = profile.get("detr_dataset_class")
    if dclass:
        merged_data["dataset_class"] = str(dclass)

    if profile.get("num_classes") is not None:
        merged_data["num_classes"] = int(profile["num_classes"])

    profile_classes = profile.get("class_names", [])
    if isinstance(profile_classes, list) and profile_classes:
        merged_data["class_names"] = [str(name) for name in profile_classes]

    merged["data"] = merged_data

    if "num_workers" in profile and "num_workers" not in merged_misc:
        merged_misc["num_workers"] = profile["num_workers"]
    merged["misc"] = merged_misc
    return merged


def default_detr_registry_path() -> Path:
    return Path(__file__).resolve().parent.parent / "yolo" / "configs" / "datasets.yaml"
