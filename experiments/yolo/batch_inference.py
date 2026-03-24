#!/usr/bin/env python3
"""统一YOLO批量推理入口（v8/v10/v11/v12）"""

import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import yaml

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        print(f"{desc}...")
        return iterable

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (255, 128, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 128, 128),
]


def normalize_version(version: str) -> str:
    value = version.lower().strip()
    if value.startswith("v"):
        value = value[1:]
    if value not in {"8", "10", "11", "12"}:
        raise ValueError(f"不支持的YOLO版本: {version}，可选: v8/v10/v11/v12")
    return value


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


def resolve_dataset_key(datasets: Dict[str, Any], dataset_name: str) -> str:
    target = _normalize_key(dataset_name)
    for key, profile in datasets.items():
        aliases = profile.get("aliases", [])
        candidates = [_normalize_key(str(key))] + [_normalize_key(str(alias)) for alias in aliases]
        if target in candidates:
            return str(key)
    choices = ", ".join(sorted(datasets.keys()))
    raise ValueError(f"未知数据集: {dataset_name}，可选: {choices}")


def build_colors(count: int) -> List[Tuple[int, int, int]]:
    if count <= 0:
        return []
    result = []
    for index in range(count):
        result.append(COLORS[index % len(COLORS)])
    return result


def path_exists_safe(path: Path) -> bool:
    try:
        return path.exists()
    except (PermissionError, OSError):
        return False


def resolve_existing_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if path_exists_safe(candidate):
        return candidate

    project_root = Path(__file__).resolve().parent.parent.parent
    alt_paths = [
        project_root / path_value,
        project_root.parent / path_value,
    ]
    for alt_path in alt_paths:
        if path_exists_safe(alt_path):
            return alt_path

    raise FileNotFoundError(f"路径不存在: {path_value}，尝试过: {[str(p) for p in alt_paths]}")


def resolve_output_path(path_value: str) -> Path:
    output = Path(path_value)
    if output.is_absolute():
        return output

    project_root = Path(__file__).resolve().parent.parent.parent
    candidate_1 = project_root / path_value
    candidate_2 = project_root.parent / path_value

    if path_exists_safe(candidate_1) or path_exists_safe(candidate_1.parent):
        return candidate_1
    return candidate_2


def resolve_data_yaml_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.exists():
        return candidate

    project_root = Path(__file__).resolve().parent.parent.parent
    alt_paths = [
        project_root / path_value,
        project_root.parent / path_value,
    ]
    for alt_path in alt_paths:
        if alt_path.exists():
            return alt_path

    raise FileNotFoundError(f"数据集YAML不存在: {path_value}，尝试过: {[str(p) for p in alt_paths]}")


def derive_default_paths_from_data_yaml(data_yaml_path: Path, dataset_key: str) -> Tuple[Path, Path]:
    with data_yaml_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    dataset_root = data_yaml_path.parent
    path_prefix = str(data.get("path", "")).strip()
    if path_prefix:
        path_candidate = Path(path_prefix)
        project_root = Path(__file__).resolve().parent.parent.parent
        workspace_root = project_root.parent

        if path_candidate.is_absolute():
            if path_exists_safe(path_candidate):
                dataset_root = path_candidate
        else:
            candidates = [
                (data_yaml_path.parent / path_candidate).resolve(),
                (workspace_root / path_candidate).resolve(),
                (workspace_root / "datasets" / path_candidate).resolve(),
                data_yaml_path.parent.resolve(),
            ]
            for candidate in candidates:
                if path_exists_safe(candidate):
                    dataset_root = candidate
                    break

    split_value = data.get("val") or data.get("test") or data.get("train")
    if not split_value:
        raise ValueError(f"数据集YAML缺少 train/val/test 字段: {data_yaml_path}")

    image_dir = Path(split_value)
    if not image_dir.is_absolute():
        image_dir = dataset_root / image_dir

    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = project_root.parent / "visualizations" / "yolo_inference" / _normalize_key(dataset_key)
    return image_dir, output_dir


def load_model(checkpoint_path: str, device: str = "cuda"):
    from ultralytics import YOLO

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"模型文件不存在: {checkpoint}")
    model = YOLO(str(checkpoint))
    model.to(device)
    model.eval()
    return model


def draw_boxes(image, boxes, labels, scores, class_names: List[str], colors: List[Tuple[int, int, int]], conf_threshold=0.3):
    import cv2

    output = image.copy()
    for box, label, score in zip(boxes, labels, scores):
        if score < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, box)
        color = colors[label] if label < len(colors) else (255, 255, 255)
        class_name = class_names[label] if label < len(class_names) else f"Class_{label}"
        text = f"{class_name} {score:.2f}"
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(output, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), color, -1)
        cv2.putText(output, text, (x1, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return output


def infer_single_image(model, image_path: Path, conf: float, device: str, imgsz: int):
    results = model.predict(source=str(image_path), conf=conf, imgsz=imgsz, device=device, verbose=False)
    if not results:
        return [], [], []
    result = results[0]
    boxes, labels, scores = [], [], []
    if result.boxes is not None:
        boxes_tensor = result.boxes.xyxy.cpu().numpy()
        labels_tensor = result.boxes.cls.cpu().numpy().astype(int)
        scores_tensor = result.boxes.conf.cpu().numpy()
        for box, label, score in zip(boxes_tensor, labels_tensor, scores_tensor):
            boxes.append(box.tolist())
            labels.append(int(label))
            scores.append(float(score))
    return boxes, labels, scores


def collect_images(image_dir: Path):
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(f"*{ext}"))
        image_files.extend(image_dir.glob(f"*{ext.upper()}"))
    return sorted(image_files)


def find_latest_checkpoint():
    checkpoints = list(Path("logs").glob("*/weights/best.pt"))
    if not checkpoints:
        return None
    return str(max(checkpoints, key=lambda path: path.stat().st_mtime))


def main():
    parser = argparse.ArgumentParser(description="统一YOLO批量推理入口")
    parser.add_argument("--version", type=str, required=True, help="YOLO版本: v8/v10/v11/v12")
    parser.add_argument("--dataset", type=str, default="dairv2x", help="数据集键名或别名（在 configs/datasets.yaml 中定义）")
    parser.add_argument("--dataset_registry", type=str, default="configs/datasets.yaml", help="数据集注册表路径")
    parser.add_argument("--checkpoint", type=str, default="logs/*/weights/best.pt", help="模型检查点")
    parser.add_argument("--image_dir", type=str, default=None, help="输入图像目录（默认取数据集配置）")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录（默认取数据集配置）")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--max_images", type=int, default=None, help="最大处理数量")
    args = parser.parse_args()

    import cv2

    version = normalize_version(args.version)
    datasets = load_dataset_registry(Path(args.dataset_registry))
    dataset_key = resolve_dataset_key(datasets, args.dataset)
    profile = resolve_dataset_profile(datasets, args.dataset)

    class_names = [str(name) for name in profile.get("class_names", [])]
    if not class_names:
        raise ValueError(f"数据集 {args.dataset} 的 class_names 为空")
    colors = build_colors(len(class_names))

    checkpoint = args.checkpoint
    if "*" in checkpoint:
        latest = find_latest_checkpoint()
        if not latest:
            raise FileNotFoundError("未找到检查点，请手动指定 --checkpoint")
        checkpoint = latest

    data_yaml = profile.get("data_yaml")
    if not data_yaml:
        raise ValueError(f"数据集 {args.dataset} 缺少 data_yaml 配置")
    data_yaml_path = resolve_data_yaml_path(str(data_yaml))

    derived_image_dir, derived_output_dir = derive_default_paths_from_data_yaml(data_yaml_path, dataset_key)
    image_dir = resolve_existing_path(str(args.image_dir or profile.get("inference_image_dir") or derived_image_dir))
    if not image_dir.exists():
        raise FileNotFoundError(f"图像目录不存在: {image_dir}")

    output_dir = resolve_output_path(str(args.output_dir or profile.get("inference_output_dir") or derived_output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 开始YOLOv{version}批量推理")
    print(f"数据集: {args.dataset}")
    print(f"模型: {checkpoint}")
    print(f"输入目录: {image_dir}")
    print(f"输出目录: {output_dir}")

    model = load_model(checkpoint, args.device)
    images = collect_images(image_dir)
    if args.max_images is not None:
        images = images[:args.max_images]

    processed = 0
    total_detections = 0
    for image_path in tqdm(images, desc="推理中"):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        boxes, labels, scores = infer_single_image(model, image_path, args.conf, args.device, args.imgsz)
        rendered = draw_boxes(image, boxes, labels, scores, class_names, colors, args.conf)
        cv2.imwrite(str(output_dir / image_path.name), rendered)
        processed += 1
        total_detections += len(boxes)

    print("\n✅ 推理完成")
    print(f"处理图像: {processed}/{len(images)}")
    print(f"总检测数: {total_detections}")
    print(f"平均检测数: {total_detections / max(processed, 1):.2f}")


if __name__ == "__main__":
    main()
