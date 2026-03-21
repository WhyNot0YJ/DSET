#!/usr/bin/env python3
"""
将DAIR-V2X数据集转换为YOLO格式
"""

import json
import struct
import shutil
from pathlib import Path
from typing import List, Dict


# DAIR-V2X类别定义（8类）
CLASS_NAMES = [
    "Car", "Truck", "Van", "Bus", "Pedestrian", 
    "Cyclist", "Motorcyclist", "Trafficcone"
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# 类别合并映射：Barrowlist -> Cyclist (ID=5)
CLASS_MERGE_MAP = {
    "Barrowlist": 5,  # Cyclist 的 ID
}

# Ignore类别（训练时过滤）
IGNORE_CLASSES = [
    "PedestrianIgnore", "CarIgnore", "OtherIgnore", 
    "Unknown_movable", "Unknown_unmovable"
]


def normalize_state(value) -> int:
    """标准化状态值到 [0, 1, 2]。"""
    try:
        state = int(float(value))
    except (TypeError, ValueError):
        return 0
    if state < 0:
        return 0
    if state > 2:
        return 2
    return state


def extract_states(ann: Dict) -> Dict[str, int]:
    """从原始标注提取截断/遮挡状态（兼容常见拼写）。"""
    truncated_raw = ann.get("truncated_state", ann.get("turncated_state", 0))
    occluded_raw = ann.get("occluded_state", ann.get("occulated_state", 0))
    return {
        "truncated_state": normalize_state(truncated_raw),
        "occluded_state": normalize_state(occluded_raw)
    }


def read_jpeg_size(image_path: Path):
    """读取 JPEG 图像尺寸（width, height），不依赖第三方库。"""
    with open(image_path, 'rb') as f:
        if f.read(2) != b'\xff\xd8':
            raise ValueError(f"Not a JPEG file: {image_path}")

        while True:
            marker_prefix = f.read(1)
            if not marker_prefix:
                break
            if marker_prefix != b'\xff':
                continue

            marker = f.read(1)
            while marker == b'\xff':
                marker = f.read(1)

            if marker in {b'\xd8', b'\xd9'}:
                continue

            seg_len_bytes = f.read(2)
            if len(seg_len_bytes) != 2:
                break
            seg_len = struct.unpack('>H', seg_len_bytes)[0]

            if marker in {b'\xc0', b'\xc1', b'\xc2', b'\xc3', b'\xc5', b'\xc6', b'\xc7', b'\xc9', b'\xca', b'\xcb', b'\xcd', b'\xce', b'\xcf'}:
                payload = f.read(5)
                if len(payload) != 5:
                    break
                _, height, width = struct.unpack('>BHH', payload)
                return width, height

            f.seek(seg_len - 2, 1)

    raise ValueError(f"Failed to parse JPEG size: {image_path}")


def convert_bbox_to_yolo_format(bbox_2d: Dict, img_width: int, img_height: int) -> tuple:
    """将DAIR-V2X的bbox格式转换为YOLO格式
    
    Args:
        bbox_2d: DAIR-V2X格式的bbox {'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2}
        img_width: 图像宽度
        img_height: 图像高度
    
    Returns:
        (class_id, cx_norm, cy_norm, w_norm, h_norm) 或 None（如果是ignore类别）
    """
    x1 = float(bbox_2d["xmin"])
    y1 = float(bbox_2d["ymin"])
    x2 = float(bbox_2d["xmax"])
    y2 = float(bbox_2d["ymax"])
    
    # 检查边界框是否有效
    if x2 <= x1 or y2 <= y1:
        return None
    
    # 转换为YOLO格式：归一化的中心点坐标和宽高
    cx = (x1 + x2) / 2.0 / img_width
    cy = (y1 + y2) / 2.0 / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height
    
    # 确保坐标在[0, 1]范围内
    cx = max(0, min(1, cx))
    cy = max(0, min(1, cy))
    w = max(0, min(1, w))
    h = max(0, min(1, h))
    
    return (cx, cy, w, h)


def convert_dairv2x_to_yolo(data_root: str, output_dir: str, split: str = "train"):
    """将DAIR-V2X数据集转换为YOLO格式
    
    Args:
        data_root: DAIR-V2X数据集根目录
        output_dir: YOLO格式输出目录
        split: 数据集分割 ('train' 或 'val')
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    
    # 创建输出目录
    images_dir = output_dir / "images" / split
    labels_dir = output_dir / "labels" / split
    labels_meta_dir = output_dir / "labels_meta" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    labels_meta_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据信息
    data_info_path = data_root / "metadata" / "data_info.json"
    if not data_info_path.exists():
        raise FileNotFoundError(f"数据信息文件不存在: {data_info_path}")
    
    with open(data_info_path, 'r', encoding='utf-8') as f:
        data_info = json.load(f)
    
    # 加载分割信息
    split_path = data_root / "metadata" / "split_data.json"
    if split_path.exists():
        with open(split_path, 'r', encoding='utf-8') as f:
            split_data = json.load(f)
            indices = split_data.get(split, [])
            indices = [int(idx) for idx in indices]
    else:
        # 如果没有分割文件，使用前80%作为训练，后20%作为验证
        total_samples = len(data_info)
        if split == "train":
            indices = list(range(int(total_samples * 0.8)))
        else:
            indices = list(range(int(total_samples * 0.8), total_samples))
    
    # 转换每个样本
    converted_count = 0
    skipped_count = 0
    
    for idx in indices:
        # 图像路径
        image_path = data_root / "image" / f"{idx:06d}.jpg"
        if not image_path.exists():
            skipped_count += 1
            continue
        
        # 读取图像尺寸
        try:
            img_width, img_height = read_jpeg_size(image_path)
        except ValueError:
            skipped_count += 1
            continue
        
        # 复制图像到输出目录
        output_image_path = images_dir / f"{idx:06d}.jpg"
        shutil.copy2(image_path, output_image_path)
        
        # 加载标注
        annotation_path = data_root / "annotations" / "camera" / f"{idx:06d}.json"
        if not annotation_path.exists():
            # 创建空的标注文件
            output_label_path = labels_dir / f"{idx:06d}.txt"
            output_label_path.write_text("")
            output_meta_path = labels_meta_dir / f"{idx:06d}.json"
            output_meta_path.write_text("[]", encoding='utf-8')
            converted_count += 1
            continue
        
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        # 转换标注
        yolo_lines = []
        yolo_meta = []
        for ann in annotations:
            class_name = ann["type"]
            
            # 跳过ignore类别
            if class_name in IGNORE_CLASSES:
                continue
            
            # 处理类别合并：Barrowlist -> Cyclist
            if class_name in CLASS_MERGE_MAP:
                class_id = CLASS_MERGE_MAP[class_name]
            elif class_name in CLASS_TO_ID:
                class_id = CLASS_TO_ID[class_name]
            else:
                # 如果不在合并映射和有效类别中，跳过
                continue
            bbox_2d = ann["2d_box"]
            
            # 转换为YOLO格式
            yolo_bbox = convert_bbox_to_yolo_format(bbox_2d, img_width, img_height)
            if yolo_bbox is None:
                continue

            states = extract_states(ann)
            
            cx, cy, w, h = yolo_bbox
            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            yolo_meta.append({
                "class_id": class_id,
                "class_name": class_name,
                "bbox_yolo": {
                    "cx": round(cx, 6),
                    "cy": round(cy, 6),
                    "w": round(w, 6),
                    "h": round(h, 6)
                },
                "truncated_state": states["truncated_state"],
                "occluded_state": states["occluded_state"]
            })
        
        # 保存标注文件
        output_label_path = labels_dir / f"{idx:06d}.txt"
        output_label_path.write_text("".join(yolo_lines))

        # 保存 sidecar 元数据（与 YOLO txt 同名）
        output_meta_path = labels_meta_dir / f"{idx:06d}.json"
        output_meta_path.write_text(
            json.dumps(yolo_meta, ensure_ascii=False),
            encoding='utf-8'
        )
        
        converted_count += 1
    
    print(f"✓ 转换完成: {split}")
    print(f"  成功转换: {converted_count} 个样本")
    print(f"  跳过: {skipped_count} 个样本")
    print(f"  输出目录: {output_dir}")
    
    return converted_count


def create_yolo_yaml(output_dir: str, data_root: str = None):
    """创建YOLO格式的YAML配置文件
    
    Args:
        output_dir: YOLO格式数据集输出目录
        data_root: 原始DAIR-V2X数据集根目录（用于设置path）
    """
    output_dir = Path(output_dir)
    yaml_path = output_dir / "dairv2x.yaml"
    
    # 如果data_root未指定，使用output_dir的父目录
    if data_root is None:
        data_root = str(output_dir)
    else:
        data_root = str(Path(data_root).resolve())
    
    yaml_content = f"""# DAIR-V2X Dataset Configuration for YOLO
# Generated automatically

path: {data_root}
train: images/train
val: images/val

# Number of classes
nc: {len(CLASS_NAMES)}

# Class names
names:
"""
    for i, name in enumerate(CLASS_NAMES):
        yaml_content += f"  {i}: {name}\n"
    
    yaml_path.write_text(yaml_content, encoding='utf-8')
    print(f"✓ YAML配置文件已创建: {yaml_path}")
    
    return yaml_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="将DAIR-V2X数据集转换为YOLO格式")
    parser.add_argument("--data_root", type=str, required=True,
                       help="DAIR-V2X数据集根目录")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="YOLO格式输出目录")
    parser.add_argument("--split", type=str, choices=["train", "val", "both"], 
                       default="both",
                       help="要转换的数据集分割")
    
    args = parser.parse_args()
    
    if args.split in ["train", "both"]:
        convert_dairv2x_to_yolo(args.data_root, args.output_dir, "train")
    
    if args.split in ["val", "both"]:
        convert_dairv2x_to_yolo(args.data_root, args.output_dir, "val")
    
    # 创建YAML配置文件
    create_yolo_yaml(args.output_dir, args.data_root)

