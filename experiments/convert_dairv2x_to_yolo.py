#!/usr/bin/env python3
"""
将DAIR-V2X数据集（COCO格式）转换为YOLO格式
用于转换 /root/autodl-fs/datasets/DAIR-V2X 到 /root/autodl-fs/datasets/DAIR-V2X_YOLO
"""

import sys
import json
import shutil
from pathlib import Path
from typing import List, Dict
import cv2
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# DAIR-V2X类别定义（8类）
CLASS_NAMES = [
    "Car", "Truck", "Van", "Bus", "Pedestrian", 
    "Cyclist", "Motorcyclist", "Trafficcone"
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# Ignore类别（训练时过滤）
IGNORE_CLASSES = [
    "PedestrianIgnore", "CarIgnore", "OtherIgnore", 
    "Unknown_movable", "Unknown_unmovable"
]


def convert_bbox_to_yolo_format(bbox_2d: Dict, img_width: int, img_height: int) -> tuple:
    """将DAIR-V2X的bbox格式转换为YOLO格式
    
    Args:
        bbox_2d: DAIR-V2X格式的bbox {'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2}
        img_width: 图像宽度
        img_height: 图像高度
    
    Returns:
        (cx_norm, cy_norm, w_norm, h_norm) 或 None（如果无效）
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
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    print(f"\n开始转换 {split} 数据集...")
    print(f"  输入目录: {data_root}")
    print(f"  输出目录: {output_dir}")
    print(f"  样本数量: {len(indices)}")
    
    for idx in indices:
        # 图像路径
        image_path = data_root / "image" / f"{idx:06d}.jpg"
        if not image_path.exists():
            skipped_count += 1
            continue
        
        # 读取图像获取尺寸
        image = cv2.imread(str(image_path))
        if image is None:
            skipped_count += 1
            continue
        
        img_height, img_width = image.shape[:2]
        
        # 复制图像到输出目录
        output_image_path = images_dir / f"{idx:06d}.jpg"
        shutil.copy2(image_path, output_image_path)
        
        # 加载标注
        annotation_path = data_root / "annotations" / "camera" / f"{idx:06d}.json"
        if not annotation_path.exists():
            # 创建空的标注文件
            output_label_path = labels_dir / f"{idx:06d}.txt"
            output_label_path.write_text("")
            converted_count += 1
            continue
        
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        # 转换标注
        yolo_lines = []
        for ann in annotations:
            class_name = ann["type"]
            
            # 跳过ignore类别
            if class_name in IGNORE_CLASSES:
                continue
            
            # 检查是否是有效类别
            if class_name not in CLASS_TO_ID:
                continue
            
            class_id = CLASS_TO_ID[class_name]
            bbox_2d = ann["2d_box"]
            
            # 转换为YOLO格式
            yolo_bbox = convert_bbox_to_yolo_format(bbox_2d, img_width, img_height)
            if yolo_bbox is None:
                continue
            
            cx, cy, w, h = yolo_bbox
            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        
        # 保存标注文件
        output_label_path = labels_dir / f"{idx:06d}.txt"
        output_label_path.write_text("".join(yolo_lines))
        
        converted_count += 1
        
        # 显示进度
        if converted_count % 100 == 0:
            print(f"  已转换: {converted_count}/{len(indices)}")
    
    print(f"\n✓ {split} 数据集转换完成")
    print(f"  成功转换: {converted_count} 个样本")
    print(f"  跳过: {skipped_count} 个样本")
    print(f"  输出目录: {output_dir}")
    
    return converted_count


def create_yolo_yaml(output_dir: str):
    """创建YOLO格式的YAML配置文件
    
    Args:
        output_dir: YOLO格式数据集输出目录
    """
    output_dir = Path(output_dir)
    yaml_path = output_dir / "dairv2x.yaml"
    
    # 使用output_dir作为path（YOLO格式数据集所在目录）
    data_path = str(output_dir.resolve())
    
    yaml_content = f"""# DAIR-V2X Dataset Configuration for YOLO
# Generated automatically

path: {data_path}
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
    print(f"\n✓ YAML配置文件已创建: {yaml_path}")
    
    return yaml_path


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="将DAIR-V2X数据集（COCO格式）转换为YOLO格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 转换训练集和验证集
  python convert_dairv2x_to_yolo.py \\
    --data_root /root/autodl-fs/datasets/DAIR-V2X \\
    --output_dir /root/autodl-fs/datasets/DAIR-V2X_YOLO
  
  # 只转换训练集
  python convert_dairv2x_to_yolo.py \\
    --data_root /root/autodl-fs/datasets/DAIR-V2X \\
    --output_dir /root/autodl-fs/datasets/DAIR-V2X_YOLO \\
    --split train
        """
    )
    parser.add_argument(
        "--data_root", 
        type=str, 
        default="/root/autodl-fs/datasets/DAIR-V2X",
        help="DAIR-V2X数据集根目录（默认: /root/autodl-fs/datasets/DAIR-V2X）"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/root/autodl-fs/datasets/DAIR-V2X_YOLO",
        help="YOLO格式输出目录（默认: /root/autodl-fs/datasets/DAIR-V2X_YOLO）"
    )
    parser.add_argument(
        "--split", 
        type=str, 
        choices=["train", "val", "both"], 
        default="both",
        help="要转换的数据集分割（默认: both）"
    )
    
    args = parser.parse_args()
    
    # 验证输入目录
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"❌ 错误: 输入目录不存在: {data_root}")
        sys.exit(1)
    
    # 验证必要的文件
    if not (data_root / "metadata" / "data_info.json").exists():
        print(f"❌ 错误: 数据信息文件不存在: {data_root / 'metadata' / 'data_info.json'}")
        sys.exit(1)
    
    print("="*60)
    print("DAIR-V2X 数据集转换工具 (COCO格式 → YOLO格式)")
    print("="*60)
    print(f"输入目录: {data_root}")
    print(f"输出目录: {args.output_dir}")
    print(f"转换分割: {args.split}")
    print("="*60)
    
    # 转换数据集
    if args.split in ["train", "both"]:
        convert_dairv2x_to_yolo(args.data_root, args.output_dir, "train")
    
    if args.split in ["val", "both"]:
        convert_dairv2x_to_yolo(args.data_root, args.output_dir, "val")
    
    # 创建YAML配置文件
    create_yolo_yaml(args.output_dir)
    
    print("\n" + "="*60)
    print("✅ 转换完成！")
    print("="*60)
    print(f"YOLO格式数据集已保存到: {args.output_dir}")
    print(f"配置文件: {Path(args.output_dir) / 'dairv2x.yaml'}")
    print("\n现在可以在YOLOv8训练配置中使用:")
    print(f"  data_yaml: \"{Path(args.output_dir) / 'dairv2x.yaml'}\"")


if __name__ == "__main__":
    main()

