#!/usr/bin/env python3
"""
可视化Ground Truth标注框
用于检查数据集标注是否偏上或其他问题
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict

# DAIR-V2X类别定义
CLASS_NAMES = [
    "Car", "Truck", "Van", "Bus", "Pedestrian", 
    "Cyclist", "Motorcyclist", "Trafficcone"
]

# 颜色定义（BGR格式，用于OpenCV）
COLORS = [
    (255, 0, 0),      # Car - 蓝色
    (0, 255, 0),      # Truck - 绿色
    (255, 255, 0),    # Van - 青色
    (0, 0, 255),      # Bus - 红色
    (255, 0, 255),    # Pedestrian - 洋红
    (0, 255, 255),    # Cyclist - 黄色
    (128, 0, 128),   # Motorcyclist - 紫色
    (255, 165, 0),   # Trafficcone - 橙色
]

# Ignore类别列表
IGNORE_CLASSES = [
    "PedestrianIgnore", "CarIgnore", "OtherIgnore", 
    "Unknown_movable", "Unknown_unmovable"
]


def load_annotations(annotation_path: Path) -> List[Dict]:
    """加载标注文件"""
    if not annotation_path.exists():
        return []
    
    with open(annotation_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    class_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}
    processed_annotations = []
    
    for ann in annotations:
        class_name = ann["type"]
        
        # 跳过Ignore类别
        if class_name in IGNORE_CLASSES:
            continue
        
        # 获取2D边界框
        bbox_2d = ann["2d_box"]
        x1 = float(bbox_2d["xmin"])
        y1 = float(bbox_2d["ymin"])
        x2 = float(bbox_2d["xmax"])
        y2 = float(bbox_2d["ymax"])
        
        # 检查边界框是否有效
        if x2 <= x1 or y2 <= y1:
            continue
        
        # 处理类别合并：Barrowlist -> Cyclist
        if class_name == "Barrowlist":
            class_id = 5  # Cyclist
        elif class_name in class_to_id:
            class_id = class_to_id[class_name]
        else:
            continue  # 跳过未知类别
        
        processed_annotations.append({
            'class_id': class_id,
            'class_name': CLASS_NAMES[class_id],
            'bbox': [x1, y1, x2, y2],  # [x1, y1, x2, y2] 格式
        })
    
    return processed_annotations


def draw_gt_boxes(image: np.ndarray, annotations: List[Dict], 
                  show_labels: bool = True, line_thickness: int = 2) -> np.ndarray:
    """在图像上绘制Ground Truth框
    
    Args:
        image: BGR格式的图像
        annotations: 标注列表
        show_labels: 是否显示类别标签
        line_thickness: 线条粗细
    
    Returns:
        绘制了GT框的图像
    """
    image = image.copy()
    
    for ann in annotations:
        x1, y1, x2, y2 = map(int, ann['bbox'])
        class_id = ann['class_id']
        class_name = ann['class_name']
        
        # 确保坐标在图像范围内
        x1 = max(0, min(x1, image.shape[1] - 1))
        y1 = max(0, min(y1, image.shape[0] - 1))
        x2 = max(0, min(x2, image.shape[1] - 1))
        y2 = max(0, min(y2, image.shape[0] - 1))
        
        # 获取颜色（BGR格式）
        color = COLORS[class_id] if class_id < len(COLORS) else (255, 255, 255)
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)
        
        # 绘制标签
        if show_labels:
            label_text = class_name
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # 文本背景
            cv2.rectangle(
                image,
                (x1, y1 - text_h - baseline - 4),
                (x1 + text_w, y1),
                color,
                -1
            )
            
            # 文本
            cv2.putText(
                image,
                label_text,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
    
    return image


def visualize_single_image(data_root: Path, image_idx: int, 
                           output_path: Path = None, show: bool = True):
    """可视化单张图像的Ground Truth
    
    Args:
        data_root: 数据集根目录
        image_idx: 图像索引（如 0, 1, 2...）
        output_path: 输出图像路径（可选）
        show: 是否显示图像
    """
    # 构建路径
    image_path = data_root / "image" / f"{image_idx:06d}.jpg"
    annotation_path = data_root / "annotations" / "camera" / f"{image_idx:06d}.json"
    
    if not image_path.exists():
        print(f"❌ 图像文件不存在: {image_path}")
        return
    
    # 加载图像
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return
    
    # 加载标注
    annotations = []
    if annotation_path.exists():
        annotations = load_annotations(annotation_path)
        print(f"✓ 加载图像: {image_path.name}")
        print(f"✓ 找到 {len(annotations)} 个标注框")
    else:
        print(f"⚠️  标注文件不存在: {annotation_path}")
        print(f"✓ 加载图像: {image_path.name} (无标注)")
    
    # 绘制GT框
    if annotations:
        image_with_boxes = draw_gt_boxes(image, annotations)
    else:
        image_with_boxes = image
    
    # 显示统计信息
    if annotations:
        print(f"\n标注统计:")
        class_counts = {}
        for ann in annotations:
            class_name = ann['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count}")
        
        # 计算边界框位置统计
        y_coords = []
        for ann in annotations:
            y1, y2 = ann['bbox'][1], ann['bbox'][3]
            y_coords.extend([y1, y2])
        if y_coords:
            print(f"\nY坐标统计 (图像高度={image.shape[0]}):")
            print(f"  最小Y: {min(y_coords):.1f}")
            print(f"  最大Y: {max(y_coords):.1f}")
            print(f"  平均Y: {np.mean(y_coords):.1f}")
            print(f"  中位数Y: {np.median(y_coords):.1f}")
            print(f"  图像上半部分 (0-{image.shape[0]//2}) 的框数: {sum(1 for y in y_coords if y < image.shape[0]//2)}")
            print(f"  图像下半部分 ({image.shape[0]//2}-{image.shape[0]}) 的框数: {sum(1 for y in y_coords if y >= image.shape[0]//2)}")
    
    # 保存或显示
    if output_path:
        cv2.imwrite(str(output_path), image_with_boxes)
        print(f"\n✓ 已保存到: {output_path}")
    
    if show:
        # 转换为RGB显示
        image_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(16, 10))
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.title(f"Ground Truth - {image_path.name} ({len(annotations) if annotations else 0} boxes)", 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


def visualize_multiple_images(data_root: Path, num_images: int = 10, 
                              start_idx: int = 0, output_dir: Path = None):
    """可视化多张图像的Ground Truth
    
    Args:
        data_root: 数据集根目录
        num_images: 要可视化的图像数量
        start_idx: 起始图像索引
        output_dir: 输出目录（可选）
    """
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_images):
        image_idx = start_idx + i
        output_path = output_dir / f"gt_{image_idx:06d}.jpg" if output_dir else None
        print(f"\n{'='*60}")
        print(f"图像 {i+1}/{num_images}: index={image_idx}")
        print(f"{'='*60}")
        visualize_single_image(data_root, image_idx, output_path, show=False)


def main():
    parser = argparse.ArgumentParser(description="可视化Ground Truth标注框")
    parser.add_argument("--data_root", type=str, required=True,
                       help="数据集根目录路径")
    parser.add_argument("--image_idx", type=int, default=0,
                       help="要可视化的图像索引（单张图像模式）")
    parser.add_argument("--num_images", type=int, default=1,
                       help="要可视化的图像数量（多张图像模式）")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="起始图像索引（多张图像模式）")
    parser.add_argument("--output", type=str, default=None,
                       help="输出图像路径（单张）或目录（多张）")
    parser.add_argument("--no_show", action="store_true",
                       help="不显示图像（仅保存）")
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"❌ 数据集根目录不存在: {data_root}")
        return
    
    if args.num_images == 1:
        # 单张图像模式
        output_path = Path(args.output) if args.output else None
        visualize_single_image(
            data_root, 
            args.image_idx, 
            output_path, 
            show=not args.no_show
        )
    else:
        # 多张图像模式
        output_dir = Path(args.output) if args.output else None
        visualize_multiple_images(
            data_root,
            args.num_images,
            args.start_idx,
            output_dir
        )


if __name__ == "__main__":
    main()

