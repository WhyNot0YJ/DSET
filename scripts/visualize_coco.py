#!/usr/bin/env python3
"""
可视化COCO格式的标注框

支持标准COCO格式的JSON文件和图像文件
"""

import json
import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import matplotlib.pyplot as plt


# 默认颜色（BGR格式，用于OpenCV）
DEFAULT_COLORS = [
    (255, 0, 0),      # 红色
    (0, 255, 0),      # 绿色
    (0, 0, 255),      # 蓝色
    (255, 255, 0),    # 青色
    (255, 0, 255),    # 洋红
    (0, 255, 255),    # 黄色
    (128, 0, 128),   # 紫色
    (255, 165, 0),    # 橙色
    (0, 128, 255),    # 橙色蓝
    (128, 255, 0),   # 黄绿
]


def load_coco_annotations(coco_json_path: Path) -> Dict:
    """加载COCO格式的JSON文件"""
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_category_name(category_id: int, categories: List[Dict]) -> str:
    """根据category_id获取类别名称"""
    for cat in categories:
        if cat["id"] == category_id:
            return cat.get("name", f"class_{category_id}")
    return f"class_{category_id}"


def draw_bbox(image: np.ndarray, bbox: List[float], label: str, 
              color: tuple, line_thickness: int = 2, show_label: bool = True) -> np.ndarray:
    """
    在图像上绘制边界框
    
    Args:
        image: BGR格式的图像
        bbox: COCO格式的bbox [x, y, width, height]
        label: 类别标签
        color: BGR颜色
        line_thickness: 线条粗细
        show_label: 是否显示标签
    
    Returns:
        绘制了边界框的图像
    """
    image = image.copy()
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    
    # 转换为xyxy格式
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    
    # 确保坐标在图像范围内
    h_img, w_img = image.shape[:2]
    x1 = max(0, min(x1, w_img - 1))
    y1 = max(0, min(y1, h_img - 1))
    x2 = max(0, min(x2, w_img - 1))
    y2 = max(0, min(y2, h_img - 1))
    
    # 绘制边界框
    cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)
    
    # 绘制标签
    if show_label and label:
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
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
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return image


def visualize_coco_image(coco_data: Dict, image_file_name: str, 
                        images_dir: Path, output_path: Optional[Path] = None,
                        show: bool = True) -> Optional[np.ndarray]:
    """
    可视化COCO格式的单张图像
    
    Args:
        coco_data: COCO格式的数据字典
        image_file_name: 图像文件名
        images_dir: 图像目录
        output_path: 输出路径（可选）
        show: 是否显示图像
    
    Returns:
        绘制了标注的图像（BGR格式）
    """
    # 查找图像
    image_path = images_dir / image_file_name
    if not image_path.exists():
        # 尝试在不同位置查找
        for possible_dir in [images_dir, images_dir.parent, images_dir / "train"]:
            test_path = possible_dir / image_file_name
            if test_path.exists():
                image_path = test_path
                break
        else:
            print(f"Error: Image not found: {image_file_name}")
            print(f"  Searched in: {images_dir}")
            return None
    
    # 加载图像
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return None
    
    # 获取该图像的所有标注
    image_id = None
    for img in coco_data.get("images", []):
        if img["file_name"] == image_file_name:
            image_id = img["id"]
            break
    
    if image_id is None:
        print(f"Warning: Image {image_file_name} not found in COCO data")
        return image
    
    annotations = [
        ann for ann in coco_data.get("annotations", [])
        if ann["image_id"] == image_id
    ]
    
    print(f"Image: {image_file_name}")
    print(f"Found {len(annotations)} annotations")
    
    # 获取类别信息
    categories = coco_data.get("categories", [])
    category_map = {cat["id"]: cat for cat in categories}
    
    # 绘制所有标注
    for ann in annotations:
        category_id = ann["category_id"]
        category_name = get_category_name(category_id, categories)
        bbox = ann["bbox"]
        
        # 选择颜色
        color_idx = category_id % len(DEFAULT_COLORS)
        color = DEFAULT_COLORS[color_idx]
        
        # 绘制边界框
        image = draw_bbox(image, bbox, category_name, color)
    
    # 显示统计信息
    if annotations:
        print("\nAnnotation statistics:")
        category_counts = {}
        for ann in annotations:
            cat_id = ann["category_id"]
            cat_name = get_category_name(cat_id, categories)
            category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
        
        for cat_name, count in sorted(category_counts.items()):
            print(f"  {cat_name}: {count}")
    
    # 保存
    if output_path:
        cv2.imwrite(str(output_path), image)
        print(f"\nSaved to: {output_path}")
    
    # 显示
    if show:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(16, 10))
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.title(f"COCO Annotations - {image_file_name} ({len(annotations)} boxes)", 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    return image


def visualize_multiple_images(coco_data: Dict, images_dir: Path, 
                             num_images: int = 10, output_dir: Optional[Path] = None,
                             show: bool = False):
    """可视化多张图像"""
    images = coco_data.get("images", [])
    
    if not images:
        print("No images found in COCO data")
        return
    
    num_images = min(num_images, len(images))
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_images):
        img_info = images[i]
        file_name = img_info["file_name"]
        
        print(f"\n{'='*60}")
        print(f"Image {i+1}/{num_images}: {file_name}")
        print(f"{'='*60}")
        
        output_path = output_dir / f"vis_{i:04d}_{file_name}" if output_dir else None
        visualize_coco_image(coco_data, file_name, images_dir, output_path, show=show)


def main():
    parser = argparse.ArgumentParser(description="可视化COCO格式的标注框")
    parser.add_argument(
        "--coco_json",
        type=str,
        required=True,
        help="COCO格式的JSON文件路径"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="图像目录路径"
    )
    parser.add_argument(
        "--image_file",
        type=str,
        default=None,
        help="要可视化的图像文件名（单张模式）"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="要可视化的图像数量（多张模式）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出图像路径（单张）或目录（多张）"
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="不显示图像（仅保存）"
    )
    
    args = parser.parse_args()
    
    coco_json_path = Path(args.coco_json)
    images_dir = Path(args.images_dir)
    
    if not coco_json_path.exists():
        print(f"Error: COCO JSON file not found: {coco_json_path}")
        return
    
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return
    
    # 加载COCO数据
    print(f"Loading COCO data from: {coco_json_path}")
    coco_data = load_coco_annotations(coco_json_path)
    
    print(f"Loaded {len(coco_data.get('images', []))} images")
    print(f"Loaded {len(coco_data.get('annotations', []))} annotations")
    print(f"Loaded {len(coco_data.get('categories', []))} categories")
    
    # 可视化
    if args.image_file:
        # 单张图像模式
        output_path = Path(args.output) if args.output else None
        visualize_coco_image(
            coco_data,
            args.image_file,
            images_dir,
            output_path,
            show=not args.no_show
        )
    else:
        # 多张图像模式
        output_dir = Path(args.output) if args.output else None
        visualize_multiple_images(
            coco_data,
            images_dir,
            args.num_images,
            output_dir,
            show=not args.no_show
        )


if __name__ == "__main__":
    main()

