#!/usr/bin/env python3
"""将DAIR-V2X数据集的真实标注框绘制到图像上"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

# 类别名称和颜色（与batch_inference.py保持一致）
CLASS_NAMES = ["Car", "Truck", "Bus", "Van", "Pedestrian", "Cyclist", "Motorcyclist"]
COLORS = [
    (255, 0, 0),    # Car - 红色
    (0, 255, 0),    # Truck - 绿色
    (0, 0, 255),    # Bus - 蓝色
    (255, 128, 0),  # Van - 橙色
    (255, 255, 0),  # Pedestrian - 黄色
    (255, 0, 255),  # Cyclist - 品红
    (0, 255, 255),  # Motorcyclist - 青色
]

# 类别映射：DAIR-V2X类别 -> 模型类别ID（直接使用数据集类别名称）
CLASS_MAPPING = {
    "Car": 0,
    "Truck": 1,
    "Bus": 2,
    "Van": 3,
    "Pedestrian": 4,
    "Cyclist": 5,
    "Motorcyclist": 6,
}


def load_annotations(annotation_path: Path) -> List[Dict]:
    """加载标注文件"""
    if not annotation_path.exists():
        return []
    
    with open(annotation_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    processed_annotations = []
    for ann in annotations:
        # 获取类别（直接使用数据集中的类别名称，不转小写）
        class_name = ann["type"]
        
        # 跳过不在类别列表中的对象（如Trafficcone, Barrowlist）
        if class_name not in CLASS_MAPPING:
            continue
        
        class_id = CLASS_MAPPING[class_name]
        
        # 获取2D边界框
        bbox_2d = ann["2d_box"]
        x1 = float(bbox_2d["xmin"])
        y1 = float(bbox_2d["ymin"])
        x2 = float(bbox_2d["xmax"])
        y2 = float(bbox_2d["ymax"])
        
        # 检查边界框是否有效
        if x2 > x1 and y2 > y1:
            processed_annotations.append({
                'class_id': class_id,
                'class_name': CLASS_NAMES[class_id],
                'bbox': [x1, y1, x2, y2],  # xyxy格式
            })
    
    return processed_annotations


def draw_ground_truth_boxes(image: np.ndarray, annotations: List[Dict]) -> np.ndarray:
    """在图像上绘制真实标注框"""
    result_image = image.copy()
    
    for ann in annotations:
        x1, y1, x2, y2 = ann['bbox']
        class_id = ann['class_id']
        class_name = ann['class_name']
        
        # 确保坐标在图像范围内
        x1 = max(0, min(int(x1), image.shape[1] - 1))
        y1 = max(0, min(int(y1), image.shape[0] - 1))
        x2 = max(0, min(int(x2), image.shape[1] - 1))
        y2 = max(0, min(int(y2), image.shape[0] - 1))
        
        # 绘制边界框
        color = COLORS[class_id]
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签
        label_text = f"{class_name}"
        
        # 计算文本位置
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_y = max(text_h + 4, y1)
        cv2.rectangle(result_image, (x1, text_y - text_h - 4), (x1 + text_w, text_y), color, -1)
        cv2.putText(result_image, label_text, (x1, text_y - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result_image


def process_single_image(image_path: Path, annotation_dir: Path, output_dir: Path) -> Tuple[bool, str]:
    """处理单张图像"""
    try:
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            return False, f"无法读取图像: {image_path}"
        
        # 获取图像ID（文件名去掉扩展名）
        image_id = image_path.stem
        
        # 加载标注
        annotation_path = annotation_dir / f"{image_id}.json"
        annotations = load_annotations(annotation_path)
        
        # 绘制真实框
        result_image = draw_ground_truth_boxes(image, annotations)
        
        # 保存结果
        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), result_image)
        
        return True, f"成功处理，标注数: {len(annotations)}"
    except Exception as e:
        return False, str(e)


def visualize_ground_truth(
    image_dir: str,
    annotation_dir: str,
    output_dir: str = None,
    max_images: int = None,
    image_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp')
):
    """批量可视化真实标注框"""
    # 设置目录
    image_dir = Path(image_dir)
    annotation_dir = Path(annotation_dir)
    
    if not image_dir.exists():
        raise ValueError(f"图像目录不存在: {image_dir}")
    if not annotation_dir.exists():
        raise ValueError(f"标注目录不存在: {annotation_dir}")
    
    if output_dir is None:
        output_dir = image_dir.parent / f"{image_dir.name}_gt_visualization"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 获取所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_dir.glob(f"*{ext}")))
        image_files.extend(list(image_dir.glob(f"*{ext.upper()}")))
    
    if len(image_files) == 0:
        print(f"警告: 在 {image_dir} 中未找到图像文件")
        return
    
    # 限制处理数量
    total_images = len(image_files)
    if max_images is not None and max_images > 0:
        image_files = image_files[:max_images]
        print(f"找到 {total_images} 张图像，将处理前 {len(image_files)} 张")
    else:
        print(f"找到 {len(image_files)} 张图像")
    
    # 批量处理
    success_count = 0
    total_annotations = 0
    failed_images = []
    
    for image_path in image_files:
        success, message = process_single_image(image_path, annotation_dir, output_dir)
        
        if success:
            success_count += 1
            # 统计标注数量
            image_id = image_path.stem
            annotation_path = annotation_dir / f"{image_id}.json"
            annotations = load_annotations(annotation_path)
            total_annotations += len(annotations)
        else:
            failed_images.append((image_path.name, message))
    
    # 打印统计信息
    print("\n" + "="*50)
    print("处理完成!")
    print(f"成功处理: {success_count}/{len(image_files)} 张图像")
    print(f"总标注数: {total_annotations} 个目标")
    if failed_images:
        print(f"失败: {len(failed_images)} 张图像")
        for img_name, error in failed_images[:5]:  # 只显示前5个错误
            print(f"  - {img_name}: {error}")
        if len(failed_images) > 5:
            print(f"  ... 还有 {len(failed_images) - 5} 个错误")
    print(f"结果保存在: {output_dir}")
    print("="*50)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="可视化DAIR-V2X数据集的真实标注框")
    parser.add_argument("--image_dir", type=str, 
                       default="datasets/DAIR-V2X/image",
                       help="输入图像目录路径")
    parser.add_argument("--annotation_dir", type=str,
                       default="datasets/DAIR-V2X/annotations/camera",
                       help="标注文件目录路径")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出图像目录路径（默认：图像目录_gt_visualization）")
    parser.add_argument("--max_images", type=int, default=None,
                       help="最大处理图像数量（默认：处理所有图像）")
    
    args = parser.parse_args()
    
    visualize_ground_truth(
        args.image_dir,
        args.annotation_dir,
        args.output_dir,
        args.max_images
    )

