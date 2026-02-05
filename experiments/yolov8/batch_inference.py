#!/usr/bin/env python3
"""YOLOv8 批量推理脚本 - 处理整个图像目录"""

import sys
import argparse
import yaml
import torch
import cv2
import numpy as np
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    class SimpleProgress:
        def __init__(self, iterable, desc=""):
            self.iterable = iterable
            self.desc = desc
            self.total = len(iterable)
            self.current = 0
            print(f"{desc}: 开始处理 {self.total} 个文件...")
        
        def __iter__(self):
            for item in self.iterable:
                self.current += 1
                if self.current % 10 == 0 or self.current == self.total:
                    print(f"  进度: {self.current}/{self.total} ({100*self.current/self.total:.1f}%)")
                yield item
    
    def tqdm(iterable, desc=""):
        return SimpleProgress(iterable, desc) if not HAS_TQDM else iterable

# 添加项目路径
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root.parent) not in sys.path:
    sys.path.insert(0, str(project_root.parent))

from ultralytics import YOLO

# 类别名称和颜色（用于可视化）- 8类
CLASS_NAMES = [
    "Car", "Truck", "Van", "Bus", "Pedestrian", 
    "Cyclist", "Motorcyclist", "Trafficcone"
]
COLORS = [
    (255, 0, 0),      # Car - 红色
    (0, 255, 0),      # Truck - 绿色
    (255, 128, 0),    # Van - 橙色
    (0, 0, 255),      # Bus - 蓝色
    (255, 255, 0),    # Pedestrian - 黄色
    (255, 0, 255),    # Cyclist - 品红
    (0, 255, 255),    # Motorcyclist - 青色
    (128, 128, 128),  # Trafficcone - 灰色
]


def load_model(checkpoint_path: str, device: str = "cuda"):
    """加载YOLO模型"""
    print(f"📦 加载模型: {checkpoint_path}")
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")
    
    model = YOLO(checkpoint_path)
    model.to(device)
    model.eval()
    
    print(f"✓ 模型加载完成")
    return model


def draw_boxes(image, boxes, labels, scores, conf_threshold=0.3):
    """在图像上绘制检测框
    
    Args:
        image: 输入图像 (BGR格式，OpenCV格式)
        boxes: 边界框列表，格式为 [[x1, y1, x2, y2], ...]
        labels: 类别标签列表
        scores: 置信度列表
        conf_threshold: 置信度阈值
    
    Returns:
        绘制了检测框的图像
    """
    image = image.copy()
    
    for box, label, score in zip(boxes, labels, scores):
        if score < conf_threshold:
            continue
        
        x1, y1, x2, y2 = map(int, box)
        
        # 获取类别颜色
        color = COLORS[label] if label < len(COLORS) else (255, 255, 255)
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 准备标签文本
        class_name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else f"Class_{label}"
        label_text = f"{class_name} {score:.2f}"
        
        # 计算文本大小
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # 绘制文本背景
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # 绘制文本
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


def inference_image(model, image_path: str, conf_threshold: float = 0.3, device: str = "cuda"):
    """对单张图像进行推理
    
    Args:
        model: YOLO模型
        image_path: 图像路径
        conf_threshold: 置信度阈值
        device: 设备
    
    Returns:
        (boxes, labels, scores): 检测结果
    """
    # 使用YOLO的predict方法
    results = model.predict(
        source=str(image_path),
        conf=conf_threshold,
        device=device,
        verbose=False
    )
    
    # 解析结果
    if len(results) == 0:
        return [], [], []
    
    result = results[0]
    
    # 提取检测结果
    boxes = []
    labels = []
    scores = []
    
    if result.boxes is not None:
        boxes_tensor = result.boxes.xyxy.cpu().numpy()  # [N, 4] xyxy格式
        labels_tensor = result.boxes.cls.cpu().numpy().astype(int)  # [N]
        scores_tensor = result.boxes.conf.cpu().numpy()  # [N]
        
        for box, label, score in zip(boxes_tensor, labels_tensor, scores_tensor):
            boxes.append(box.tolist())
            labels.append(int(label))
            scores.append(float(score))
    
    return boxes, labels, scores


def batch_inference(
    model,
    image_dir: str,
    output_dir: str,
    conf_threshold: float = 0.5,
    device: str = "cuda",
    max_images: int = None
):
    """批量推理
    
    Args:
        model: YOLO模型
        image_dir: 输入图像目录
        output_dir: 输出目录
        conf_threshold: 置信度阈值
        device: 设备
        max_images: 最大处理图像数（None表示处理所有）
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_dir.glob(f"*{ext}")))
        image_files.extend(list(image_dir.glob(f"*{ext.upper()}")))
    
    image_files = sorted(image_files)
    
    if max_images is not None:
        image_files = image_files[:max_images]
    
    print(f"📸 找到 {len(image_files)} 张图像")
    
    # 批量处理
    processed_count = 0
    total_detections = 0
    
    for image_path in tqdm(image_files, desc="处理图像"):
        try:
            # 推理
            boxes, labels, scores = inference_image(
                model, str(image_path), conf_threshold, device
            )
            
            # 加载原始图像用于绘制
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"⚠️  无法读取图像: {image_path}")
                continue
            
            # 绘制检测框
            result_image = draw_boxes(image, boxes, labels, scores, conf_threshold)
            
            # 保存结果
            output_path = output_dir / image_path.name
            cv2.imwrite(str(output_path), result_image)
            
            processed_count += 1
            total_detections += len(boxes)
            
        except Exception as e:
            print(f"⚠️  处理图像失败 {image_path}: {e}")
            continue
    
    print(f"\n✅ 批量推理完成！")
    print(f"  处理图像数: {processed_count}/{len(image_files)}")
    print(f"  总检测数: {total_detections}")
    print(f"  平均每张图像: {total_detections/max(processed_count, 1):.2f} 个检测")
    print(f"  结果保存在: {output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOv8批量推理脚本')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径（.pt文件）')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='输入图像目录')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='置信度阈值（默认: 0.5）')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备（默认: cuda）')
    parser.add_argument('--max_images', type=int, default=None,
                       help='最大处理图像数（默认: 处理所有）')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 YOLOv8 批量推理")
    print("="*60)
    print(f"模型: {args.checkpoint}")
    print(f"输入目录: {args.image_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"置信度阈值: {args.conf}")
    print(f"设备: {args.device}")
    if args.max_images:
        print(f"最大处理数: {args.max_images}")
    print("="*60)
    
    # 加载模型
    model = load_model(args.checkpoint, args.device)
    
    # 批量推理
    batch_inference(
        model=model,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        conf_threshold=args.conf,
        device=args.device,
        max_images=args.max_images
    )


if __name__ == '__main__':
    main()

