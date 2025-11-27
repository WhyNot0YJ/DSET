#!/usr/bin/env python3
"""
重新评估YOLOv8模型，使用max_det=100进行公平对比

这个脚本使用已训练的YOLOv8模型，在验证集上使用max_det=100重新评估，
以便与DETR系列模型（num_queries=100）进行公平对比。
"""

import sys
import argparse
import yaml
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root.parent) not in sys.path:
    sys.path.insert(0, str(project_root.parent))

from ultralytics import YOLO


def reevaluate_model(model_path: str, data_yaml: str, max_det: int = 100, device: str = "cuda"):
    """
    重新评估模型，使用指定的max_det值
    
    Args:
        model_path: 模型权重路径（.pt文件）
        data_yaml: 数据配置文件路径
        max_det: 最大检测框数量（默认100，与DETR系列对齐）
        device: 设备
    """
    print("="*80)
    print(f"重新评估YOLOv8模型 (max_det={max_det})")
    print("="*80)
    print(f"模型路径: {model_path}")
    print(f"数据配置: {data_yaml}")
    print(f"最大检测框数: {max_det}")
    print("="*80)
    
    # 加载模型
    if not Path(model_path).exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model = YOLO(model_path)
    
    # 验证参数
    val_kwargs = {
        'data': str(data_yaml),
        'imgsz': 640,
        'device': device,
        'max_det': max_det,  # 关键：设置最大检测框数量
        'conf': None,  # 使用默认置信度阈值
        'iou': 0.7,  # NMS IoU阈值
        'verbose': True,
    }
    
    print(f"\n开始验证（max_det={max_det}）...")
    print("-"*80)
    
    # 运行验证
    results = model.val(**val_kwargs)
    
    # 打印结果
    print("\n" + "="*80)
    print("评估结果:")
    print("="*80)
    
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"mAP@0.5: {metrics.get('metrics/mAP50(B)', 0.0):.4f}")
        print(f"mAP@0.75: {metrics.get('metrics/mAP75(B)', 0.0):.4f}")
        print(f"mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 0.0):.4f}")
        print(f"Precision: {metrics.get('metrics/precision(B)', 0.0):.4f}")
        print(f"Recall: {metrics.get('metrics/recall(B)', 0.0):.4f}")
    else:
        # 尝试从results对象中提取
        if hasattr(results, 'box'):
            box_metrics = results.box
            print(f"mAP@0.5: {box_metrics.map50:.4f}")
            print(f"mAP@0.5:0.95: {box_metrics.map:.4f}")
    
    print("="*80)
    print(f"\n✓ 评估完成（max_det={max_det}）")
    print("="*80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="重新评估YOLOv8模型（使用max_det=100）")
    parser.add_argument("--model", type=str, required=True,
                       help="模型权重路径（.pt文件）")
    parser.add_argument("--data", type=str, required=True,
                       help="数据配置文件路径（.yaml）")
    parser.add_argument("--max_det", type=int, default=100,
                       help="最大检测框数量（默认100，与DETR系列对齐）")
    parser.add_argument("--device", type=str, default="cuda",
                       help="设备 (cuda/cpu)")
    
    args = parser.parse_args()
    
    reevaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        max_det=args.max_det,
        device=args.device
    )


if __name__ == '__main__':
    main()

