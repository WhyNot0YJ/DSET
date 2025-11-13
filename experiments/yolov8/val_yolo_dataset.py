#!/usr/bin/env python3
"""
基于 YOLO 格式数据集评估 YOLOv8，并输出：
- runs/detect/<exp_name>/ 下的可视化结果图
- 使用 Ultralytics 内置的验证功能计算 mAP 指标
- 生成 PR 曲线、混淆矩阵等科研图表

要求：
- 数据目录：/home/yujie/proj/task-selective-det/experiments/datasets/coco_yolo
  - images/val2017/ 下为验证集图像
  - labels/ 下为 YOLO 格式标注文件
  - coco_yolo.yaml 为数据集配置文件

可选参数见 --help。
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from ultralytics import YOLO


def ensure_yolo_data(yolo_root: Path) -> None:
    """检查 YOLO 格式数据集是否就绪。"""
    images_dir = yolo_root / "images" / "val2017"
    labels_dir = yolo_root / "labels" / "val2017"
    yaml_file = yolo_root / "coco_yolo.yaml"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"未找到图像目录: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"未找到标注目录: {labels_dir}")
    if not yaml_file.exists():
        raise FileNotFoundError(f"未找到配置文件: {yaml_file}")
    
    # 检查是否有图像文件
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    if not image_files:
        raise FileNotFoundError(f"图像目录为空: {images_dir}")
    
    # 检查是否有标注文件
    label_files = list(labels_dir.glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"标注目录为空: {labels_dir}")
    
    print(f"[INFO] 找到 {len(image_files)} 张图像，{len(label_files)} 个标注文件")


def run_validation_with_plots(model: YOLO, data_yaml: Path, out_dir: Path, imgsz: int, conf: float, exp_name: str) -> None:
    """使用 Ultralytics 内置验证功能进行完整评估并生成图表。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("[INFO] 开始验证评估...")
    results = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        conf=conf,
        plots=True,  # 生成所有图表
        save=True,   # 保存检测结果
        save_txt=False,  # 不保存 txt 格式
        project=str(out_dir.parent),  # runs/detect
        name=exp_name,                # <exp_name>
        exist_ok=True,
        verbose=True,
    )
    
    # 打印主要指标
    if hasattr(results, 'box'):
        metrics = results.box
        print("\n" + "="*60)
        print("验证结果摘要:")
        print("="*60)
        print(f"mAP50: {metrics.map50:.4f}")
        print(f"mAP50-95: {metrics.map:.4f}")
        print(f"Precision: {metrics.mp:.4f}")
        print(f"Recall: {metrics.mr:.4f}")
        print("="*60)
    
    return results


def run_prediction_only(model: YOLO, images_dir: Path, out_dir: Path, imgsz: int, conf: float, exp_name: str) -> None:
    """仅运行预测并保存可视化结果，不进行验证评估。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("[INFO] 开始预测并保存可视化结果...")
    results = model.predict(
        source=str(images_dir),
        imgsz=imgsz,
        conf=conf,
        save=True,
        save_txt=False,
        project=str(out_dir.parent),  # runs/detect
        name=exp_name,                # <exp_name>
        exist_ok=True,
        verbose=True,
    )
    
    print(f"[INFO] 预测完成，结果保存在: {out_dir}")


def save_metrics_summary(results, out_dir: Path) -> None:
    """保存评估指标摘要到文本文件。"""
    if not hasattr(results, 'box'):
        print("[WARN] 无法获取评估指标")
        return
    
    metrics = results.box
    metrics_file = out_dir / "metrics_summary.txt"
    
    lines = [
        "YOLO 数据集验证结果摘要",
        "="*50,
        f"mAP50: {metrics.map50:.4f}",
        f"mAP50-95: {metrics.map:.4f}",
        f"Precision: {metrics.mp:.4f}",
        f"Recall: {metrics.mr:.4f}",
        "",
        "各类别 mAP50:",
    ]
    
    # 添加各类别指标
    if hasattr(metrics, 'ap_class_index') and hasattr(metrics, 'ap50'):
        for i, (class_idx, ap50) in enumerate(zip(metrics.ap_class_index, metrics.ap50)):
            lines.append(f"  类别 {class_idx}: {ap50:.4f}")
    
    metrics_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] 指标摘要已保存: {metrics_file}")


def main():
    ap = argparse.ArgumentParser(description="基于 YOLO 格式数据集评估 YOLOv8")
    ap.add_argument("--model", default="yolov8n.pt", help="模型路径，如 yolov8n.pt 或训练好的 best.pt")
    ap.add_argument("--data", default="/workspace/experiments/datasets/coco_yolo/coco_yolo.yaml", 
                    help="YOLO 格式数据集配置文件")
    ap.add_argument("--imgsz", type=int, default=640, help="评估分辨率")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    ap.add_argument("--exp", default="yolo_val", help="输出实验名，存放于 runs/detect/<exp>")
    ap.add_argument("--prediction-only", action="store_true", 
                    help="仅运行预测（不进行验证评估），适用于快速可视化")
    args = ap.parse_args()

    yolo_root = Path(args.data).parent
    data_yaml = Path(args.data)
    
    # 检查数据集
    ensure_yolo_data(yolo_root)

    # 输出目录
    runs_detect_dir = Path(__file__).resolve().parent / "runs" / "detect"
    out_dir = runs_detect_dir / args.exp
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("YOLO 格式数据集评估 - YOLOv8")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"数据配置: {data_yaml}")
    print(f"输出: {out_dir}")
    print(f"模式: {'仅预测' if args.prediction_only else '完整验证'}")
    print("=" * 60)

    model = YOLO(args.model)

    if args.prediction_only:
        # 仅预测模式
        images_dir = yolo_root / "images" / "val2017"
        run_prediction_only(model, images_dir, out_dir, args.imgsz, args.conf, args.exp)
    else:
        # 完整验证模式
        results = run_validation_with_plots(model, data_yaml, out_dir, args.imgsz, args.conf, args.exp)
        save_metrics_summary(results, out_dir)

    print(f"\n[SUCCESS] 评估完成。结果保存在: {out_dir}")
    print("\n生成的文件包括:")
    print("- 可视化检测结果图像")
    if not args.prediction_only:
        print("- PR 曲线图 (PR_curve.png)")
        print("- 混淆矩阵 (confusion_matrix.png)")
        print("- F1 曲线图 (F1_curve.png)")
        print("- 指标摘要 (metrics_summary.txt)")


if __name__ == "__main__":
    main()
