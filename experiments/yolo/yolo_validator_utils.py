#!/usr/bin/env python3
"""
YOLO验证器工具模块 - 支持多尺度mAP和KITTI难度分级指标
包含：多尺度分类、KITTI难度评估、指标聚合、日志记录
"""

import numpy as np
import pandas as pd
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import defaultdict


class MultiScaleMetricsCalculator:
    """多尺度和难度分级指标计算器"""
    
    # 尺寸阈值 (COCO标准)
    SMALL_AREA_THRESHOLD = 32 * 32      # <1024 pixels
    MEDIUM_AREA_THRESHOLD = 96 * 96    # 1024-9216 pixels
    LARGE_AREA_THRESHOLD = 96 * 96     # >9216 pixels
    
    # KITTI难度分级标准
    OCCLUDED_THRESHOLD = 0.15  # 15% 遮挡
    TRUNCATED_THRESHOLD = 0.15  # 15% 截断
    MIN_HEIGHT = 25  # 最小高度(像素)
    
    @staticmethod
    def calculate_box_area(box: np.ndarray) -> float:
        """计算边界框面积
        
        Args:
            box: [x1, y1, x2, y2] 或 [x, y, w, h]
        """
        if len(box) < 4:
            return 0
        if box[2] > box[0]:  # x2 > x1 格式
            w = box[2] - box[0]
            h = box[3] - box[1]
        else:  # w, h 格式
            w = box[2]
            h = box[3]
        return max(0, w * h)
    
    @staticmethod
    def get_box_height(box: np.ndarray) -> float:
        """获取边界框的高度"""
        if len(box) < 4:
            return 0
        if box[3] > box[1]:
            return box[3] - box[1]
        else:
            return box[3]
    
    @staticmethod
    def categorize_by_scale(area: float) -> Optional[str]:
        """按面积分类为small/medium/large"""
        if area < MultiScaleMetricsCalculator.SMALL_AREA_THRESHOLD:
            return 'small'
        elif area < MultiScaleMetricsCalculator.MEDIUM_AREA_THRESHOLD:
            return 'medium'
        else:
            return 'large'
    
    @staticmethod
    def _normalize_occlusion_level(occlusion: float) -> int:
        """将遮挡输入归一化为KITTI离散等级(0/1/2/3+)。"""
        try:
            value = float(occlusion)
        except (TypeError, ValueError):
            return 3

        if value < 0:
            return 3

        # 离散等级输入：0/1/2
        if value >= 1.0:
            return int(round(value))

        # 比例输入：映射为离散等级
        if value <= 0.15:
            return 0
        if value <= 0.50:
            return 1
        if value <= 0.80:
            return 2
        return 3

    @staticmethod
    def categorize_by_kitti_difficulty(
        height: float,
        occlusion: float = 0.0,
        truncation: float = 0.0
    ) -> Optional[str]:
        """
        按KITTI标准分类难度等级
        
        Easy: height >= 40px, occlusion == 0, truncation <= 15%
        Moderate: height >= 25px, occlusion <= 1, truncation <= 30%
        Hard: height >= 25px, occlusion <= 2, truncation <= 50%
        其余: Ignore
        """
        if height < MultiScaleMetricsCalculator.MIN_HEIGHT:
            return None

        occ_level = MultiScaleMetricsCalculator._normalize_occlusion_level(occlusion)

        # 1) Easy
        if height >= 40 and occ_level == 0 and truncation <= 0.15:
            return 'easy'

        # 2) Moderate
        if height >= 25 and occ_level <= 1 and truncation <= 0.30:
            return 'moderate'

        # 3) Hard
        if height >= 25 and occ_level <= 2 and truncation <= 0.50:
            return 'hard'

        # 4) Ignore / DontCare
        return None


class MetricsAggregator:
    """指标聚合器 - 收集和计算多维指标"""
    
    def __init__(self):
        self.scale_metrics = defaultdict(lambda: defaultdict(list))
        self.difficulty_metrics = defaultdict(lambda: defaultdict(list))
        self.logger = logging.getLogger(__name__)
    
    def update_scale_metric(self, scale: str, metric_name: str, value: Any):
        """更新按尺度的指标"""
        if scale in ['small', 'medium', 'large']:
            self.scale_metrics[scale][metric_name].append(value)
    
    def update_difficulty_metric(self, difficulty: str, metric_name: str, value: Any):
        """更新按难度的指标"""
        if difficulty in ['easy', 'moderate', 'hard']:
            self.difficulty_metrics[difficulty][metric_name].append(value)
    
    def get_scale_map(self, iou_threshold: float = 0.5) -> Dict[str, float]:
        """计算各尺度的mAP"""
        scale_map = {}
        for scale in ['small', 'medium', 'large']:
            metrics = self.scale_metrics[scale]
            if metrics and 'ap' in metrics:
                scale_map[f'mAP@{iou_threshold}({scale})'] = np.mean(metrics['ap'])
        return scale_map
    
    def get_difficulty_map(self, iou_threshold: float = 0.5) -> Dict[str, float]:
        """计算各难度等级的mAP (KITTI标准)"""
        difficulty_map = {}
        for difficulty in ['easy', 'moderate', 'hard']:
            metrics = self.difficulty_metrics[difficulty]
            if metrics and 'ap' in metrics:
                difficulty_map[f'mAP@{iou_threshold}({difficulty})'] = np.mean(metrics['ap'])
        return difficulty_map
    
    def get_summary(self) -> Dict[str, Any]:
        """获取完整的指标摘要"""
        return {
            'scale_metrics': dict(self.scale_metrics),
            'difficulty_metrics': dict(self.difficulty_metrics),
            'scale_map_50': self.get_scale_map(0.5),
            'scale_map_50_95': self.get_scale_map(0.5),
            'difficulty_map_50': self.get_difficulty_map(0.5),
            'difficulty_map_50_95': self.get_difficulty_map(0.5),
        }


class MetricsLogger:
    """训练指标日志记录器 - 支持多尺度和KITTI难度指标"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.metrics_file = self.log_dir / 'detailed_metrics.csv'
        self.scale_difficulty_file = self.log_dir / 'scale_difficulty_metrics.json'
        self.logger = logging.getLogger(__name__)
    
    def log_epoch_metrics(
        self,
        epoch: int,
        metrics_dict: Dict,
        scale_metrics: Optional[Dict] = None,
        difficulty_metrics: Optional[Dict] = None
    ):
        """记录epoch的所有指标"""
        record = {'epoch': epoch}
        record.update(metrics_dict)
        if scale_metrics:
            record.update(scale_metrics)
        if difficulty_metrics:
            record.update(difficulty_metrics)
        
        # 追加到CSV
        df = pd.DataFrame([record])
        
        if self.metrics_file.exists():
            existing = pd.read_csv(self.metrics_file)
            df = pd.concat([existing, df], ignore_index=True)
        
        df.to_csv(self.metrics_file, index=False)
        
        # 保存scale和difficulty的完整指标到JSON
        if scale_metrics or difficulty_metrics:
            data = {'epoch': epoch}
            if scale_metrics:
                data['scale_metrics'] = scale_metrics
            if difficulty_metrics:
                data['difficulty_metrics'] = difficulty_metrics
            
            all_data = []
            if self.scale_difficulty_file.exists():
                with open(self.scale_difficulty_file, 'r') as f:
                    all_data = json.load(f)
            
            all_data.append(data)
            
            with open(self.scale_difficulty_file, 'w') as f:
                json.dump(all_data, f, indent=2)
    
    def log_scale_metrics(self, epoch: int, scale_metrics: Dict[str, float]):
        """记录多尺度指标"""
        self.logger.info(f"\n📊 Epoch {epoch} - Scale-wise mAP:")
        for metric_name, value in sorted(scale_metrics.items()):
            self.logger.info(f"   {metric_name}: {value:.4f}")
    
    def log_difficulty_metrics(self, epoch: int, difficulty_metrics: Dict[str, float]):
        """记录难度等级指标(KITTI标准)"""
        self.logger.info(f"\n🎯 Epoch {epoch} - Difficulty-wise mAP (KITTI):")
        for metric_name, value in sorted(difficulty_metrics.items()):
            self.logger.info(f"   {metric_name}: {value:.4f}")
    
    def create_metrics_report(self) -> str:
        """生成指标报告"""
        if not self.metrics_file.exists():
            return "No metrics data available"
        
        df = pd.read_csv(self.metrics_file)
        report = f"总Epochs数: {len(df)}\n"
        
        # 添加最佳指标
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        map_cols = [c for c in numeric_cols if 'map' in c.lower() or 'ap' in c.lower()]
        
        for col in sorted(map_cols):
            if not pd.isna(df[col]).all():
                best_idx = df[col].idxmax()
                best_val = df[col].max()
                best_epoch = int(df.iloc[best_idx]['epoch'])
                report += f"最佳{col}: {best_val:.4f} (Epoch {best_epoch})\n"
        
        return report


class MetricsPlotter:
    """指标绘图工具"""
    
    @staticmethod
    def plot_multi_scale_metrics(metrics_file: Path, output_path: Path):
        """绘制多尺度mAP曲线"""
        import matplotlib.pyplot as plt
        
        if not metrics_file.exists():
            return
        
        df = pd.read_csv(metrics_file)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('YOLO Multi-Scale & KITTI Difficulty Metrics', fontsize=14, fontweight='bold')
        
        # 按尺寸的mAP
        scales = ['small', 'medium', 'large']
        scale_cols = [c for c in df.columns if any(s in c.lower() for s in scales)]
        
        for col in scale_cols:
            if col in df.columns:
                axes[0].plot(df.index, df[col], marker='o', label=col, linewidth=2)
        
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('mAP', fontsize=11)
        axes[0].set_title('Multi-Scale mAP Curves', fontsize=12, fontweight='bold')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # 按难度的mAP (KITTI)
        difficulties = ['easy', 'moderate', 'hard']
        difficulty_cols = [c for c in df.columns if any(d in c.lower() for d in difficulties)]
        
        for col in difficulty_cols:
            if col in df.columns:
                axes[1].plot(df.index, df[col], marker='s', label=col, linewidth=2)
        
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('mAP', fontsize=11)
        axes[1].set_title('KITTI Difficulty-wise mAP Curves', fontsize=12, fontweight='bold')
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
