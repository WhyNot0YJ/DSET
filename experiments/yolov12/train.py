#!/usr/bin/env python3
"""
YOLOv12训练脚本 - 支持DAIR-V2X数据集
"""

import sys
import os
import argparse
import yaml
import torch
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import polars as pl
# 添加项目路径
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root.parent) not in sys.path:
    sys.path.insert(0, str(project_root.parent))

# Debug: Check environment
try:
    import ultralytics
except ImportError:
    # Fallback: Attempt to use yolov8's ultralytics if local one fails
    yolov8_path = project_root.parent / "yolov8"
    if yolov8_path.exists() and str(yolov8_path) not in sys.path:
         print(f"Warning: Local ultralytics not found, attempting to use {yolov8_path}")
         sys.path.insert(0, str(yolov8_path))

# 导入ultralytics（本地副本）
try:
    from ultralytics import YOLO
except ImportError as e:
    print(f"Critical Error: Could not import ultralytics. Search paths: {sys.path}")
    # Check if local directory exists
    local_lib = project_root / "ultralytics"
    print(f"Local ultralytics dir exists: {local_lib.exists()}")
    if local_lib.exists():
         print(f"Local ultralytics contents: {list(local_lib.glob('*'))}")
    raise e


# DAIR-V2X类别定义（8类）
CLASS_NAMES = [
    "Car", "Truck", "Van", "Bus", "Pedestrian", 
    "Cyclist", "Motorcyclist", "Trafficcone"
]


class YOLOv12Trainer:
    """YOLOv12训练器 - 适配DAIR-V2X数据集"""
    
    def __init__(self, config: Dict, config_path: Optional[str] = None):
        """
        初始化训练器
        
        Args:
            config: 配置字典
            config_path: 配置文件路径（用于保存）
        """
        self.config = config
        self.config_path = config_path
        
        # 设置日志
        self.setup_logging()
        
        # 验证配置
        self._validate_config()
        
        # 获取配置参数
        self.model_config = config.get('model', {})
        self.training_config = config.get('training', {})
        self.data_config = config.get('data', {})
        self.checkpoint_config = config.get('checkpoint', {})
        self.misc_config = config.get('misc', {})
        
        # 类别信息
        self.class_names = CLASS_NAMES
        self.num_classes = len(CLASS_NAMES)
        
        self.logger.info(f"✓ 初始化YOLOv12训练器")
        self.logger.info(f"  类别数量: {self.num_classes}")
        self.logger.info(f"  类别: {', '.join(self.class_names)}")
    
    def _validate_config(self):
        """验证配置文件"""
        required_keys = {
            'model': ['model_name'],
            'training': ['epochs', 'batch_size'],
            'data': ['data_yaml']
        }
        
        missing_keys = []
        for section, keys in required_keys.items():
            if section not in self.config:
                missing_keys.append(f"缺少配置节: {section}")
                continue
            for key in keys:
                if key not in self.config[section]:
                    missing_keys.append(f"{section}.{key}")
        
        if missing_keys:
            error_msg = f"配置文件缺少必需的配置项:\n"
            error_msg += "\n".join(f"  - {key}" for key in missing_keys)
            raise ValueError(error_msg)
    
    def setup_logging(self):
        """设置日志系统"""
        # 检查是否从检查点恢复
        resume_checkpoint = getattr(self, '_resume_checkpoint_path', None)
        
        if resume_checkpoint and Path(resume_checkpoint).exists():
            # 恢复训练：使用检查点所在目录
            self.log_dir = Path(resume_checkpoint).parent
            self.experiment_name = self.log_dir.name
        else:
            # 新训练：创建带时间戳的目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.config.get('model', {}).get('model_name', 'yolov12n')
            # 去掉.pt后缀（如果存在）
            if model_name.endswith('.pt'):
                model_name = model_name[:-3]
            self.experiment_name = f"yolo_{model_name.replace('yolov8', 'v8').replace('yolov12', 'v12').replace('yolo11', 'v11')}"
            # 直接从config获取，因为checkpoint_config还未初始化
            checkpoint_config = self.config.get('checkpoint', {})
            log_base = checkpoint_config.get('log_dir', 'logs')
            self.log_dir = Path(f"{log_base}/{self.experiment_name}_{timestamp}")
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置日志处理器
        handlers = [
            logging.FileHandler(self.log_dir / 'training.log', mode='a'),
            logging.StreamHandler()
        ]
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers,
            force=True
        )
        
        self.logger = logging.getLogger(__name__)
        
        # 保存配置文件（仅新训练时）
        if not resume_checkpoint:
            config_save_path = self.log_dir / 'config.yaml'
            with open(config_save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            self.logger.info(f"✓ 配置已保存到: {config_save_path}")
    
    def create_model(self):
        """创建YOLO模型"""
        model_name = self.model_config.get('model_name', 'yolov12n.pt')
        pretrained_weights = self.model_config.get('pretrained_weights', None)
        
        # 如果指定了预训练权重，尝试解析路径
        if pretrained_weights:
            # 如果是相对路径，基于配置文件所在目录或项目根目录解析
            pretrained_path = Path(pretrained_weights)
            if not pretrained_path.is_absolute():
                # 尝试相对于配置文件所在目录
                if self.config_path:
                    config_dir = Path(self.config_path).parent
                    pretrained_path = config_dir / pretrained_weights
                # 如果还是不存在，尝试相对于项目根目录
                if not pretrained_path.exists():
                    project_root = Path(__file__).parent.resolve()
                    pretrained_path = project_root / pretrained_weights
            
            if pretrained_path.exists():
                self.logger.info(f"✓ 加载预训练权重: {pretrained_path}")
                model = YOLO(str(pretrained_path))
            else:
                self.logger.warning(f"⚠️  预训练权重文件不存在: {pretrained_path}")
                self.logger.info(f"   将使用模型名称自动加载: {model_name}")
                model = YOLO(model_name)
        else:
            self.logger.info(f"✓ 创建模型: {model_name}")
            model = YOLO(model_name)
        
        # YOLO模型在训练时会自动从data.yaml读取类别数并调整模型
        # 这里我们只需要确保data.yaml中的类别数正确即可
        # YOLO的train()方法会自动处理类别数的修改
        self.logger.info(f"  模型将在训练时自动适配 {self.num_classes} 类（从data.yaml读取）")
        
        return model
    
    def start_training(self, resume_checkpoint: Optional[str] = None, epochs_override: Optional[int] = None):
        """开始训练
        
        Args:
            resume_checkpoint: 恢复训练的检查点路径
            epochs_override: 覆盖配置文件中的epochs（用于测试模式）
        """
        self._resume_checkpoint_path = resume_checkpoint
        
        # 设置日志（需要在设置resume_checkpoint之后）
        self.setup_logging()
        
        self.logger.info("=" * 80)
        self.logger.info("🚀 开始YOLOv12训练")
        self.logger.info("=" * 80)
        
        # 创建模型
        model = self.create_model()
        
        # 获取训练参数（如果提供了epochs_override，则使用它）
        config_epochs = self.training_config.get('epochs', 100)
        epochs = epochs_override if epochs_override is not None else config_epochs
        
        # 如果使用了epochs覆盖，显示提示
        if epochs_override is not None:
            self.logger.info(f"⚠️  测试模式：使用命令行参数覆盖epochs ({config_epochs} → {epochs_override})")
        
        batch_size = self.training_config.get('batch_size', 16)
        imgsz = self.training_config.get('imgsz', 640)
        device = self.misc_config.get('device', 'cuda')
        workers = self.misc_config.get('num_workers', 8)
        
        # 数据配置
        data_yaml = self.data_config.get('data_yaml')
        if not Path(data_yaml).exists():
            raise FileNotFoundError(f"数据配置文件不存在: {data_yaml}")
        
        # 训练参数
        # 注意：ultralytics会在project/name目录下创建训练结果
        # 我们设置project为log_dir的父目录，name为log_dir的名称，这样ultralytics的结果会保存在我们的log_dir下
        train_kwargs = {
            'data': str(data_yaml),
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': imgsz,
            'device': device,
            'workers': workers,
            'project': str(self.log_dir.parent),  # 例如: "logs"
            'name': self.log_dir.name,  # 例如: "yolo_v8l_20251117_120940" (使用完整目录名，而不是experiment_name)
            'exist_ok': True,
            'plots': True,
            'save': True,
            'save_period': self.training_config.get('save_period', 10),
            'val': True,
            'verbose': True,  # 启用详细输出，显示训练过程
        }
        
        # 优化器配置（与RT-DETR对齐）
        if 'optimizer' in self.training_config:
            train_kwargs['optimizer'] = self.training_config['optimizer']
        
        # 学习率配置
        if 'lr0' in self.training_config:
            train_kwargs['lr0'] = self.training_config['lr0']
        if 'lrf' in self.training_config:
            train_kwargs['lrf'] = self.training_config['lrf']
        if 'momentum' in self.training_config:
            train_kwargs['momentum'] = self.training_config['momentum']
        if 'weight_decay' in self.training_config:
            train_kwargs['weight_decay'] = self.training_config['weight_decay']
        if 'warmup_epochs' in self.training_config:
            train_kwargs['warmup_epochs'] = self.training_config['warmup_epochs']
        if 'warmup_momentum' in self.training_config:
            train_kwargs['warmup_momentum'] = self.training_config['warmup_momentum']
        if 'warmup_bias_lr' in self.training_config:
            train_kwargs['warmup_bias_lr'] = self.training_config['warmup_bias_lr']
        
        # 学习率调度器配置（与RT-DETR对齐）
        if 'cos_lr' in self.training_config:
            train_kwargs['cos_lr'] = self.training_config['cos_lr']
        
        # 随机种子和确定性（与RT-DETR对齐）
        if 'seed' in self.training_config:
            train_kwargs['seed'] = self.training_config['seed']
        if 'deterministic' in self.training_config:
            train_kwargs['deterministic'] = self.training_config['deterministic']
        
        # Early Stopping配置（与RT-DETR对齐）
        if 'patience' in self.training_config:
            train_kwargs['patience'] = self.training_config['patience']
        
        # 颜色增强
        if 'hsv_h' in self.training_config:
            train_kwargs['hsv_h'] = self.training_config['hsv_h']
        if 'hsv_s' in self.training_config:
            train_kwargs['hsv_s'] = self.training_config['hsv_s']
        if 'hsv_v' in self.training_config:
            train_kwargs['hsv_v'] = self.training_config['hsv_v']
            
        # 几何增强
        if 'degrees' in self.training_config:
            train_kwargs['degrees'] = self.training_config['degrees']
        if 'translate' in self.training_config:
            train_kwargs['translate'] = self.training_config['translate']
        if 'scale' in self.training_config:
            train_kwargs['scale'] = self.training_config['scale']
        if 'flipud' in self.training_config:
            train_kwargs['flipud'] = self.training_config['flipud']
        if 'fliplr' in self.training_config:
            train_kwargs['fliplr'] = self.training_config['fliplr']
        if 'mosaic' in self.training_config:
            train_kwargs['mosaic'] = self.training_config['mosaic']
        if 'mixup' in self.training_config:
            train_kwargs['mixup'] = self.training_config['mixup']
        
        # 检测框数量限制（默认300）
        if 'max_det' in self.training_config:
            train_kwargs['max_det'] = self.training_config['max_det']
            self.logger.info(f"  检测框数量限制: {self.training_config['max_det']}")
        
        # 恢复训练
        if resume_checkpoint and Path(resume_checkpoint).exists():
            self.logger.info(f"📦 从检查点恢复训练: {resume_checkpoint}")
            train_kwargs['resume'] = True
            # YOLO的resume参数可以是True或检查点路径
            if Path(resume_checkpoint).is_file():
                train_kwargs['resume'] = str(resume_checkpoint)
        
        # 显示关键配置信息（与RT-DETR对齐的格式）
        self.logger.info("📝 训练配置:")
        self.logger.info(f"  数据集路径: {data_yaml}")
        self.logger.info(f"  训练轮数: {epochs}")
        self.logger.info(f"  批次大小: {batch_size}")
        self.logger.info(f"  优化器: {self.training_config.get('optimizer', 'auto')}")
        self.logger.info(f"  初始学习率: {self.training_config.get('lr0', 0.01)}")
        self.logger.info(f"  Weight decay: {self.training_config.get('weight_decay', 0.0001)}")
        self.logger.info(f"  输出目录: {self.log_dir}")
        pretrained_weights_display = self.model_config.get('pretrained_weights', None)
        if pretrained_weights_display:
            self.logger.info(f"  预训练权重: {pretrained_weights_display}")
        if resume_checkpoint:
            self.logger.info(f"  恢复检查点: {resume_checkpoint}")
        self.logger.info("=" * 80)
        
        # 训练配置摘要（与RT-DETR对齐）
        self.logger.info("训练配置摘要:")
        self.logger.info(f"  - 训练轮数: {epochs}")
        self.logger.info(f"  - 批次大小: {batch_size}")
        self.logger.info(f"  - 优化器: {self.training_config.get('optimizer', 'auto')}")
        self.logger.info(f"  - 初始学习率: {self.training_config.get('lr0', 0.01)}")
        self.logger.info(f"  - Weight decay: {self.training_config.get('weight_decay', 0.0001)}")
        self.logger.info(f"  - Warmup轮数: {self.training_config.get('warmup_epochs', 3.0)}")
        self.logger.info(f"  - 设备: {device}")
        self.logger.info("=" * 80)
        
        # 开始训练
        self.logger.info(f"开始训练 {epochs} epochs")
        try:
            results = model.train(**train_kwargs)
            
            # 训练完成后，解析结果并按照RT-DETR格式输出
            self.logger.info("=" * 80)
            self.logger.info("✅ 训练完成！")
            self.logger.info("=" * 80)
            
            # 解析并打印训练结果（从results.csv读取）
            self._parse_and_print_training_results()
            
            # 生成与RT-DETR一致的训练曲线图
            self._plot_training_curves()
            
            # 统一文件命名格式（与RT-DETR对齐）
            self._align_file_naming()
            
            # 打印最佳模型路径（统一格式）
            best_model_path = self.log_dir / "best_model.pth"
            if best_model_path.exists():
                self.logger.info(f"✓ 最佳模型: {best_model_path}")
            else:
                # 如果统一后的文件不存在，检查原始位置
                original_best = self.log_dir / "weights" / "best.pt"
                if original_best.exists():
                    self.logger.info(f"✓ 最佳模型（原始位置）: {original_best}")
            
            # 尝试从results中提取最佳指标（如果ultralytics返回了这些信息）
            if hasattr(results, 'results_dict'):
                results_dict = results.results_dict
                if 'metrics/mAP50-95(B)' in results_dict:
                    best_map = results_dict['metrics/mAP50-95(B)']
                    self.logger.info(f"✓ 最佳mAP@0.5:0.95: {best_map:.4f}")
                if 'metrics/mAP50(B)' in results_dict:
                    best_map50 = results_dict['metrics/mAP50(B)']
                    self.logger.info(f"✓ 最佳mAP@0.5: {best_map50:.4f}")
            
            self.logger.info(f"✓ 所有输出已保存到: {self.log_dir}")
            self.logger.info("=" * 80)
            
            return results
        except Exception as e:
            self.logger.error(f"训练失败: {e}")
            raise
    
    def _parse_and_print_training_results(self):
        """解析ultralytics的results.csv并按照RT-DETR格式重新打印"""
        try:
            # ultralytics会在project/name目录下生成results.csv
            # 根据train_kwargs的设置，应该是self.log_dir/results.csv
            results_csv = self.log_dir / "results.csv"
            if not results_csv.exists():
                self.logger.warning(f"未找到results.csv文件: {results_csv}")
                return
            
            # 读取CSV文件
            df = pd.read_csv(results_csv)
            
            # 提取关键列（ultralytics的列名）
            # 计算总损失：train/box_loss + train/cls_loss + train/dfl_loss
            train_loss_cols = []
            val_loss_cols = []
            map50_col = None
            map50_95_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'train/box_loss' in col_lower or 'train/cls_loss' in col_lower or 'train/dfl_loss' in col_lower:
                    train_loss_cols.append(col)
                elif 'val/box_loss' in col_lower or 'val/cls_loss' in col_lower or 'val/dfl_loss' in col_lower:
                    val_loss_cols.append(col)
                elif 'metrics/map50(b)' in col_lower and map50_col is None:
                    map50_col = col
                elif 'metrics/map50-95(b)' in col_lower and map50_95_col is None:
                    map50_95_col = col
            
            # 计算总损失
            if train_loss_cols:
                df['train_loss'] = df[train_loss_cols].sum(axis=1)
            else:
                df['train_loss'] = 0.0
                
            if val_loss_cols:
                df['val_loss'] = df[val_loss_cols].sum(axis=1)
            else:
                df['val_loss'] = 0.0
            
            # 按照RT-DETR格式打印每个epoch的结果
            self.logger.info("=" * 80)
            self.logger.info("训练过程摘要（按RT-DETR格式）:")
            self.logger.info("=" * 80)
            
            for idx, row in df.iterrows():
                epoch = int(row.get('epoch', idx + 1))
                train_loss = row.get('train_loss', 0.0)
                val_loss = row.get('val_loss', 0.0)
                
                # 按照RT-DETR格式打印
                self.logger.info(f"Epoch {epoch}:")
                self.logger.info(f"  训练损失: {train_loss:.2f} | 验证损失: {val_loss:.2f}")
                
                # 如果有mAP信息，也打印
                if map50_col and not pd.isna(row.get(map50_col)):
                    map50 = row.get(map50_col, 0.0)
                    self.logger.info(f"  mAP@0.5: {map50:.4f}")
                if map50_95_col and not pd.isna(row.get(map50_95_col)):
                    map50_95 = row.get(map50_95_col, 0.0)
                    self.logger.info(f"  mAP@0.5:0.95: {map50_95:.4f}")
            
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.warning(f"解析训练结果失败: {e}")
    
    def _plot_training_curves(self):
        """生成与RT-DETR一致的训练曲线图"""
        try:
            results_csv = self.log_dir / "results.csv"
            if not results_csv.exists():
                self.logger.warning(f"未找到results.csv文件: {results_csv}")
                return
            
            # 读取CSV文件
            df = pd.read_csv(results_csv)
            
            # 提取数据
            epochs = df.get('epoch', range(1, len(df) + 1)).values
            
            # 计算总损失
            train_loss_cols = []
            val_loss_cols = []
            for col in df.columns:
                col_lower = col.lower()
                if 'train/box_loss' in col_lower or 'train/cls_loss' in col_lower or 'train/dfl_loss' in col_lower:
                    train_loss_cols.append(col)
                elif 'val/box_loss' in col_lower or 'val/cls_loss' in col_lower or 'val/dfl_loss' in col_lower:
                    val_loss_cols.append(col)
            
            train_loss = df[train_loss_cols].sum(axis=1).values if train_loss_cols else None
            val_loss = df[val_loss_cols].sum(axis=1).values if val_loss_cols else None
            
            # 提取mAP指标
            map50 = None
            map50_95 = None
            for col in df.columns:
                col_lower = col.lower()
                if 'metrics/map50(b)' in col_lower and map50 is None:
                    map50 = df[col].values
                elif 'metrics/map50-95(b)' in col_lower and map50_95 is None:
                    map50_95 = df[col].values
            
            # 提取学习率
            lr = None
            for col in df.columns:
                if 'lr' in col.lower() or 'learning_rate' in col.lower():
                    lr = df[col].values
                    break
            
            # 创建与RT-DETR一致的训练曲线图
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            title = 'YOLOv12 Training Curves'
            fig.suptitle(title, fontsize=16, fontweight='bold')
            
            # 1. 损失曲线
            ax = axes[0]
            if train_loss is not None:
                ax.plot(epochs, train_loss, 'b-o', 
                        label='Train Loss', linewidth=2, markersize=4)
            if val_loss is not None:
                ax.plot(epochs, val_loss, 'r-s', 
                        label='Val Loss', linewidth=2, markersize=4)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('Loss Curves', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # 2. mAP曲线
            ax = axes[1]
            if map50 is not None:
                ax.plot(epochs, map50, 'g-^', 
                        label='mAP@0.5', linewidth=2, markersize=4)
            if map50_95 is not None:
                ax.plot(epochs, map50_95, 'm-d', 
                        label='mAP@[0.5:0.95]', linewidth=2, markersize=4)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('mAP', fontsize=12)
            ax.set_title('mAP Metrics', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # 3. 学习率曲线
            ax = axes[2]
            if lr is not None:
                ax.plot(epochs, lr, 'orange', linewidth=2)
                ax.set_yscale('log')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Learning Rate', fontsize=12)
            ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = self.log_dir / 'training_curves.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✓ 训练曲线已保存到: {save_path}")
            
        except Exception as e:
            self.logger.warning(f"绘制训练曲线失败: {e}")
    
    def _align_file_naming(self):
        """统一文件命名格式，与RT-DETR对齐"""
        try:
            import shutil
            
            # 1. 将best.pt复制为best_model.pth（统一最佳模型命名）
            original_best = self.log_dir / "weights" / "best.pt"
            aligned_best = self.log_dir / "best_model.pth"
            if original_best.exists() and not aligned_best.exists():
                shutil.copy2(original_best, aligned_best)
                self.logger.info(f"✓ 已创建统一命名的最佳模型: {aligned_best}")
            
            # 2. 将last.pt复制为latest_checkpoint.pth（统一检查点命名）
            original_last = self.log_dir / "weights" / "last.pt"
            aligned_last = self.log_dir / "latest_checkpoint.pth"
            if original_last.exists() and not aligned_last.exists():
                shutil.copy2(original_last, aligned_last)
                self.logger.info(f"✓ 已创建统一命名的最新检查点: {aligned_last}")
            
            # 3. 将results.csv复制为training_history.csv（统一训练历史命名）
            original_results = self.log_dir / "results.csv"
            aligned_history = self.log_dir / "training_history.csv"
            if original_results.exists() and not aligned_history.exists():
                shutil.copy2(original_results, aligned_history)
                self.logger.info(f"✓ 已创建统一命名的训练历史: {aligned_history}")
            
        except Exception as e:
            self.logger.warning(f"统一文件命名失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOv12训练脚本')
    parser.add_argument('--config', type=str, required=True,
                       help='YAML配置文件路径')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='从检查点恢复训练（检查点文件路径）')
    parser.add_argument('--resume', action='store_true',
                       help='自动从最新检查点恢复训练')
    parser.add_argument('--epochs', type=int, default=None,
                       help='覆盖配置文件中的epochs（用于测试模式）')
    
    args = parser.parse_args()
    
    # 加载配置
    if not Path(args.config).exists():
        raise FileNotFoundError(f"配置文件不存在: {args.config}")
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 如果启用自动恢复，查找最新的检查点
    if args.resume and not args.resume_from_checkpoint:
        log_base = config.get('checkpoint', {}).get('log_dir', 'logs')
        log_dir = Path(log_base)
        if log_dir.exists():
            # 查找所有包含weights/best.pt的目录
            checkpoints = list(log_dir.glob("*/weights/best.pt"))
            if checkpoints:
                # 按修改时间排序，取最新的
                latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
                args.resume_from_checkpoint = str(latest_checkpoint)
                print(f"📦 找到最新检查点: {args.resume_from_checkpoint}")
    
    # 创建训练器
    trainer = YOLOv12Trainer(config, config_path=args.config)
    
    # 开始训练（如果提供了--epochs参数，则覆盖配置文件中的值）
    trainer.start_training(resume_checkpoint=args.resume_from_checkpoint, epochs_override=args.epochs)


if __name__ == '__main__':
    main()

