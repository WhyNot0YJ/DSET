#!/usr/bin/env python3
"""
YOLOv8训练脚本 - 支持DAIR-V2X数据集
"""

import sys
import os
import argparse
import yaml
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

# 添加项目路径
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root.parent) not in sys.path:
    sys.path.insert(0, str(project_root.parent))

# 导入ultralytics（本地副本）
from ultralytics import YOLO

# DAIR-V2X类别定义（10类）
CLASS_NAMES = [
    "Car", "Truck", "Van", "Bus", "Pedestrian", 
    "Cyclist", "Tricyclist", "Motorcyclist", "Barrowlist", "Trafficcone"
]


class YOLOv8Trainer:
    """YOLOv8训练器 - 适配DAIR-V2X数据集"""
    
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
        
        self.logger.info(f"✓ 初始化YOLOv8训练器")
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
            model_name = self.config.get('model', {}).get('model_name', 'yolov8n')
            self.experiment_name = f"yolo_{model_name.replace('yolov8', 'v8').replace('yolo11', 'v11')}"
            log_base = self.checkpoint_config.get('log_dir', 'logs')
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
        model_name = self.model_config.get('model_name', 'yolov8n.pt')
        pretrained_weights = self.model_config.get('pretrained_weights', None)
        
        # 如果指定了预训练权重，使用它；否则使用模型名称
        if pretrained_weights and Path(pretrained_weights).exists():
            self.logger.info(f"✓ 加载预训练权重: {pretrained_weights}")
            model = YOLO(pretrained_weights)
        else:
            self.logger.info(f"✓ 创建模型: {model_name}")
            model = YOLO(model_name)
        
        # YOLO模型在训练时会自动从data.yaml读取类别数并调整模型
        # 这里我们只需要确保data.yaml中的类别数正确即可
        # YOLO的train()方法会自动处理类别数的修改
        self.logger.info(f"  模型将在训练时自动适配 {self.num_classes} 类（从data.yaml读取）")
        
        return model
    
    def start_training(self, resume_checkpoint: Optional[str] = None):
        """开始训练"""
        self._resume_checkpoint_path = resume_checkpoint
        
        # 设置日志（需要在设置resume_checkpoint之后）
        self.setup_logging()
        
        self.logger.info("="*60)
        self.logger.info("🚀 开始YOLOv8训练")
        self.logger.info("="*60)
        
        # 创建模型
        model = self.create_model()
        
        # 获取训练参数
        epochs = self.training_config.get('epochs', 100)
        batch_size = self.training_config.get('batch_size', 16)
        imgsz = self.training_config.get('imgsz', 640)
        device = self.misc_config.get('device', 'cuda')
        workers = self.misc_config.get('num_workers', 8)
        
        # 数据配置
        data_yaml = self.data_config.get('data_yaml')
        if not Path(data_yaml).exists():
            raise FileNotFoundError(f"数据配置文件不存在: {data_yaml}")
        
        # 训练参数
        train_kwargs = {
            'data': str(data_yaml),
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': imgsz,
            'device': device,
            'workers': workers,
            'project': str(self.log_dir.parent),
            'name': self.experiment_name,
            'exist_ok': True,
            'plots': True,
            'save': True,
            'save_period': self.training_config.get('save_period', 10),
            'val': True,
        }
        
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
        
        # 数据增强配置
        if 'hsv_h' in self.training_config:
            train_kwargs['hsv_h'] = self.training_config['hsv_h']
        if 'hsv_s' in self.training_config:
            train_kwargs['hsv_s'] = self.training_config['hsv_s']
        if 'hsv_v' in self.training_config:
            train_kwargs['hsv_v'] = self.training_config['hsv_v']
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
        
        # 恢复训练
        if resume_checkpoint and Path(resume_checkpoint).exists():
            self.logger.info(f"📦 从检查点恢复训练: {resume_checkpoint}")
            train_kwargs['resume'] = True
            # YOLO的resume参数可以是True或检查点路径
            if Path(resume_checkpoint).is_file():
                train_kwargs['resume'] = str(resume_checkpoint)
        
        self.logger.info(f"训练参数:")
        self.logger.info(f"  数据配置: {data_yaml}")
        self.logger.info(f"  训练轮数: {epochs}")
        self.logger.info(f"  批次大小: {batch_size}")
        self.logger.info(f"  图像尺寸: {imgsz}")
        self.logger.info(f"  设备: {device}")
        self.logger.info(f"  工作进程: {workers}")
        self.logger.info(f"  日志目录: {self.log_dir}")
        
        # 开始训练
        try:
            results = model.train(**train_kwargs)
            self.logger.info("="*60)
            self.logger.info("✅ 训练完成！")
            self.logger.info("="*60)
            
            # 打印最佳模型路径
            best_model_path = self.log_dir / "weights" / "best.pt"
            if best_model_path.exists():
                self.logger.info(f"最佳模型: {best_model_path}")
            
            return results
        except Exception as e:
            self.logger.error(f"训练失败: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOv8训练脚本')
    parser.add_argument('--config', type=str, required=True,
                       help='YAML配置文件路径')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='从检查点恢复训练（检查点文件路径）')
    parser.add_argument('--resume', action='store_true',
                       help='自动从最新检查点恢复训练')
    
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
    trainer = YOLOv8Trainer(config, config_path=args.config)
    
    # 开始训练
    trainer.start_training(resume_checkpoint=args.resume_from_checkpoint)


if __name__ == '__main__':
    main()

