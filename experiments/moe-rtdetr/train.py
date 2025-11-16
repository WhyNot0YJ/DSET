#!/usr/bin/env python3
"""自适应专家RT-DETR训练脚本 - DAIR-V2X数据集

细粒度MoE架构：Decoder FFN层集成自适应专家层

主要特性：
- 细粒度MoE：每个Decoder层的FFN使用自适应专家层
- 支持多种backbone架构
- 混合精度训练
- EMA模型
- 学习率预热
- COCO格式评估
- 检查点恢复
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import re
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from datetime import datetime
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

# 添加项目路径
project_root = Path(__file__).parent.resolve()
# 确保当前工作目录在路径中（重要：当从不同目录运行时）
if str(os.getcwd()) not in sys.path:
    sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))  # 添加experiments目录

# 导入随机种子工具
from seed_utils import set_seed, seed_worker

# 导入自定义模块（与 dset 保持完全相同的导入顺序）
from src.misc.training_visualizer import TrainingVisualizer
from src.misc.early_stopping import EarlyStopping

# 先导入其他模块，最后导入 data 相关模块（避免循环导入）
# 导入RT-DETR组件
from src.zoo.rtdetr import HybridEncoder, RTDETRTransformerv2, RTDETRCriterionv2, HungarianMatcher
from src.nn.backbone.presnet import PResNet
from src.nn.backbone.hgnetv2 import HGNetv2
from src.nn.backbone.csp_resnet import CSPResNet
from src.nn.backbone.csp_darknet import CSPDarkNet
from src.nn.backbone.test_resnet import MResNet

# 导入优化器增强模块
from src.optim.ema import ModelEMA
from src.optim.amp import GradScaler
from src.optim.warmup import WarmupLR

# 最后导入 data 相关模块（此时 src 包已完全初始化）
from src.data.dataset.dairv2x_detection import DAIRV2XDetection
from src.nn.postprocessor.detr_postprocessor import DetDETRPostProcessor
from src.nn.postprocessor.box_revert import BoxProcessFormat
import cv2

# 导入 batch_inference 中的函数（确保逻辑一致）
try:
    from batch_inference import postprocess_outputs, draw_boxes, inference_from_preprocessed_image
    USE_BATCH_INFERENCE_LOGIC = True
except ImportError:
    USE_BATCH_INFERENCE_LOGIC = False


def create_backbone(backbone_type: str, **kwargs) -> nn.Module:
    """创建backbone的工厂函数。
    
    Args:
        backbone_type: backbone类型（presnet18/34/50/101, hgnetv2_l等）
        **kwargs: backbone特定参数（会覆盖默认配置）
    
    Returns:
        nn.Module: backbone模型实例
        
    Raises:
        ValueError: 不支持的backbone类型
    """
    # PResNet配置（通过正则表达式解析depth）
    if backbone_type.startswith('presnet'):
        depth_match = re.search(r'(\d+)', backbone_type)
        if depth_match:
            depth = int(depth_match.group(1))
        else:
            raise ValueError(f"无法从backbone类型 {backbone_type} 解析depth")
        
        default_params = {
            'depth': depth,
            'variant': 'd',
            'return_idx': [1, 2, 3],
            'freeze_at': -1,  # moe-rtdetr不使用冻结
            'freeze_norm': False,
            'pretrained': False
        }
        default_params.update(kwargs)
        return PResNet(**default_params)
    
    # HGNetv2配置
    elif backbone_type.startswith('hgnetv2'):
        name_map = {'hgnetv2_l': 'L', 'hgnetv2_x': 'X', 'hgnetv2_h': 'H'}
        if backbone_type not in name_map:
            raise ValueError(f"不支持的HGNetv2类型: {backbone_type}")
        
        default_params = {
            'name': name_map[backbone_type],
            'return_idx': [1, 2, 3],
            'freeze_at': -1,
            'freeze_norm': False,
            'pretrained': False
        }
        default_params.update(kwargs)
        return HGNetv2(**default_params)
    
    # CSPResNet配置
    elif backbone_type.startswith('cspresnet'):
        name_map = {'cspresnet_s': 's', 'cspresnet_m': 'm', 'cspresnet_l': 'l', 'cspresnet_x': 'x'}
        if backbone_type not in name_map:
            raise ValueError(f"不支持的CSPResNet类型: {backbone_type}")
        
        default_params = {
            'name': name_map[backbone_type],
            'return_idx': [1, 2, 3],
            'pretrained': False
        }
        default_params.update(kwargs)
        return CSPResNet(**default_params)
    
    # CSPDarkNet配置
    elif backbone_type == 'cspdarknet':
        default_params = {'return_idx': [2, 3, -1]}
        default_params.update(kwargs)
        return CSPDarkNet(**default_params)
    
    # Modified ResNet
    elif backbone_type == 'mresnet':
        default_params = {'num_blocks': [2, 2, 2, 2]}
        default_params.update(kwargs)
        return MResNet(**default_params)
    
    else:
        raise ValueError(f"不支持的backbone类型: {backbone_type}")






class AdaptiveExpertRTDETR(nn.Module):
    """自适应专家RT-DETR模型（细粒度MoE架构）。
    
    架构设计：
    1. 共享Backbone：提取多尺度特征
    2. 共享Encoder：增强特征表达
    3. 自适应专家Decoder：FFN层使用AdaptiveExpertLayer（每层独立Router + N个专家FFN）
    4. 统一输出：直接输出检测结果，无需额外融合
    """
    
    def __init__(self, config_name: str = "A", hidden_dim: int = 256, 
                 num_queries: int = 300, top_k: int = 2, backbone_type: str = "presnet34",
                 num_decoder_layers: int = 3, encoder_in_channels: list = None, 
                 encoder_expansion: float = 1.0, num_experts: int = None,
                 moe_balance_weight: float = None):
        """初始化自适应专家RT-DETR模型。
        
        Args:
            config_name: 专家配置名称（保留用于兼容性，但不再用于确定专家数量）
            hidden_dim: 隐藏层维度
            num_queries: 查询数量
            top_k: 路由器Top-K选择
            backbone_type: Backbone类型
            num_decoder_layers: Decoder层数
            encoder_in_channels: Encoder输入通道数
            encoder_expansion: Encoder expansion参数
            num_experts: 专家数量（优先使用，如果未提供则通过config_name映射）
            moe_balance_weight: MoE负载均衡损失权重（可选，默认自动调整）
        """
        super().__init__()
        
        self.config_name = config_name
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.top_k = top_k
        self.backbone_type = backbone_type
        self.image_size = 640
        self.num_decoder_layers = num_decoder_layers
        
        # Encoder配置
        self.encoder_in_channels = encoder_in_channels or [512, 1024, 2048]
        self.encoder_expansion = encoder_expansion
        
        # MoE配置：支持自定义权重
        if moe_balance_weight is not None:
            self.moe_balance_weight = moe_balance_weight
        
        # 获取专家数量：优先使用直接传入的num_experts，否则通过config_name映射（向后兼容）
        if num_experts is not None:
            self.num_experts = num_experts
        else:
            configs = {"A": 6, "B": 3, "C": 2}
            self.num_experts = configs.get(config_name, 6)
        
        # ========== 共享组件 ==========
        self.backbone = self._build_backbone()
        self.encoder = self._build_encoder()
        
        # ========== 细粒度MoE Decoder ==========
        # 使用传入的decoder层数参数
        
        self.decoder = RTDETRTransformerv2(
            num_classes=8,  # 8类：Car, Truck, Van, Bus, Pedestrian, Cyclist, Motorcyclist, Trafficcone
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_layers=num_decoder_layers,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation='relu',
            feat_channels=[256, 256, 256],
            feat_strides=[8, 16, 32],
            num_levels=3,
            # 细粒度MoE配置
            use_moe=True,
            num_experts=self.num_experts,
            moe_top_k=top_k
        )
        
        print(f"✓ MoE Decoder配置: {num_decoder_layers}层, {self.num_experts}个专家, top_k={top_k}")
        
        # RT-DETR损失函数
        self.detr_criterion = self._build_detr_criterion()
        
    def _build_backbone(self) -> nn.Module:
        """构建backbone。"""
        return create_backbone(self.backbone_type)
    
    def _build_encoder(self) -> nn.Module:
        """构建encoder - 使用配置参数。"""
        input_size = [self.image_size, self.image_size]
        
        return HybridEncoder(
            in_channels=self.encoder_in_channels,
            feat_strides=[8, 16, 32],
            hidden_dim=256,
            use_encoder_idx=[2],
            num_encoder_layers=1,
            expansion=self.encoder_expansion,
            nhead=8,
            dropout=0.0,
            act='silu',
            eval_spatial_size=input_size
        )
    
    def _build_detr_criterion(self) -> RTDETRCriterionv2:
        """构建RT-DETR损失函数。"""
        matcher = HungarianMatcher(
            weight_dict={'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
            use_focal_loss=False,
            alpha=0.25,
            gamma=2.0
        )
        
        # 主损失权重
        main_weight_dict = {
            'loss_vfl': 1.0,
            'loss_bbox': 5.0,
            'loss_giou': 2.0
        }
        
        num_decoder_layers = self.num_decoder_layers
        aux_weight_dict = {}
        for i in range(num_decoder_layers - 1):  # 前N-1层
            aux_weight_dict[f'loss_vfl_aux_{i}'] = 1.0
            aux_weight_dict[f'loss_bbox_aux_{i}'] = 5.0
            aux_weight_dict[f'loss_giou_aux_{i}'] = 2.0
        
        # Encoder辅助损失（通常1层）
        aux_weight_dict['loss_vfl_enc_0'] = 1.0
        aux_weight_dict['loss_bbox_enc_0'] = 5.0
        aux_weight_dict['loss_giou_enc_0'] = 2.0
        
        # Denoising辅助损失（如果启用num_denoising>0）
        # RT-DETR默认num_denoising=100，我们也需要添加这些损失的权重
        # 使用动态读取的层数
        num_denoising_layers = num_decoder_layers  # 和decoder层数一致
        for i in range(num_denoising_layers):
            aux_weight_dict[f'loss_vfl_dn_{i}'] = 1.0
            aux_weight_dict[f'loss_bbox_dn_{i}'] = 5.0
            aux_weight_dict[f'loss_giou_dn_{i}'] = 2.0
        
        # 合并所有权重
        weight_dict = {**main_weight_dict, **aux_weight_dict}
        
        criterion = RTDETRCriterionv2(
            matcher=matcher,
            weight_dict=weight_dict,
            losses=['vfl', 'boxes'],
            alpha=0.75,
            gamma=2.0,
            num_classes=8,  # 8类：Car, Truck, Van, Bus, Pedestrian, Cyclist, Motorcyclist, Trafficcone
            boxes_weight_format=None,
            share_matched_indices=False
        )
        
        return criterion
    
    
    def forward(self, images: torch.Tensor, 
                targets: Optional[List[Dict]] = None) -> Dict[str, torch.Tensor]:
        """前向传播。
        
        Args:
            images: [B, C, H, W] 输入图像
            targets: 训练目标列表（可选）
        
        Returns:
            Dict: 包含检测结果和损失的字典
        """
        # 共享特征提取
        backbone_features = self.backbone(images)
        encoder_features = self.encoder(backbone_features)
        
        # MoE Decoder前向（内部自动处理路由和专家融合）
        decoder_output = self.decoder(encoder_features, targets)
        
        # 构建输出字典
        output = {
            'pred_logits': decoder_output.get('pred_logits'),
            'pred_boxes': decoder_output.get('pred_boxes'),
            'bboxes': decoder_output.get('pred_boxes'),
            'class_scores': decoder_output.get('pred_logits'),
        }
        
        if targets is not None:
            # 计算检测损失（训练和验证都需要）
            detection_loss_dict = self.detr_criterion(decoder_output, targets)
            detection_loss = sum(v for v in detection_loss_dict.values() 
                               if isinstance(v, torch.Tensor))
            
            # 获取MoE负载均衡损失（仅训练时）
            if self.training:
                moe_load_balance_loss = decoder_output.get('moe_load_balance_loss', 
                                                          torch.tensor(0.0, device=images.device))
            else:
                moe_load_balance_loss = torch.tensor(0.0, device=images.device)
            
            # 总损失：检测损失 + MoE负载均衡损失
            # 支持从实例变量读取MoE权重（如果设置），否则使用默认值
            if hasattr(self, 'moe_balance_weight'):
                balance_weight = self.moe_balance_weight
            else:
                # 动态调整MoE损失权重（top_k=1时需要更强的约束）
                if hasattr(self.decoder, 'moe_top_k') and self.decoder.moe_top_k == 1:
                    balance_weight = 0.1  # top_k=1时使用更大的权重
                else:
                    balance_weight = 0.05  # top_k>1时使用较小的权重
            
            total_loss = detection_loss + balance_weight * moe_load_balance_loss
            
            output['detection_loss'] = detection_loss
            output['moe_load_balance_loss'] = moe_load_balance_loss
            output['total_loss'] = total_loss
            output['loss_dict'] = detection_loss_dict
        
        return output


class AdaptiveExpertTrainer:
    """自适应专家RT-DETR训练器。
    
    负责模型训练、验证、检查点管理等功能。
    """
    
    def __init__(self, config: Dict, config_file_path: Optional[str] = None):
        """初始化训练器。
        
        Args:
            config: 训练配置字典
            config_file_path: 配置文件路径（如果使用配置文件），用于验证
        """
        self.config = config
        self.config_file_path = config_file_path
        
        # 如果使用配置文件，验证必需的配置项
        if config_file_path:
            self._validate_config_file()
        
        # 如果使用配置文件，device必须存在，否则报错
        if config_file_path:
            if 'misc' not in self.config or 'device' not in self.config['misc']:
                raise ValueError(f"配置文件 {config_file_path} 缺少必需的配置项: misc.device")
            device_str = self.config['misc']['device']
        else:
            device_str = self.config.get('misc', {}).get('device', 'cuda')
        self.device = torch.device(device_str)
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.best_map = 0.0  # 记录最佳mAP
        self.global_step = 0
        # 从配置中读取 resume_from_checkpoint（支持两种格式）
        self.resume_from_checkpoint = self.config.get('resume_from_checkpoint', None)
        if self.resume_from_checkpoint is None and 'checkpoint' in self.config:
            self.resume_from_checkpoint = self.config['checkpoint'].get('resume_from_checkpoint', None)
        
        # 梯度裁剪参数（从配置读取）
        self.clip_max_norm = self.config.get('training', {}).get('clip_max_norm', 10.0)
        
        # 初始化组件
        self._setup_logging()
        self.model = self._create_model()
        self.train_loader, self.val_loader = self._create_data_loaders()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.warmup_scheduler = self._create_warmup_scheduler()
        self.ema = self._create_ema()
        self.scaler = self._create_scaler()
        
        self.visualizer = TrainingVisualizer(log_dir=self.log_dir, model_type='moe', experiment_name=self.experiment_name)
        self.early_stopping = self._create_early_stopping()
        
        # 初始化推理相关组件
        self._setup_inference_components()
        
        # 恢复检查点
        if self.resume_from_checkpoint:
            self._resume_from_checkpoint()
    
    def _validate_config_file(self):
        """验证配置文件是否包含所有必需的配置项"""
        required_keys = {
            'model': ['config_name', 'backbone', 'hidden_dim', 'num_queries', 'num_decoder_layers', 'top_k'],
            'training': ['epochs', 'batch_size', 'pretrained_lr', 'new_lr', 'warmup_epochs'],
            'data': ['data_root'],
            'misc': ['device', 'num_workers']
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
            error_msg = f"配置文件 {self.config_file_path} 缺少必需的配置项:\n"
            error_msg += "\n".join(f"  - {key}" for key in missing_keys)
            raise ValueError(error_msg)
    
    def _setup_logging(self) -> None:
        """设置日志系统。"""
        if self.resume_from_checkpoint:
            checkpoint_path = Path(self.resume_from_checkpoint)
            self.log_dir = checkpoint_path.parent
            # 从目录名中提取实验名称（去掉时间戳部分）
            dir_name = self.log_dir.name
            # 假设格式为 moe6_rtdetr_r50_20240101_120000，提取 moe6_rtdetr_r50
            parts = dir_name.rsplit('_', 2)  # 分割最后两部分（日期和时间）
            if len(parts) >= 2:
                self.experiment_name = '_'.join(parts[:-2]) if len(parts) > 2 else parts[0]
            else:
                self.experiment_name = dir_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 从配置中获取backbone类型，加入到目录名中
            backbone_type = self.config.get('model', {}).get('backbone', 'unknown')
            # 移除presnet前缀，只保留数字部分（如presnet18 -> r18, presnet34 -> r34）
            backbone_short = backbone_type.replace('presnet', 'r').replace('pres', 'r') if 'presnet' in backbone_type or 'pres' in backbone_type else backbone_type
            # 直接从配置文件读取专家数量，如果未配置则通过config_name映射（向后兼容）
            num_experts = self.config.get('model', {}).get('num_experts', None)
            if num_experts is None:
                # 向后兼容：通过config_name映射
                config_name = self.config.get('model', {}).get('config_name', 'A')
                configs = {'A': 6, 'B': 3, 'C': 2}
                num_experts = configs.get(config_name, 6)
            expert_num = str(num_experts)
            # 生成实验名称（不带时间戳）
            self.experiment_name = f"moe{expert_num}_rtdetr_{backbone_short}"
            self.log_dir = Path(f"logs/{self.experiment_name}_{timestamp}")
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        if self.resume_from_checkpoint:
            self.logger.info(f"恢复训练，日志目录: {self.log_dir}")
        
        if not self.resume_from_checkpoint:
            with open(self.log_dir / 'config.yaml', 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False)
    
    def _create_model(self) -> AdaptiveExpertRTDETR:
        """创建模型。"""
        # 从配置文件读取encoder配置
        encoder_config = self.config['model']['encoder']
        encoder_in_channels = encoder_config['in_channels']
        encoder_expansion = encoder_config['expansion']
        
        # 从配置文件读取专家数量，如果未配置则使用None（会通过config_name映射）
        num_experts = self.config['model'].get('num_experts', None)
        
        # 从配置文件读取MoE权重
        moe_balance_weight = self.config.get('training', {}).get('moe_balance_weight', None)
        
        model = AdaptiveExpertRTDETR(
            config_name=self.config['model'].get('config_name', 'A'),
            hidden_dim=self.config['model']['hidden_dim'],
            num_queries=self.config['model']['num_queries'],
            top_k=self.config['model']['top_k'],
            backbone_type=self.config['model']['backbone'],
            num_decoder_layers=self.config['model']['num_decoder_layers'],
            encoder_in_channels=encoder_in_channels,
            encoder_expansion=encoder_expansion,
            num_experts=num_experts,
            moe_balance_weight=moe_balance_weight
        )
        
        # 加载预训练权重
        pretrained_weights = self.config['model'].get('pretrained_weights', None)
        if pretrained_weights:
            self._load_pretrained_weights(model, pretrained_weights)
        
        model = model.to(self.device)
        
        # 启用GPU优化设置
        if torch.cuda.is_available():
            # 启用cudnn benchmark以加速卷积操作（输入尺寸固定时）
            torch.backends.cudnn.benchmark = True
            # 启用TensorFloat-32（RTX 5090支持，可加速某些操作）
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.logger.info("✓ 已启用GPU优化: cudnn.benchmark=True, TF32=True")
        
        self.logger.info(f"✓ 创建MOE RT-DETR模型")
        self.logger.info(f"  专家数量: {model.num_experts}")
        self.logger.info(f"  配置: {model.config_name}")
        self.logger.info(f"  Backbone: {model.backbone_type}")
        self.logger.info(f"  Encoder: in_channels={encoder_in_channels}, expansion={encoder_expansion}")
        
        return model
    
    def _load_pretrained_weights(self, model: AdaptiveExpertRTDETR, pretrained_path: str) -> None:
        """从本地文件加载预训练权重
        
        Args:
            pretrained_path: 本地权重文件路径（如 'pretrained/rtdetrv2_r50vd_6x_coco_ema.pth'）
        """
        try:
            pretrained_file = Path(pretrained_path)
            if not pretrained_file.exists():
                self.logger.warning(f"预训练权重文件不存在: {pretrained_path}")
                self.logger.info("将从随机初始化开始训练")
                return
            
            self.logger.info(f"从本地文件加载预训练权重: {pretrained_path}")
            checkpoint = torch.load(pretrained_file, map_location='cpu')
            
            # 处理不同的checkpoint格式
            if isinstance(checkpoint, dict):
                if 'ema' in checkpoint and 'module' in checkpoint['ema']:
                    # EMA格式: {'ema': {'module': {...}}}
                    state_dict = checkpoint['ema']['module']
                    self.logger.info("✓ 检测到EMA checkpoint格式")
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # 过滤掉类别相关参数（形状不匹配）
            filtered_state_dict = {}
            skipped_class_params = 0
            
            for k, v in state_dict.items():
                # 跳过类别相关的参数（这些参数的形状会不匹配）
                if any(keyword in k for keyword in ['class_embed', 'score_head', 'denoising_class_embed']):
                    skipped_class_params += 1
                    continue
                filtered_state_dict[k] = v
            
            # 加载过滤后的参数
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
            
            # 统计加载结果
            total_params = len(filtered_state_dict)
            loaded_params = total_params - len(missing_keys)
            
            self.logger.info(f"✓ 成功加载预训练权重: {loaded_params}/{total_params} 个参数")
            
            # 报告跳过的类别参数
            if skipped_class_params > 0:
                self.logger.info(f"  - 跳过类别相关参数: {skipped_class_params} 个（COCO 80类 → DAIR-V2X 8类）")
            
            # 统计各部分的参数
            backbone_loaded = sum(1 for k in filtered_state_dict.keys() if k not in missing_keys and 'backbone' in k)
            encoder_loaded = sum(1 for k in filtered_state_dict.keys() if k not in missing_keys and 'encoder' in k)
            decoder_loaded = sum(1 for k in filtered_state_dict.keys() if k not in missing_keys and 'decoder' in k)
            
            self.logger.info(f"  - Backbone: {backbone_loaded} 个参数")
            self.logger.info(f"  - Encoder: {encoder_loaded} 个参数")
            self.logger.info(f"  - Decoder: {decoder_loaded} 个参数")
            
            if len(missing_keys) > 0:
                self.logger.info(f"  - 预训练模型缺少参数: {len(missing_keys)} 个（当前模型新增）")
                if len(missing_keys) <= 5:
                    self.logger.info(f"    示例: {list(missing_keys)}")
                else:
                    self.logger.info(f"    示例: {list(missing_keys)[:3]} ...")
            
            if len(unexpected_keys) > 0:
                self.logger.info(f"  - 模型新增参数: {len(unexpected_keys)} 个（将随机初始化）")
                
        except Exception as e:
            self.logger.error(f"✗ 加载预训练权重失败: {e}")
            self.logger.info("将从随机初始化开始训练")
    
    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """创建数据加载器。"""
        # 修改：移除不必要的max()，使用配置值
        batch_size = self.config['training']['batch_size']
        target_size = self.model.image_size
        
        # 修改：训练时启用mosaic增强
        use_mosaic = self.config['training'].get('use_mosaic', True)
        
        train_dataset = DAIRV2XDetection(
            data_root=self.config['data']['data_root'],
            split='train',
            use_mosaic=use_mosaic,
            target_size=target_size
        )
        
        val_dataset = DAIRV2XDetection(
            data_root=self.config['data']['data_root'],
            split='val',
            use_mosaic=False,
            target_size=target_size
        )
        
        # 从misc配置中读取num_workers和pin_memory
        num_workers = self.config.get('misc', {}).get('num_workers', 16)
        pin_memory = self.config.get('misc', {}).get('pin_memory', True)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        self.val_dataset = val_dataset
        
        self.logger.info(f"✓ 创建数据加载器")
        self.logger.info(f"  训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def _collate_fn(self, batch: List[Tuple]) -> Tuple[torch.Tensor, List[Dict]]:
        """数据整理函数。"""
        images, targets = zip(*batch)
        
        if isinstance(images[0], np.ndarray):
            images = torch.stack([
                torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 
                for img in images
            ], dim=0)
        else:
            images = torch.stack(images, 0)
        
        return images, list(targets)
    
    def _create_optimizer(self) -> optim.Adam:
        """创建优化器。"""
        # 预训练参数：backbone + encoder
        pretrained_params = list(self.model.backbone.parameters()) + \
                           list(self.model.encoder.parameters())
        
        # 新参数：Decoder（包含内部的自适应专家层）
        decoder_params = list(self.model.decoder.parameters())
        
        # 确保学习率是浮点数类型
        pretrained_lr = float(self.config['training']['pretrained_lr'])
        new_lr = float(self.config['training']['new_lr'])
        weight_decay = float(self.config['training'].get('weight_decay', 0.0001))
        
        optimizer = optim.Adam([
            {'params': pretrained_params, 'lr': pretrained_lr},
            {'params': decoder_params, 'lr': new_lr}
        ], weight_decay=weight_decay)
        
        self.logger.info(f"✓ 创建优化器 (pretrained_lr={pretrained_lr}, new_lr={new_lr}, weight_decay={weight_decay})")
        self.logger.info(f"  预训练参数: {len(pretrained_params)} | Decoder参数: {len(decoder_params)}")
        
        return optimizer
    
    def _create_scheduler(self):
        """创建学习率调度器。"""
        scheduler_type = self.config.get('training', {}).get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            # 从配置文件读取eta_min，默认1e-7
            eta_min = self.config.get('training', {}).get('eta_min', 1e-7)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config['training']['epochs'],
                eta_min=eta_min
            )
            self.logger.info(f"✓ 使用CosineAnnealingLR调度器 (eta_min={eta_min})")
        else:
            # MultiStepLR
            milestones = self.config.get('training', {}).get('milestones', [60, 80])
            gamma = float(self.config.get('training', {}).get('gamma', 0.1))
            scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=gamma
            )
            self.logger.info(f"✓ 使用MultiStepLR调度器 (milestones={milestones})")
        
        return scheduler
    
    def _create_warmup_scheduler(self) -> WarmupLR:
        """创建学习率预热调度器。"""
        # 修改：warmup epochs从默认10改为3
        warmup_epochs = self.config.get('training', {}).get('warmup_epochs', 3)
        # 确保warmup_end_lr是浮点数
        warmup_end_lr = float(self.config['training']['new_lr'])
        warmup_scheduler = WarmupLR(
            self.optimizer,
            warmup_epochs=warmup_epochs,
            warmup_start_lr=1e-7,
            warmup_end_lr=warmup_end_lr
        )
        self.logger.info(f"✓ 学习率预热 {warmup_epochs} epochs")
        return warmup_scheduler
    
    def _create_ema(self) -> ModelEMA:
        """创建EMA模型。"""
        ema_decay = self.config.get('training', {}).get('ema_decay', 0.9999)
        return ModelEMA(self.model, decay=ema_decay)
    
    def _create_scaler(self):
        """创建混合精度训练器。"""
        return torch.amp.GradScaler('cuda')
    
    def _create_early_stopping(self) -> Optional[EarlyStopping]:
        """创建Early Stopping。"""
        training_config = self.config.get('training', {})
        patience = training_config.get('early_stopping_patience', None)
        
        if patience is None or patience <= 0:
            self.logger.info("⏱️  Early Stopping: 未启用")
            return None
        
        metric_name = training_config.get('early_stopping_metric', 'mAP_0.5_0.95')
        mode = 'max' if 'mAP' in metric_name or 'AP' in metric_name else 'min'
        
        self.logger.info(f"⏱️  Early Stopping: 启用 (patience={patience}, metric={metric_name}, mode={mode})")
        
        return EarlyStopping(
            patience=patience,
            mode=mode,
            min_delta=0.0001,
            metric_name=metric_name,
            logger=self.logger
        )
    
    def _setup_inference_components(self) -> None:
        """初始化推理相关组件"""
        # 创建后处理器
        self.postprocessor = DetDETRPostProcessor(
            num_classes=8,  # 8类：Car, Truck, Van, Bus, Pedestrian, Cyclist, Motorcyclist, Trafficcone
            use_focal_loss=True,
            num_top_queries=300,
            box_process_format=BoxProcessFormat.RESIZE
        )
        
        # 创建推理输出目录
        self.inference_output_dir = self.log_dir / "inference_samples"
        self.inference_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 类别名称和颜色（用于推理可视化）- 8类正式检测类别
        self.class_names = [
            "Car", "Truck", "Van", "Bus", "Pedestrian", 
            "Cyclist", "Motorcyclist", "Trafficcone"
        ]
        self.colors = [
            (255, 0, 0),      # Car - 红色
            (0, 255, 0),      # Truck - 绿色
            (255, 128, 0),    # Van - 橙色
            (0, 0, 255),      # Bus - 蓝色
            (255, 255, 0),    # Pedestrian - 黄色
            (255, 0, 255),    # Cyclist - 品红
            (0, 255, 255),    # Motorcyclist - 青色
            (128, 128, 128),  # Trafficcone - 灰色
        ]
        
        self.logger.info(f"推理输出目录: {self.inference_output_dir}")
    
    def _inference_single_image_from_batch(self, images, targets, batch_idx, image_idx=0, suffix=None):
        """从batch中选择一张图片进行推理并保存结果（直接复用batch_inference.py的逻辑）
        
        Args:
            images: 图像tensor
            targets: 目标列表
            batch_idx: batch索引
            image_idx: 图像在batch中的索引
            suffix: 文件名后缀（默认使用epoch，如"epoch_0"或"best_model"）
        """
        try:
            # 使用EMA模型进行推理
            self.ema.module.eval()
            
            # 选择batch中的指定图片
            single_image = images[image_idx:image_idx+1]  # [1, 3, H, W]
            single_target = targets[image_idx] if image_idx < len(targets) else None
            
            if single_target is None:
                return
            
            # 获取image_id用于命名和查找原始图像
            image_id = single_target['image_id'].item() if 'image_id' in single_target else batch_idx
            
            # 获取原始图像路径
            data_root = Path(self.config['data']['data_root'])
            orig_image_path = data_root / "image" / f"{image_id:06d}.jpg"
            
            if not orig_image_path.exists():
                return
            
            # 使用batch_inference.py中的函数进行推理（完全复用逻辑）
            if USE_BATCH_INFERENCE_LOGIC:
                result_image = inference_from_preprocessed_image(
                    single_image,
                    self.ema.module,
                    self.postprocessor,
                    orig_image_path,
                    conf_threshold=0.3,
                    target_size=640,
                    device=str(self.device),
                    class_names=self.class_names,
                    colors=self.colors,
                    verbose=False  # 训练时不打印调试信息
                )
                
                if result_image is None:
                    self.ema.module.train()
                    return
                
                # 保存结果：图片名_suffix.jpg
                image_name = orig_image_path.stem
                if suffix is None:
                    suffix = f"epoch_{self.current_epoch}"
                output_filename = f"{image_name}_{suffix}.jpg"
                output_path = self.inference_output_dir / output_filename
                cv2.imwrite(str(output_path), result_image)
            else:
                # 备用逻辑（如果无法导入batch_inference，使用简化版本）
                with torch.no_grad():
                    outputs = self.ema.module(single_image)
                
                # 简化的后处理和绘制（不推荐，但作为备用）
                eval_sizes = torch.tensor([[640, 640]], device=self.device)
                results = self.postprocessor(outputs, eval_sizes=eval_sizes)
                
                if len(results) > 0:
                    result = results[0]
                    labels = result['labels'].cpu().numpy()
                    boxes = result['boxes'].cpu().numpy()
                    scores = result['scores'].cpu().numpy()
                    
                    mask = scores >= 0.3
                    labels = labels[mask]
                    boxes = boxes[mask]
                    scores = scores[mask]
                    
                    if len(labels) > 0:
                        orig_image = cv2.imread(str(orig_image_path))
                        if orig_image is not None:
                            result_image = draw_boxes(
                                orig_image.copy(), labels, boxes, scores,
                                class_names=self.class_names,
                                colors=self.colors
                            )
                            image_name = orig_image_path.stem
                            if suffix is None:
                                suffix = f"epoch_{self.current_epoch}"
                            output_filename = f"{image_name}_{suffix}.jpg"
                            output_path = self.inference_output_dir / output_filename
                            cv2.imwrite(str(output_path), result_image)
            
            # 恢复训练模式
            self.ema.module.train()
            
        except Exception as e:
            # 如果推理失败，不影响训练
            if hasattr(self, 'logger'):
                self.logger.debug(f"推理失败（不影响训练）: {e}")
            if hasattr(self, 'ema') and hasattr(self.ema, 'module'):
                self.ema.module.train()
    
    def _run_inference_on_best_model(self, best_ema_state=None):
        """使用best_model运行推理，输出5张验证图像的推理结果
        
        Args:
            best_ema_state: best_model的EMA模型state_dict，如果提供则使用它进行推理
        """
        try:
            # 保存当前EMA模型状态（推理后恢复）
            original_ema_state = None
            if best_ema_state is not None and hasattr(self, 'ema') and self.ema:
                original_ema_state = self.ema.state_dict()
                # 加载best_model的EMA参数
                self.ema.load_state_dict(best_ema_state)
            
            # 从验证数据加载器中获取一个batch用于推理
            inference_images, inference_targets = next(iter(self.val_loader))
            inference_images = inference_images.to(self.device)
            inference_targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                 for k, v in t.items()} for t in inference_targets]
            
            # 打印前5张推理结果
            batch_size = len(inference_targets)
            num_inference_images = min(5, batch_size)
            self.logger.info(f"  生成best_model推理结果（前{num_inference_images}张）...")
            
            for img_idx in range(num_inference_images):
                self._inference_single_image_from_batch(
                    inference_images, inference_targets, 0, image_idx=img_idx,
                    suffix=f"best_model_epoch_{self.current_epoch}"
                )
            
            self.logger.info(f"  ✓ 推理结果已保存到: {self.inference_output_dir}")
            
            # 恢复原始EMA模型状态
            if original_ema_state is not None and hasattr(self, 'ema') and self.ema:
                self.ema.load_state_dict(original_ema_state)
                
        except Exception as e:
            # 如果推理失败，不影响训练，但尝试恢复EMA状态
            if hasattr(self, 'logger'):
                self.logger.warning(f"best_model推理失败（不影响训练）: {e}")
            if original_ema_state is not None and hasattr(self, 'ema') and self.ema:
                try:
                    self.ema.load_state_dict(original_ema_state)
                except:
                    pass
    
    def _resume_from_checkpoint(self) -> None:
        """从检查点恢复训练。"""
        try:
            checkpoint_path = Path(self.resume_from_checkpoint)
            if not checkpoint_path.exists():
                self.logger.warning(f"检查点不存在: {checkpoint_path}")
                return
            
            self.logger.info(f"从检查点恢复: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 恢复状态
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint.get('epoch', 0) + 1
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            self.best_map = checkpoint.get('best_map', 0.0)
            self.global_step = checkpoint.get('global_step', 0)
            
            if 'ema_state_dict' in checkpoint:
                self.ema.load_state_dict(checkpoint['ema_state_dict'])
            if 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            if 'visualizer_state' in checkpoint:
                self.visualizer.load_state_dict(checkpoint['visualizer_state'])
            if 'early_stopping_state' in checkpoint and self.early_stopping:
                self.early_stopping.load_state_dict(checkpoint['early_stopping_state'])
            
            self.logger.info(f"✓ 恢复成功 (epoch={self.current_epoch}, step={self.global_step}, "
                           f"best_loss={self.best_loss:.4f})")
            
        except Exception as e:
            self.logger.error(f"恢复检查点失败: {e}")
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch。"""
        self.model.train()
        total_loss = 0.0
        detection_loss = 0.0
        moe_lb_loss = 0.0  # MoE load balance loss
        
        # 统计细粒度MoE的专家使用率（跨所有Decoder层聚合）
        expert_usage_count = [0] * self.model.num_experts
        total_tokens = 0
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            
            images = images.to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            # 前向传播
            self.optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = self.model(images, targets)
                loss = outputs.get('total_loss', torch.tensor(0.0, device=self.device))
            
            # 反向传播（添加梯度裁剪）
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪（防止梯度爆炸）
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_max_norm)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.ema.update(self.model)
            
            # 统计损失
            total_loss += loss.item()
            if isinstance(outputs, dict):
                if 'detection_loss' in outputs:
                    detection_loss += outputs['detection_loss'].item()
                if 'moe_load_balance_loss' in outputs:
                    moe_lb_loss += outputs['moe_load_balance_loss'].item()
            
            # 收集细粒度MoE的专家使用统计
            if self.model.decoder.use_moe:
                for layer in self.model.decoder.decoder.layers:
                    if hasattr(layer, 'adaptive_expert_layer') and layer.adaptive_expert_layer.router_logits_cache is not None:
                        router_logits = layer.adaptive_expert_layer.router_logits_cache  # [N, num_experts]
                        # 计算每个token选择的top-k专家
                        _, top_indices = torch.topk(router_logits, self.model.decoder.moe_top_k, dim=-1)  # [N, K]
                        # 统计每个专家被选中的次数
                        for expert_id in range(self.model.num_experts):
                            expert_usage_count[expert_id] += (top_indices == expert_id).sum().item()
                        total_tokens += router_logits.shape[0] * self.model.decoder.moe_top_k
            
            if batch_idx % 50 == 0:
                det_loss_val = outputs.get('detection_loss', torch.tensor(0.0)).item() if isinstance(outputs, dict) else 0.0
                moe_loss_val = outputs.get('moe_load_balance_loss', torch.tensor(0.0)).item() if isinstance(outputs, dict) else 0.0
                self.logger.info(f'Epoch {self.current_epoch} | Batch {batch_idx} | '
                               f'Loss: {loss.item():.2f} (Det: {det_loss_val:.2f}, MoE: {moe_loss_val:.4f})')
            
            self.global_step += 1
        
        # 计算平均值
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        avg_detection_loss = detection_loss / num_batches
        avg_moe_lb_loss = moe_lb_loss / num_batches
        
        # 计算专家使用率
        expert_usage_rate = []
        if total_tokens > 0:
            for count in expert_usage_count:
                expert_usage_rate.append(count / total_tokens)
        else:
            expert_usage_rate = [1.0 / self.model.num_experts] * self.model.num_experts
        
        return {
            'total_loss': avg_loss,
            'detection_loss': avg_detection_loss,
            'moe_load_balance_loss': avg_moe_lb_loss,
            'expert_usage': expert_usage_count,
            'expert_usage_rate': expert_usage_rate
        }
    
    def validate(self) -> Dict[str, float]:
        """验证模型并计算mAP。"""
        self.ema.module.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        total_raw_predictions = 0  # 原始query总数
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(self.val_loader):
                images = images.to(self.device)
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in t.items()} for t in targets]
                
                outputs = self.ema.module(images, targets)
                
                if isinstance(outputs, dict):
                    if 'total_loss' in outputs:
                        total_loss += outputs['total_loss'].item()
                    
                    # 统计原始预测数
                    if 'class_scores' in outputs:
                        total_raw_predictions += outputs['class_scores'].shape[0] * outputs['class_scores'].shape[1]
                    
                    # 收集预测结果
                    if 'class_scores' in outputs and 'bboxes' in outputs:
                        self._collect_predictions(outputs, targets, batch_idx, all_predictions, all_targets)
        
        # 保存预测结果用于后续打印每个类别mAP（避免重复计算）
        self._last_val_predictions = all_predictions
        self._last_val_targets = all_targets
        
        # 计算mAP（同时计算每个类别的mAP并保存，避免在保存best_model时重复计算）
        mAP_metrics = self._compute_map_metrics(all_predictions, all_targets, print_per_category=False)
        # 保存每个类别的mAP，避免在_print_best_model_per_category_map中重复计算
        self._last_per_category_map = mAP_metrics.get('per_category_map', {})
        
        avg_loss = total_loss / len(self.val_loader)
        
        return {
            'total_loss': avg_loss,
            'mAP_0.5': mAP_metrics.get('mAP_0.5', 0.0),
            'mAP_0.75': mAP_metrics.get('mAP_0.75', 0.0),
            'mAP_0.5_0.95': mAP_metrics.get('mAP_0.5_0.95', 0.0),
            'num_predictions': len(all_predictions),
            'num_raw_predictions': total_raw_predictions,
            'num_targets': len(all_targets)
        }
    
    def _collect_predictions(self, outputs: Dict, targets: List[Dict], batch_idx: int,
                            all_predictions: List, all_targets: List) -> None:
        """收集预测结果用于mAP计算。保留所有有效预测框，不做top-k限制。"""
        pred_logits = outputs['class_scores']  # [B, Q, C]
        pred_boxes = outputs['bboxes']  # [B, Q, 4]
        
        batch_size = pred_logits.shape[0]
        
        for i in range(batch_size):
            # VFL损失使用sigmoid，所以推理时也应该使用sigmoid
            pred_scores_sigmoid = torch.sigmoid(pred_logits[i])  # [Q, C]
            max_scores, pred_classes = torch.max(pred_scores_sigmoid, dim=-1)  # [Q]
            
            # 过滤无效框（padding框），保留所有有效预测框
            valid_boxes_mask = ~torch.all(pred_boxes[i] == 1.0, dim=1)
            valid_indices = torch.where(valid_boxes_mask)[0]
            if len(valid_indices) > 0:
                filtered_boxes = pred_boxes[i][valid_indices]
                filtered_classes = pred_classes[valid_indices]
                filtered_scores = max_scores[valid_indices]
                
                # 转换为COCO格式
                if filtered_boxes.shape[0] > 0:
                    boxes_coco = torch.zeros_like(filtered_boxes)
                    if filtered_boxes.max() <= 1.0:
                        # 归一化坐标 -> 像素坐标
                        boxes_coco[:, 0] = (filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2) * self.model.image_size
                        boxes_coco[:, 1] = (filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2) * self.model.image_size
                        boxes_coco[:, 2] = filtered_boxes[:, 2] * self.model.image_size
                        boxes_coco[:, 3] = filtered_boxes[:, 3] * self.model.image_size
                    else:
                        boxes_coco = filtered_boxes.clone()
                    
                    # Clamp坐标
                    boxes_coco[:, 0] = torch.clamp(boxes_coco[:, 0], 0, self.model.image_size)
                    boxes_coco[:, 1] = torch.clamp(boxes_coco[:, 1], 0, self.model.image_size)
                    boxes_coco[:, 2] = torch.clamp(boxes_coco[:, 2], 1, self.model.image_size)
                    boxes_coco[:, 3] = torch.clamp(boxes_coco[:, 3], 1, self.model.image_size)
                    
                    for j in range(boxes_coco.shape[0]):
                        all_predictions.append({
                            'image_id': batch_idx * self.config['training']['batch_size'] + i,
                            'category_id': int(filtered_classes[j].item()) + 1,
                            'bbox': boxes_coco[j].cpu().numpy().tolist(),
                            'score': float(filtered_scores[j].item())
                        })
            
            # 处理真实标签（评估时包含iscrowd字段，COCOeval会自动处理）
            if i < len(targets) and 'labels' in targets[i] and 'boxes' in targets[i]:
                true_labels = targets[i]['labels']
                true_boxes = targets[i]['boxes']
                
                if len(true_labels) > 0:
                    img_size = self.model.image_size
                    max_val = float(true_boxes.max().item()) if true_boxes.numel() > 0 else 0.0
                    scale = img_size if max_val <= 1.0 + 1e-6 else 1.0
                    
                    true_boxes_coco = torch.zeros_like(true_boxes)
                    true_boxes_coco[:, 0] = (true_boxes[:, 0] - true_boxes[:, 2] / 2) * scale
                    true_boxes_coco[:, 1] = (true_boxes[:, 1] - true_boxes[:, 3] / 2) * scale
                    true_boxes_coco[:, 2] = true_boxes[:, 2] * scale
                    true_boxes_coco[:, 3] = true_boxes[:, 3] * scale
                    
                    true_boxes_coco[:, 0] = torch.clamp(true_boxes_coco[:, 0], 0, img_size)
                    true_boxes_coco[:, 1] = torch.clamp(true_boxes_coco[:, 1], 0, img_size)
                    true_boxes_coco[:, 2] = torch.clamp(true_boxes_coco[:, 2], 1, img_size)
                    true_boxes_coco[:, 3] = torch.clamp(true_boxes_coco[:, 3], 1, img_size)
                    
                    # 获取iscrowd字段（评估时存在）
                    has_iscrowd = 'iscrowd' in targets[i]
                    iscrowd_values = targets[i]['iscrowd'] if has_iscrowd else torch.zeros(len(true_labels), dtype=torch.int64)
                    
                    for j in range(len(true_labels)):
                        ann_dict = {
                            'image_id': batch_idx * self.config['training']['batch_size'] + i,
                            'category_id': int(true_labels[j].item()) + 1,
                            'bbox': true_boxes_coco[j].cpu().numpy().tolist(),
                            'area': float((true_boxes_coco[j, 2] * true_boxes_coco[j, 3]).item())
                        }
                        # 评估时添加iscrowd字段，让COCOeval自动处理
                        if has_iscrowd:
                            ann_dict['iscrowd'] = int(iscrowd_values[j].item())
                        all_targets.append(ann_dict)
    
    def _print_best_model_per_category_map(self):
        """使用best_model时打印详细的每类mAP（8类），重新计算以输出COCO详细评估表格"""
        try:
            # 使用已保存的预测结果重新计算，以便输出COCO详细评估表格
            if hasattr(self, '_last_val_predictions') and hasattr(self, '_last_val_targets'):
                # 重新计算mAP，print_per_category=True会输出COCO详细评估表格
                mAP_metrics = self._compute_map_metrics(self._last_val_predictions, self._last_val_targets, print_per_category=True)
                per_category_map = mAP_metrics.get('per_category_map', {})
            else:
                # 如果没有保存的结果，则重新计算（兼容性处理）
                self.logger.warning("未找到保存的验证结果，重新计算每个类别mAP...")
                self.ema.module.eval()
                all_predictions = []
                all_targets = []
                
                with torch.no_grad():
                    for batch_idx, (images, targets) in enumerate(self.val_loader):
                        images = images.to(self.device)
                        targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                   for k, v in t.items()} for t in targets]
                        
                        outputs = self.ema.module(images, targets)
                        
                        if 'class_scores' in outputs and 'bboxes' in outputs:
                            self._collect_predictions(outputs, targets, batch_idx, all_predictions, all_targets)
                
                mAP_metrics = self._compute_map_metrics(all_predictions, all_targets, print_per_category=True)
                per_category_map = mAP_metrics.get('per_category_map', {})
        except Exception as e:
            self.logger.warning(f"打印best_model每类mAP失败: {e}")
    
    def _compute_map_metrics(self, predictions: List[Dict], targets: List[Dict], print_per_category: bool = False) -> Dict[str, float]:
        """计算mAP指标。
        
        Args:
            predictions: 预测结果列表
            targets: 真实标签列表
            print_per_category: 是否打印每个类别的详细mAP（默认False，只在best_model时打印）
        """
        try:
            if len(predictions) == 0:
                return {
                    'mAP_0.5': 0.0,
                    'mAP_0.75': 0.0,
                    'mAP_0.5_0.95': 0.0
                }
            
            # 获取类别信息
            if hasattr(self, 'val_dataset') and hasattr(self.val_dataset, 'get_categories'):
                categories = self.val_dataset.get_categories()
            else:
                categories = [
                    {'id': 1, 'name': 'Car'},
                    {'id': 2, 'name': 'Truck'},
                    {'id': 3, 'name': 'Van'},
                    {'id': 4, 'name': 'Bus'},
                    {'id': 5, 'name': 'Pedestrian'},
                    {'id': 6, 'name': 'Cyclist'},
                    {'id': 7, 'name': 'Motorcyclist'},
                    {'id': 8, 'name': 'Trafficcone'}
                ]
            
            # 创建COCO格式数据
            coco_gt = {
                'images': [],
                'annotations': [],
                'categories': categories,
                'info': {
                    'description': 'DAIR-V2X Dataset',
                    'version': '1.0',
                    'year': 2024
                }
            }
            
            # 添加图像信息
            image_ids = set(target['image_id'] for target in targets)
            for img_id in image_ids:
                coco_gt['images'].append({
                    'id': img_id, 
                    'width': self.model.image_size, 
                    'height': self.model.image_size
                })
            
            # 添加标注
            for i, target in enumerate(targets):
                target['id'] = i + 1
                coco_gt['annotations'].append(target)
            
            # 使用pycocotools评估（抑制所有输出以节省时间）
            from io import StringIO
            import sys
            
            coco_gt_obj = COCO()
            coco_gt_obj.dataset = coco_gt
            # 抑制createIndex的输出
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                coco_gt_obj.createIndex()
            finally:
                sys.stdout = old_stdout
            
            # 抑制loadRes的输出
            sys.stdout = StringIO()
            try:
                coco_dt = coco_gt_obj.loadRes(predictions)
            finally:
                sys.stdout = old_stdout
            
            coco_eval = COCOeval(coco_gt_obj, coco_dt, 'bbox')
            # 如果print_per_category=True（保存best_model时），输出COCO详细评估表格；否则抑制输出
            if print_per_category:
                # 只抑制中间过程输出，保留summary表格
                sys.stdout = StringIO()
                try:
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                finally:
                    sys.stdout = old_stdout
                # 输出summary表格
                coco_eval.summarize()
            else:
                # 完全抑制输出
                sys.stdout = StringIO()
                try:
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                    coco_eval.summarize()
                finally:
                    sys.stdout = old_stdout
            
            # 提取每个类别的 mAP@0.5:0.95
            category_map = {cat['id']: cat['name'] for cat in categories}
            per_category_map = {}
            
            # 方法：为每个类别单独计算 AP
            # 通过设置 catIds 参数，只评估特定类别
            cat_ids = coco_eval.params.catIds
            
            for cat_id, cat_name in category_map.items():
                if cat_id in cat_ids:
                    try:
                        # 为当前类别创建单独的 COCOeval 对象
                        coco_eval_cat = COCOeval(coco_gt_obj, coco_dt, 'bbox')
                        coco_eval_cat.params.catIds = [cat_id]  # 只评估当前类别
                        # 抑制所有输出（evaluate、accumulate、summarize都会产生输出）
                        sys.stdout = StringIO()
                        try:
                            coco_eval_cat.evaluate()
                            coco_eval_cat.accumulate()
                            coco_eval_cat.summarize()
                        finally:
                            sys.stdout = old_stdout
                        
                        # 检查 stats 是否存在且有足够的元素
                        # stats[0] = AP@0.5:0.95, 需要确保至少有1个元素
                        if hasattr(coco_eval_cat, 'stats') and len(coco_eval_cat.stats) > 0:
                            per_category_map[cat_name] = float(coco_eval_cat.stats[0])
                        else:
                            # 如果没有检测结果，stats 可能为空，设为0
                            per_category_map[cat_name] = 0.0
                    except (IndexError, AttributeError, ValueError) as e:
                        # 捕获可能的索引错误、属性错误或值错误
                        # 如果该类别没有检测结果，这些错误是正常的
                        per_category_map[cat_name] = 0.0
                    except Exception as e:
                        # 其他异常也捕获，确保不会中断整个评估过程
                        self.logger.debug(f"类别 {cat_name} AP计算失败: {e}")
                        per_category_map[cat_name] = 0.0
                else:
                    per_category_map[cat_name] = 0.0
            
            # 只在best_model时打印每个类别的详细mAP
            if print_per_category:
                self.logger.info("  每个类别的 mAP@0.5:0.95:")
                category_order = ['Car', 'Truck', 'Van', 'Bus', 'Pedestrian', 
                                'Cyclist', 'Motorcyclist', 'Trafficcone']
                for cat_name in category_order:
                    map_val = per_category_map.get(cat_name, 0.0)
                    self.logger.info(f"    {cat_name:12s}: {map_val:.4f}")
            
            result = {
                'mAP_0.5': coco_eval.stats[1],
                'mAP_0.75': coco_eval.stats[2],
                'mAP_0.5_0.95': coco_eval.stats[0],
                'per_category_map': per_category_map  # 保存每个类别的mAP
            }
            
            # 添加每个类别的指标
            for cat_name in per_category_map.keys():
                result[f'mAP_{cat_name}'] = per_category_map[cat_name]
            
            return result
            
        except Exception as e:
            self.logger.warning(f"mAP计算失败: {e}")
            return {
                'mAP_0.5': 0.0,
                'mAP_0.75': 0.0,
                'mAP_0.5_0.95': 0.0
            }
    
    def _safe_save(self, checkpoint: Dict, path: Path, desc: str = "检查点") -> bool:
        """安全保存checkpoint - 带重试和错误处理。"""
        import time
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # 先保存到临时文件
                temp_path = path.with_suffix('.pth.tmp')
                torch.save(checkpoint, temp_path)
                
                # 确保写入完成
                import os
                os.sync()
                
                # 重命名为目标文件（原子操作）
                temp_path.replace(path)
                self.logger.info(f"💾 保存{desc}: {path}")
                return True
                
            except Exception as e:
                self.logger.warning(f"保存{desc}失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # 等待1秒后重试
                else:
                    self.logger.error(f"⚠️  保存{desc}最终失败，跳过并继续训练")
                    return False
        
        return False
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """保存最佳模型检查点。"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'warmup_scheduler_state_dict': self.warmup_scheduler.state_dict(),
            'ema_state_dict': self.ema.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'config': self.config,
            'best_loss': self.best_loss,
            'best_map': self.best_map,
            'global_step': self.global_step,
            'visualizer_state': self.visualizer.state_dict()
        }
        
        if self.early_stopping:
            checkpoint['early_stopping_state'] = self.early_stopping.state_dict()
        
        if is_best:
            # 保存当前EMA模型的state_dict（用于推理时确保使用best_model的参数）
            best_ema_state = None
            if hasattr(self, 'ema') and self.ema:
                best_ema_state = self.ema.state_dict()
            
            best_path = self.log_dir / 'best_model.pth'
            self._safe_save(checkpoint, best_path, "最佳模型")
            
            # 在best_model时重新计算并打印详细的每类mAP（8类）
            self._print_best_model_per_category_map()
    
    def save_latest_checkpoint(self, epoch: int) -> None:
        """保存最新检查点用于断点续训（带重试机制）"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'warmup_scheduler_state_dict': self.warmup_scheduler.state_dict(),
            'ema_state_dict': self.ema.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'config': self.config,
            'best_loss': self.best_loss,
            'best_map': self.best_map,
            'global_step': self.global_step,
            'visualizer_state': self.visualizer.state_dict()
        }
        
        if self.early_stopping:
            checkpoint['early_stopping_state'] = self.early_stopping.state_dict()
        
        latest_path = self.log_dir / 'latest_checkpoint.pth'
        self._safe_save(checkpoint, latest_path, "最新检查点")
    
    def train(self) -> None:
        """主训练循环。"""
        epochs = self.config['training']['epochs']
        self.logger.info(f"开始训练 {epochs} epochs")
        self.logger.info(f"✓ 梯度裁剪: max_norm={self.clip_max_norm}")
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # 训练
            train_metrics = self.train_epoch()
            
            # 验证
            val_metrics = self.validate()
            
            # 学习率调度
            if self.current_epoch < self.warmup_scheduler.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                self.scheduler.step()
            
            # 输出日志
            self.logger.info(f"Epoch {epoch}:")
            self.logger.info(f"  训练损失: {train_metrics.get('total_loss', 0.0):.2f} | 验证损失: {val_metrics.get('total_loss', 0.0):.2f}")
            # mAP只在best_model时输出，不在这里输出
            self.logger.info(f"  预测/目标: {val_metrics['num_predictions']}/{val_metrics['num_targets']}")
            
            # 显示详细损失（前20个epoch每次显示，之后每5个epoch显示）
            should_show_details = (epoch < 20) or (epoch % 5 == 0)
            if should_show_details:
                self.logger.info(f"  检测损失: {train_metrics['detection_loss']:.2f}")
                self.logger.info(f"  MoE负载均衡损失: {train_metrics['moe_load_balance_loss']:.4f}")
                # 显示专家使用率
                usage_str = [f"{rate*100:.2f}%" for rate in train_metrics['expert_usage_rate']]
                self.logger.info(f"  专家使用率: [{', '.join(usage_str)}]")
            
            # 记录训练指标到可视化器
            current_lr = self.optimizer.param_groups[0]['lr']
            self.visualizer.record(
                epoch=epoch,
                train_loss=train_metrics.get('total_loss', 0.0),
                val_loss=val_metrics.get('total_loss', 0.0),
                mAP_0_5=val_metrics.get('mAP_0.5', 0.0),
                mAP_0_75=val_metrics.get('mAP_0.75', 0.0),
                mAP_0_5_0_95=val_metrics.get('mAP_0.5_0.95', 0.0),
                learning_rate=current_lr,
                expert_usage=train_metrics.get('expert_usage_rate', []),  # 细粒度MoE专家使用率
                router_loss=train_metrics.get('moe_load_balance_loss', 0.0)  # 记录MoE负载均衡损失
            )
            
            # 保存检查点 - 同时考虑loss和mAP
            is_best_loss = val_metrics.get('total_loss', float('inf')) < self.best_loss
            is_best_map = val_metrics.get('mAP_0.5_0.95', 0.0) > self.best_map
            
            if is_best_loss:
                self.best_loss = val_metrics.get('total_loss', float('inf'))
                self.logger.info(f"  🎉 新的最佳验证损失: {self.best_loss:.2f}")
            
            if is_best_map:
                self.best_map = val_metrics.get('mAP_0.5_0.95', 0.0)
                self.logger.info(f"  🎉 新的最佳mAP: {self.best_map:.4f}")
                self.save_checkpoint(epoch, is_best=True)
            
            # Early Stopping检查
            if self.early_stopping:
                # 获取要监控的指标值
                metric_name = self.early_stopping.metric_name
                if 'mAP_0.5_0.95' in metric_name or 'mAP_0.5:0.95' in metric_name:
                    metric_value = val_metrics.get('mAP_0.5_0.95', 0.0)
                elif 'mAP_0.5' in metric_name:
                    metric_value = val_metrics.get('mAP_0.5', 0.0)
                elif 'mAP_0.75' in metric_name:
                    metric_value = val_metrics.get('mAP_0.75', 0.0)
                elif 'loss' in metric_name.lower():
                    metric_value = val_metrics.get('total_loss', float('inf'))
                else:
                    metric_value = val_metrics.get('mAP_0.5_0.95', 0.0)  # 默认
                
                if self.early_stopping(metric_value, epoch):
                    self.logger.info(f"Early Stopping在epoch {epoch}触发，停止训练")
                    break
            
            # 每个epoch都保存latest用于断点续训（不会堆积文件）
            self.save_latest_checkpoint(epoch)
            
            # 绘制训练曲线（每个epoch都更新）
            try:
                self.visualizer.plot()
            except Exception as e:
                self.logger.warning(f"绘制训练曲线失败: {e}")
        
        self.logger.info("✓ 训练完成！")
        
        # 最后绘制一次完整的训练曲线并导出CSV
        try:
            self.visualizer.plot()
            self.visualizer.export_to_csv()
            self.logger.info(f"✓ 训练曲线已保存到: {self.log_dir}/training_curves.png")
            self.logger.info(f"✓ 训练历史已导出到: {self.log_dir}/training_history.csv")
        except Exception as e:
            self.logger.error(f"绘制最终训练曲线失败: {e}")
        
        # 训练结束时使用best_model输出5张推理图像
        self.logger.info("=" * 60)
        self.logger.info("使用best_model生成推理结果（5张图像）...")
        try:
            best_model_path = self.log_dir / 'best_model.pth'
            if best_model_path.exists():
                # 加载best_model的checkpoint
                checkpoint = torch.load(best_model_path, map_location=self.device)
                best_ema_state = checkpoint.get('ema_state_dict', None)
                
                # 使用best_model进行推理
                self._run_inference_on_best_model(best_ema_state)
            else:
                self.logger.warning("未找到best_model.pth，跳过推理")
        except Exception as e:
            self.logger.warning(f"训练结束时推理失败（不影响训练结果）: {e}")


def main() -> None:
    """主函数。"""
    parser = argparse.ArgumentParser(description='自适应专家RT-DETR训练')
    parser.add_argument('--config', type=str, default='A', 
                       help='专家配置 (A: 6专家, B: 3专家) 或YAML配置文件路径')
    parser.add_argument('--backbone', type=str, default='presnet34', 
                       choices=['presnet18', 'presnet34', 'presnet50', 'presnet101',
                               'hgnetv2_l', 'hgnetv2_x', 'hgnetv2_h',
                               'cspresnet_s', 'cspresnet_m', 'cspresnet_l', 'cspresnet_x',
                               'cspdarknet', 'mresnet'],
                       help='Backbone类型')
    parser.add_argument('--data_root', type=str, default='datasets/DAIR-V2X', 
                       help='DAIR-V2X数据集路径')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小 (RTX 5090优化)')
    parser.add_argument('--pretrained_lr', type=float, default=1e-5, help='预训练组件学习率')
    parser.add_argument('--new_lr', type=float, default=1e-4, help='新组件学习率')
    parser.add_argument('--top_k', type=int, default=3, help='路由器Top-K')
    parser.add_argument('--pretrained_weights', type=str, default=None,
                       help='预训练权重路径（RT-DETR COCO预训练模型）')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='从检查点恢复训练（检查点文件路径）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子，用于确保实验可重复性（默认：42）')
    parser.add_argument('--deterministic', action='store_true',
                       help='使用确定性算法（会降低速度但保证完全可重复）')
    
    args = parser.parse_args()
    
    # 设置随机种子（必须在所有操作之前）
    print("\n" + "="*60)
    print("🔧 初始化训练环境")
    print("="*60)
    set_seed(args.seed, deterministic=args.deterministic)
    
    # 加载配置
    config_file_path = None
    if args.config and args.config.endswith('.yaml'):
        # 从YAML文件加载配置
        config_file_path = args.config
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"📄 从配置文件加载: {args.config}")
        
        # 确保学习率相关值是浮点数（YAML中的科学计数法可能被解析为字符串）
        if 'training' in config:
            if 'pretrained_lr' in config['training']:
                config['training']['pretrained_lr'] = float(config['training']['pretrained_lr'])
            if 'new_lr' in config['training']:
                config['training']['new_lr'] = float(config['training']['new_lr'])
            if 'eta_min' in config['training']:
                config['training']['eta_min'] = float(config['training']['eta_min'])
        
        # 只允许显式传递的命令行参数覆盖配置文件（不等于默认值的才覆盖）
        if args.backbone != 'presnet34':
            config['model']['backbone'] = args.backbone
        if args.epochs != 100:
            config['training']['epochs'] = args.epochs
        if args.batch_size != 32:
            config['training']['batch_size'] = args.batch_size
        if args.pretrained_lr != 1e-5:
            config['training']['pretrained_lr'] = args.pretrained_lr
        if args.new_lr != 1e-4:
            config['training']['new_lr'] = args.new_lr
        if args.top_k != 3:
            config['model']['top_k'] = args.top_k
        if args.data_root != 'datasets/DAIR-V2X':
            config['data']['data_root'] = args.data_root
        if args.pretrained_weights:
            config['model']['pretrained_weights'] = args.pretrained_weights
        
        # 命令行参数覆盖配置文件中的 resume_from_checkpoint
        if args.resume_from_checkpoint:
            if 'checkpoint' not in config:
                config['checkpoint'] = {}
            config['checkpoint']['resume_from_checkpoint'] = args.resume_from_checkpoint
    else:
        # 创建默认配置
        config = {
            'model': {
                'config': args.config,
                'hidden_dim': 256,
                'num_queries': 300,
                'top_k': args.top_k,
                'backbone': args.backbone
            },
            'data': {
                'data_root': args.data_root
            },
            'training': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'pretrained_lr': args.pretrained_lr,
                'new_lr': args.new_lr,
                'use_mosaic': True, 
                'warmup_epochs': 3,
                'ema_decay': 0.9999
            }
        }
        
        if args.pretrained_weights:
            config['model']['pretrained_weights'] = args.pretrained_weights
        
        if args.resume_from_checkpoint:
            config['checkpoint'] = {'resume_from_checkpoint': args.resume_from_checkpoint}
    
    # 创建训练器
    trainer = AdaptiveExpertTrainer(config, config_file_path=config_file_path)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
