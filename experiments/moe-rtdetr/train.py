#!/usr/bin/env python3
"""自适应专家RT-DETR训练脚本 - 细粒度MoE架构（Decoder FFN层集成自适应专家层）"""

import os
import sys
import argparse
from pathlib import Path

_experiments_root = Path(__file__).resolve().parent.parent
if str(_experiments_root) not in sys.path:
    sys.path.insert(0, str(_experiments_root))
from common.dataset_registry import (
    load_dataset_registry,
    resolve_dataset_profile,
    apply_detr_dataset_profile,
    default_detr_registry_path,
)
from common.vram_batch import (
    compute_vram_batch_adjustment,
    format_vram_batch_log,
    resolve_cuda_device_index,
)
from common.det_eval_metrics import (
    kitti_difficulty_from_coco_ann,
    coco_ap_at_iou50_all,
    coco_area_ap_at_iou50,
    dataset_dir_name,
)
from common.detr_eval_utils import (
    evaluate_best_model_after_training,
    log_detr_eval_summary,
    write_detr_eval_csv,
)
from common.detr_data_root import resolve_detr_data_root
from common.eval_schedule import (
    should_run_validation,
    describe_eval_schedule,
)

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import re
from torch.utils.data import DataLoader
import logging
from datetime import datetime
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

project_root = Path(__file__).parent.resolve()
if str(os.getcwd()) not in sys.path:
    sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))

from seed_utils import set_seed, seed_worker
from src.misc.training_visualizer import TrainingVisualizer
from src.misc.early_stopping import EarlyStopping
from src.zoo.rtdetr import HybridEncoder, RTDETRTransformerv2, RTDETRCriterionv2, HungarianMatcher
from src.nn.backbone.presnet import PResNet
from src.nn.backbone.hgnetv2 import HGNetv2
from src.nn.backbone.csp_resnet import CSPResNet
from src.nn.backbone.csp_darknet import CSPDarkNet
from src.nn.backbone.test_resnet import MResNet
from src.optim.ema import ModelEMA
from src.optim.amp import GradScaler
from src.optim.warmup import WarmupLR
from src.data.dataset.dairv2x_detection import DAIRV2XDetection
from src.data.dataset.coco_folder_detection import CocoFolderDetection
from src.nn.postprocessor.detr_postprocessor import DetDETRPostProcessor
from src.nn.postprocessor.box_revert import BoxProcessFormat
import cv2
import torchvision.transforms as T

try:
    from batch_inference import postprocess_outputs, draw_boxes, inference_from_preprocessed_image
    USE_BATCH_INFERENCE_LOGIC = True
except ImportError:
    USE_BATCH_INFERENCE_LOGIC = False


def create_backbone(backbone_type: str, **kwargs) -> nn.Module:
    """创建backbone的工厂函数"""
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
            'freeze_at': 0,
            'freeze_norm': True,
            'pretrained': False
        }
        default_params.update(kwargs)
        return PResNet(**default_params)
    
    elif backbone_type.startswith('hgnetv2'):
        name_map = {'hgnetv2_l': 'L', 'hgnetv2_x': 'X', 'hgnetv2_h': 'H'}
        if backbone_type not in name_map:
            raise ValueError(f"不支持的HGNetv2类型: {backbone_type}")
        
        default_params = {
            'name': name_map[backbone_type],
            'return_idx': [1, 2, 3],
            'freeze_at': 0,
            'freeze_norm': True,
            'pretrained': False
        }
        default_params.update(kwargs)
        return HGNetv2(**default_params)
    
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
    """自适应专家RT-DETR模型（细粒度MoE架构）"""
    
    def __init__(self, hidden_dim: int = 256, 
                 num_queries: int = 300, top_k: int = 2, backbone_type: str = "presnet34",
                 num_decoder_layers: int = 3, encoder_in_channels: list = None, 
                 encoder_expansion: float = 1.0, num_experts: int = 6,
                 moe_balance_weight: float = None,
                 num_classes: int = 8):
        """
        Args:
            hidden_dim: 隐藏层维度
            num_queries: 查询数量
            top_k: 路由器Top-K选择
            backbone_type: Backbone类型
            num_decoder_layers: Decoder层数
            encoder_in_channels: Encoder输入通道数
            encoder_expansion: Encoder expansion参数
            num_experts: 专家数量（必需）
            moe_balance_weight: MoE负载均衡损失权重
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.top_k = top_k
        self.backbone_type = backbone_type
        self.image_size = 640
        self.num_decoder_layers = num_decoder_layers
        
        self.encoder_in_channels = encoder_in_channels or [512, 1024, 2048]
        self.encoder_expansion = encoder_expansion
        self.num_classes = num_classes
        
        if moe_balance_weight is not None:
            self.moe_balance_weight = moe_balance_weight
        
        # 设置专家数量
        self.num_experts = num_experts
        
        self.backbone = self._build_backbone()
        self.encoder = self._build_encoder()
        
        self.decoder = RTDETRTransformerv2(
            num_classes=num_classes,
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
            use_moe=True,
            num_experts=self.num_experts,
            moe_top_k=top_k
        )
        
        print(f"✓ MoE Decoder配置: {num_decoder_layers}层, {self.num_experts}个专家, top_k={top_k}")
        self.detr_criterion = self._build_detr_criterion()
        
    def _build_backbone(self) -> nn.Module:
        """构建backbone"""
        return create_backbone(self.backbone_type)
    
    def _build_encoder(self) -> nn.Module:
        """构建encoder"""
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
            act='silu'
        )
    
    def _build_detr_criterion(self) -> RTDETRCriterionv2:
        """构建RT-DETR损失函数。"""
        matcher = HungarianMatcher(
            weight_dict={'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
            use_focal_loss=False,
            alpha=0.25,
            gamma=2.0
        )
        
        main_weight_dict = {
            'loss_vfl': 1.0,
            'loss_bbox': 5.0,
            'loss_giou': 2.0
        }
        
        num_decoder_layers = self.num_decoder_layers
        aux_weight_dict = {}
        for i in range(num_decoder_layers - 1):
            aux_weight_dict[f'loss_vfl_aux_{i}'] = 1.0
            aux_weight_dict[f'loss_bbox_aux_{i}'] = 5.0
            aux_weight_dict[f'loss_giou_aux_{i}'] = 2.0
        
        aux_weight_dict['loss_vfl_enc_0'] = 1.0
        aux_weight_dict['loss_bbox_enc_0'] = 5.0
        aux_weight_dict['loss_giou_enc_0'] = 2.0
        
        num_denoising_layers = num_decoder_layers
        for i in range(num_denoising_layers):
            aux_weight_dict[f'loss_vfl_dn_{i}'] = 1.0
            aux_weight_dict[f'loss_bbox_dn_{i}'] = 5.0
            aux_weight_dict[f'loss_giou_dn_{i}'] = 2.0
        
        weight_dict = {**main_weight_dict, **aux_weight_dict}
        
        criterion = RTDETRCriterionv2(
            matcher=matcher,
            weight_dict=weight_dict,
            losses=['vfl', 'boxes'],
            alpha=0.75,
            gamma=2.0,
            num_classes=self.num_classes,
            boxes_weight_format=None,
            share_matched_indices=False
        )
        
        return criterion
    
    
    def forward(self, images: torch.Tensor, 
                targets: Optional[List[Dict]] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: [B, C, H, W] 输入图像
            targets: 训练目标列表（可选）
        
        Returns:
            Dict: 包含检测结果和损失的字典
        """
        backbone_features = self.backbone(images)
        encoder_features = self.encoder(backbone_features)
        decoder_output = self.decoder(encoder_features, targets)
        
        output = {
            'pred_logits': decoder_output.get('pred_logits'),
            'pred_boxes': decoder_output.get('pred_boxes'),
            'bboxes': decoder_output.get('pred_boxes'),
            'class_scores': decoder_output.get('pred_logits'),
        }
        
        if targets is not None:
            detection_loss_dict = self.detr_criterion(decoder_output, targets)
            detection_loss = sum(v for v in detection_loss_dict.values() 
                               if isinstance(v, torch.Tensor))
            
            if self.training:
                moe_load_balance_loss = decoder_output.get('moe_load_balance_loss', 
                                                          torch.tensor(0.0, device=images.device))
            else:
                moe_load_balance_loss = torch.tensor(0.0, device=images.device)
            
            if hasattr(self, 'moe_balance_weight'):
                balance_weight = self.moe_balance_weight
            else:
                if hasattr(self.decoder, 'moe_top_k') and self.decoder.moe_top_k == 1:
                    balance_weight = 0.1
                else:
                    balance_weight = 0.05
            
            total_loss = detection_loss + balance_weight * moe_load_balance_loss
            
            output['detection_loss'] = detection_loss
            output['moe_load_balance_loss'] = moe_load_balance_loss
            output['total_loss'] = total_loss
            output['loss_dict'] = detection_loss_dict
        
        return output


class AdaptiveExpertTrainer:
    """自适应专家RT-DETR训练器"""
    
    def __init__(self, config: Dict, config_file_path: Optional[str] = None):
        """
        Args:
            config: 训练配置字典
            config_file_path: 配置文件路径
        """
        self.config = config
        self.config_file_path = config_file_path
        
        if config_file_path:
            self._validate_config_file()
        
        if config_file_path:
            if 'misc' not in self.config or 'device' not in self.config['misc']:
                raise ValueError(f"配置文件 {config_file_path} 缺少必需的配置项: misc.device")
            device_str = self.config['misc']['device']
        else:
            device_str = self.config.get('misc', {}).get('device', 'cuda')
        self.device = torch.device(device_str)
        
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.best_loss_epoch = -1
        self.best_map = 0.0
        self.best_map_epoch = -1
        self.best_mAP_50 = 0.0
        self.best_mAP_50_epoch = -1
        self.best_mAP_075 = 0.0
        self.best_mAP_075_epoch = -1
        self.global_step = 0
        self.resume_from_checkpoint = self.config.get('resume_from_checkpoint', None)
        if self.resume_from_checkpoint is None and 'checkpoint' in self.config:
            self.resume_from_checkpoint = self.config['checkpoint'].get('resume_from_checkpoint', None)
        
        self.clip_max_norm = self.config.get('training', {}).get('clip_max_norm', 10.0)

        self.num_classes = int(self.config.get("data", {}).get("num_classes", 8))
        self.class_names = [
            "Car", "Truck", "Van", "Bus", "Pedestrian",
            "Cyclist", "Motorcyclist", "Trafficcone",
        ]
        self.colors = [
            (255, 0, 0),
            (0, 255, 0),
            (255, 128, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (128, 128, 128),
        ]
        self._init_dataset_derived_fields()

        # [新增] 读取 close_mosaic_epochs 参数
        # 优先从 data_augmentation 读取，兼容 augmentation
        aug_config = self.config.get('data_augmentation', {})
        if not aug_config:
            aug_config = self.config.get('augmentation', {})
        self.close_mosaic_epochs = aug_config.get('close_mosaic_epochs', 0)
        
        self._setup_logging()
        self._apply_vram_batch_size_rule()
        self.model = self._create_model()
        self.train_loader, self.val_loader = self._create_data_loaders()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.warmup_scheduler = self._create_warmup_scheduler()
        self.ema = self._create_ema()
        self.scaler = self._create_scaler()
        
        self.visualizer = TrainingVisualizer(log_dir=self.log_dir, model_type='moe', experiment_name=self.experiment_name)
        self.early_stopping = self._create_early_stopping()
        self._setup_inference_components()
        
        if self.resume_from_checkpoint:
            self._resume_from_checkpoint()
            
    def _apply_vram_batch_size_rule(self):
        """根据 VRAM 动态调整 batch_size（与 rt-detr / yolo / cas_detr 共用 common.vram_batch）。"""
        if not torch.cuda.is_available():
            return

        misc = self.config.setdefault('misc', {})
        device_str = str(misc.get('device', 'cuda'))
        idx = resolve_cuda_device_index(device_str)

        orig_bs = int(self.config['training']['batch_size'])
        orig_nw = int(misc.get('num_workers', 4))
        orig_pf = int(misc.get('prefetch_factor', 1))

        r = compute_vram_batch_adjustment(
            orig_bs, orig_nw, orig_pf, device_index=idx
        )
        if r is None:
            return

        self.config['training']['batch_size'] = r.batch_size
        misc['num_workers'] = r.num_workers
        misc['prefetch_factor'] = r.prefetch_factor

        if getattr(self, 'logger', None):
            self.logger.info(format_vram_batch_log(r))
    
    def _validate_config_file(self):
        """验证配置文件是否包含所有必需的配置项"""
        required_keys = {
            'model': ['num_experts', 'backbone', 'hidden_dim', 'num_queries', 'num_decoder_layers', 'top_k'],
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

    def _init_dataset_derived_fields(self) -> None:
        dc = self.config.setdefault("data", {})
        self.num_classes = int(dc.get("num_classes", self.num_classes))
        cn = dc.get("class_names")
        if isinstance(cn, list) and cn:
            self.class_names = [str(x) for x in cn]
        while len(self.colors) < len(self.class_names):
            self.colors.append((128, 128, 128))
        self.colors = self.colors[: len(self.class_names)]
        self._dair_truncation_categorical = (
            dc.get("dataset_class") == "DAIRV2XDetection"
            or "DAIR-V2X" in str(dc.get("data_root", ""))
        )

    def _resolve_dataset_class(self):
        name = self.config.get("data", {}).get("dataset_class", "DAIRV2XDetection")
        if name == "DAIRV2XDetection":
            return DAIRV2XDetection
        if name == "CocoFolderDetection":
            return CocoFolderDetection
        raise ValueError(f"未知 dataset_class: {name}")

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
            # 直接从配置文件读取专家数量
            num_experts = self.config.get('model', {}).get('num_experts', 6)
            expert_num = str(num_experts)
            # 生成实验名称（不带时间戳）
            self.experiment_name = f"moe{expert_num}_rtdetr_{backbone_short}"
            ds_dir = dataset_dir_name(self.config)
            self.log_dir = Path(f"logs/{ds_dir}/{self.experiment_name}_{timestamp}")
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
        """创建模型"""
        encoder_config = self.config['model']['encoder']
        encoder_in_channels = encoder_config['in_channels']
        encoder_expansion = encoder_config['expansion']
        
        num_experts = self.config['model'].get('num_experts', 6)
        moe_balance_weight = self.config.get('training', {}).get('moe_balance_weight', None)
        
        model = AdaptiveExpertRTDETR(
            hidden_dim=self.config['model']['hidden_dim'],
            num_queries=self.config['model']['num_queries'],
            top_k=self.config['model']['top_k'],
            backbone_type=self.config['model']['backbone'],
            num_decoder_layers=self.config['model']['num_decoder_layers'],
            encoder_in_channels=encoder_in_channels,
            encoder_expansion=encoder_expansion,
            num_experts=num_experts,
            moe_balance_weight=moe_balance_weight,
            num_classes=self.num_classes,
        )
        
        pretrained_weights = self.config['model'].get('pretrained_weights', None)
        if pretrained_weights:
            self._load_pretrained_weights(model, pretrained_weights)
        
        model = model.to(self.device)
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.logger.info("✓ 已启用GPU优化: cudnn.benchmark=True, TF32=True")
        
        self.logger.info(f"✓ 创建MOE RT-DETR模型")
        self.logger.info(f"  专家数量: {model.num_experts}")
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
            checkpoint = torch.load(pretrained_file, map_location='cpu', weights_only=False)
            
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
            # 注意：missing_keys 可能包含预训练模型中不存在的参数（如不同专家数量的模型）
            # 只统计预训练模型中实际存在的 missing_keys
            actual_missing_keys = [k for k in missing_keys if k in filtered_state_dict]
            total_params = len(filtered_state_dict)
            loaded_params = total_params - len(actual_missing_keys)
            
            self.logger.info(f"✓ 成功加载预训练权重: {loaded_params}/{total_params} 个参数")
            
            # 报告跳过的类别参数
            if skipped_class_params > 0:
                self.logger.info(f"  - 跳过类别相关参数: {skipped_class_params} 个（COCO 80类 → DAIR-V2X 8类）")
            
            # 统计各部分的参数（只统计预训练模型中实际存在的参数）
            backbone_loaded = sum(1 for k in filtered_state_dict.keys() if k not in actual_missing_keys and 'backbone' in k)
            encoder_loaded = sum(1 for k in filtered_state_dict.keys() if k not in actual_missing_keys and 'encoder' in k)
            decoder_loaded = sum(1 for k in filtered_state_dict.keys() if k not in actual_missing_keys and 'decoder' in k)
            
            self.logger.info(f"  - Backbone: {backbone_loaded} 个参数")
            self.logger.info(f"  - Encoder: {encoder_loaded} 个参数")
            self.logger.info(f"  - Decoder: {decoder_loaded} 个参数")
            
            if len(actual_missing_keys) > 0:
                self.logger.info(f"  - 预训练模型缺少参数: {len(actual_missing_keys)} 个（当前模型新增）")
                if len(actual_missing_keys) <= 5:
                    self.logger.info(f"    示例: {list(actual_missing_keys)}")
                else:
                    self.logger.info(f"    示例: {list(actual_missing_keys)[:3]} ...")
            
            # 如果 missing_keys 中有预训练模型中不存在的参数，说明是模型结构差异
            model_only_missing = [k for k in missing_keys if k not in filtered_state_dict]
            if len(model_only_missing) > 0:
                self.logger.debug(f"  - 模型结构差异导致的 missing_keys: {len(model_only_missing)} 个（预训练模型中不存在，不影响加载统计）")
            
            if len(unexpected_keys) > 0:
                self.logger.info(f"  - 模型新增参数: {len(unexpected_keys)} 个（将随机初始化）")
                
        except Exception as e:
            self.logger.error(f"✗ 加载预训练权重失败: {e}")
            self.logger.info("将从随机初始化开始训练")
    
    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """创建数据加载器。"""
        from src.data.dataloader import BatchImageCollateFuncion
        
        raw_root = self.config['data']['data_root']
        resolved_root = resolve_detr_data_root(raw_root)
        if resolved_root != raw_root:
            self.logger.info(f"✓ 数据集根目录: {raw_root} → {resolved_root}")
        self.config['data']['data_root'] = resolved_root

        # 修改：移除不必要的max()，使用配置值
        batch_size = self.config['training']['batch_size']
        target_size = self.model.image_size
        
        # 获取基础 Dataloader 配置
        num_workers = self.config.get('misc', {}).get('num_workers', 16)
        pin_memory = self.config.get('misc', {}).get('pin_memory', True)
        prefetch_factor = self.config.get('misc', {}).get('prefetch_factor', 4)
        
        # 从config中读取augmentation配置
        augmentation_config = self.config.get('augmentation', {})
        # 如果target_size被指定，覆盖augmentation_config中的值
        if target_size != 640:
            augmentation_config = augmentation_config.copy()
            augmentation_config['target_size'] = target_size

        ds_cls = self._resolve_dataset_class()
        train_dataset = ds_cls(
            data_root=self.config['data']['data_root'],
            split='train',
            augmentation_config=augmentation_config
        )
        
        val_dataset = ds_cls(
            data_root=self.config['data']['data_root'],
            split='val',
            augmentation_config=augmentation_config
        )

        # 多尺度训练配置 (从config中读取或使用默认值)
        scales = self.config.get('augmentation', {}).get('scales', [576, 608, 640, 640, 640, 672, 704])
        stop_epoch = self.config.get('augmentation', {}).get('stop_epoch', 71)
        train_collate_fn = BatchImageCollateFuncion(scales=scales, stop_epoch=stop_epoch)
        # Keep validation deterministic at the dataset's fixed resize target.
        val_collate_fn = BatchImageCollateFuncion(scales=None, stop_epoch=stop_epoch)

        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            collate_fn=train_collate_fn,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            collate_fn=val_collate_fn,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )
        
        self.val_dataset = val_dataset
        
        self.logger.info(f"✓ 创建数据加载器")
        self.logger.info(f"  训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}")
        self.logger.info(f"  数据加载配置: num_workers={num_workers}, prefetch_factor={prefetch_factor}, pin_memory={pin_memory}")
        
        return train_loader, val_loader
    
    def _create_optimizer(self) -> optim.AdamW:
        """创建优化器（使用分组学习率，与rt-detr保持一致）。"""
        # 获取配置中的学习率，确保是浮点数类型
        new_lr = float(self.config['training']['new_lr'])
        pretrained_lr = float(self.config['training']['pretrained_lr'])
        weight_decay = float(self.config['training'].get('weight_decay', 0.0001))
        
        # 分组参数（与rt-detr保持一致的分组策略）
        param_groups = []
        
        # 定义新增结构的关键词（MoE、CaS_DETR等）
        # 基于实际代码中的模块命名：
        # - decoder.layers.X.decoder_moe_layer.* (MoE-RTDETR的decoder MoE)
        new_structure_keywords = [
            'decoder_moe_layer'   # decoder中的MoE层
        ]
        
        # 1. 预训练参数组（backbone、encoder、decoder的标准层，排除norm层和新增结构）
        pretrained_params = []
        pretrained_names = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # 判断是否为预训练部分（backbone、encoder、decoder）
                is_pretrained = any(part in name for part in ['backbone', 'encoder', 'decoder'])
                # 排除norm层
                is_norm = any(norm in name for norm in ['norm', 'bn', 'gn', 'ln'])
                # 排除新增结构（即使它们在encoder/decoder中）
                is_new_structure = any(keyword in name.lower() for keyword in new_structure_keywords)
                
                if is_pretrained and not is_norm and not is_new_structure:
                    pretrained_params.append(param)
                    pretrained_names.append(name)
        
        if pretrained_params:
            param_groups.append({
                'params': pretrained_params,
                'lr': pretrained_lr,
                'weight_decay': weight_decay
            })
            self.logger.info(f"✓ 预训练参数组: {len(pretrained_params)} 个参数, lr={pretrained_lr}")
        
        # 2. Norm层参数（无weight decay）
        norm_params = []
        norm_names = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and any(norm in name for norm in ['norm', 'bn', 'gn', 'ln']):
                norm_params.append(param)
                norm_names.append(name)
        
        if norm_params:
            param_groups.append({
                'params': norm_params,
                'lr': new_lr,
                'weight_decay': 0.0  # Norm层不使用weight decay
            })
            self.logger.info(f"✓ Norm层参数组: {len(norm_params)} 个参数, lr={new_lr}, wd=0")
        
        # 3. 新参数组（MoE层、CaS_DETR层等新增结构，即使它们在encoder/decoder中）
        new_params = []
        new_names = []
        processed_params = set(id(p) for p in pretrained_params + norm_params)
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and id(param) not in processed_params:
                new_params.append(param)
                new_names.append(name)
        
        if new_params:
            param_groups.append({
                'params': new_params,
                'lr': new_lr,
                'weight_decay': weight_decay
            })
            self.logger.info(f"✓ 新参数组: {len(new_params)} 个参数, lr={new_lr}")
        
        optimizer = optim.AdamW(
            param_groups,
            betas=(0.9, 0.999)
        )
        
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
        
        return EarlyStopping(patience=patience, metric_name=metric_name, logger=self.logger)

    def _early_stopping_metric_improved(self, val_metrics: Dict[str, float]) -> bool:
        """与 is_best_* 同一套比较规则，在更新 best_* 之前调用。"""
        name = self.config['training'].get('early_stopping_metric', 'mAP_0.5_0.95')
        if 'loss' in name.lower():
            return val_metrics.get('total_loss', float('inf')) < self.best_loss
        if 'mAP_0.75' in name:
            return val_metrics.get('mAP_0.75', 0.0) > self.best_mAP_075
        if 'mAP_0.5_0.95' in name or 'mAP_0.5:0.95' in name:
            return val_metrics.get('mAP_0.5_0.95', 0.0) > self.best_map
        if 'mAP_0.5' in name:
            return val_metrics.get('mAP_0.5', 0.0) > self.best_mAP_50
        return val_metrics.get('mAP_0.5_0.95', 0.0) > self.best_map

    def _early_stopping_best_snapshot(self) -> Tuple[float, int]:
        """更新 best_* 之后，用于早停日志的「当前监控指标最佳值 / epoch」。"""
        name = self.config['training'].get('early_stopping_metric', 'mAP_0.5_0.95')
        if 'loss' in name.lower():
            return self.best_loss, self.best_loss_epoch
        if 'mAP_0.75' in name:
            return self.best_mAP_075, self.best_mAP_075_epoch
        if 'mAP_0.5_0.95' in name or 'mAP_0.5:0.95' in name:
            return self.best_map, self.best_map_epoch
        if 'mAP_0.5' in name:
            return self.best_mAP_50, self.best_mAP_50_epoch
        return self.best_map, self.best_map_epoch

    def _setup_inference_components(self) -> None:
        """初始化推理相关组件"""
        # 创建后处理器
        self.postprocessor = DetDETRPostProcessor(
            num_classes=self.num_classes,
            use_focal_loss=True,
            num_top_queries=300,
            box_process_format=BoxProcessFormat.RESIZE
        )
        
        # 创建推理输出目录
        self.inference_output_dir = self.log_dir / "inference_samples"
        self.inference_output_dir.mkdir(parents=True, exist_ok=True)
        
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
            
            ds = self.val_dataset
            orig_image_path = None
            if hasattr(ds, "get_image_path"):
                orig_image_path = ds.get_image_path(int(image_id))
            if orig_image_path is None or not orig_image_path.exists():
                data_root = Path(self.config['data']['data_root'])
                orig_image_path = data_root / "image" / f"{int(image_id):06d}.jpg"
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
                
                # 根据 box_revert 文档，orig_sizes 应该是 (w, h)
                _, _, h, w = single_image.shape
                eval_sizes = torch.tensor([[w, h]], device=self.device)
                
                results = self.postprocessor(outputs, orig_sizes=eval_sizes)
                
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
    
    def _build_test_dataloader_optional(self):
        if not self.config.get("data", {}).get("eval_test_after_training", True):
            return None
        from src.data.dataloader import BatchImageCollateFuncion

        ds_cls = self._resolve_dataset_class()
        augmentation_config = self.config.get("augmentation", {})
        try:
            test_dataset = ds_cls(
                data_root=self.config["data"]["data_root"],
                split="test",
                augmentation_config=augmentation_config,
            )
        except FileNotFoundError:
            return None
        if len(test_dataset) == 0:
            return None

        scales = self.config.get("augmentation", {}).get(
            "scales", [576, 608, 640, 640, 640, 672, 704]
        )
        stop_epoch = self.config.get("augmentation", {}).get("stop_epoch", 71)
        # Keep test-time evaluation deterministic at the dataset's fixed resize target.
        collate_fn = BatchImageCollateFuncion(scales=None, stop_epoch=stop_epoch)
        num_workers = self.config.get("misc", {}).get("num_workers", 16)
        pin_memory = self.config.get("misc", {}).get("pin_memory", True)
        prefetch_factor = self.config.get("misc", {}).get("prefetch_factor", 4)
        return DataLoader(
            test_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )

    def _run_ema_eval_on_dataloader(self, dataloader, split_label: str,
                                     best_epoch=None, bench_dict=None):
        all_predictions = []
        all_targets = []
        current_h, current_w = 640, 640

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                _, _, current_h, current_w = images.shape
                images = images.to(self.device, non_blocking=True)
                targets = [
                    {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                    for t in targets
                ]

                outputs = self.ema.module(images, targets)
                has_predictions = ("pred_logits" in outputs and "pred_boxes" in outputs) or (
                    "class_scores" in outputs and "bboxes" in outputs
                )
                if has_predictions:
                    self._collect_predictions(
                        outputs, targets, batch_idx, all_predictions, all_targets, current_w, current_h
                    )

        if len(all_predictions) == 0 or len(all_targets) == 0:
            self.logger.warning(f"best_model 在 {split_label} 上无有效预测或标注，跳过 AP")
            return

        metrics = self._compute_map_metrics(
            all_predictions,
            all_targets,
            img_h=current_h,
            img_w=current_w,
            print_per_category=True,
            compute_difficulty=True,
        )

        log_detr_eval_summary(self.logger, split_label, metrics, bench_dict)
        csv_path = write_detr_eval_csv(
            self.log_dir, self.config, self.experiment_name,
            split_label, metrics, self.class_names, bench_dict,
        )
        self.logger.info(f"✓ best_model [{split_label}] 评估完成 (epoch={best_epoch})  → {csv_path}")

    def _evaluate_best_model_and_print_all_ap(self):
        """训练结束后在 val 上评估；若存在 test 则再评一次。"""
        evaluate_best_model_after_training(
            log_dir=self.log_dir,
            device=self.device,
            config=self.config,
            experiment_name=self.experiment_name,
            logger=self.logger,
            ema=self.ema,
            val_loader=self.val_loader,
            build_test_loader_fn=self._build_test_dataloader_optional,
            run_eval_fn=self._run_ema_eval_on_dataloader,
        )

    def _run_inference_on_best_model(self, best_ema_state=None, best_epoch=None):
        """使用best_model运行推理，输出5张验证图像的推理结果
        
        Args:
            best_ema_state: best_model的EMA模型state_dict，如果提供则使用它进行推理
            best_epoch: best_model保存时的epoch，用于文件名
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
            inference_images = inference_images.to(self.device, non_blocking=True)
            inference_targets = [{k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                                 for k, v in t.items()} for t in inference_targets]
            
            # 打印前5张推理结果
            batch_size = len(inference_targets)
            num_inference_images = min(5, batch_size)
            # 使用best_epoch作为文件名，如果没有提供则使用current_epoch（向后兼容）
            epoch_for_filename = best_epoch if best_epoch is not None else self.current_epoch
            self.logger.info(f"  生成best_model推理结果（前{num_inference_images}张，epoch={epoch_for_filename}）...")
            
            for img_idx in range(num_inference_images):
                self._inference_single_image_from_batch(
                    inference_images, inference_targets, 0, image_idx=img_idx,
                    suffix=f"best_model_epoch_{epoch_for_filename}"
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
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # 恢复状态
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint.get('epoch', 0) + 1
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            self.best_loss_epoch = checkpoint.get('best_loss_epoch', -1)
            self.best_map = checkpoint.get('best_map', 0.0)
            self.best_map_epoch = checkpoint.get('best_map_epoch', -1)
            self.best_mAP_50 = checkpoint.get('best_mAP_50', 0.0)
            self.best_mAP_50_epoch = checkpoint.get('best_mAP_50_epoch', -1)
            self.best_mAP_075 = checkpoint.get('best_mAP_075', 0.0)
            self.best_mAP_075_epoch = checkpoint.get('best_mAP_075_epoch', -1)
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
        total_loss = torch.tensor(0.0, device=self.device)
        detection_loss = torch.tensor(0.0, device=self.device)
        moe_lb_loss = torch.tensor(0.0, device=self.device)
        
        # 统计细粒度MoE的专家使用率（跨所有Decoder层聚合）
        expert_usage_count = [0] * self.model.num_experts
        total_tokens = 0
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            
            images = images.to(self.device, non_blocking=True)
            targets = [{k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
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
            
            # 统计损失（GPU 上累加，避免每 batch .item() 同步）
            total_loss += loss.detach()
            if isinstance(outputs, dict):
                if 'detection_loss' in outputs:
                    detection_loss += outputs['detection_loss'].detach()
                if 'moe_load_balance_loss' in outputs:
                    moe_lb_loss += outputs['moe_load_balance_loss'].detach()
            
            # 收集细粒度MoE的专家使用统计（使用torch.bincount向量化）
            if self.model.decoder.use_moe:
                for layer in self.model.decoder.decoder.layers:
                    if hasattr(layer, 'decoder_moe_layer') and layer.decoder_moe_layer.router_logits_cache is not None:
                        router_logits = layer.decoder_moe_layer.router_logits_cache  # [N, num_experts]
                        # 计算每个token选择的top-k专家
                        _, top_indices = torch.topk(router_logits, self.model.decoder.moe_top_k, dim=-1)  # [N, K]
                        
                        # 使用 torch.bincount 在 GPU 上直接统计（向量化）
                        flat_indices = top_indices.flatten()
                        counts = torch.bincount(flat_indices, minlength=self.model.num_experts)
                        
                        # 只把最终的几个数字搬回CPU
                        current_counts = counts.cpu().tolist()
                        
                        # 累加
                        for i in range(self.model.num_experts):
                            if i < len(current_counts):
                                expert_usage_count[i] += current_counts[i]
                        
                        total_tokens += router_logits.shape[0] * self.model.decoder.moe_top_k
            
            if batch_idx % 100 == 0:
                det_loss_val = outputs.get('detection_loss', torch.tensor(0.0)).item() if isinstance(outputs, dict) else 0.0
                moe_loss_val = outputs.get('moe_load_balance_loss', torch.tensor(0.0)).item() if isinstance(outputs, dict) else 0.0
                self.logger.info(f'Epoch {self.current_epoch} | Batch {batch_idx} | '
                               f'Loss: {loss.item():.2f} (Det: {det_loss_val:.2f}, MoE: {moe_loss_val:.4f})')
            
            self.global_step += 1
        
        # 计算平均值（一次性 .item() 同步）
        num_batches = len(self.train_loader)
        avg_loss = total_loss.item() / num_batches
        avg_detection_loss = detection_loss.item() / num_batches
        avg_moe_lb_loss = moe_lb_loss.item() / num_batches
        
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
        
        # 初始化默认尺寸 (防止 val_loader 为空)
        current_h, current_w = 640, 640
        
        # 前30个epoch只计算loss，不进行cocoEval评估
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(self.val_loader):
                # 动态获取 Tensor 尺寸
                B, C, H_tensor, W_tensor = images.shape
                current_h, current_w = H_tensor, W_tensor

                images = images.to(self.device, non_blocking=True)
                targets = [{k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                           for k, v in t.items()} for t in targets]
                
                outputs = self.ema.module(images, targets)
                
                if isinstance(outputs, dict):
                    if 'total_loss' in outputs:
                        total_loss += outputs['total_loss'].item()
                    
                    # 收集预测结果（只在需要计算mAP时收集，前30个epoch跳过）
                    if 'class_scores' in outputs and 'bboxes' in outputs:
                        self._collect_predictions(outputs, targets, batch_idx, all_predictions, all_targets, W_tensor, H_tensor)
        
        # 保存预测结果用于后续打印每个类别mAP（避免重复计算）
        self._last_val_predictions = all_predictions
        self._last_val_targets = all_targets
        
        avg_loss = total_loss / len(self.val_loader)
        
        # 计算mAP（不计算每个类别的mAP，只在best_model时计算）
        mAP_metrics = self._compute_map_metrics(all_predictions, all_targets, 
                                              img_h=current_h, img_w=current_w,
                                              print_per_category=False)
        
        return {
            'total_loss': avg_loss,
            'mAP_0.5': mAP_metrics.get('mAP_0.5', 0.0),
            'mAP_0.75': mAP_metrics.get('mAP_0.75', 0.0),
            'mAP_0.5_0.95': mAP_metrics.get('mAP_0.5_0.95', 0.0)
        }
    
    @staticmethod
    def _cxcywh_to_xywh_orig(boxes: torch.Tensor, img_w: int, img_h: int,
                              orig_w: float, orig_h: float) -> np.ndarray:
        """cxcywh (归一化或像素) → COCO xywh in original image coords. Returns numpy."""
        scale = min(img_w / orig_w, img_h / orig_h)
        normalized = boxes.max() <= 1.01
        if normalized:
            cx = boxes[:, 0] * img_w
            cy = boxes[:, 1] * img_h
            bw = boxes[:, 2] * img_w
            bh = boxes[:, 3] * img_h
        else:
            cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        out = torch.stack([
            ((cx - bw / 2) / scale).clamp(0, orig_w),
            ((cy - bh / 2) / scale).clamp(0, orig_h),
            (bw / scale).clamp(1, orig_w),
            (bh / scale).clamp(1, orig_h),
        ], dim=1)
        return out.cpu().numpy()

    def _collect_predictions(self, outputs: Dict, targets: List[Dict], batch_idx: int,
                            all_predictions: List, all_targets: List, img_w: int, img_h: int) -> None:
        """向量化收集预测/GT，单次 .cpu().numpy() 替代逐元素 .item()。"""
        if 'class_scores' in outputs:
            pred_logits = outputs['class_scores']
            pred_boxes = outputs['bboxes']
        elif 'pred_logits' in outputs:
            pred_logits = outputs['pred_logits']
            pred_boxes = outputs['pred_boxes']
        else:
            return

        bs = self.config['training']['batch_size']
        batch_size = pred_logits.shape[0]

        for i in range(batch_size):
            image_id = batch_idx * bs + i
            orig_h, orig_w = targets[i]['orig_size'].tolist()

            scores_sig = torch.sigmoid(pred_logits[i])
            max_scores, pred_cls = scores_sig.max(dim=-1)
            valid = ~torch.all(pred_boxes[i] == 1.0, dim=1)
            if valid.any():
                fb = pred_boxes[i][valid]
                fc = pred_cls[valid]
                fs = max_scores[valid]
                boxes_np = self._cxcywh_to_xywh_orig(fb, img_w, img_h, orig_w, orig_h)
                cls_np = (fc.cpu().numpy() + 1).tolist()
                scores_np = fs.cpu().numpy().tolist()
                all_predictions.extend(
                    {'image_id': image_id, 'category_id': c, 'bbox': boxes_np[j].tolist(), 'score': s}
                    for j, (c, s) in enumerate(zip(cls_np, scores_np))
                )

            if i >= len(targets) or 'labels' not in targets[i] or 'boxes' not in targets[i]:
                continue
            true_labels = targets[i]['labels']
            true_boxes = targets[i]['boxes']
            if len(true_labels) == 0:
                continue

            gt_boxes_np = self._cxcywh_to_xywh_orig(true_boxes, img_w, img_h, orig_w, orig_h)
            gt_cls_np = (true_labels.cpu().numpy() + 1)
            gt_area = gt_boxes_np[:, 2] * gt_boxes_np[:, 3]
            gt_h = gt_boxes_np[:, 3]

            has_iscrowd = 'iscrowd' in targets[i]
            has_occ = 'occluded_state' in targets[i]
            has_trunc = 'truncated_state' in targets[i]
            iscrowd_np = targets[i]['iscrowd'].cpu().numpy() if has_iscrowd else None
            occ_np = targets[i]['occluded_state'].cpu().numpy() if has_occ else None
            trunc_np = targets[i]['truncated_state'].cpu().numpy() if has_trunc else None

            for j in range(len(gt_cls_np)):
                ann = {
                    'image_id': image_id,
                    'category_id': int(gt_cls_np[j]),
                    'bbox': gt_boxes_np[j].tolist(),
                    'area': float(gt_area[j]),
                    'bbox_height': float(gt_h[j]),
                }
                if iscrowd_np is not None:
                    ann['iscrowd'] = int(iscrowd_np[j])
                if occ_np is not None:
                    ann['occluded_state'] = float(occ_np[j])
                if trunc_np is not None:
                    ann['truncated_state'] = float(trunc_np[j])
                all_targets.append(ann)

    def _get_kitti_difficulty(self, target: Dict) -> str:
        """与 YOLO 侧 ``categorize_by_kitti_difficulty`` + DAIR 截断映射一致。"""
        return kitti_difficulty_from_coco_ann(
            target,
            dair_categorical_trunc=self._dair_truncation_categorical,
        )

    def _run_coco_eval(self, predictions: List[Dict], targets: List[Dict], categories: List[Dict],
                       img_h: int, img_w: int, print_summary: bool = False):
        """执行一次 COCOeval，返回 coco_eval 对象；无目标时返回 None。"""
        if len(targets) == 0:
            return None

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

        image_ids = set(target['image_id'] for target in targets)
        for img_id in image_ids:
            coco_gt['images'].append({'id': img_id, 'width': img_w, 'height': img_h})

        for i, target in enumerate(targets):
            ann = target.copy()
            ann['id'] = i + 1
            coco_gt['annotations'].append(ann)

        from io import StringIO
        import sys

        coco_gt_obj = COCO()
        coco_gt_obj.dataset = coco_gt

        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            coco_gt_obj.createIndex()
            coco_dt = coco_gt_obj.loadRes(predictions)
            coco_eval = COCOeval(coco_gt_obj, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
        finally:
            sys.stdout = old_stdout

        if print_summary:
            coco_eval.summarize()
        else:
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                coco_eval.summarize()
            finally:
                sys.stdout = old_stdout

        return coco_eval

    def _compute_difficulty_aps(self, predictions: List[Dict], targets: List[Dict], categories: List[Dict],
                                img_h: int, img_w: int) -> Dict[str, float]:
        """计算 KITTI 风格 AP_easy / AP_moderate / AP_hard。"""
        import copy
        easy_targets = []
        moderate_targets = []
        hard_targets = []

        for target in targets:
            level = self._get_kitti_difficulty(target)
            
            # 独立模式：每档只评该档 GT，其余难度作 iscrowd=1（不计 FP）
            t_easy = copy.deepcopy(target)
            if level != 'easy':
                t_easy['iscrowd'] = 1
            easy_targets.append(t_easy)

            t_mod = copy.deepcopy(target)
            if level != 'moderate':
                t_mod['iscrowd'] = 1
            moderate_targets.append(t_mod)

            t_hard = copy.deepcopy(target)
            if level != 'hard':
                t_hard['iscrowd'] = 1
            hard_targets.append(t_hard)

        easy_eval = self._run_coco_eval(predictions, easy_targets, categories, img_h, img_w, print_summary=False)
        moderate_eval = self._run_coco_eval(predictions, moderate_targets, categories, img_h, img_w, print_summary=False)
        hard_eval = self._run_coco_eval(predictions, hard_targets, categories, img_h, img_w, print_summary=False)

        return {
            "AP_easy": coco_ap_at_iou50_all(easy_eval),
            "AP_moderate": coco_ap_at_iou50_all(moderate_eval),
            "AP_hard": coco_ap_at_iou50_all(hard_eval),
        }
    
    def _extract_per_category_ap_from_eval(
        self, coco_eval, categories: List[Dict]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """从一次 COCOeval 的 precision[T,R,K,A,M] 提取各类别 AP@50 与 AP@0.5:0.95（T=0 为 IoU=0.5）。"""
        per_cat_50 = {cat["name"]: 0.0 for cat in categories}
        per_cat_5095 = {cat["name"]: 0.0 for cat in categories}
        if coco_eval is None or not hasattr(coco_eval, "eval") or "precision" not in coco_eval.eval:
            return per_cat_50, per_cat_5095

        try:
            precision = coco_eval.eval["precision"]
            area_index = 0
            max_det_index = len(coco_eval.params.maxDets) - 1
            cat_id_to_index = {cat_id: idx for idx, cat_id in enumerate(coco_eval.params.catIds)}

            for cat in categories:
                cat_id = cat["id"]
                cat_name = cat["name"]
                if cat_id not in cat_id_to_index:
                    continue

                cat_index = cat_id_to_index[cat_id]
                p50 = precision[0, :, cat_index, area_index, max_det_index]
                v50 = p50[p50 > -1]
                per_cat_50[cat_name] = float(np.mean(v50)) if v50.size > 0 else 0.0
                p5095 = precision[:, :, cat_index, area_index, max_det_index]
                v5095 = p5095[p5095 > -1]
                per_cat_5095[cat_name] = float(np.mean(v5095)) if v5095.size > 0 else 0.0
        except Exception as e:
            self.logger.debug(f"从COCOeval提取每类AP失败: {e}")

        return per_cat_50, per_cat_5095
    
    def _compute_map_metrics(self, predictions: List[Dict], targets: List[Dict], 
                             img_h: int = 640, img_w: int = 640,
                             print_per_category: bool = False,
                             compute_difficulty: bool = False) -> Dict[str, float]:
        """计算mAP指标。
        
        Args:
            predictions: 预测结果列表
            targets: 真实标签列表
            img_h: 图像高度 (Tensor Shape)
            img_w: 图像宽度 (Tensor Shape)
            print_per_category: 是否打印每个类别的详细mAP（默认False，只在best_model时打印）
        """
        try:
            if len(predictions) == 0:
                return {
                    'mAP_0.5': 0.0,
                    'mAP_0.75': 0.0,
                    'mAP_0.5_0.95': 0.0,
                    'AP_small': 0.0,
                    'AP_medium': 0.0,
                    'AP_large': 0.0,
                    'AP_small_50': 0.0,
                    'AP_medium_50': 0.0,
                    'AP_large_50': 0.0,
                    'AP_easy': 0.0,
                    'AP_moderate': 0.0,
                    'AP_hard': 0.0
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
                    'width': img_w,
                    'height': img_h
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
            
            # 每类 AP@50 / AP@0.5:0.95 从同一次 COCOeval 的 precision 张量提取
            per_cat_50: Dict[str, float] = {}
            per_cat_5095: Dict[str, float] = {}
            if print_per_category:
                per_cat_50, per_cat_5095 = self._extract_per_category_ap_from_eval(
                    coco_eval, categories
                )
            
            difficulty_metrics = {
                'AP_easy': 0.0,
                'AP_moderate': 0.0,
                'AP_hard': 0.0,
            }
            if compute_difficulty:
                difficulty_metrics = self._compute_difficulty_aps(predictions, targets, categories, img_h, img_w)
            
            s50, m50, l50 = coco_area_ap_at_iou50(coco_eval)
            result = {
                'mAP_0.5': coco_eval.stats[1],
                'mAP_0.75': coco_eval.stats[2],
                'mAP_0.5_0.95': coco_eval.stats[0],
                'AP_small': coco_eval.stats[3] if len(coco_eval.stats) > 3 else 0.0,
                'AP_medium': coco_eval.stats[4] if len(coco_eval.stats) > 4 else 0.0,
                'AP_large': coco_eval.stats[5] if len(coco_eval.stats) > 5 else 0.0,
                'AP_small_50': s50,
                'AP_medium_50': m50,
                'AP_large_50': l50,
                'AP_easy': difficulty_metrics['AP_easy'],
                'AP_moderate': difficulty_metrics['AP_moderate'],
                'AP_hard': difficulty_metrics['AP_hard'],
            }
            
            for cat_name in per_cat_5095.keys():
                result[f"AP50_{cat_name}"] = per_cat_50.get(cat_name, 0.0)
                result[f"AP5095_{cat_name}"] = per_cat_5095[cat_name]
            
            return result
            
        except Exception as e:
            self.logger.warning(f"mAP计算失败: {e}")
            return {
                'mAP_0.5': 0.0,
                'mAP_0.75': 0.0,
                'mAP_0.5_0.95': 0.0,
                'AP_small': 0.0,
                'AP_medium': 0.0,
                'AP_large': 0.0,
                'AP_small_50': 0.0,
                'AP_medium_50': 0.0,
                'AP_large_50': 0.0,
                'AP_easy': 0.0,
                'AP_moderate': 0.0,
                'AP_hard': 0.0
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
            'best_loss_epoch': getattr(self, 'best_loss_epoch', -1),
            'best_map': self.best_map,
            'best_map_epoch': getattr(self, 'best_map_epoch', -1),
            'best_mAP_50': getattr(self, 'best_mAP_50', 0.0),
            'best_mAP_50_epoch': getattr(self, 'best_mAP_50_epoch', -1),
            'best_mAP_075': getattr(self, 'best_mAP_075', 0.0),
            'best_mAP_075_epoch': getattr(self, 'best_mAP_075_epoch', -1),
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
            'best_loss_epoch': getattr(self, 'best_loss_epoch', -1),
            'best_map': self.best_map,
            'best_map_epoch': getattr(self, 'best_map_epoch', -1),
            'best_mAP_50': getattr(self, 'best_mAP_50', 0.0),
            'best_mAP_50_epoch': getattr(self, 'best_mAP_50_epoch', -1),
            'best_mAP_075': getattr(self, 'best_mAP_075', 0.0),
            'best_mAP_075_epoch': getattr(self, 'best_mAP_075_epoch', -1),
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
        eval_sched = self.config.get("training", {}).get("eval_schedule")
        self.logger.info(f"✓ 验证策略: {describe_eval_schedule(eval_sched)}")

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # 更新训练集 epoch
            if hasattr(self.train_loader, 'set_epoch'):
                self.train_loader.set_epoch(epoch)
            elif hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            if hasattr(self.train_loader, 'dataset') and hasattr(self.train_loader.dataset, 'set_epoch'):
                self.train_loader.dataset.set_epoch(epoch)
            if hasattr(self.train_loader.collate_fn, 'set_epoch'):
                self.train_loader.collate_fn.set_epoch(epoch)
            
            # 训练一个epoch
            train_metrics = self.train_epoch()
            
            should_validate = should_run_validation(epoch, eval_sched)

            if should_validate:
                val_metrics = self.validate()
            else:
                val_metrics = {}
            
            # 学习率调度
            if self.current_epoch < self.warmup_scheduler.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                self.scheduler.step()
            
            # 输出日志
            self.logger.info(f"Epoch {epoch}:")
            if should_validate:
                current_map = val_metrics.get('mAP_0.5_0.95', 0.0)
                current_map_50 = val_metrics.get('mAP_0.5', 0.0)
                self.logger.info(f"  训练损失: {train_metrics.get('total_loss', 0.0):.2f} | 验证损失: {val_metrics.get('total_loss', 0.0):.2f}")
                self.logger.info(f"  📊 当前mAP: {current_map:.4f} (mAP@50: {current_map_50:.4f})")
            else:
                self.logger.info(f"  训练损失: {train_metrics.get('total_loss', 0.0):.2f} | 验证损失: Skipped")
            
            # 显示详细损失（与验证频率保持一致，或者始终显示）
            # 这里改为始终显示详细损失，因为不验证时也需要监控训练Loss
            should_show_details = True
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
            
            # 保存检查点 - 仅在实际验证时更新 best_*；早停只计数
            if should_validate:
                es_improved = self._early_stopping_metric_improved(val_metrics)
                vl = val_metrics.get('total_loss', float('inf'))
                v5095 = val_metrics.get('mAP_0.5_0.95', 0.0)
                v50 = val_metrics.get('mAP_0.5', 0.0)
                v75 = val_metrics.get('mAP_0.75', 0.0)
                is_best_loss = vl < self.best_loss
                is_best_map = v5095 > self.best_map
                is_best_map50 = v50 > self.best_mAP_50
                is_best_map75 = v75 > self.best_mAP_075

                if is_best_loss:
                    self.best_loss = vl
                    self.best_loss_epoch = epoch
                    self.logger.info(f"  🎉 新的最佳验证损失: {self.best_loss:.2f}")

                if is_best_map:
                    self.best_map = v5095
                    self.best_map_epoch = epoch
                    self.logger.info(f"  🎉 新的最佳mAP: {self.best_map:.4f}")
                    self.save_checkpoint(epoch, is_best=True)

                if is_best_map50:
                    self.best_mAP_50 = v50
                    self.best_mAP_50_epoch = epoch
                if is_best_map75:
                    self.best_mAP_075 = v75
                    self.best_mAP_075_epoch = epoch

                if self.early_stopping:
                    bv, be = self._early_stopping_best_snapshot()
                    if self.early_stopping.step(es_improved, best_value=bv, best_epoch=be):
                        self.logger.info(f"Early Stopping在epoch {epoch}触发，停止训练")
                        break
            
            if epoch % 5 == 0 or epoch == epochs - 1:
                self.save_latest_checkpoint(epoch)
                try:
                    self.visualizer.plot()
                except Exception as e:
                    self.logger.warning(f"绘制训练曲线失败: {e}")
        
        self.logger.info("✓ 训练完成！")
        try:
            self.visualizer.plot()
            self.visualizer.export_to_csv()
        except Exception as e:
            self.logger.warning(f"绘制训练曲线失败: {e}")

        self._evaluate_best_model_and_print_all_ap()
        try:
            best_model_path = self.log_dir / 'best_model.pth'
            latest_checkpoint_path = self.log_dir / 'latest_checkpoint.pth'
            checkpoint_path = best_model_path if best_model_path.exists() else latest_checkpoint_path
            if checkpoint_path.exists():
                if checkpoint_path == latest_checkpoint_path:
                    self.logger.warning("未找到best_model.pth，改用latest_checkpoint.pth进行推理")
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                best_ema_state = checkpoint.get('ema_state_dict', None)
                best_epoch = checkpoint.get('epoch', None)
                self._run_inference_on_best_model(best_ema_state, best_epoch=best_epoch)
            else:
                self.logger.warning("未找到best_model.pth和latest_checkpoint.pth，跳过推理")
        except Exception as e:
            self.logger.warning(f"训练结束时推理失败（不影响训练结果）: {e}")


def main() -> None:
    """主函数。"""
    
    # ==========================================
    # [新增] 解决 AutoDL/Docker 共享内存不足导致的死锁问题
    # ==========================================
    import torch.multiprocessing
    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
        print("✓ 已设置多进程共享策略为: file_system (防止共享内存溢出)")
    except:
        pass
    # ==========================================
    
    parser = argparse.ArgumentParser(description='自适应专家RT-DETR训练')
    parser.add_argument('--config', type=str, default='A', 
                       help='专家配置 (A: 6专家, B: 3专家) 或YAML配置文件路径')
    parser.add_argument('--backbone', type=str, default='presnet34', 
                       choices=['presnet18', 'presnet34', 'presnet50', 'presnet101',
                               'hgnetv2_l', 'hgnetv2_x', 'hgnetv2_h',
                               'cspresnet_s', 'cspresnet_m', 'cspresnet_l', 'cspresnet_x',
                               'cspdarknet', 'mresnet'],
                       help='Backbone类型')
    parser.add_argument('--data_root', type=str, default='/root/autodl-fs/datasets/DAIR-V2X', 
                       help='DAIR-V2X数据集路径')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小 (RTX 5090优化)')
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
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='数据集键名或别名（与 yolo/configs/datasets.yaml 一致）',
    )
    parser.add_argument(
        '--dataset_registry',
        type=str,
        default=str(default_detr_registry_path()),
        help='数据集注册表 YAML 路径',
    )
    
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
        if args.batch_size != 16:
            config['training']['batch_size'] = args.batch_size
        if args.pretrained_lr != 1e-5:
            config['training']['pretrained_lr'] = args.pretrained_lr
        if args.new_lr != 1e-4:
            config['training']['new_lr'] = args.new_lr
        if args.top_k != 3:
            config['model']['top_k'] = args.top_k
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
                'use_mosaic': False,  # 禁用Mosaic，不适合路测探头场景（会破坏空间关系）
                'warmup_epochs': 3,
                'ema_decay': 0.9999
            },
            'misc': {
                'device': 'cuda',
                'num_workers': 16,
                'pin_memory': True
            },
            'data_augmentation': {
                'brightness': 0.4,
                'contrast': 0.4,
                'saturation': 0.7,
                'hue': 0.015,
                'flip_prob': 0.5,
                'color_jitter_prob': 0.0
            }
        }
        
        if args.pretrained_weights:
            config['model']['pretrained_weights'] = args.pretrained_weights
        
        if args.resume_from_checkpoint:
            config['checkpoint'] = {'resume_from_checkpoint': args.resume_from_checkpoint}

    registry_path = Path(args.dataset_registry)
    if not registry_path.is_absolute():
        registry_path = Path(__file__).resolve().parent / registry_path
    datasets_map = load_dataset_registry(registry_path)
    if args.dataset:
        profile = resolve_dataset_profile(datasets_map, args.dataset)
        config = apply_detr_dataset_profile(config, profile)
        print(f"🗂️  DETR 数据集: {args.dataset} -> data_root={config.get('data', {}).get('data_root')}")
    elif args.config and str(args.config).endswith('.yaml') and args.data_root != '/root/autodl-fs/datasets/DAIR-V2X':
        config.setdefault('data', {})['data_root'] = args.data_root
    elif not (args.config and str(args.config).endswith('.yaml')):
        if args.data_root != '/root/autodl-fs/datasets/DAIR-V2X':
            config.setdefault('data', {})['data_root'] = args.data_root
    
    # 创建训练器
    trainer = AdaptiveExpertTrainer(config, config_file_path=config_file_path)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
