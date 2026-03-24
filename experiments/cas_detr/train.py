#!/usr/bin/env python3
"""CaS_DETR Training Script - Dual-Sparse Expert Transformer (Token Pruning + Decoder MoE)"""

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
    write_eval_csv,
    dataset_display_name,
    dataset_dir_name,
    model_display_name,
)

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import re
import gc
from torch.utils.data import DataLoader
import logging
from datetime import datetime
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
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
    """Factory function to create backbone"""
    if backbone_type.startswith('presnet'):
        depth_match = re.search(r'(\d+)', backbone_type)
        if depth_match:
            depth = int(depth_match.group(1))
        else:
            raise ValueError(f"Cannot parse depth from backbone type {backbone_type}")
        
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
            raise ValueError(f"Unsupported HGNetv2 type: {backbone_type}")
        
        default_params = {
            'name': name_map[backbone_type],
            'return_idx': [1, 2, 3],
            'freeze_at': 0,
            'freeze_norm': True,
            'pretrained': False
        }
        default_params.update(kwargs)
        return HGNetv2(**default_params)
    
    # CSPResNet config
    elif backbone_type.startswith('cspresnet'):
        name_map = {'cspresnet_s': 's', 'cspresnet_m': 'm', 'cspresnet_l': 'l', 'cspresnet_x': 'x'}
        if backbone_type not in name_map:
            raise ValueError(f"Unsupported CSPResNet type: {backbone_type}")
        
        default_params = {
            'name': name_map[backbone_type],
            'return_idx': [1, 2, 3],
            'pretrained': False
        }
        default_params.update(kwargs)
        return CSPResNet(**default_params)
    
    # CSPDarkNet config
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
        raise ValueError(f"Unsupported backbone type: {backbone_type}")






class CaS_DETRRTDETR(nn.Module):
    """CaS_DETR (Dual-Sparse Expert Transformer) RT-DETR model.
    
    Architecture Design:
    1. Shared Backbone: Extracts multi-scale features
    2. CaS_DETR Encoder (Dual-Sparse):
       - Token Pruning: Prunes redundant tokens
    3. MoE Decoder: MoELayer in FFN
    4. Unified Output: Directly outputs detection results
    """
    
    def __init__(self, hidden_dim: int = 256,
                 decoder_hidden_dim: Optional[int] = None,
                 num_queries: int = 300, top_k: int = 2, backbone_type: str = "presnet34",
                 num_decoder_layers: int = 3, encoder_in_channels: list = None, 
                 encoder_expansion: float = 1.0, num_experts: int = 6,
                 num_encoder_layers: int = 1,
                 use_encoder_idx: list = None,
                 token_keep_ratio: Union[float, Dict[int, float]] = None,
                 # MoE weight config
                 decoder_moe_balance_weight: float = None,
                 moe_balance_warmup_epochs: int = 0,
                # CASS (Context-Aware Soft Supervision) config
                use_cass: bool = False,
                cass_loss_weight: float = 0.2,
                cass_expansion_ratio: float = 0.3,
                cass_min_size: float = 1.0,
                # CASS Loss config
                cass_loss_type: str = 'vfl',  # 'focal' or 'vfl'
                cass_focal_alpha: float = 0.75,
                cass_focal_beta: float = 2.0,
                # MoE noise_std config
                moe_noise_std: float = 0.1,
                num_classes: int = 8):
        """Initialize CaS_DETR RT-DETR model.
        
        Args:
            hidden_dim: Encoder hidden dimension
            decoder_hidden_dim: Decoder hidden dimension (defaults to hidden_dim if None)
            num_queries: Number of queries
            top_k: Top-K experts for router
            backbone_type: Backbone type
            num_decoder_layers: Number of decoder layers
            encoder_in_channels: Encoder input channels
            encoder_expansion: Encoder expansion parameter
            num_experts: Number of decoder experts (required)
            num_encoder_layers: Number of encoder transformer layers
            use_encoder_idx: Which feature pyramid levels (P3, P4, P5) to process with Transformer Encoder
            token_keep_ratio: Patch retention ratio, can be float (uniform) or dict mapping layer index to ratio (e.g., {2: 0.9})
            decoder_moe_balance_weight: Decoder MoE balance loss weight
            moe_balance_warmup_epochs: Number of epochs before applying MOE balance loss (default: 0)
            use_cass: Whether to use Context-Aware Soft Supervision for token pruning
            cass_loss_weight: CASS loss weight
            cass_expansion_ratio: Context band expansion ratio (0.2-0.8)
            cass_min_size: Minimum box size on feature map (protects small objects)
            cass_loss_type: Loss type ('focal' for Focal Loss, 'vfl' for Varifocal Loss)
            cass_focal_alpha: Focal/VFL alpha parameter (positive sample weight)
            cass_focal_beta: Focal/VFL beta/gamma parameter (hard example mining strength)
            
        Note:
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim or hidden_dim
        self.num_queries = num_queries
        self.top_k = top_k
        self.backbone_type = backbone_type
        self.num_decoder_layers = num_decoder_layers
        
        # Encoder配置
        self.encoder_in_channels = encoder_in_channels
        self.encoder_expansion = encoder_expansion
        self.num_encoder_layers = num_encoder_layers
        self.use_encoder_idx = use_encoder_idx
        
        # CaS_DETR双稀疏配置（ 必然启用，无需存储）
        self.token_keep_ratio = token_keep_ratio
        
        # CASS configuration
        self.use_cass = use_cass
        self.cass_loss_weight = cass_loss_weight
        self.cass_expansion_ratio = cass_expansion_ratio
        self.cass_min_size = cass_min_size
        # CASS Loss configuration
        self.cass_loss_type = cass_loss_type
        self.cass_focal_alpha = cass_focal_alpha
        self.cass_focal_beta = cass_focal_beta
        
        # MoE noise_std configuration
        self.moe_noise_std = moe_noise_std
        self.num_classes = num_classes
        
        # MoE和Token Pruning权重配置
        if decoder_moe_balance_weight is not None:
            self.decoder_moe_balance_weight = decoder_moe_balance_weight
        
        # MOE Balance Warmup配置：在前N个epoch内不应用MOE平衡损失，让专家自然分化
        self.moe_balance_warmup_epochs = moe_balance_warmup_epochs
        
        # 设置专家数量
        self.num_experts = num_experts
        
        # Current epoch for control (Token Pruning is always enabled from epoch 0)
        self.current_epoch = 0
        
        # ========== Shared Components ==========
        self.backbone = self._build_backbone()
        self.encoder = self._build_encoder()
        
        # ========== Fine-grained MoE Decoder ==========
        # Use passed decoder layers argument
        
        self.decoder = RTDETRTransformerv2(
            num_classes=num_classes,
            hidden_dim=self.decoder_hidden_dim,
            num_queries=num_queries,
            num_layers=num_decoder_layers,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation='relu',
            feat_channels=[self.decoder_hidden_dim, self.decoder_hidden_dim, self.decoder_hidden_dim],
            feat_strides=[8, 16, 32],
            num_levels=3,
            # Fine-grained MoE config
            use_moe=True,
            num_experts=self.num_experts,
            moe_top_k=top_k,
            moe_noise_std=self.moe_noise_std
        )
        
        print(f"✓ MoE Decoder config: {num_decoder_layers} layers, {self.num_experts} experts, top_k={top_k}")
        
        # RT-DETR Loss
        self.detr_criterion = self._build_detr_criterion()
    
    def set_epoch(self, epoch: int):
        """Set current epoch for warmup control.
        
        Args:
            epoch: Current training epoch
        """
        self.current_epoch = epoch
        # Also update encoder's epoch (for token pruning mechanism)
        if hasattr(self, 'encoder') and hasattr(self.encoder, 'set_epoch'):
            self.encoder.set_epoch(epoch)
        
    def _build_backbone(self) -> nn.Module:
        """Build backbone."""
        return create_backbone(self.backbone_type)
    
    def _build_encoder(self) -> nn.Module:
        """Build encoder - Supports CaS_DETR dual-sparse mechanism."""
        # Support Shared MoE
        
        return HybridEncoder(
            in_channels=self.encoder_in_channels,
            feat_strides=[8, 16, 32],
            hidden_dim=self.hidden_dim,
            use_encoder_idx=self.use_encoder_idx,
            num_encoder_layers=self.num_encoder_layers,
            expansion=self.encoder_expansion,
            nhead=8,
            dropout=0.0,
            act='silu',
            # CaS_DETR dual-sparse params
            token_keep_ratio=self.token_keep_ratio,
            # CASS parameters
            use_cass=self.use_cass,
            cass_expansion_ratio=self.cass_expansion_ratio,
            cass_min_size=self.cass_min_size,
            cass_decay_type='gaussian',
            # CASS Loss parameters
            cass_loss_type=self.cass_loss_type,
            cass_focal_alpha=self.cass_focal_alpha,
            cass_focal_beta=self.cass_focal_beta,
            # MoE noise_std parameter
            moe_noise_std=self.moe_noise_std
        )
    
    def _build_detr_criterion(self) -> RTDETRCriterionv2:
        """Build RT-DETR loss function."""
        matcher = HungarianMatcher(
            weight_dict={'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
            use_focal_loss=False,
            alpha=0.25,
            gamma=2.0
        )
        
        # Main loss weights
        main_weight_dict = {
            'loss_vfl': 1.0,
            'loss_bbox': 5.0,
            'loss_giou': 2.0
        }
        
        # Dynamically read decoder layers
        num_decoder_layers = self.num_decoder_layers
        aux_weight_dict = {}
        for i in range(num_decoder_layers - 1):  # First N-1 layers
            aux_weight_dict[f'loss_vfl_aux_{i}'] = 1.0
            aux_weight_dict[f'loss_bbox_aux_{i}'] = 5.0
            aux_weight_dict[f'loss_giou_aux_{i}'] = 2.0
        
        # Encoder auxiliary loss (usually 1 layer)
        aux_weight_dict['loss_vfl_enc_0'] = 1.0
        aux_weight_dict['loss_bbox_enc_0'] = 5.0
        aux_weight_dict['loss_giou_enc_0'] = 2.0
        
        # Denoising auxiliary loss
        num_denoising_layers = num_decoder_layers
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
            num_classes=self.num_classes,
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
        
        # CaS_DETR Encoder（双稀疏：Token Pruning）
        # ⚠️  和 Token Pruning 必然启用（CaS_DETR核心特性）
        encoder_features, encoder_info = self.encoder(backbone_features, return_encoder_info=True)
        
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
            
            # ========== CaS_DETR双稀疏损失 ==========
            # 1. Decoder MoE负载均衡损失（仅训练时）
            if self.training:
                decoder_moe_loss = decoder_output.get('moe_load_balance_loss', 
                                                     torch.tensor(0.0, device=images.device))
            else:
                decoder_moe_loss = torch.tensor(0.0, device=images.device)
            
            # 检查是否在 MOE Balance Warmup 期间
            # 在 warmup 期间，MOE 平衡损失权重设为 0，让专家自然分化
            in_moe_balance_warmup = self.current_epoch < self.moe_balance_warmup_epochs
            
            # Decoder MoE权重
            if in_moe_balance_warmup:
                decoder_moe_weight = 0.0
            elif hasattr(self, 'decoder_moe_balance_weight'):
                decoder_moe_weight = self.decoder_moe_balance_weight
            else:
                decoder_moe_weight = 0.05
            
            # 2. CASS (Context-Aware Soft Supervision) Loss
            # Provides explicit supervision for token importance predictor using GT bboxes
            if self.use_cass and self.training and encoder_info and targets is not None:
                importance_scores_list = encoder_info.get('importance_scores_list', [])
                feat_shapes_list = encoder_info.get('feat_shapes_list', [])
                
                if importance_scores_list and feat_shapes_list and \
                   hasattr(self.encoder, 'shared_token_pruner') and self.encoder.shared_token_pruner:
                    cass_loss = torch.tensor(0.0, device=images.device)
                    
                    # Get image shape (assuming all images in batch have same size)
                    img_shape = (images.shape[2], images.shape[3])  # (H, W)
                    
                    # Extract gt_bboxes from targets
                    # Note: boxes from _collate_fn are always normalized (cx, cy, w, h in [0, 1])
                    gt_bboxes = []
                    for t in targets:
                        if t is not None and 'boxes' in t:
                            boxes = t['boxes']
                            # Convert normalized (cx, cy, w, h) to absolute (x1, y1, x2, y2)
                            if boxes.numel() > 0:
                                boxes_abs = boxes.clone()
                                cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                                # 转换为归一化的 (x1, y1, x2, y2)
                                boxes_abs[:, 0] = cx - w / 2  # x1 (归一化)
                                boxes_abs[:, 1] = cy - h / 2  # y1 (归一化)
                                boxes_abs[:, 2] = cx + w / 2  # x2 (归一化)
                                boxes_abs[:, 3] = cy + h / 2  # y2 (归一化)
                                # 转换为绝对坐标
                                boxes_abs[:, 0] *= img_shape[1]  # x1 (绝对)
                                boxes_abs[:, 1] *= img_shape[0]  # y1 (绝对)
                                boxes_abs[:, 2] *= img_shape[1]  # x2 (绝对)
                                boxes_abs[:, 3] *= img_shape[0]  # y2 (绝对)
                                gt_bboxes.append(boxes_abs)
                            else:
                                gt_bboxes.append(boxes)
                        else:
                            gt_bboxes.append(torch.empty(0, 4, device=images.device))
                    
                    pruner = self.encoder.shared_token_pruner
                    scores = importance_scores_list[0]
                    feat_shape = feat_shapes_list[0]
                    if pruner.use_cass:
                        cass_loss = pruner.compute_cass_loss_from_info(
                            info={'token_importance_scores': scores},
                            gt_bboxes=gt_bboxes,
                            feat_shape=feat_shape,
                            img_shape=img_shape
                        )
                        if cass_loss.device != images.device:
                            cass_loss = cass_loss.to(images.device)
                else:
                    cass_loss = torch.tensor(0.0, device=images.device)
            else:
                # 未启用 CASS：CASS Loss 为 0
                cass_loss = torch.tensor(0.0, device=images.device)
            
            # CASS Loss weight
            cass_weight = self.cass_loss_weight if hasattr(self, 'cass_loss_weight') else 0.2
            
            # 总损失：L = L_task + Decoder MoE损失 + CASS损失
            total_loss = detection_loss + \
                        decoder_moe_weight * decoder_moe_loss + \
                        cass_weight * cass_loss
            
            output['detection_loss'] = detection_loss
            output['decoder_moe_loss'] = decoder_moe_loss
            output['cass_loss'] = cass_loss
            output['moe_load_balance_loss'] = decoder_moe_loss  # 保持向后兼容
            output['total_loss'] = total_loss
            output['loss_dict'] = detection_loss_dict
            
            output['decoder_moe_weight'] = decoder_moe_weight
            output['cass_weight'] = cass_weight
            
            # 添加encoder info到输出（用于监控）
            if encoder_info:
                output['encoder_info'] = encoder_info
        
        return output


class CaS_DETRTrainer:
    """CaS_DETR (Dual-Sparse Expert Transformer) 训练器。
    
    负责模型训练、验证、检查点管理等功能。
    支持双稀疏机制的渐进式训练。
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
        
        # 初始化组件
        self._setup_logging()
        self._apply_vram_batch_size_rule()
        self.model = self._create_model()
        
        pretrained_weights = self.config['model'].get('pretrained_weights', None)
        if pretrained_weights:
            self._load_pretrained_weights(self.model, pretrained_weights)
            
        self.train_loader, self.val_loader = self._create_data_loaders()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.warmup_scheduler = self._create_warmup_scheduler()
        self.ema = self._create_ema()
        self.scaler = self._create_scaler()
        
        self.visualizer = TrainingVisualizer(log_dir=self.log_dir, model_type='cas_detr', experiment_name=self.experiment_name)
        self.early_stopping = self._create_early_stopping()
        
        # 初始化推理相关组件
        self._setup_inference_components()
        
        # 恢复检查点
        if self.resume_from_checkpoint:
            self._resume_from_checkpoint()
            
    def _apply_vram_batch_size_rule(self):
        """根据 VRAM 动态调整 batch_size（与 rt-detr / yolo / moe-rtdetr 共用 common.vram_batch）。"""
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
            # 假设格式为 cas_detr6_r50_20240101_120000，提取 cas_detr6_r50
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
            # 从配置文件读取decoder专家数量
            num_decoder_experts = self.config.get('model', {}).get('num_experts', 6)
            cas_detr_config = self.config.get('model', {}).get('cas_detr', {})
            # 生成实验名称（不带时间戳）
            self.experiment_name = f"cas_detr{num_decoder_experts}_{backbone_short}"
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
    
    def _create_model(self) -> CaS_DETRRTDETR:
        """创建CaS_DETR模型（支持双稀疏）。"""
        # 从配置文件读取encoder配置
        encoder_config = self.config['model']['encoder']
        encoder_in_channels = encoder_config['in_channels']
        encoder_expansion = encoder_config['expansion']
        use_encoder_idx = encoder_config.get('use_encoder_idx', [1, 2])
        
        # 从配置文件读取专家数量
        num_experts = self.config['model'].get('num_experts', 6)
        
        # 从配置文件读取 MoE noise_std
        moe_noise_std = self.config['model'].get('moe_noise_std', 0.1)
        
        # CaS_DETR双稀疏配置
        cas_detr_config = self.config['model'].get('cas_detr', {})
        # ⚠️ 注意： 和 Token Pruning 必然启用（CaS_DETR核心特性），无需配置
        token_keep_ratio = cas_detr_config.get('token_keep_ratio', 0.7)
        
        # CASS (Context-Aware Soft Supervision) 配置
        use_cass = cas_detr_config.get('use_cass', False)
        cass_loss_weight = cas_detr_config.get('cass_loss_weight', 0.2)
        cass_expansion_ratio = cas_detr_config.get('cass_expansion_ratio', 0.3)
        cass_min_size = cas_detr_config.get('cass_min_size', 1.0)
        # CASS Loss 配置
        cass_loss_type = cas_detr_config.get('cass_loss_type', 'vfl')  # 'focal' or 'vfl'
        cass_focal_alpha = cas_detr_config.get('cass_focal_alpha', 0.75)
        cass_focal_beta = cas_detr_config.get('cass_focal_beta', 2.0)
        
        # 从配置文件读取MoE权重
        decoder_moe_balance_weight = self.config.get('training', {}).get('decoder_moe_balance_weight', None)
        # MOE Balance Warmup: 在前N个epoch内不应用MOE平衡损失
        moe_balance_warmup_epochs = self.config.get('training', {}).get('moe_balance_warmup_epochs', 0)
        
        # 从配置文件读取num_encoder_layers，默认为1
        num_encoder_layers = self.config.get('model', {}).get('encoder', {}).get('num_encoder_layers', 1)
        decoder_hidden_dim = self.config['model'].get('decoder_hidden_dim', None)
        
        model = CaS_DETRRTDETR(
            hidden_dim=self.config['model']['hidden_dim'],
            decoder_hidden_dim=decoder_hidden_dim,
            num_queries=self.config['model']['num_queries'],
            top_k=self.config['model']['top_k'],
            backbone_type=self.config['model']['backbone'],
            num_decoder_layers=self.config['model']['num_decoder_layers'],
            encoder_in_channels=encoder_in_channels,
            encoder_expansion=encoder_expansion,
            num_experts=num_experts,
            num_encoder_layers=num_encoder_layers,
            use_encoder_idx=use_encoder_idx,
            # CaS_DETR双稀疏参数（ 必然启用，无需传递）
            token_keep_ratio=token_keep_ratio,
            # MoE权重配置
            decoder_moe_balance_weight=decoder_moe_balance_weight,
            moe_balance_warmup_epochs=moe_balance_warmup_epochs,
            # CASS配置
            use_cass=use_cass,
            cass_loss_weight=cass_loss_weight,
            cass_expansion_ratio=cass_expansion_ratio,
            cass_min_size=cass_min_size,
            # CASS Loss配置
            cass_loss_type=cass_loss_type,
            cass_focal_alpha=cass_focal_alpha,
            cass_focal_beta=cass_focal_beta,
            # MoE noise_std 配置
            moe_noise_std=moe_noise_std,
            num_classes=self.num_classes,
        )
        
        # [修复] 移除 _create_model 内部的加载逻辑，统一在 CaS_DETRTrainer.__init__ 中处理
        
        model = model.to(self.device)
        
        # 启用GPU优化设置
        if torch.cuda.is_available():
            # 启用cudnn benchmark以加速卷积操作（输入尺寸固定时）
            torch.backends.cudnn.benchmark = True
            # 启用TensorFloat-32（RTX 5090支持，可加速某些操作）
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.logger.info("✓ 已启用GPU优化: cudnn.benchmark=True, TF32=True")
        
        # 获取实际的num_encoder_layers用于日志输出
        num_encoder_layers = self.config.get('model', {}).get('encoder', {}).get('num_encoder_layers', 1)
        
        self.logger.info(f"✓ 创建CaS_DETR RT-DETR模型")
        self.logger.info(f"  Decoder专家数量: {model.num_experts}")
        self.logger.info(f"  Backbone: {model.backbone_type}")
        self.logger.info(f"  Encoder: in_channels={encoder_in_channels}, expansion={encoder_expansion}, num_layers={num_encoder_layers}")
        self.logger.info(f"  设计: 层间共享")
        self.logger.info(f"  双稀疏配置（CaS_DETR核心特性，必然启用）:")
        self.logger.info(f"    - Token Pruning: 启用（与  兼容）")
        self.logger.info(f"      → keep_ratio={token_keep_ratio}")
        self.logger.info(f"  损失权重配置:")
        self.logger.info(f"    - CASS Supervision: {use_cass} (loss_type={cass_loss_type}, weight={cass_loss_weight}, expansion={cass_expansion_ratio}, min_size={cass_min_size})")
        if use_cass:
            self.logger.info(f"      → CASS Loss params: alpha={cass_focal_alpha}, beta={cass_focal_beta}")
        self.logger.info(f"    - Decoder MoE: {decoder_moe_balance_weight if decoder_moe_balance_weight else 'auto'}")
        if moe_balance_warmup_epochs > 0:
            self.logger.info(f"    - MOE Balance Warmup: {moe_balance_warmup_epochs} epochs (延迟平衡策略：前{moe_balance_warmup_epochs}个epoch不应用MOE平衡损失)")
        
        return model
    
    def _load_pretrained_weights(self, model: CaS_DETRRTDETR, pretrained_path: str) -> None:
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
            
            # [优化] 手动逐个参数加载，解决维度不匹配导致整个加载失败的问题
            # [新增] 实现专家克隆：将标准FFN权重复制到MoE专家
            model_state_dict = model.state_dict()
            load_count = 0
            mismatch_count = 0
            expert_clone_count = 0
            
            final_state_dict = {}
            # 第一步：收集需要克隆的FFN权重
            decoder_ffn_weights_to_clone = {}  # {layer_idx: {'linear1.weight': tensor, 'linear1.bias': tensor, ...}}
            encoder_ffn_weights_to_clone = {}  # {layer_idx: {'linear1.weight': tensor, 'linear1.bias': tensor, ...}}
            # 跟踪成功克隆的FFN权重键名（用于从missing_keys中排除）
            successfully_cloned_ffn_keys = set()
            
            for k, v in filtered_state_dict.items():
                # 检测Decoder层的FFN权重
                # 支持两种格式：decoder.layers.X.linear1 或 decoder.decoder.layers.X.linear1
                decoder_ffn_match = None
                if ('decoder.layers.' in k or 'decoder.decoder.layers.' in k) and ('linear1' in k or 'linear2' in k):
                    # 尝试匹配 decoder.decoder.layers.X.linear1.weight (优先)
                    match1 = re.search(r'decoder\.decoder\.layers\.(\d+)\.(linear\d)\.(weight|bias)', k)
                    if match1:
                        decoder_ffn_match = match1
                    else:
                        # 尝试匹配 decoder.layers.X.linear1.weight
                        match2 = re.search(r'decoder\.layers\.(\d+)\.(linear\d)\.(weight|bias)', k)
                        if match2:
                            decoder_ffn_match = match2
                
                if decoder_ffn_match:
                    layer_idx = int(decoder_ffn_match.group(1))
                    linear_name = decoder_ffn_match.group(2)  # 'linear1' or 'linear2'
                    param_type = decoder_ffn_match.group(3)  # 'weight' or 'bias'
                    
                    if layer_idx not in decoder_ffn_weights_to_clone:
                        decoder_ffn_weights_to_clone[layer_idx] = {}
                    decoder_ffn_weights_to_clone[layer_idx][f'{linear_name}.{param_type}'] = v
                    continue  # 暂时跳过，稍后处理
                
                # 检测Encoder层的FFN权重
                # 支持多种格式：
                # - encoder.encoder.0.layers.0.linear1.weight (HybridEncoder中的多尺度encoder)
                # - encoder.encoder.layers.X.linear1.weight
                # - encoder.layers.X.linear1.weight
                encoder_ffn_match = None
                if ('encoder.layers.' in k or 'encoder.encoder' in k) and ('linear1' in k or 'linear2' in k):
                    # 尝试匹配 encoder.encoder.0.layers.0.linear1.weight (优先，HybridEncoder格式)
                    match1 = re.search(r'encoder\.encoder\.\d+\.layers\.(\d+)\.(linear\d)\.(weight|bias)', k)
                    if match1:
                        encoder_ffn_match = match1
                    else:
                        # 尝试匹配 encoder.encoder.layers.X.linear1.weight
                        match2 = re.search(r'encoder\.encoder\.layers\.(\d+)\.(linear\d)\.(weight|bias)', k)
                        if match2:
                            encoder_ffn_match = match2
                        else:
                            # 尝试匹配 encoder.layers.X.linear1.weight
                            match3 = re.search(r'encoder\.layers\.(\d+)\.(linear\d)\.(weight|bias)', k)
                            if match3:
                                encoder_ffn_match = match3
                
                if encoder_ffn_match:
                    layer_idx = int(encoder_ffn_match.group(1))
                    linear_name = encoder_ffn_match.group(2)  # 'linear1' or 'linear2'
                    param_type = encoder_ffn_match.group(3)  # 'weight' or 'bias'
                    
                    if layer_idx not in encoder_ffn_weights_to_clone:
                        encoder_ffn_weights_to_clone[layer_idx] = {}
                    encoder_ffn_weights_to_clone[layer_idx][f'{linear_name}.{param_type}'] = v
                    continue  # 暂时跳过，稍后处理
                
                # 其他权重正常处理
                if k in model_state_dict:
                    if v.shape == model_state_dict[k].shape:
                        final_state_dict[k] = v
                        load_count += 1
                    else:
                        mismatch_count += 1
                        # 只在调试级别打印，避免日志刷屏
                        # self.logger.debug(f"维度不匹配跳过: {k} {v.shape} -> {model_state_dict[k].shape}")
                else:
                    # 预训练权重中有，但模型中没有（可能是 unexpected_keys）
                    pass
            
            # 第二步：将Decoder FFN权重复制到MoE专家
            decoder_num_experts = model.num_experts
            if decoder_ffn_weights_to_clone:
                self.logger.info(f"  - 找到 {len(decoder_ffn_weights_to_clone)} 个Decoder层的FFN权重，准备克隆到MoE")
            for layer_idx, ffn_params in decoder_ffn_weights_to_clone.items():
                # 检查是否有完整的FFN参数
                if 'linear1.weight' in ffn_params and 'linear1.bias' in ffn_params and \
                   'linear2.weight' in ffn_params and 'linear2.bias' in ffn_params:
                    
                    linear1_weight = ffn_params['linear1.weight']  # [dim_feedforward, d_model]
                    linear1_bias = ffn_params['linear1.bias']  # [dim_feedforward]
                    linear2_weight = ffn_params['linear2.weight']  # [d_model, dim_feedforward]
                    linear2_bias = ffn_params['linear2.bias']  # [d_model]
                    
                    # MoE层参数命名：decoder.decoder.layers.X.decoder_moe_layer.expert_w1
                    # 尝试两种格式
                    expert_w1_key = f'decoder.decoder.layers.{layer_idx}.decoder_moe_layer.expert_w1'
                    expert_b1_key = f'decoder.decoder.layers.{layer_idx}.decoder_moe_layer.expert_b1'
                    expert_w2_key = f'decoder.decoder.layers.{layer_idx}.decoder_moe_layer.expert_w2'
                    expert_b2_key = f'decoder.decoder.layers.{layer_idx}.decoder_moe_layer.expert_b2'
                    
                    # 如果找不到，尝试另一种格式
                    if expert_w1_key not in model_state_dict:
                        expert_w1_key = f'decoder.layers.{layer_idx}.decoder_moe_layer.expert_w1'
                        expert_b1_key = f'decoder.layers.{layer_idx}.decoder_moe_layer.expert_b1'
                        expert_w2_key = f'decoder.layers.{layer_idx}.decoder_moe_layer.expert_w2'
                        expert_b2_key = f'decoder.layers.{layer_idx}.decoder_moe_layer.expert_b2'
                    
                    # 检查这些键是否存在于模型中（在循环外检查，避免重复检查）
                    if expert_w1_key in model_state_dict:
                        # 获取模型中的参数（用于形状检查）
                        model_w1 = model_state_dict[expert_w1_key]
                        model_b1 = model_state_dict[expert_b1_key]
                        model_w2 = model_state_dict[expert_w2_key]
                        model_b2 = model_state_dict[expert_b2_key]
                        
                        # 检查形状是否匹配
                        # model_w1 shape: [num_experts, dim_feedforward, d_model]
                        # linear1_weight shape: [dim_feedforward, d_model]
                        if (model_w1.shape[1:] == linear1_weight.shape and 
                            model_b1.shape[1:] == linear1_bias.shape and
                            model_w2.shape[1:] == linear2_weight.shape and
                            model_b2.shape[1:] == linear2_bias.shape):
                            
                            # 标记该层的FFN权重成功克隆（用于从missing_keys中排除）
                            # 需要同时标记两种可能的键名格式
                            successfully_cloned_ffn_keys.add(f'decoder.decoder.layers.{layer_idx}.linear1.weight')
                            successfully_cloned_ffn_keys.add(f'decoder.decoder.layers.{layer_idx}.linear1.bias')
                            successfully_cloned_ffn_keys.add(f'decoder.decoder.layers.{layer_idx}.linear2.weight')
                            successfully_cloned_ffn_keys.add(f'decoder.decoder.layers.{layer_idx}.linear2.bias')
                            successfully_cloned_ffn_keys.add(f'decoder.layers.{layer_idx}.linear1.weight')
                            successfully_cloned_ffn_keys.add(f'decoder.layers.{layer_idx}.linear1.bias')
                            successfully_cloned_ffn_keys.add(f'decoder.layers.{layer_idx}.linear2.weight')
                            successfully_cloned_ffn_keys.add(f'decoder.layers.{layer_idx}.linear2.bias')
                            
                            # 克隆到每个专家
                            for expert_idx in range(decoder_num_experts):
                                # 复制权重并添加噪声
                                cloned_w1 = linear1_weight.clone()
                                cloned_w1 += torch.randn_like(cloned_w1) * 0.01
                                cloned_b1 = linear1_bias.clone()
                                cloned_b1 += torch.randn_like(cloned_b1) * 0.01
                                cloned_w2 = linear2_weight.clone()
                                cloned_w2 += torch.randn_like(cloned_w2) * 0.01
                                cloned_b2 = linear2_bias.clone()
                                cloned_b2 += torch.randn_like(cloned_b2) * 0.01
                                
                                # 直接赋值（因为expert_w1是Parameter，需要按索引赋值）
                                # 注意：这里我们需要在加载后手动赋值，因为state_dict不支持索引赋值
                                # 所以我们先存储这些值，在load_state_dict之后赋值
                                if not hasattr(self, '_expert_clone_params'):
                                    self._expert_clone_params = []
                                
                                self._expert_clone_params.append({
                                    'type': 'decoder',
                                    'layer_idx': layer_idx,
                                    'expert_idx': expert_idx,
                                    'w1': cloned_w1,
                                    'b1': cloned_b1,
                                    'w2': cloned_w2,
                                    'b2': cloned_b2
                                })
                                expert_clone_count += 4  # 4个参数：w1, b1, w2, b2
                        else:
                            self.logger.warning(f"Decoder层{layer_idx}形状不匹配: "
                                              f"model_w1={model_w1.shape[1:]}, linear1_weight={linear1_weight.shape}")
                    else:
                        self.logger.debug(f"Decoder层{layer_idx}未找到MoE层参数，跳过专家克隆")
            
            # 第二步（续）：将Encoder FFN权重复制到MoE专家
            
            if encoder_ffn_weights_to_clone:
                self.logger.info(f"  - 找到 {len(encoder_ffn_weights_to_clone)} 个Encoder层的FFN权重，准备克隆到MoE")
            
            for layer_idx, ffn_params in encoder_ffn_weights_to_clone.items():
                # 检查是否有完整的FFN参数
                if 'linear1.weight' in ffn_params and 'linear1.bias' in ffn_params and \
                   'linear2.weight' in ffn_params and 'linear2.bias' in ffn_params:
                    
                    linear1_weight = ffn_params['linear1.weight']  # [dim_feedforward, d_model]
                    linear1_bias = ffn_params['linear1.bias']  # [dim_feedforward]
                    linear2_weight = ffn_params['linear2.weight']  # [d_model, dim_feedforward]
                    linear2_bias = ffn_params['linear2.bias']  # [d_model]
                    
                    # MoE层参数命名：尝试多种格式
                    # 注意：预训练权重中是 encoder.encoder.0.layers.0.linear1，但模型中是 encoder.encoder.layers.0.moe_layer
                    # 所以layer_idx应该对应到模型的layers索引
                    expert_w1_key = f'encoder.encoder.layers.{layer_idx}.moe_layer.expert_w1'
                    expert_b1_key = f'encoder.encoder.layers.{layer_idx}.moe_layer.expert_b1'
                    expert_w2_key = f'encoder.encoder.layers.{layer_idx}.moe_layer.expert_w2'
                    expert_b2_key = f'encoder.encoder.layers.{layer_idx}.moe_layer.expert_b2'
                    
                    # 如果找不到，尝试其他格式
                    if expert_w1_key not in model_state_dict:
                        # 尝试 encoder.layers.X.moe_layer.expert_w1
                        expert_w1_key = f'encoder.layers.{layer_idx}.moe_layer.expert_w1'
                        expert_b1_key = f'encoder.layers.{layer_idx}.moe_layer.expert_b1'
                        expert_w2_key = f'encoder.layers.{layer_idx}.moe_layer.expert_w2'
                        expert_b2_key = f'encoder.layers.{layer_idx}.moe_layer.expert_b2'
                        
                        # 如果还是找不到，尝试 encoder.encoder.0.layers.X.moe_layer (匹配预训练权重格式)
                        if expert_w1_key not in model_state_dict:
                            expert_w1_key = f'encoder.encoder.0.layers.{layer_idx}.moe_layer.expert_w1'
                            expert_b1_key = f'encoder.encoder.0.layers.{layer_idx}.moe_layer.expert_b1'
                            expert_w2_key = f'encoder.encoder.0.layers.{layer_idx}.moe_layer.expert_w2'
                            expert_b2_key = f'encoder.encoder.0.layers.{layer_idx}.moe_layer.expert_b2'
                    
                    # 检查这些键是否存在于模型中（在循环外检查，避免重复检查）
                    if expert_w1_key in model_state_dict:
                        # 获取模型中的参数（用于形状检查）
                        model_w1 = model_state_dict[expert_w1_key]
                        model_b1 = model_state_dict[expert_b1_key]
                        model_w2 = model_state_dict[expert_w2_key]
                        model_b2 = model_state_dict[expert_b2_key]
                        
                        # 检查形状是否匹配
                        # model_w1 shape: [num_experts, dim_feedforward, d_model]
                        # linear1_weight shape: [dim_feedforward, d_model]
                        shape_match = (model_w1.shape[1:] == linear1_weight.shape and 
                                      model_b1.shape[1:] == linear1_bias.shape and
                                      model_w2.shape[1:] == linear2_weight.shape and
                                      model_b2.shape[1:] == linear2_bias.shape)
                        
                        if shape_match:
                            # 标记该层的FFN权重成功克隆（用于从missing_keys中排除）
                            # 需要同时标记两种可能的键名格式
                            successfully_cloned_ffn_keys.add(f'encoder.encoder.layers.{layer_idx}.linear1.weight')
                            successfully_cloned_ffn_keys.add(f'encoder.encoder.layers.{layer_idx}.linear1.bias')
                            successfully_cloned_ffn_keys.add(f'encoder.encoder.layers.{layer_idx}.linear2.weight')
                            successfully_cloned_ffn_keys.add(f'encoder.encoder.layers.{layer_idx}.linear2.bias')
                            successfully_cloned_ffn_keys.add(f'encoder.layers.{layer_idx}.linear1.weight')
                            successfully_cloned_ffn_keys.add(f'encoder.layers.{layer_idx}.linear1.bias')
                            successfully_cloned_ffn_keys.add(f'encoder.layers.{layer_idx}.linear2.weight')
                            successfully_cloned_ffn_keys.add(f'encoder.layers.{layer_idx}.linear2.bias')
                            
                            # 克隆到每个专家
                            for expert_idx in range(encoder_num_experts):
                                # 复制权重并添加噪声
                                cloned_w1 = linear1_weight.clone()
                                cloned_w1 += torch.randn_like(cloned_w1) * 0.01
                                cloned_b1 = linear1_bias.clone()
                                cloned_b1 += torch.randn_like(cloned_b1) * 0.01
                                cloned_w2 = linear2_weight.clone()
                                cloned_w2 += torch.randn_like(cloned_w2) * 0.01
                                cloned_b2 = linear2_bias.clone()
                                cloned_b2 += torch.randn_like(cloned_b2) * 0.01
                                
                                # 存储克隆参数
                                if not hasattr(self, '_expert_clone_params'):
                                    self._expert_clone_params = []
                                
                                self._expert_clone_params.append({
                                    'type': 'encoder',
                                    'layer_idx': layer_idx,
                                    'expert_idx': expert_idx,
                                    'w1': cloned_w1,
                                    'b1': cloned_b1,
                                    'w2': cloned_w2,
                                    'b2': cloned_b2
                                })
                                expert_clone_count += 4  # 4个参数：w1, b1, w2, b2
                        else:
                            self.logger.debug(f"Encoder层{layer_idx}形状不匹配，跳过专家克隆")
                    else:
                        self.logger.debug(f"Encoder层{layer_idx}未找到MoE层参数，跳过专家克隆")
            
            # 使用 strict=False 加载匹配的部分
            missing_keys, unexpected_keys = model.load_state_dict(final_state_dict, strict=False)
            
            # [修复] 从missing_keys中排除成功克隆的FFN权重（这些权重通过专家克隆方式加载）
            actual_missing_keys = [k for k in missing_keys if k not in successfully_cloned_ffn_keys]
            cloned_ffn_count = len(missing_keys) - len(actual_missing_keys)
            
            # 第三步：手动赋值MoE专家权重（在load_state_dict之后）
            if hasattr(self, '_expert_clone_params') and self._expert_clone_params:
                decoder_clone_count = 0
                encoder_clone_count = 0
                
                for clone_info in self._expert_clone_params:
                    layer_idx = clone_info['layer_idx']
                    expert_idx = clone_info['expert_idx']
                    clone_type = clone_info['type']
                    
                    if clone_type == 'decoder':
                        # 获取Decoder MoE层：model.decoder 是 RTDETRTransformerv2，它包含 decoder.decoder (TransformerDecoder)
                        decoder_layer = model.decoder.decoder.layers[layer_idx]
                        if hasattr(decoder_layer, 'decoder_moe_layer'):
                            moe_layer = decoder_layer.decoder_moe_layer
                            # 直接赋值（expert_w1是Parameter，支持索引赋值）
                            with torch.no_grad():
                                moe_layer.expert_w1.data[expert_idx] = clone_info['w1']
                                moe_layer.expert_b1.data[expert_idx] = clone_info['b1']
                                moe_layer.expert_w2.data[expert_idx] = clone_info['w2']
                                moe_layer.expert_b2.data[expert_idx] = clone_info['b2']
                            decoder_clone_count += 4
                    
                    elif clone_type == 'encoder':
                        # 获取层：model.encoder 是 HybridEncoder，它包含 encoder.encoder (TransformerEncoder)
                        encoder_layer = model.encoder.encoder.layers[layer_idx]
                        if hasattr(encoder_layer, 'moe_layer'):
                            moe_layer = encoder_layer.moe_layer
                            # 直接赋值（expert_w1是Parameter，支持索引赋值）
                            with torch.no_grad():
                                moe_layer.expert_w1.data[expert_idx] = clone_info['w1']
                                moe_layer.expert_b1.data[expert_idx] = clone_info['b1']
                                moe_layer.expert_w2.data[expert_idx] = clone_info['w2']
                                moe_layer.expert_b2.data[expert_idx] = clone_info['b2']
                            encoder_clone_count += 4
                
                # 清理临时数据
                delattr(self, '_expert_clone_params')
                
                if decoder_clone_count > 0:
                    self.logger.info(f"  - Decoder专家克隆: {decoder_clone_count} 个参数 ({decoder_num_experts} 个专家)")
                if encoder_clone_count > 0:
                    self.logger.info(f"  - Encoder专家克隆: {encoder_clone_count} 个参数 ({encoder_num_experts} 个专家)")
                
                # 将专家克隆的参数数量计入成功加载的总数
                load_count += decoder_clone_count + encoder_clone_count
            
            self.logger.info(f"✓ 成功加载权重参数: {load_count} 个")
            if expert_clone_count > 0:
                self.logger.info(f"  - 专家克隆总计: {expert_clone_count} 个参数")
            if mismatch_count > 0:
                self.logger.info(f"  - 维度不匹配跳过: {mismatch_count} 个参数")
            
            # 统计各部分的加载情况
            backbone_loaded = sum(1 for k in final_state_dict.keys() if 'backbone' in k)
            encoder_loaded = sum(1 for k in final_state_dict.keys() if 'encoder' in k)
            decoder_loaded = sum(1 for k in final_state_dict.keys() if 'decoder' in k)
            
            self.logger.info(f"  - Backbone 加载: {backbone_loaded} 个参数")
            self.logger.info(f"  - Encoder 加载: {encoder_loaded} 个参数")
            self.logger.info(f"  - Decoder 加载: {decoder_loaded} 个参数")
            
            if cloned_ffn_count > 0:
                self.logger.info(f"  - FFN权重通过专家克隆加载: {cloned_ffn_count} 个参数（已从Missing中排除）")
                
        except Exception as e:
            self.logger.error(f"✗ 加载预训练权重失败: {e}")
            self.logger.info("将从随机初始化开始训练")
    
    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """创建初始数据加载器。"""
        # 使用配置文件中的 batch_size（已经过 vram_batch_size_rule 调整）
        current_batch_size = self.config['training']['batch_size']
        
        self.logger.info(f"📦 初始化训练: epoch={self.current_epoch}, 当前使用 batch_size={current_batch_size}")
        
        train_loader = self._build_train_loader(current_batch_size)
        
        # 验证集配置 (从config读取augmentation配置)
        augmentation_config = self.config.get('augmentation', {})
        ds_cls = self._resolve_dataset_class()
        val_dataset = ds_cls(
            data_root=self.config['data']['data_root'],
            split='val',
            augmentation_config=augmentation_config
        )
        self.val_dataset = val_dataset
        
        num_workers = self.config.get('misc', {}).get('num_workers', 16)
        pin_memory = self.config.get('misc', {}).get('pin_memory', True)
        prefetch_factor = self.config.get('misc', {}).get('prefetch_factor', 4)
        
        from src.data.dataloader import BatchImageCollateFuncion
        stop_epoch = self.config.get('augmentation', {}).get('stop_epoch', 71)
        val_collate_fn = BatchImageCollateFuncion(stop_epoch=stop_epoch) # 验证集不使用 multi-scale
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=current_batch_size, # 验证集同步使用调整后的 BS
            shuffle=False,
            num_workers=num_workers,
            collate_fn=val_collate_fn,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )
        
        return train_loader, val_loader

    def _build_train_loader(self, batch_size: int) -> DataLoader:
        """根据指定的 batch_size 构建训练加载器。"""
        from src.data.dataloader import BatchImageCollateFuncion
        
        # 从config中读取augmentation配置
        augmentation_config = self.config.get('augmentation', {})
        
        ds_cls = self._resolve_dataset_class()
        train_dataset = ds_cls(
            data_root=self.config['data']['data_root'],
            split='train',
            augmentation_config=augmentation_config
        )
        
        num_workers = self.config.get('misc', {}).get('num_workers', 16)
        pin_memory = self.config.get('misc', {}).get('pin_memory', True)
        prefetch_factor = self.config.get('misc', {}).get('prefetch_factor', 4)

        # 多尺度训练配置 (从config中读取或使用默认值)
        scales = self.config.get('augmentation', {}).get('scales', [480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800])
        stop_epoch = self.config.get('augmentation', {}).get('stop_epoch', 71)
        train_collate_fn = BatchImageCollateFuncion(scales=scales, stop_epoch=stop_epoch)
        
        return DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            collate_fn=train_collate_fn,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )
    
    def _collate_fn(self, batch: List[Tuple]) -> Tuple[torch.Tensor, List[Dict]]:
        """数据整理函数。"""
        images, targets = zip(*batch)
        
        # 1. 处理图像 (保持 Tensor 格式)
        if not isinstance(images[0], torch.Tensor):
            processed_images = [T.functional.to_tensor(img) for img in images]
        else:
            processed_images = list(images)

        # 2. 计算 Batch 最大尺寸
        sizes = [img.shape[-2:] for img in processed_images]
        stride = 32
        max_h_raw = max(s[0] for s in sizes)
        max_w_raw = max(s[1] for s in sizes)
        # 向上取整到 32 倍数
        max_h = (max_h_raw + stride - 1) // stride * stride
        max_w = (max_w_raw + stride - 1) // stride * stride
        
        # 3. 创建画布并填充 (左上角对齐)
        batch_images = torch.zeros(len(processed_images), 3, max_h, max_w, 
                                   dtype=processed_images[0].dtype)
        
        for i, img in enumerate(processed_images):
            h, w = img.shape[-2:]
            batch_images[i, :, :h, :w] = img

        # 4. Normalize targets based on final Batch size
        # [修复] 坐标归一化基准不统一问题
        # 关键：所有样本的 boxes 必须使用相同的归一化基准（batch 的最大尺寸 max_w, max_h）
        # 这样去噪分支和主分支才能使用相同的归一化基准，避免数值崩溃
        new_targets = []
        for i, t in enumerate(list(targets)):
            # [FIX] Use deepcopy or clone to ensure original data is not modified
            new_t = t.copy()
            # Must clone, otherwise boxes[:, 0] = ... modifies source tensor
            boxes = new_t['boxes'].clone()
            
            # 手动归一化：除以 max_w 和 max_h（batch 最大尺寸，不是单个样本尺寸）
            # 格式是 cx, cy, w, h
            # x轴数据 (cx, w) 除以 max_w
            # y轴数据 (cy, h) 除以 max_h
            # 注意：所有样本都使用相同的 max_w 和 max_h，确保归一化基准统一
            boxes[:, 0] = boxes[:, 0] / max_w
            boxes[:, 1] = boxes[:, 1] / max_h
            boxes[:, 2] = boxes[:, 2] / max_w
            boxes[:, 3] = boxes[:, 3] / max_h
            
            # 限制数值在 0-1 之间 (防止浮点溢出)
            boxes = torch.clamp(boxes, 0.0, 1.0)
            
            new_t['boxes'] = boxes
            # [修复] 保存归一化基准，确保去噪分支和主分支使用相同的基准
            new_t['normalization_size'] = torch.tensor([max_w, max_h], dtype=torch.float32)
            new_targets.append(new_t)
        
        return batch_images, new_targets
    
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
        # - decoder.layers.X.decoder_moe_layer.* (CaS_DETR的decoder MoE)
        # - encoder.shared_token_pruner.* (CaS_DETR的token pruning)
        # - importance_predictor (token pruning中的重要性预测器)
        new_structure_keywords = [
            'decoder_moe_layer',  # decoder中的MoE层
            'shared_token_pruner',    # token pruning模块
            'importance_predictor'     # importance predictor
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

        current_batch_size = self.config["training"]["batch_size"]
        num_workers = self.config.get("misc", {}).get("num_workers", 16)
        pin_memory = self.config.get("misc", {}).get("pin_memory", True)
        prefetch_factor = self.config.get("misc", {}).get("prefetch_factor", 4)
        stop_epoch = self.config.get("augmentation", {}).get("stop_epoch", 71)
        val_collate_fn = BatchImageCollateFuncion(stop_epoch=stop_epoch)
        return DataLoader(
            test_dataset,
            batch_size=current_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=val_collate_fn,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )

    def _run_ema_eval_on_dataloader(self, dataloader, split_label: str, best_epoch=None):
        all_predictions = []
        all_targets = []
        image_id_to_size = {}
        current_h, current_w = 640, 640
        bs = self.config["training"]["batch_size"]

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                _, _, current_h, current_w = images.shape
                images = images.to(self.device, non_blocking=True)
                targets = [
                    {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                    for t in targets
                ]
                for i, _target in enumerate(targets):
                    image_id = batch_idx * bs + i
                    image_id_to_size[image_id] = (current_w, current_h)

                outputs = self.ema.module(images, targets)
                has_predictions = ("class_scores" in outputs and "bboxes" in outputs) or (
                    "pred_logits" in outputs and "pred_boxes" in outputs
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
            image_id_to_size=image_id_to_size,
            img_h=current_h,
            img_w=current_w,
            print_per_category=True,
            compute_difficulty=True,
        )

        m = metrics
        self.logger.info(
            f"  best_model [{split_label}]  "
            f"mAP50={m.get('mAP_0.5',0):.4f}  mAP75={m.get('mAP_0.75',0):.4f}  mAP={m.get('mAP_0.5_0.95',0):.4f}\n"
            f"    E/M/H@0.5: {m.get('AP_easy',0):.4f}/{m.get('AP_moderate',0):.4f}/{m.get('AP_hard',0):.4f}  |  "
            f"S/M/L@0.5: {m.get('AP_small_50',0):.4f}/{m.get('AP_medium_50',0):.4f}/{m.get('AP_large_50',0):.4f}  |  "
            f"S/M/L@0.5:0.95: {m.get('AP_small',0):.4f}/{m.get('AP_medium',0):.4f}/{m.get('AP_large',0):.4f}"
        )
        summary_csv = self.log_dir.parent / "eval_metrics.csv"
        write_eval_csv(
            summary_csv,
            model=model_display_name(self.config, self.experiment_name),
            dataset=dataset_display_name(self.config),
            eval_split=split_label,
            metrics=metrics,
            class_names=self.class_names,
            append=summary_csv.exists(),
        )
        self.logger.info(f"✓ best_model [{split_label}] 评估完成 (epoch={best_epoch})  → {summary_csv}")

    def _evaluate_best_model_and_print_all_ap(self):
        """训练结束后在 val 上评估；若存在 test 则再评一次。"""
        best_model_path = self.log_dir / 'best_model.pth'
        latest_checkpoint_path = self.log_dir / 'latest_checkpoint.pth'
        if best_model_path.exists():
            checkpoint_path = best_model_path
        elif latest_checkpoint_path.exists():
            checkpoint_path = latest_checkpoint_path
            self.logger.warning("未找到best_model.pth，改用latest_checkpoint.pth进行训练结束评估")
        else:
            self.logger.warning("未找到best_model.pth和latest_checkpoint.pth，跳过训练结束评估")
            return

        original_ema_state = None
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            best_ema_state = checkpoint.get('ema_state_dict', None)
            best_epoch = checkpoint.get('epoch', None)

            if hasattr(self, 'ema') and self.ema and best_ema_state is not None:
                original_ema_state = self.ema.state_dict()
                self.ema.load_state_dict(best_ema_state)

            self.ema.module.eval()
            self._run_ema_eval_on_dataloader(self.val_loader, split_label='val', best_epoch=best_epoch)

            test_loader = self._build_test_dataloader_optional()
            if test_loader is not None:
                self.logger.info("在 test 划分上进行训练结束评估（与 YOLO 一致，仅一次）…")
                self._run_ema_eval_on_dataloader(test_loader, split_label='test', best_epoch=best_epoch)
            else:
                self.logger.info("无可用 test 数据或未配置 test，跳过 test 评估。")

        except Exception as e:
            self.logger.warning(f"训练结束后的best_model评估失败: {e}")
        finally:
            if original_ema_state is not None and hasattr(self, 'ema') and self.ema:
                try:
                    self.ema.load_state_dict(original_ema_state)
                except Exception:
                    pass

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
    
    def _save_token_visualization(self, epoch: int) -> None:
        """保存 Token 重要性热力图（适配全局多尺度剪枝）。"""
        try:
            viz_dir = self.log_dir / "visualizations" / f"epoch_{epoch}"
            viz_dir.mkdir(parents=True, exist_ok=True)
            self.ema.module.eval()
            
            # 获取验证数据
            images, targets = next(iter(self.val_loader))
            B, _, H_tensor, W_tensor = images.shape
            
            with torch.no_grad():
                # 显式传递 targets 确保 forward 返回 encoder_info
                outputs = self.ema.module(images.to(self.device, non_blocking=True), 
                    [{k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets])
            
            enc_info = outputs.get('encoder_info', {})
            # 🚀 核心修改：直接使用 HybridEncoder 准备好的 layer_wise_heatmaps
            heatmaps_2d_list = enc_info.get('layer_wise_heatmaps', [])
            
            if not heatmaps_2d_list:
                self.logger.warning(f"📸 Epoch {epoch}: 可视化跳过，layer_wise_heatmaps 为空。")
                return

            # 获取使用的 encoder indices 来确定 level 名称（S3=0, S4=1, S5=2）
            use_encoder_idx = self.model.encoder.use_encoder_idx if hasattr(self.model.encoder, 'use_encoder_idx') else [1, 2]
            level_names = [f"S{idx+3}" for idx in use_encoder_idx]  # S3=0, S4=1, S5=2
            
            # 遍历所有 level，为每个 level 生成热力图
            for level_idx, (heatmap_tensor, level_name) in enumerate(zip(heatmaps_2d_list, level_names)):
                # heatmaps_2d_list 里的形状是 [B, 1, H_i, W_i]
                scores_prob = torch.sigmoid(heatmap_tensor)
                h_feat, w_feat = scores_prob.shape[2], scores_prob.shape[3]
                
                for i in range(min(3, len(targets))):
                    img_id = targets[i]['image_id'].item()
                    data_root = Path(self.config['data']['data_root'])
                    
                    # 尝试命名匹配
                    possible_paths = [
                        data_root / "image" / f"{img_id:06d}.jpg",
                        data_root / "image" / f"{img_id}.jpg"
                    ]
                    orig_img = None
                    for p in possible_paths:
                        if p.exists():
                            orig_img = cv2.imread(str(p))
                            break
                    
                    if orig_img is None: continue

                    orig_h, orig_w = orig_img.shape[:2]

                    # 物理空间校准
                    valid_h_feat = int(round(orig_h * (h_feat / H_tensor)))
                    valid_w_feat = int(round(orig_w * (w_feat / W_tensor)))
                    
                    s_2d = scores_prob[i, 0].cpu().numpy()
                    s_valid = s_2d[:valid_h_feat, :valid_w_feat]
                    
                    s_norm = (s_valid - s_valid.min()) / (s_valid.max() - s_valid.min() + 1e-8)
                    heatmap = cv2.applyColorMap((s_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    heatmap = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    
                    overlay = cv2.addWeighted(orig_img, 0.4, heatmap, 0.6, 0)
                    cv2.imwrite(str(viz_dir / f"sample_{img_id}_{level_name}_heatmap.jpg"), overlay)
                
            self.logger.info(f"📸 Epoch {epoch}: 已保存 {len(heatmaps_2d_list)} 个尺度({', '.join(level_names)})的重要性热力图至 {viz_dir}")
        except Exception as e:
            self.logger.error(f"可视化模块运行崩溃: {e}", exc_info=True)
        except Exception as e:
            self.logger.error(f"可视化模块运行崩溃: {e}", exc_info=True)
    
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
        """训练一个epoch（支持CaS_DETR渐进式训练，采用即产即清原则优化）。"""
        self.model.train()
        
        # 设置模型的epoch（Token Pruning从epoch 0开始就启用）
        if hasattr(self.model, 'set_epoch'):
            self.model.set_epoch(self.current_epoch)
        
        # [极致优化] 核心优化：在GPU上直接维护计数器，不再缓存巨大的Logits列表
        num_decoder_experts = self.model.num_experts
        
        # 统计整个Epoch的累加器（在GPU上，极小的内存占用）
        decoder_expert_usage_total = torch.zeros(num_decoder_experts, dtype=torch.long, device=self.device)
        total_dec_tokens = 0
        
        # 损失统计（GPU tensor 累加，避免每 batch .item() 同步）
        total_loss = torch.tensor(0.0, device=self.device)
        detection_loss = torch.tensor(0.0, device=self.device)
        moe_lb_loss = torch.tensor(0.0, device=self.device)
        cass_loss_sum = torch.tensor(0.0, device=self.device)
        token_pruning_ratios = []
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = [{k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            # 前向传播
            # [内存优化] 使用 set_to_none=True 提升内存效率
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                outputs = self.model(images, targets)
                loss = outputs.get('total_loss', torch.tensor(0.0, device=self.device))
                if not isinstance(loss, torch.Tensor):
                    loss = torch.tensor(loss, device=self.device)
                if loss.dim() > 0:
                    loss = loss.sum()
            
            # 反向传播（添加梯度裁剪）
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_max_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.ema.update(self.model)
            
            # 显存清理（每个 batch 后执行，防止碎片化）
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 统计各种Loss（GPU tensor 累加，避免每 batch .item() 同步）
            total_loss += loss.detach() if isinstance(loss, torch.Tensor) else float(loss)
            if isinstance(outputs, dict):
                if 'detection_loss' in outputs:
                    det_loss_val = outputs['detection_loss']
                    detection_loss += det_loss_val.detach() if isinstance(det_loss_val, torch.Tensor) else float(det_loss_val)
                if 'decoder_moe_loss' in outputs:
                    moe_loss_val = outputs['decoder_moe_loss']
                    moe_lb_loss += moe_loss_val.detach() if isinstance(moe_loss_val, torch.Tensor) else float(moe_loss_val)
                if 'cass_loss' in outputs:
                    cass_loss_val = outputs['cass_loss']
                    cass_loss_sum += cass_loss_val.detach() if isinstance(cass_loss_val, torch.Tensor) else float(cass_loss_val)
            
            # [极致优化] 即产即清：不保留Logits列表，计算完TopK和bincount后立即释放显存
            # 处理统计
            if isinstance(outputs, dict) and 'encoder_info' in outputs:
                enc_info = outputs['encoder_info']
                # Token Pruning比例
                if 'token_pruning_ratios' in enc_info and enc_info['token_pruning_ratios']:
                    avg_ratio = sum(enc_info['token_pruning_ratios']) / len(enc_info['token_pruning_ratios'])
                    token_pruning_ratios.append(avg_ratio)
                
                enc_logits = enc_info.get('moe_router_logits', [])
                if enc_logits and isinstance(enc_logits, list) and len(enc_logits) > 0:
                    # [安全脱钩] 使用 detach() 创建统计副本，确保不影响反向传播
                    enc_logits_detached = [logits.detach() if isinstance(logits, torch.Tensor) else logits for logits in enc_logits]
                    enc_logits_tensor = torch.cat(enc_logits_detached, dim=0)
                    _, enc_indices = torch.topk(enc_logits_tensor, enc_top_k, dim=-1)
                    # 显式释放临时张量
                    del enc_logits_detached, enc_logits_tensor, enc_indices
            
            # 处理Decoder MoE统计（即产即清）
            if self.model.decoder.use_moe:
                for layer in self.model.decoder.decoder.layers:
                    if hasattr(layer, 'decoder_moe_layer'):
                        dec_logits = layer.decoder_moe_layer.router_logits_cache
                        if dec_logits:
                            # 处理列表格式的logits
                            if isinstance(dec_logits, list) and len(dec_logits) > 0:
                                # [安全脱钩] 使用 detach() 创建统计副本，确保不影响反向传播
                                dec_logits_detached = [logits.detach() if isinstance(logits, torch.Tensor) else logits for logits in dec_logits]
                                dec_logits_tensor = torch.cat(dec_logits_detached, dim=0)
                                del dec_logits_detached
                            elif isinstance(dec_logits, torch.Tensor) and dec_logits.numel() > 0:
                                # [安全脱钩] 使用 detach() 创建统计副本
                                dec_logits_tensor = dec_logits.detach()
                            else:
                                continue
                            
                            # 仅在GPU上计算TopK索引并计数，完成后Logits即可被释放
                            _, dec_indices = torch.topk(dec_logits_tensor, self.model.decoder.moe_top_k, dim=-1)
                            decoder_expert_usage_total.add_(torch.bincount(dec_indices.flatten(), minlength=num_decoder_experts))
                            total_dec_tokens += dec_indices.numel()
                            # 显式释放临时张量
                            del dec_logits_tensor, dec_indices
            
            # [优化] 日志打印逻辑：每100个batch只显示基本loss信息
            if batch_idx % 100 == 0:
                det_loss_val = outputs.get('detection_loss', torch.tensor(0.0)).item() if isinstance(outputs, dict) else 0.0
                moe_loss_val = outputs.get('moe_load_balance_loss', torch.tensor(0.0)).item() if isinstance(outputs, dict) else 0.0
                
                self.logger.info(f'Epoch {self.current_epoch} | Batch {batch_idx} | '
                               f'Loss: {loss.item():.2f} (Det: {det_loss_val:.2f}, MoE: {moe_loss_val:.4f})')
            
            self.global_step += 1
        
        # Epoch结束，计算平均值并返回统计结果（此处统一 .item() 同步一次）
        num_batches = len(self.train_loader)
        avg_loss = (total_loss / num_batches).item()
        avg_detection_loss = (detection_loss / num_batches).item()
        avg_decoder_moe_lb_loss = (moe_lb_loss / num_batches).item()
        avg_cass_loss = (cass_loss_sum / num_batches).item()
        
        # 计算专家使用率（从GPU累加器转换）
        decoder_expert_usage_count = decoder_expert_usage_total.cpu().tolist()
        
        expert_usage_rate = []
        if total_dec_tokens > 0:
            for count in decoder_expert_usage_count:
                expert_usage_rate.append(count / total_dec_tokens)
        else:
            expert_usage_rate = [1.0 / num_decoder_experts] * num_decoder_experts
        
        # 计算平均Token Pruning比例
        avg_token_pruning_ratio = sum(token_pruning_ratios) / len(token_pruning_ratios) if token_pruning_ratios else 0.0
        
        # [内存优化] 统计完专家使用率后，手动清空 router_logits_cache（即产即清）
        # 确保这只是针对统计日志的清理，不影响 detr_criterion 的计算
        if self.model.decoder.use_moe:
            for layer in self.model.decoder.decoder.layers:
                if hasattr(layer, 'decoder_moe_layer'):
                    if hasattr(layer.decoder_moe_layer, 'router_logits_cache'):
                        layer.decoder_moe_layer.router_logits_cache = []
        
        # 准备返回结果
        result = {
            'total_loss': avg_loss,
            'detection_loss': avg_detection_loss,
            'decoder_moe_loss': avg_decoder_moe_lb_loss,
            'cass_loss': avg_cass_loss,  # CASS supervision loss
            'token_pruning_ratio': avg_token_pruning_ratio,
            'moe_load_balance_loss': avg_decoder_moe_lb_loss,  # 总MoE损失（向后兼容）
            'expert_usage': decoder_expert_usage_count,
            'expert_usage_rate': expert_usage_rate,
        }
        
        # [内存优化] 释放临时统计变量
        del decoder_expert_usage_total
        del decoder_expert_usage_count
        
        return result
    
    def validate(self) -> Dict[str, float]:
        """验证模型并计算mAP。"""
        self.ema.module.eval()
        
        # 设置encoder的epoch（验证时也需要，虽然Token Pruning从epoch 0就启用）
        # 1. 更新训练模型 (保持原样)
        # 设置模型的epoch（Token Pruning从epoch 0开始就启用）
        # 这会同时更新encoder的epoch（在model.set_epoch内部调用）
        if hasattr(self.model, 'set_epoch'):
            self.model.set_epoch(self.current_epoch)
        
        # =========================================================
        # [修复] 必须同时更新 EMA 模型的 epoch，否则验证时不会剪枝！
        # EMA模型是deepcopy的独立副本，需要单独设置epoch
        # =========================================================
        if hasattr(self.ema.module, 'set_epoch'):
            self.ema.module.set_epoch(self.current_epoch)
            # 调试：验证EMA模型的pruning状态（pruning现在从epoch 0开始就启用）
            if hasattr(self.ema.module, 'encoder') and hasattr(self.ema.module.encoder, 'shared_token_pruner') and self.ema.module.encoder.shared_token_pruner:
                pruner = self.ema.module.encoder.shared_token_pruner
                keep_ratio = pruner.keep_ratio if hasattr(pruner, 'keep_ratio') else 'N/A'
                self.logger.debug(f"[验证] Epoch {self.current_epoch}: EMA pruner.keep_ratio={keep_ratio}")
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        image_id_to_size = {}
        
        # 统计验证时的剪枝比例
        val_pruning_ratios = []
        
        # 验证逻辑
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(self.val_loader):
                # 动态获取 Tensor 尺寸
                B, C, H_tensor, W_tensor = images.shape

                images = images.to(self.device, non_blocking=True)
                targets = [{k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                           for k, v in t.items()} for t in targets]
                
                # 记录该 batch 的尺寸信息
                for i, target in enumerate(targets):
                    image_id = batch_idx * self.config['training']['batch_size'] + i
                    image_id_to_size[image_id] = (W_tensor, H_tensor)
                
                outputs = self.ema.module(images, targets)
                
                # 收集验证时的剪枝信息
                if isinstance(outputs, dict) and 'encoder_info' in outputs:
                    encoder_info = outputs['encoder_info']
                    if 'token_pruning_ratios' in encoder_info and encoder_info['token_pruning_ratios']:
                        avg_ratio = sum(encoder_info['token_pruning_ratios']) / len(encoder_info['token_pruning_ratios'])
                        val_pruning_ratios.append(avg_ratio)
                
                if isinstance(outputs, dict):
                    if 'total_loss' in outputs:
                        total_loss += outputs['total_loss'].item()
                    
                    # 收集预测结果（只在需要计算mAP时收集，前30个epoch跳过）
                    if 'class_scores' in outputs and 'bboxes' in outputs:
                        self._collect_predictions(outputs, targets, batch_idx, all_predictions, all_targets, W_tensor, H_tensor)
        
        # 保存预测结果用于后续打印每个类别mAP（避免重复计算）
        self._last_val_predictions = all_predictions
        self._last_val_targets = all_targets
        self._last_val_image_id_to_size = image_id_to_size
        
        avg_loss = total_loss / len(self.val_loader)
        
        # 计算平均验证时的剪枝比例
        avg_val_pruning_ratio = sum(val_pruning_ratios) / len(val_pruning_ratios) if val_pruning_ratios else 0.0
        
        # 打印验证时的剪枝状态（每次验证都打印，用于监控）
        if avg_val_pruning_ratio > 0.0:
            self.logger.info(f"  ✓ 验证时Token Pruning生效: {avg_val_pruning_ratio:.2%} tokens被剪枝")
        else:
            # pruning_ratio=0.0 可能是因为 keep_ratio >= 1.0 或配置问题
            self.logger.warning(f"  ⚠ 验证时Token Pruning未生效 (pruning_ratio=0.0)! 请检查keep_ratio配置或EMA模型epoch设置")
        
        # [修复] 计算 mAP 时，传递 image_id_to_size 以支持多尺度验证精度
        mAP_metrics = self._compute_map_metrics(all_predictions, all_targets, 
                                              image_id_to_size=image_id_to_size,
                                              print_per_category=False)
        
        return {
            'total_loss': avg_loss,
            'mAP_0.5': mAP_metrics.get('mAP_0.5', 0.0),
            'mAP_0.75': mAP_metrics.get('mAP_0.75', 0.0),
            'mAP_0.5_0.95': mAP_metrics.get('mAP_0.5_0.95', 0.0),
            'mAP_s': mAP_metrics.get('mAP_s', 0.0),
            'mAP_m': mAP_metrics.get('mAP_m', 0.0),
            'mAP_l': mAP_metrics.get('mAP_l', 0.0),
            'val_token_pruning_ratio': avg_val_pruning_ratio  # 添加验证时的剪枝比例
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
                    ann['occluded_state'] = int(occ_np[j])
                if trunc_np is not None:
                    ann['truncated_state'] = int(trunc_np[j])
                all_targets.append(ann)

    def _get_kitti_difficulty(self, target: Dict) -> str:
        """与 YOLO 侧 ``categorize_by_kitti_difficulty`` + DAIR 截断映射一致。"""
        return kitti_difficulty_from_coco_ann(
            target,
            dair_categorical_trunc=self._dair_truncation_categorical,
        )

    def _run_coco_eval(self, predictions: List[Dict], targets: List[Dict], categories: List[Dict],
                       img_h: int, img_w: int, image_id_to_size: Dict[int, Tuple[int, int]] = None,
                       print_summary: bool = False):
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
            if image_id_to_size and img_id in image_id_to_size:
                w, h = image_id_to_size[img_id]
            else:
                w, h = img_w, img_h
            coco_gt['images'].append({'id': img_id, 'width': w, 'height': h})

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
                                img_h: int, img_w: int,
                                image_id_to_size: Dict[int, Tuple[int, int]] = None) -> Dict[str, float]:
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

        easy_eval = self._run_coco_eval(predictions, easy_targets, categories, img_h, img_w, image_id_to_size=image_id_to_size, print_summary=False)
        moderate_eval = self._run_coco_eval(predictions, moderate_targets, categories, img_h, img_w, image_id_to_size=image_id_to_size, print_summary=False)
        hard_eval = self._run_coco_eval(predictions, hard_targets, categories, img_h, img_w, image_id_to_size=image_id_to_size, print_summary=False)

        # AP@IoU=0.50（stats[1]），与 YOLO 训练后 KITTI 难度 mAP@0.5 对齐；勿用 stats[0]（0.5:0.95）
        return {
            "AP_easy": coco_ap_at_iou50_all(easy_eval),
            "AP_moderate": coco_ap_at_iou50_all(moderate_eval),
            "AP_hard": coco_ap_at_iou50_all(hard_eval),
        }
    
    def _extract_per_category_ap_from_eval(self, coco_eval, categories: List[Dict]) -> Dict[str, float]:
        """从一次 COCOeval 结果中提取各类别 AP@0.5:0.95，避免重复评估。"""
        per_category_map = {cat['name']: 0.0 for cat in categories}
        if coco_eval is None or not hasattr(coco_eval, 'eval') or 'precision' not in coco_eval.eval:
            return per_category_map

        try:
            precision = coco_eval.eval['precision']
            area_index = 0
            max_det_index = len(coco_eval.params.maxDets) - 1
            cat_id_to_index = {cat_id: idx for idx, cat_id in enumerate(coco_eval.params.catIds)}

            for cat in categories:
                cat_id = cat['id']
                cat_name = cat['name']
                if cat_id not in cat_id_to_index:
                    continue

                cat_index = cat_id_to_index[cat_id]
                precision_cat = precision[:, :, cat_index, area_index, max_det_index]
                valid = precision_cat[precision_cat > -1]
                per_category_map[cat_name] = float(np.mean(valid)) if valid.size > 0 else 0.0
        except Exception as e:
            self.logger.debug(f"从COCOeval提取每类AP失败: {e}")

        return per_category_map
    
    def _compute_map_metrics(self, predictions: List[Dict], targets: List[Dict], 
                             image_id_to_size: Dict[int, Tuple[int, int]] = None,
                             img_h: int = 640, img_w: int = 640,
                             print_per_category: bool = False,
                             compute_difficulty: bool = False) -> Dict[str, float]:
        """计算mAP指标。
        
        Args:
            predictions: 预测结果列表
            targets: 真实标签列表
            image_id_to_size: 图像ID到(W, H)的映射字典（推荐）
            img_h: 默认图像高度
            img_w: 默认图像宽度
            print_per_category: 是否打印每个类别的详细mAP（默认False，只在best_model时打印）
        """
        try:
            if len(predictions) == 0:
                return {
                    'mAP_0.5': 0.0,
                    'mAP_0.75': 0.0,
                    'mAP_0.5_0.95': 0.0,
                    'mAP_s': 0.0,
                    'mAP_m': 0.0,
                    'mAP_l': 0.0,
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
            
            # [修复] 动态设置每张图像的正确尺寸
            image_ids = set(target['image_id'] for target in targets)
            for img_id in image_ids:
                if image_id_to_size and img_id in image_id_to_size:
                    w, h = image_id_to_size[img_id]
                else:
                    w, h = img_w, img_h
                coco_gt['images'].append({
                    'id': img_id, 
                    'width': w,
                    'height': h
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

            s50, m50, l50 = coco_area_ap_at_iou50(coco_eval)
            
            # 每类 AP 从同一次 COCOeval 的 precision 张量提取，避免重复评估
            per_category_map = self._extract_per_category_ap_from_eval(coco_eval, categories) if print_per_category else {}
            
            difficulty_metrics = {
                'AP_easy': 0.0,
                'AP_moderate': 0.0,
                'AP_hard': 0.0,
            }
            if compute_difficulty:
                difficulty_metrics = self._compute_difficulty_aps(
                    predictions, targets, categories, img_h, img_w,
                    image_id_to_size=image_id_to_size
                )
            
            result = {
                'mAP_0.5': coco_eval.stats[1],
                'mAP_0.75': coco_eval.stats[2],
                'mAP_0.5_0.95': coco_eval.stats[0],
                'mAP_s': coco_eval.stats[3] if len(coco_eval.stats) > 3 else 0.0,  # Small objects
                'mAP_m': coco_eval.stats[4] if len(coco_eval.stats) > 4 else 0.0,  # Medium objects
                'mAP_l': coco_eval.stats[5] if len(coco_eval.stats) > 5 else 0.0,  # Large objects
                'AP_small': coco_eval.stats[3] if len(coco_eval.stats) > 3 else 0.0,
                'AP_medium': coco_eval.stats[4] if len(coco_eval.stats) > 4 else 0.0,
                'AP_large': coco_eval.stats[5] if len(coco_eval.stats) > 5 else 0.0,
                'AP_small_50': s50,
                'AP_medium_50': m50,
                'AP_large_50': l50,
                'AP_easy': difficulty_metrics['AP_easy'],
                'AP_moderate': difficulty_metrics['AP_moderate'],
                'AP_hard': difficulty_metrics['AP_hard'],
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
                'mAP_0.5_0.95': 0.0,
                'mAP_s': 0.0,
                'mAP_m': 0.0,
                'mAP_l': 0.0,
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
            
            # [内存优化] Checkpoint 原子化管理：保存后立即回收临时对象
            del checkpoint
            if best_ema_state is not None:
                del best_ema_state
            gc.collect()
    
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
        
        # [内存优化] Checkpoint 原子化管理：保存后立即回收临时对象
        del checkpoint
        gc.collect()
    
    def train(self) -> None:
        """主训练循环。"""
        epochs = self.config['training']['epochs']
        self.logger.info(f"开始训练 {epochs} epochs")
        self.logger.info(f"✓ 梯度裁剪: max_norm={self.clip_max_norm}")
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch

            # 更新训练集 epoch（验证集无论什么 epoch 都不增强，无需更新）
            if hasattr(self.train_loader, 'set_epoch'):
                self.train_loader.set_epoch(epoch)
            elif hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            if hasattr(self.train_loader.collate_fn, 'set_epoch'):
                self.train_loader.collate_fn.set_epoch(epoch)

            base_batch_size = self.config['training']['batch_size']
            current_target_batch_size = base_batch_size
            
            # 如果当前加载器的 batch_size 与目标不一致，则重建加载器
            if self.train_loader.batch_size != current_target_batch_size:
                self.logger.info(f"🔄 动态调整 Batch Size: {self.train_loader.batch_size} -> {current_target_batch_size} (Epoch {epoch})")
                # 销毁旧的迭代器（如果有）并重建
                del self.train_loader
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.train_loader = self._build_train_loader(current_target_batch_size)
                
                # 重建后必须重新设置 epoch 状态
                self.train_loader.set_epoch(epoch)
                if hasattr(self.train_loader.collate_fn, 'set_epoch'):
                    self.train_loader.collate_fn.set_epoch(epoch)

            # 训练
            train_metrics = self.train_epoch()
            
            # 验证策略：
            # - 前50 epoch：每10轮验证一次
            # - 50-70 epoch：每5轮验证一次
            # - 70-90 epoch：每2轮验证一次
            # - 90 epoch以后：每轮验证
            should_validate = False
            if epoch < 50:
                if (epoch + 1) % 10 == 0:
                    should_validate = True
            elif epoch < 70:
                if (epoch + 1) % 5 == 0:
                    should_validate = True
            elif epoch < 90:
                if (epoch + 1) % 2 == 0:
                    should_validate = True
            else:
                should_validate = True
            
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
                self.logger.info(f"  Decoder MoE损失: {train_metrics.get('decoder_moe_loss', 0.0):.4f}")
                if self.model.use_cass:
                    self.logger.info(f"  CASS Loss: {train_metrics.get('cass_loss', 0.0):.4f}")
                self.logger.info(f"  MoE总损失: {train_metrics['moe_load_balance_loss']:.4f}")
                # 显示专家使用率（每个epoch显示一次）
                usage_str = [f"{rate*100:.2f}%" for rate in train_metrics['expert_usage_rate']]
                self.logger.info(f"  Decoder专家使用率: [{', '.join(usage_str)}]")
            
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
                # CaS_DETR特有的可视化参数
                detection_loss=train_metrics.get('detection_loss', 0.0),
                decoder_moe_loss=train_metrics.get('decoder_moe_loss', 0.0),
                token_pruning_ratio=train_metrics.get('token_pruning_ratio', 0.0),
                # 传递encoder和decoder专家使用率
                decoder_expert_usage=train_metrics.get('expert_usage_rate', [])
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
            
            # Early Stopping检查（前30个epoch不检查mAP相关的指标）
            if self.early_stopping and should_validate:
                # 获取要监控的指标值
                metric_name = self.early_stopping.metric_name
                # 如果监控的是mAP相关指标且epoch < 30，跳过Early Stopping检查
                is_map_metric = any(x in metric_name for x in ['mAP', 'AP'])
                if is_map_metric and epoch < 30:
                    # 前30个epoch不进行mAP评估，跳过Early Stopping检查
                    pass
                else:
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
            
            # 每5个epoch保存latest + 绘制曲线（减少IO开销）
            if epoch % 5 == 0 or epoch == self.config['training']['epochs'] - 1:
                self.save_latest_checkpoint(epoch)
                try:
                    self.visualizer.plot()
                except Exception as e:
                    self.logger.warning(f"绘制训练曲线失败: {e}")
            
            # [内存优化] 在每个 Epoch 结束、保存完模型后，显式清理临时指标变量
            del train_metrics, val_metrics
            if should_validate:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 每11个epoch保存Token重要性热力图（第11、21、31...次）
            if (epoch + 1) % 10 == 0:
                try:
                    self._save_token_visualization(epoch)
                except Exception as e:
                    self.logger.debug(f"Token可视化失败（不影响训练）: {e}")
        
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

    if torch.cuda.is_available():
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        torch.cuda.empty_cache()
        print("✓ 已启用显存碎片整理策略: expandable_segments=True")

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
                'decoder_hidden_dim': 256,
                'num_queries': 300,
                'top_k': args.top_k,
                'backbone': args.backbone,
                'num_decoder_layers': 3,
                'encoder': {
                    'in_channels': [512, 1024, 2048],
                    'expansion': 1.0,
                    'num_encoder_layers': 1,
                    'use_encoder_idx': [1, 2]
                }
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
                # [修改] 大幅提升光照变化的强度，对齐 YOLOv10
                'brightness': 0.4,   # 原 0.15 -> 0.4
                'contrast': 0.4,     # 原 0.15 -> 0.4
                'saturation': 0.7,   # 原 0.1 -> 0.7
                'hue': 0.015,        # 原 0.05 -> 0.015
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
    elif args.config and str(args.config).endswith('.yaml') and args.data_root != 'datasets/DAIR-V2X':
        config.setdefault('data', {})['data_root'] = args.data_root
    elif not (args.config and str(args.config).endswith('.yaml')):
        if args.data_root != 'datasets/DAIR-V2X':
            config.setdefault('data', {})['data_root'] = args.data_root
    
    # 创建训练器
    trainer = CaS_DETRTrainer(config, config_file_path=config_file_path)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
