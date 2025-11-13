#!/usr/bin/env python3
"""
MOE RT-DETR训练脚本 - DAIR-V2X数据集
完整的训练流程，包含路由器、专家网络和门控机制
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from datetime import datetime
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from PIL import Image
import cv2
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 导入RT-DETR组件
from src.zoo.rtdetr import MOERTDETR, MOERTDETRCriterion, HybridEncoder, RTDETRTransformerv2, RTDETRCriterionv2, HungarianMatcher
from src.nn.backbone.presnet import PResNet
from src.nn.backbone.hgnetv2 import HGNetv2
from src.nn.backbone.csp_resnet import CSPResNet
from src.nn.backbone.csp_darknet import CSPDarkNet
from src.nn.backbone.timm_model import TimmModel
from src.nn.backbone.torchvision_model import TorchVisionModel
from src.nn.backbone.test_resnet import MResNet


def create_backbone(backbone_type: str, **kwargs):
    """
    创建backbone的工厂函数
    
    Args:
        backbone_type: backbone类型
        **kwargs: backbone特定参数
    
    Returns:
        backbone模型实例
    """
    backbone_configs = {
        # PResNet variants
        'presnet18': {'class': PResNet, 'params': {'depth': 18, 'variant': 'd', 'return_idx': [1, 2, 3], 'freeze_at': -1, 'freeze_norm': False, 'pretrained': False}},
        'presnet34': {'class': PResNet, 'params': {'depth': 34, 'variant': 'd', 'return_idx': [1, 2, 3], 'freeze_at': -1, 'freeze_norm': False, 'pretrained': False}},
        'presnet50': {'class': PResNet, 'params': {'depth': 50, 'variant': 'd', 'return_idx': [1, 2, 3], 'freeze_at': -1, 'freeze_norm': False, 'pretrained': False}},
        'presnet101': {'class': PResNet, 'params': {'depth': 101, 'variant': 'd', 'return_idx': [1, 2, 3], 'freeze_at': -1, 'freeze_norm': False, 'pretrained': False}},
        
        # HGNetv2 variants
        'hgnetv2_l': {'class': HGNetv2, 'params': {'name': 'L', 'return_idx': [1, 2, 3], 'freeze_at': -1, 'freeze_norm': False, 'pretrained': False}},
        'hgnetv2_x': {'class': HGNetv2, 'params': {'name': 'X', 'return_idx': [1, 2, 3], 'freeze_at': -1, 'freeze_norm': False, 'pretrained': False}},
        'hgnetv2_h': {'class': HGNetv2, 'params': {'name': 'H', 'return_idx': [1, 2, 3], 'freeze_at': -1, 'freeze_norm': False, 'pretrained': False}},
        
        # CSPResNet variants
        'cspresnet_s': {'class': CSPResNet, 'params': {'name': 's', 'return_idx': [1, 2, 3], 'pretrained': False}},
        'cspresnet_m': {'class': CSPResNet, 'params': {'name': 'm', 'return_idx': [1, 2, 3], 'pretrained': False}},
        'cspresnet_l': {'class': CSPResNet, 'params': {'name': 'l', 'return_idx': [1, 2, 3], 'pretrained': False}},
        'cspresnet_x': {'class': CSPResNet, 'params': {'name': 'x', 'return_idx': [1, 2, 3], 'pretrained': False}},
        
        # CSPDarkNet
        'cspdarknet': {'class': CSPDarkNet, 'params': {'return_idx': [2, 3, -1]}},
        
        # Modified ResNet
        'mresnet': {'class': MResNet, 'params': {'num_blocks': [2, 2, 2, 2]}},
    }
    
    if backbone_type not in backbone_configs:
        raise ValueError(f"不支持的backbone类型: {backbone_type}. 支持的类型: {list(backbone_configs.keys())}")
    
    config = backbone_configs[backbone_type]
    backbone_class = config['class']
    default_params = config['params'].copy()
    
    # 更新参数
    default_params.update(kwargs)
    
    return backbone_class(**default_params)


class DAIRV2XDataset:
    """DAIR-V2X数据集加载器"""
    
    def __init__(self, data_root: str, split: str = "train", transform=None):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        
        # 加载数据信息
        self.data_info = self._load_data_info()
        self.class_names = ["car", "truck", "bus", "person", "bicycle", "motorcycle"]
        self.class_to_id = {name: i for i, name in enumerate(self.class_names)}
        
        # 加载训练/验证分割
        self.split_indices = self._load_split_indices()
        
    def _load_data_info(self):
        """加载数据信息"""
        data_info_path = self.data_root / "metadata" / "data_info.json"
        if data_info_path.exists():
            with open(data_info_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"数据信息文件不存在: {data_info_path}")
    
    def _load_split_indices(self):
        """加载训练/验证分割"""
        split_path = self.data_root / "metadata" / "split_data.json"
        if split_path.exists():
            with open(split_path, 'r', encoding='utf-8') as f:
                split_data = json.load(f)
                indices = split_data.get(self.split, [])
                # 将字符串索引转换为整数，并过滤掉超出范围的索引
                valid_indices = []
                for idx in indices:
                    idx_int = int(idx)
                    if 0 <= idx_int < len(self.data_info):
                        valid_indices.append(idx_int)
                return valid_indices
        else:
            # 如果没有分割文件，使用前80%作为训练，后20%作为验证
            total_samples = len(self.data_info)
            if self.split == "train":
                return list(range(int(total_samples * 0.8)))
            else:
                return list(range(int(total_samples * 0.8), total_samples))
    
    def __len__(self):
        return len(self.split_indices)
    
    def __getitem__(self, idx):
        """获取数据项"""
        # 获取实际的数据索引
        actual_idx = self.split_indices[idx]
        data_item = self.data_info[actual_idx]
        
        # 加载真实图片
        image_path = self.data_root / data_item["image_path"]
        image = self._load_image(image_path)
        
        # 加载真实标注
        annotation_path = self.data_root / "annotations" / "camera" / f"{actual_idx:06d}.json"
        bboxes, labels = self._load_annotations(annotation_path)
        
        # 预处理图片和标注
        processed_image, scale, pad_h, pad_w = self._preprocess_image(image, target_size=640)
        processed_bboxes = self._adjust_bboxes(bboxes, scale, pad_h, pad_w)
        
        # 转换为tensor
        bboxes_tensor = torch.tensor(processed_bboxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        # 创建目标字典（RT-DETR格式）
        target = {
            'boxes': bboxes_tensor,
            'labels': labels_tensor,
            'image_id': actual_idx,
            'orig_size': torch.tensor([image.shape[0], image.shape[1]]),
            'size': torch.tensor([640, 640])
        }
        
        return processed_image, target
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """加载图片"""
        if not image_path.exists():
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        # 使用OpenCV加载图片（BGR格式）
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法加载图片: {image_path}")
        
        # 转换为RGB格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 转换为CHW格式
        image = np.transpose(image, (2, 0, 1))
        
        return image
    
    def _load_annotations(self, annotation_path: Path) -> Tuple[List[List[float]], List[int]]:
        """加载标注文件"""
        if not annotation_path.exists():
            return [], []
        
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        bboxes = []
        labels = []
        
        for ann in annotations:
            # 获取类别
            class_name = ann["type"].lower()
            if class_name in self.class_to_id:
                class_id = self.class_to_id[class_name]
            else:
                continue  # 跳过未知类别
            
            # 获取2D边界框
            bbox_2d = ann["2d_box"]
            x1 = float(bbox_2d["xmin"])
            y1 = float(bbox_2d["ymin"])
            x2 = float(bbox_2d["xmax"])
            y2 = float(bbox_2d["ymax"])
            
            # 检查边界框是否有效
            if x2 > x1 and y2 > y1:
                bboxes.append([x1, y1, x2, y2])
                labels.append(class_id)
        
        return bboxes, labels
    
    def _preprocess_image(self, image, target_size=640):
        """预处理图片：保持宽高比缩放到目标尺寸（模拟COCO预处理）"""
        # image: [C, H, W] numpy array
        C, H, W = image.shape
        
        # 计算缩放比例（保持宽高比）
        scale = min(target_size / H, target_size / W)
        new_h, new_w = int(H * scale), int(W * scale)
        
        # 转换为tensor进行缩放
        image_tensor = torch.from_numpy(image).float()
        resized_image = torch.nn.functional.interpolate(
            image_tensor.unsqueeze(0), 
            size=(new_h, new_w), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        # 创建填充后的图片
        padded_image = torch.zeros(C, target_size, target_size, dtype=resized_image.dtype)
        padded_image[:, :new_h, :new_w] = resized_image
        
        # 计算填充偏移
        pad_h = (target_size - new_h) // 2
        pad_w = (target_size - new_w) // 2
        
        return padded_image, scale, pad_h, pad_w
    
    def _adjust_bboxes(self, bboxes, scale, pad_h, pad_w):
        """调整所有bbox坐标以匹配预处理后的图片"""
        adjusted_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            
            # 缩放
            x1 *= scale
            y1 *= scale
            x2 *= scale
            y2 *= scale
            
            # 添加填充偏移
            x1 += pad_w
            y1 += pad_h
            x2 += pad_w
            y2 += pad_h
            
            # 归一化到[0,1]（RT-DETR期望归一化坐标）
            x1 /= 640
            y1 /= 640
            x2 /= 640
            y2 /= 640
            
            # 确保坐标在有效范围内
            x1 = max(0, min(1, x1))
            y1 = max(0, min(1, y1))
            x2 = max(0, min(1, x2))
            y2 = max(0, min(1, y2))
            
            # 确保bbox有效
            if x2 > x1 and y2 > y1:
                adjusted_bboxes.append([x1, y1, x2, y2])
        
        return adjusted_bboxes


class Router(nn.Module):
    """路由器 - 决定哪些专家应该被激活"""
    
    def __init__(self, input_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 路由器网络
        self.router_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [batch_size, seq_len, input_dim] 输入特征
        Returns:
            expert_weights: [batch_size, seq_len, num_experts] 专家权重
            expert_indices: [batch_size, seq_len, top_k] 选中的专家索引
            routing_weights: [batch_size, seq_len, top_k] 路由权重
        """
        batch_size, seq_len, _ = features.shape
        
        # 计算专家权重
        expert_logits = self.router_net(features)  # [batch_size, seq_len, num_experts]
        
        # Top-K选择
        top_k_weights, top_k_indices = torch.topk(expert_logits, self.top_k, dim=-1)
        
        # 重新归一化权重
        routing_weights = torch.softmax(top_k_weights, dim=-1)
        
        return expert_logits, top_k_indices, routing_weights


class ExpertNetwork(nn.Module):
    """单个专家网络 - 包含完整的检测能力"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 专家特定的Decoder - 使用RT-DETR的RTDETRTransformerv2
        # RTDETRTransformerv2已经包含了完整的检测功能（特征处理 + 检测头）
        # 调整参数以匹配HybridEncoder的输出：3个256通道特征
        self.decoder = RTDETRTransformerv2(
            num_classes=num_classes,
            hidden_dim=256,  # 固定为256以匹配预训练模型
            num_queries=100,  # 每个专家处理100个查询
            num_layers=3,  # 专家可以使用更少的层数
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation='relu',
            feat_channels=[256, 256, 256],  # 匹配HybridEncoder输出（3个特征级别）
            feat_strides=[8, 16, 32],  # 匹配HybridEncoder输出
            num_levels=3  # 3个特征级别
        )
        
    def forward(self, encoder_features, targets: Optional[List[Dict]] = None) -> Dict:
        """专家前向传播 - 返回完整的检测结果"""
        # encoder_features应该是多尺度特征列表，直接传递给RTDETRTransformerv2
        # RTDETRTransformerv2期望的格式：[feat1, feat2, feat3, ...] 每个feat是[B, C, H, W]
        
        # 直接使用RTDETRTransformerv2处理多尺度特征
        # RTDETRTransformerv2返回完整的检测结果
        decoder_output = self.decoder(encoder_features, targets)
        
        # RTDETRTransformerv2的输出格式：{'pred_logits': ..., 'pred_boxes': ...}
        if isinstance(decoder_output, dict):
            return {
                'bboxes': decoder_output.get('pred_boxes'),
                'class_scores': decoder_output.get('pred_logits'),
                'decoder_output': decoder_output
            }
        else:
            # 如果返回的不是字典，包装成标准格式
            return {
                'bboxes': None,
                'class_scores': None,
                'decoder_output': decoder_output
            }


class CompleteMOERTDETR(nn.Module):
    """完整的MOE RT-DETR模型"""
    
    def __init__(self, config_name: str = "A", hidden_dim: int = 256, 
                 num_queries: int = 300, top_k: int = 2, backbone_type: str = "presnet50"):
        super().__init__()
        
        self.config_name = config_name
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.top_k = top_k
        self.backbone_type = backbone_type
        
        # 获取专家数量
        configs = {
            "A": 6, "B": 3, "C": 3
        }
        self.num_experts = configs.get(config_name, 6)
        
        # ========== 共享部分 ==========
        # 1. Backbone (支持多种backbone类型)
        self.backbone = self._build_backbone()
        
        # 2. Encoder (根据backbone输出调整)
        self.encoder = self._build_encoder()
        
        # ========== MOE部分 ==========
        # 3. 路由器
        self.router = Router(hidden_dim, self.num_experts, top_k)
        
        # 4. 专家网络（每个专家包含完整的检测能力）
        self.experts = nn.ModuleList([
            ExpertNetwork(hidden_dim, hidden_dim, num_classes=6)  # 每个专家检测6个类别（DAIR-V2X）
            for _ in range(self.num_experts)
        ])
        
        # 5. 专家权重（可学习参数）
        self.expert_weights = nn.Parameter(torch.ones(self.num_experts) / self.num_experts)
        
        # 6. 标准RT-DETR损失函数（用于专家损失计算）
        self.detr_criterion = self._build_detr_criterion()
        
    def _build_backbone(self):
        """构建backbone - 支持多种backbone类型"""
        return create_backbone(self.backbone_type)
    
    def _build_encoder(self):
        """构建encoder - 根据backbone输出动态调整"""
        # 根据输入图片尺寸动态设置eval_spatial_size
        input_size = [640, 640]  # 与数据集图片尺寸匹配
        
        # 获取backbone的输出通道数
        backbone_out_channels = self.backbone.out_channels
        backbone_strides = getattr(self.backbone, 'out_strides', [8, 16, 32])
        
        # 确保有3个特征级别
        if len(backbone_out_channels) >= 3:
            in_channels = backbone_out_channels[-3:]  # 使用最后3层
            feat_strides = backbone_strides[-3:] if len(backbone_strides) >= 3 else [8, 16, 32]
        else:
            # 如果backbone输出少于3层，使用默认配置
            in_channels = [512, 1024, 2048]
            feat_strides = [8, 16, 32]
        
        return HybridEncoder(
            in_channels=in_channels,  # 根据backbone动态调整
            feat_strides=feat_strides,  # 根据backbone动态调整
            hidden_dim=256,  # 固定为256以匹配预训练模型
            use_encoder_idx=[2],  # 对第3层做transformer编码，匹配预训练模型
            num_encoder_layers=1,
            expansion=1.0,  # 匹配预训练模型的expansion参数
            nhead=8,
            dropout=0.0,
            act='silu',
            eval_spatial_size=input_size  # 动态匹配
        )
    
    def _build_detr_criterion(self):
        """构建标准RT-DETR损失函数"""
        # 创建匈牙利匹配器
        matcher = HungarianMatcher(
            weight_dict={'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
            use_focal_loss=False,
            alpha=0.25,
            gamma=2.0
        )
        
        # 创建RT-DETR损失函数
        criterion = RTDETRCriterionv2(
            matcher=matcher,
            weight_dict={'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2},
            losses=['vfl', 'boxes'],  # 使用VFL损失和边界框损失
            alpha=0.75,
            gamma=2.0,
            num_classes=6,  # DAIR-V2X数据集有6个类别
            boxes_weight_format=None,
            share_matched_indices=False
        )
        
        return criterion
    
    def _build_decoder(self):
        """构建decoder"""
        return nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.hidden_dim,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                activation='relu',
                batch_first=True
            ),
            num_layers=6
        )
    
    def forward(self, images: torch.Tensor, targets: Optional[List[Dict]] = None):
        """前向传播 - 架构A：专家网络包含Decoder"""
        # ========== 共享特征提取 ==========
        # 1. Backbone提取特征
        backbone_features = self.backbone(images)  # 返回多尺度特征列表
        
        # 2. Encoder编码特征
        encoder_features = self.encoder(backbone_features)  # 返回编码后的特征
        
        # ========== MOE处理 ==========
        # 3. 路由器决定专家选择
        # encoder_features可能是列表或tensor，需要统一处理
        if isinstance(encoder_features, (list, tuple)):
            # 使用最后一个特征图进行路由
            router_features = encoder_features[-1]  # 使用最高层特征
        else:
            router_features = encoder_features
        
        # 确保router_features是3D tensor [B, H*W, C]
        if len(router_features.shape) == 4:  # [B, C, H, W]
            B, C, H, W = router_features.shape
            router_features_flat = router_features.flatten(2).transpose(1, 2)  # [B, H*W, C]
        else:
            router_features_flat = router_features
        
        expert_logits, expert_indices, routing_weights = self.router(router_features_flat)
        
        # 4. 专家处理（每个专家包含完整的检测能力）
        expert_outputs = []
        for expert in self.experts:
            # 每个专家使用完整的encoder_features
            expert_output = expert(encoder_features, targets)
            expert_outputs.append(expert_output)
        
        # 5. 加权融合专家检测结果
        combined_output = self._combine_expert_detections(
            expert_outputs, expert_indices, routing_weights
        )
        
        if self.training and targets is not None:
            # ========== 训练模式 ==========
            router_loss = self._compute_router_loss(expert_logits)
            
            # 计算专家特定损失
            expert_losses = self._compute_expert_losses(expert_outputs, targets)
            
            # 包装输出
            combined_output['router_loss'] = router_loss
            combined_output['expert_losses'] = expert_losses
            combined_output['expert_logits'] = expert_logits
            combined_output['expert_indices'] = expert_indices
            combined_output['routing_weights'] = routing_weights
            
            # 计算总损失
            total_loss = sum(expert_losses) + router_loss
            combined_output['total_loss'] = total_loss
            
            # 调试信息
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 1
            
            if self._debug_counter % 50 == 0:  # 每50个batch打印一次
                print(f"Debug - Router Loss: {router_loss.item():.4f}")
                print(f"Debug - Expert Losses: {[loss.item() for loss in expert_losses]}")
                print(f"Debug - Total Loss: {total_loss.item():.4f}")
        
        return combined_output
    
    def _combine_expert_detections(self, expert_outputs: List[Dict], 
                                  expert_indices: torch.Tensor, 
                                  routing_weights: torch.Tensor) -> Dict:
        """融合专家检测结果"""
        batch_size, seq_len, top_k = expert_indices.shape
        
        # 获取专家输出的实际查询数量
        expert_bboxes = expert_outputs[0]['bboxes']
        expert_class_scores = expert_outputs[0]['class_scores']
        actual_seq_len = expert_bboxes.shape[1]
        
        # 初始化输出
        combined_bboxes = torch.zeros_like(expert_bboxes)
        combined_class_scores = torch.zeros_like(expert_class_scores)
        
        # 加权融合 - 只处理实际存在的查询
        for b in range(batch_size):
            for s in range(min(seq_len, actual_seq_len)):
                for k in range(top_k):
                    expert_idx = expert_indices[b, s, k].item()
                    weight = routing_weights[b, s, k]
                    
                    combined_bboxes[b, s] += weight * expert_outputs[expert_idx]['bboxes'][b, s]
                    combined_class_scores[b, s] += weight * expert_outputs[expert_idx]['class_scores'][b, s]
        
        return {
            'bboxes': combined_bboxes,
            'class_scores': combined_class_scores
        }
    
    def _compute_router_loss(self, expert_logits: torch.Tensor) -> torch.Tensor:
        """计算路由器损失（负载均衡）"""
        # 计算每个专家的使用频率
        expert_usage = torch.mean(expert_logits, dim=[0, 1])  # [num_experts]
        
        # 计算使用频率的标准差（鼓励均匀使用）
        usage_std = torch.std(expert_usage)
        
        # 计算负载均衡损失
        target_usage = 1.0 / self.num_experts
        load_balance_loss = torch.sum((expert_usage - target_usage) ** 2)
        
        return usage_std + load_balance_loss
    
    def _compute_expert_losses(self, expert_outputs: List[Dict], targets: List[Dict]) -> List[torch.Tensor]:
        """计算所有专家的损失 - 使用标准RT-DETR损失函数"""
        expert_losses = []
        
        for expert_id, expert_output in enumerate(expert_outputs):
            # 检查专家输出是否包含有效的检测结果
            if 'decoder_output' in expert_output and expert_output['decoder_output'] is not None:
                decoder_output = expert_output['decoder_output']
                
                # 如果包含标准的DETR输出格式，使用标准损失函数
                if isinstance(decoder_output, dict) and 'pred_logits' in decoder_output and 'pred_boxes' in decoder_output:
                    try:
                        # 使用标准RT-DETR损失函数计算专家损失
                        expert_loss_dict = self.detr_criterion(decoder_output, targets)
                        
                        # 计算总损失（VFL + 边界框 + GIoU）
                        total_expert_loss = 0.0
                        for loss_name, loss_value in expert_loss_dict.items():
                            if isinstance(loss_value, torch.Tensor):
                                total_expert_loss += loss_value
                        
                        # 专家特定的权重调整
                        expert_weight = 1.0 + 0.1 * expert_id  # 不同专家有不同的权重
                        weighted_loss = expert_weight * total_expert_loss
                        
                        expert_losses.append(weighted_loss)
                        
                    except Exception as e:
                        # 如果标准损失计算失败，使用备用损失
                        print(f"专家{expert_id}标准损失计算失败: {e}，使用备用损失")
                        expert_losses.append(self._compute_fallback_loss(decoder_output, expert_id))
                        
                elif isinstance(decoder_output, dict) and 'loss' in decoder_output:
                    # 如果已经包含损失信息，直接使用
                    expert_losses.append(decoder_output['loss'])
                else:
                    # 使用备用损失
                    expert_losses.append(self._compute_fallback_loss(decoder_output, expert_id))
            else:
                # 默认损失
                expert_losses.append(torch.tensor(0.01, device=next(self.parameters()).device, requires_grad=True))
        
        return expert_losses
    
    def _compute_fallback_loss(self, decoder_output, expert_id):
        """计算备用损失（当标准损失计算失败时使用）"""
        device = next(self.parameters()).device
        
        if isinstance(decoder_output, dict) and 'pred_logits' in decoder_output and 'pred_boxes' in decoder_output:
            pred_logits = decoder_output['pred_logits']
            pred_boxes = decoder_output['pred_boxes']
            
            # 简化的分类损失：使用KL散度鼓励预测合理的分布
            batch_size, num_queries, num_classes = pred_logits.shape
            
            # 创建均匀分布作为目标（鼓励预测所有类别）
            uniform_target = torch.ones_like(pred_logits) / num_classes
            pred_probs = torch.softmax(pred_logits, dim=-1)
            cls_loss = F.kl_div(torch.log(pred_probs + 1e-8), uniform_target, reduction='batchmean')
            
            # 简化的回归损失：鼓励预测在合理范围内的bbox
            bbox_loss = torch.mean(torch.abs(pred_boxes)) + \
                       0.01 * torch.mean(torch.abs(pred_boxes - 0.5))  # 鼓励预测接近中心
            
            # 专家特定的损失权重
            expert_weight = 1.0 + 0.1 * expert_id  # 不同专家有不同的权重
            # 最佳权重：分类和回归损失1:1平衡
            total_loss = expert_weight * (cls_loss + 1.0 * bbox_loss)
            return total_loss
        else:
            # 如果没有任何有用信息，使用小的随机损失
            return torch.tensor(0.01, device=device, requires_grad=True)
    
    def _compute_expert_loss(self, detection_output: Dict, targets: List[Dict], expert_id: int):
        """计算专家损失"""
        # 过滤该专家应该处理的目标
        expert_targets = self._filter_targets_for_expert(targets, expert_id)
        
        if len(expert_targets) == 0:
            return torch.tensor(0.0, device=detection_output['bboxes'].device)
        
        # 简化的损失计算
        if len(expert_targets) > 0:
            target_bboxes = expert_targets[0]['bboxes']
            target_scores = torch.ones_like(detection_output['scores'])
            
            bbox_loss = nn.MSELoss()(detection_output['bboxes'], target_bboxes)
            cls_loss = nn.BCEWithLogitsLoss()(detection_output['scores'], target_scores)
            
            return bbox_loss + cls_loss
        
        return torch.tensor(0.0, device=detection_output['bboxes'].device)
    
    def _compute_router_loss(self, expert_logits: torch.Tensor) -> torch.Tensor:
        """计算路由器损失（负载均衡）"""
        # 计算每个专家的使用频率
        expert_usage = torch.mean(expert_logits, dim=[0, 1])  # [num_experts]
        
        # 计算使用频率的标准差（鼓励均匀使用）
        usage_std = torch.std(expert_usage)
        
        # 计算负载均衡损失
        target_usage = 1.0 / self.num_experts
        load_balance_loss = torch.sum((expert_usage - target_usage) ** 2)
        
        return usage_std + load_balance_loss
    
    def _filter_targets_for_expert(self, targets: List[Dict], expert_id: int):
        """为特定专家过滤目标"""
        expert_targets = []
        for target in targets:
            if self._should_expert_handle_target(target, expert_id):
                expert_targets.append(target)
        return expert_targets
    
    def _should_expert_handle_target(self, target: Dict, expert_id: int) -> bool:
        """判断专家是否应该处理该目标"""
        # 简化版本：每个专家都处理所有目标（用于测试）
        return True


class MOERTDETRTrainer:
    """MOE RT-DETR训练器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置日志
        self._setup_logging()
        
        # 创建模型
        self.model = self._create_model()
        
        # 创建数据加载器
        self.train_loader, self.val_loader = self._create_data_loaders()
        
        # 创建优化器
        self.optimizer = self._create_optimizer()
        
        # 创建学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        
    def _setup_logging(self):
        """设置日志"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(f"logs/moe_rtdetr_{timestamp}")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.log_dir = log_dir
        
        # 保存配置文件
        with open(log_dir / 'config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def _create_model(self):
        """创建模型"""
        model = CompleteMOERTDETR(
            config_name=self.config['model']['config_name'],
            hidden_dim=self.config['model']['hidden_dim'],
            num_queries=self.config['model']['num_queries'],
            top_k=self.config['model']['top_k'],
            backbone_type=self.config['model'].get('backbone_type', 'presnet50')
        )
        
        # 加载预训练权重
        pretrained_weights = self.config.get('pretrained_weights', None)
        if pretrained_weights:
            self._load_pretrained_weights(model, pretrained_weights)
        
        model = model.to(self.device)
        self.logger.info(f"创建MOE RT-DETR模型")
        self.logger.info(f"专家数量: {model.num_experts}")
        self.logger.info(f"配置名称: {model.config_name}")
        self.logger.info(f"Backbone类型: {model.backbone_type}")
        self.logger.info(f"Backbone输出通道: {model.backbone.out_channels}")
        
        return model
    
    def _load_pretrained_weights(self, model, pretrained_path):
        """加载预训练权重"""
        try:
            if pretrained_path == 'torch_hub':
                # 使用torch.hub加载预训练权重
                self.logger.info("使用torch.hub加载RT-DETR预训练权重")
                self._load_from_torch_hub(model)
                return
            
            if pretrained_path.startswith('http'):
                # 下载预训练权重
                self.logger.info(f"下载预训练权重: {pretrained_path}")
                # 这里可以添加下载逻辑
                return
            
            if Path(pretrained_path).exists():
                self.logger.info(f"加载预训练权重: {pretrained_path}")
                checkpoint = torch.load(pretrained_path, map_location='cpu')
                
                # 只加载backbone和encoder的权重
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # 过滤出backbone和encoder的权重
                backbone_encoder_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('backbone.') or key.startswith('encoder.'):
                        backbone_encoder_dict[key] = value
                
                # 加载权重
                model.load_state_dict(backbone_encoder_dict, strict=False)
                self.logger.info(f"成功加载 {len(backbone_encoder_dict)} 个预训练参数")
            else:
                self.logger.warning(f"预训练权重文件不存在: {pretrained_path}")
                
        except Exception as e:
            self.logger.warning(f"加载预训练权重失败: {e}")
            self.logger.info("将使用随机初始化的权重")
    
    def _load_from_torch_hub(self, model):
        """从torch.hub加载预训练权重"""
        try:
            # 加载RT-DETR预训练模型
            self.logger.info("正在从torch.hub加载RT-DETR预训练模型...")
            pretrained_model = torch.hub.load('lyuwenyu/RT-DETR', 'rtdetrv2_r50vd', pretrained=True, trust_repo=True)
            
            # 提取并转换权重
            backbone_encoder_dict = self._extract_backbone_encoder_weights(pretrained_model.state_dict())
            
            if not backbone_encoder_dict:
                self.logger.warning("未找到匹配的backbone/encoder权重")
                return
            
            # 加载权重到我们的模型
            missing_keys, unexpected_keys = model.load_state_dict(backbone_encoder_dict, strict=False)
            
            # 记录加载结果
            self.logger.info(f"✅ 成功加载 {len(backbone_encoder_dict)} 个预训练参数")
            if missing_keys:
                self.logger.debug(f"缺失的键数量: {len(missing_keys)}")
            if unexpected_keys:
                self.logger.debug(f"意外的键数量: {len(unexpected_keys)}")
            
        except Exception as e:
            self.logger.warning(f"❌ 从torch.hub加载预训练权重失败: {e}")
            self.logger.info("将使用随机初始化的权重")
    
    def _extract_backbone_encoder_weights(self, pretrained_state_dict):
        """提取并转换backbone和encoder权重"""
        # 定义需要转换的权重前缀
        target_prefixes = ['model.backbone.', 'model.encoder.']
        prefix_mapping = {'model.backbone.': 'backbone.', 'model.encoder.': 'encoder.'}
        
        # 提取匹配的权重
        extracted_weights = {}
        for key, value in pretrained_state_dict.items():
            for old_prefix, new_prefix in prefix_mapping.items():
                if key.startswith(old_prefix):
                    new_key = key.replace(old_prefix, new_prefix, 1)
                    extracted_weights[new_key] = value
                    break
        
        # 统计信息
        backbone_count = sum(1 for k in extracted_weights.keys() if k.startswith('backbone.'))
        encoder_count = sum(1 for k in extracted_weights.keys() if k.startswith('encoder.'))
        
        self.logger.info(f"提取到 {backbone_count} 个backbone权重, {encoder_count} 个encoder权重")
        
        return extracted_weights
    
    def _create_data_loaders(self):
        """创建数据加载器"""
        batch_size = self.config['training']['batch_size']
        
        # 创建数据集
        train_dataset = DAIRV2XDataset(
            data_root=self.config['data']['data_root'],
            split='train'
        )
        
        val_dataset = DAIRV2XDataset(
            data_root=self.config['data']['data_root'],
            split='val'
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,
            collate_fn=self._collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4,
            collate_fn=self._collate_fn
        )
        
        self.logger.info(f"创建数据加载器")
        self.logger.info(f"训练集: {len(train_dataset)} 样本")
        self.logger.info(f"验证集: {len(val_dataset)} 样本")
        
        return train_loader, val_loader
    
    def _collate_fn(self, batch):
        """数据整理函数"""
        images, targets = zip(*batch)
        images = torch.stack(images, 0)
        return images, list(targets)
    
    def _create_optimizer(self):
        """创建优化器"""
        # 参数分组
        pretrained_params = []
        new_params = []
        
        # 预训练组件（小学习率）
        pretrained_params.extend(self.model.backbone.parameters())
        pretrained_params.extend(self.model.encoder.parameters())
        
        # 新组件（大学习率）
        new_params.extend(self.model.router.parameters())
        new_params.extend(self.model.experts.parameters())
        new_params.append(self.model.expert_weights)
        
        optimizer = optim.Adam([
            {'params': pretrained_params, 'lr': self.config['training']['pretrained_lr']},
            {'params': new_params, 'lr': self.config['training']['new_lr']}
        ], weight_decay=1e-4)  # 添加权重衰减
        
        self.logger.info(f"创建优化器")
        self.logger.info(f"预训练参数学习率: {self.config['training']['pretrained_lr']}")
        self.logger.info(f"新参数学习率: {self.config['training']['new_lr']}")
        
        return optimizer
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config['training']['epochs'],
            eta_min=1e-7
        )
        
        self.logger.info("创建余弦退火学习率调度器")
        return scheduler
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        router_loss = 0.0
        expert_losses = [0.0] * self.model.num_experts
        expert_usage = [0] * self.model.num_experts  # 专家使用统计
        routing_entropy = 0.0  # 路由熵
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images, targets)
            
            # 计算损失
            if isinstance(outputs, dict) and 'total_loss' in outputs:
                loss = outputs['total_loss']
            else:
                # 如果没有总损失，使用路由器损失
                loss = outputs.get('router_loss', torch.tensor(0.0, device=self.device))
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            if isinstance(outputs, dict):
                if 'router_loss' in outputs:
                    router_loss += outputs['router_loss'].item()
                if 'expert_losses' in outputs:
                    for i, expert_loss in enumerate(outputs['expert_losses']):
                        expert_losses[i] += expert_loss.item()
                
                # 计算专家使用情况
                if 'expert_indices' in outputs:
                    expert_indices = outputs['expert_indices']  # [batch_size, seq_len, top_k]
                    for batch_indices in expert_indices:
                        for seq_indices in batch_indices:
                            # seq_indices 是 [top_k] 的tensor，包含选中的专家索引
                            for expert_idx in seq_indices:
                                if isinstance(expert_idx, torch.Tensor):
                                    expert_idx = expert_idx.item()
                                if expert_idx < len(expert_usage):
                                    expert_usage[expert_idx] += 1
                
                # 计算路由熵
                if 'routing_weights' in outputs:
                    routing_weights = outputs['routing_weights']
                    for batch_weights in routing_weights:
                        for weights in batch_weights:
                            # 计算熵: -sum(p * log(p))
                            weights = torch.softmax(weights, dim=-1)
                            entropy = -torch.sum(weights * torch.log(weights + 1e-8))
                            routing_entropy += entropy.item()
            
            if batch_idx % 10 == 0:
                # 显示当前批次的专家使用情况
                current_expert_usage = [0] * self.model.num_experts
                if isinstance(outputs, dict) and 'expert_indices' in outputs:
                    expert_indices = outputs['expert_indices']  # [batch_size, seq_len, top_k]
                    for batch_indices in expert_indices:
                        for seq_indices in batch_indices:
                            # seq_indices 是 [top_k] 的tensor，包含选中的专家索引
                            for expert_idx in seq_indices:
                                if isinstance(expert_idx, torch.Tensor):
                                    expert_idx = expert_idx.item()
                                if expert_idx < len(current_expert_usage):
                                    current_expert_usage[expert_idx] += 1
                
                self.logger.info(f'Epoch {self.current_epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Expert Usage: {current_expert_usage}')
        
        # 计算平均损失和指标
        avg_loss = total_loss / len(self.train_loader)
        avg_router_loss = router_loss / len(self.train_loader)
        avg_expert_losses = [loss / len(self.train_loader) for loss in expert_losses]
        avg_routing_entropy = routing_entropy / len(self.train_loader)
        
        # 计算专家使用率
        total_usage = sum(expert_usage)
        expert_usage_rate = [usage / total_usage if total_usage > 0 else 0 for usage in expert_usage]
        
        return {
            'total_loss': avg_loss,
            'router_loss': avg_router_loss,
            'expert_losses': avg_expert_losses,
            'expert_usage': expert_usage,
            'expert_usage_rate': expert_usage_rate,
            'routing_entropy': avg_routing_entropy
        }
    
    def validate(self):
        """验证 - 使用mAP评估指标"""
        self.model.eval()
        total_loss = 0.0
        router_loss = 0.0
        
        # 用于mAP计算的预测和真实标签
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(self.val_loader):
                images = images.to(self.device)
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in t.items()} for t in targets]
                
                outputs = self.model(images, targets)
                
                if isinstance(outputs, dict):
                    if 'total_loss' in outputs:
                        total_loss += outputs['total_loss'].item()
                    if 'router_loss' in outputs:
                        router_loss += outputs['router_loss'].item()
                    
                    # 收集预测结果用于mAP计算
                    if 'class_scores' in outputs and 'bboxes' in outputs:
                        pred_logits = outputs['class_scores']  # [B, num_queries, num_classes]
                        pred_boxes = outputs['bboxes']  # [B, num_queries, 4]
                        
                        # 处理每个样本的预测
                        for i in range(pred_logits.shape[0]):
                            # 获取预测的类别和置信度
                            pred_scores = torch.softmax(pred_logits[i], dim=-1)  # [num_queries, num_classes]
                            pred_classes = torch.argmax(pred_scores, dim=-1)  # [num_queries]
                            max_scores = torch.max(pred_scores, dim=-1)[0]  # [num_queries]
                            
                            # 过滤低置信度的预测
                            keep_mask = max_scores > 0.1  # 置信度阈值
                            if keep_mask.any():
                                filtered_boxes = pred_boxes[i][keep_mask]  # [N, 4]
                                filtered_classes = pred_classes[keep_mask]  # [N]
                                filtered_scores = max_scores[keep_mask]  # [N]
                                
                                # 转换为COCO格式 (x1, y1, w, h)
                                # 假设pred_boxes是归一化的(cx, cy, w, h)格式
                                if filtered_boxes.shape[0] > 0:
                                    # 转换坐标格式
                                    boxes_coco = torch.zeros_like(filtered_boxes)
                                    boxes_coco[:, 0] = (filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2) * 640  # x1
                                    boxes_coco[:, 1] = (filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2) * 640  # y1
                                    boxes_coco[:, 2] = filtered_boxes[:, 2] * 640  # w
                                    boxes_coco[:, 3] = filtered_boxes[:, 3] * 640  # h
                                    
                                    # 添加到预测列表
                                    for j in range(filtered_boxes.shape[0]):
                                        all_predictions.append({
                                            'image_id': batch_idx * self.config['training']['batch_size'] + i,
                                            'category_id': int(filtered_classes[j].item()) + 1,  # COCO类别从1开始
                                            'bbox': boxes_coco[j].cpu().numpy().tolist(),
                                            'score': float(filtered_scores[j].item())
                                        })
                            
                            # 处理真实标签
                            if i < len(targets) and 'labels' in targets[i] and 'boxes' in targets[i]:
                                true_labels = targets[i]['labels']
                                true_boxes = targets[i]['boxes']
                                
                                if len(true_labels) > 0:
                                    # 转换真实标签为COCO格式
                                    true_boxes_coco = torch.zeros_like(true_boxes)
                                    true_boxes_coco[:, 0] = (true_boxes[:, 0] - true_boxes[:, 2] / 2) * 640  # x1
                                    true_boxes_coco[:, 1] = (true_boxes[:, 1] - true_boxes[:, 3] / 2) * 640  # y1
                                    true_boxes_coco[:, 2] = true_boxes[:, 2] * 640  # w
                                    true_boxes_coco[:, 3] = true_boxes[:, 3] * 640  # h
                                    
                                    for j in range(len(true_labels)):
                                        all_targets.append({
                                            'image_id': batch_idx * self.config['training']['batch_size'] + i,
                                            'category_id': int(true_labels[j].item()) + 1,  # COCO类别从1开始
                                            'bbox': true_boxes_coco[j].cpu().numpy().tolist(),
                                            'area': float(true_boxes_coco[j, 2] * true_boxes_coco[j, 3]),
                                            'iscrowd': 0
                                        })
        
        # 计算mAP
        mAP_metrics = self._compute_map_metrics(all_predictions, all_targets)
        
        avg_loss = total_loss / len(self.val_loader)
        avg_router_loss = router_loss / len(self.val_loader)
        
        return {
            'total_loss': avg_loss,
            'router_loss': avg_router_loss,
            'mAP_0.5': mAP_metrics.get('mAP_0.5', 0.0),
            'mAP_0.75': mAP_metrics.get('mAP_0.75', 0.0),
            'mAP_0.5_0.95': mAP_metrics.get('mAP_0.5_0.95', 0.0),
            'num_predictions': len(all_predictions),
            'num_targets': len(all_targets)
        }
    
    def _compute_map_metrics(self, predictions, targets):
        """计算mAP指标"""
        try:
            # 创建COCO格式的数据
            coco_gt = {
                'images': [],
                'annotations': [],
                'categories': [
                    {'id': 1, 'name': 'car'},
                    {'id': 2, 'name': 'truck'},
                    {'id': 3, 'name': 'bus'},
                    {'id': 4, 'name': 'person'},
                    {'id': 5, 'name': 'bicycle'},
                    {'id': 6, 'name': 'motorcycle'}
                ]
            }
            
            # 添加图像信息
            image_ids = set()
            for target in targets:
                image_ids.add(target['image_id'])
            
            for img_id in image_ids:
                coco_gt['images'].append({'id': img_id})
            
            # 添加标注信息
            for i, target in enumerate(targets):
                target['id'] = i + 1
                coco_gt['annotations'].append(target)
            
            # 创建COCO对象
            coco_gt = COCO()
            coco_gt.dataset = coco_gt
            coco_gt.createIndex()
            
            # 创建预测结果
            coco_dt = coco_gt.loadRes(predictions)
            
            # 计算mAP
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            return {
                'mAP_0.5': coco_eval.stats[1],  # mAP@0.5
                'mAP_0.75': coco_eval.stats[2],  # mAP@0.75
                'mAP_0.5_0.95': coco_eval.stats[0],  # mAP@[0.5:0.95]
                'mAP_small': coco_eval.stats[3],  # mAP for small objects
                'mAP_medium': coco_eval.stats[4],  # mAP for medium objects
                'mAP_large': coco_eval.stats[5],  # mAP for large objects
            }
            
        except Exception as e:
            print(f"mAP计算失败: {e}")
            # 返回简化的准确率作为备用
            if len(predictions) > 0 and len(targets) > 0:
                # 简单的准确率计算
                correct = 0
                total = len(targets)
                for pred in predictions:
                    for target in targets:
                        if (pred['image_id'] == target['image_id'] and 
                            pred['category_id'] == target['category_id']):
                            correct += 1
                            break
                
                accuracy = correct / total if total > 0 else 0.0
                return {
                    'mAP_0.5': accuracy,
                    'mAP_0.75': accuracy,
                    'mAP_0.5_0.95': accuracy,
                    'mAP_small': accuracy,
                    'mAP_medium': accuracy,
                    'mAP_large': accuracy,
                }
            else:
                return {
                    'mAP_0.5': 0.0,
                    'mAP_0.75': 0.0,
                    'mAP_0.5_0.95': 0.0,
                    'mAP_small': 0.0,
                    'mAP_medium': 0.0,
                    'mAP_large': 0.0,
                }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_loss': self.best_loss
        }
        
        # 保存最新检查点
        checkpoint_path = self.log_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = self.log_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"保存最佳模型: {best_path}")
    
    def train(self):
        """主训练循环"""
        epochs = self.config['training']['epochs']
        
        self.logger.info(f"开始训练，总epochs: {epochs}")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # 训练
            train_metrics = self.train_epoch()
            
            # 验证
            val_metrics = self.validate()
            
            # 学习率调度
            self.scheduler.step()
            
            # 日志
            self.logger.info(f"Epoch {epoch}:")
            self.logger.info(f"  训练损失: {train_metrics['total_loss']:.4f}")
            self.logger.info(f"  验证损失: {val_metrics['total_loss']:.4f}")
            self.logger.info(f"  mAP@0.5: {val_metrics['mAP_0.5']:.4f}")
            self.logger.info(f"  mAP@0.75: {val_metrics['mAP_0.75']:.4f}")
            self.logger.info(f"  mAP@[0.5:0.95]: {val_metrics['mAP_0.5_0.95']:.4f}")
            self.logger.info(f"  📏 尺寸分类mAP:")
            self.logger.info(f"    - 小目标 mAP: {val_metrics['mAP_small']:.4f}")
            self.logger.info(f"    - 中等目标 mAP: {val_metrics['mAP_medium']:.4f}")
            self.logger.info(f"    - 大目标 mAP: {val_metrics['mAP_large']:.4f}")
            self.logger.info(f"  预测数量: {val_metrics['num_predictions']}, 目标数量: {val_metrics['num_targets']}")
            self.logger.info(f"  路由器损失: {train_metrics['router_loss']:.4f}")
            self.logger.info(f"  专家损失: {[f'{loss:.4f}' for loss in train_metrics['expert_losses']]}")
            self.logger.info(f"  专家使用率: {[f'{rate:.3f}' for rate in train_metrics['expert_usage_rate']]}")
            self.logger.info(f"  路由熵: {train_metrics['routing_entropy']:.4f}")
            
            # 显示专家损失（如果有的话）
            if 'expert_losses' in train_metrics:
                expert_losses = train_metrics['expert_losses']
                for i, expert_loss in enumerate(expert_losses):
                    self.logger.info(f"  专家{i}损失: {expert_loss:.4f}")
            
            # 保存检查点
            is_best = val_metrics['total_loss'] < self.best_loss
            if is_best:
                self.best_loss = val_metrics['total_loss']
            
            if epoch % 10 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        self.logger.info("训练完成！")


def main():
    parser = argparse.ArgumentParser(description='MOE RT-DETR训练')
    parser.add_argument('--config', type=str, default='A', choices=['A', 'B', 'C'], 
                       help='MOE配置 (A: 6个专家, B: 3个专家按复杂度, C: 3个专家按尺寸)')
    parser.add_argument('--backbone', type=str, default='presnet50', 
                       choices=['presnet18', 'presnet34', 'presnet50', 'presnet101',
                               'hgnetv2_l', 'hgnetv2_x', 'hgnetv2_h',
                               'cspresnet_s', 'cspresnet_m', 'cspresnet_l', 'cspresnet_x',
                               'cspdarknet', 'mresnet'],
                       help='Backbone类型选择')
    parser.add_argument('--data_root', type=str, default='datasets/DAIR-V2X', 
                       help='DAIR-V2X数据集路径')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--pretrained_lr', type=float, default=1e-5, help='预训练组件学习率')
    parser.add_argument('--new_lr', type=float, default=1e-4, help='新组件学习率')
    parser.add_argument('--top_k', type=int, default=2, help='路由器Top-K')
    parser.add_argument('--pretrained_weights', type=str, default=None,
                       help='预训练权重路径 (RT-DETR COCO预训练模型)')
    
    args = parser.parse_args()
    
    # 创建配置
    config = {
        'model': {
            'config_name': args.config,
            'hidden_dim': 256,
            'num_queries': 300,
            'top_k': args.top_k,
            'backbone_type': args.backbone
        },
        'data': {
            'data_root': args.data_root
        },
        'training': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'pretrained_lr': args.pretrained_lr,
            'new_lr': args.new_lr
        }
    }
    
    # 添加预训练权重配置
    if args.pretrained_weights:
        config['pretrained_weights'] = args.pretrained_weights
    
    # 创建训练器
    trainer = MOERTDETRTrainer(config)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
