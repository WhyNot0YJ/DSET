"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 
from typing import List, Dict, Optional, Tuple

from ...core import register


__all__ = ['RTDETR', 'MOERTDETR', 'Router', 'ExpertNetwork']


@register()
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module, 
        encoder: nn.Module, 
        decoder: nn.Module, 
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        
    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)        
        x = self.decoder(x, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 


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
    """单个专家网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 专家特定的处理网络
        self.expert_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """专家前向传播"""
        return self.expert_net(x)


@register()
class MOERTDETR(nn.Module):
    """MOE RT-DETR模型"""
    __inject__ = ['backbone', 'encoder', 'decoder']
    
    def __init__(self, 
                 backbone: nn.Module, 
                 encoder: nn.Module, 
                 decoder: nn.Module,
                 num_experts: int = 6,
                 top_k: int = 2,
                 config_name: str = "A"):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.num_experts = num_experts
        self.top_k = top_k
        self.config_name = config_name
        
        # MOE配置
        self.config = self._get_moe_config(config_name)
        
        # 路由器
        self.router = Router(256, num_experts, top_k)  # 假设hidden_dim=256
        
        # 专家网络
        self.experts = nn.ModuleList([
            ExpertNetwork(256, 256, 256) for _ in range(num_experts)
        ])
        
        # 专家权重（可学习参数）
        self.expert_weights = nn.Parameter(torch.ones(num_experts) / num_experts)
        
    def _get_moe_config(self, config_name: str) -> Dict:
        """获取MOE配置"""
        configs = {
            "A": {
                "num_experts": 6,
                "expert_mapping": {
                    0: "car", 1: "truck", 2: "bus",
                    3: "person", 4: "bicycle", 5: "motorcycle"
                },
                "class_to_expert": {
                    "car": 0, "truck": 1, "bus": 2,
                    "person": 3, "bicycle": 4, "motorcycle": 5
                }
            },
            "B": {
                "num_experts": 3,
                "expert_mapping": {
                    0: "vehicles", 1: "people", 2: "two_wheelers"
                },
                "class_to_expert": {
                    "car": 0, "truck": 0, "bus": 0,
                    "person": 1, "bicycle": 2, "motorcycle": 2
                }
            },
            "C": {
                "num_experts": 3,
                "expert_mapping": {
                    0: "large_objects", 1: "medium_objects", 2: "small_objects"
                },
                "class_to_expert": {
                    "truck": 0, "bus": 0, "car": 1,
                    "person": 2, "bicycle": 2, "motorcycle": 2
                }
            }
        }
        return configs.get(config_name, configs["A"])
    
    def forward(self, x, targets=None):
        """前向传播"""
        # 共享特征提取
        x = self.backbone(x)
        x = self.encoder(x)
        
        # 获取编码特征用于路由器
        if isinstance(x, (list, tuple)):
            encoder_features = x[0] if len(x) > 0 else x
        else:
            encoder_features = x
        
        # 确保encoder_features是3D tensor [batch_size, seq_len, hidden_dim]
        if len(encoder_features.shape) == 4:  # [B, C, H, W]
            B, C, H, W = encoder_features.shape
            encoder_features = encoder_features.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # 路由器决定专家选择
        expert_logits, expert_indices, routing_weights = self.router(encoder_features)
        
        # 专家处理
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(encoder_features)
            expert_outputs.append(expert_output)
        
        # 加权融合专家输出
        combined_features = self._combine_expert_outputs(
            expert_outputs, expert_indices, routing_weights
        )
        
        # 使用融合后的特征进行解码
        if isinstance(x, (list, tuple)):
            # 如果encoder返回的是列表，替换第一个元素
            x[0] = combined_features
        else:
            x = combined_features
        
        # 解码器处理
        x = self.decoder(x, targets)
        
        if self.training and targets is not None:
            # 训练模式：添加MOE相关信息
            if isinstance(x, dict):
                x['expert_logits'] = expert_logits
                x['expert_indices'] = expert_indices
                x['routing_weights'] = routing_weights
                x['router_loss'] = self._compute_router_loss(expert_logits)
            else:
                # 如果decoder返回的不是字典，包装成字典
                x = {
                    'outputs': x,
                    'expert_logits': expert_logits,
                    'expert_indices': expert_indices,
                    'routing_weights': routing_weights,
                    'router_loss': self._compute_router_loss(expert_logits)
                }
        
        return x
    
    def _combine_expert_outputs(self, expert_outputs: List[torch.Tensor], 
                               expert_indices: torch.Tensor, 
                               routing_weights: torch.Tensor) -> torch.Tensor:
        """融合专家输出 - 向量化版本，避免三重循环"""
        batch_size, seq_len, top_k = expert_indices.shape
        hidden_dim = expert_outputs[0].shape[-1]
        
        # ✅ 向量化修复：将所有专家输出堆叠为 [num_experts, B, seq_len, C]
        expert_outputs_stack = torch.stack(expert_outputs, dim=0)  # [num_experts, B, seq_len, C]
        
        # 使用高级索引选择对应的专家输出
        # expert_indices: [B, seq_len, top_k] -> 展平为 [B*seq_len*top_k]
        B, S, K = batch_size, seq_len, top_k
        expert_indices_flat = expert_indices.view(B * S * K)  # [B*S*K]
        batch_indices = torch.arange(B, device=expert_indices.device).view(B, 1, 1).expand(B, S, K).reshape(B * S * K)
        seq_indices = torch.arange(S, device=expert_indices.device).view(1, S, 1).expand(B, S, K).reshape(B * S * K)
        
        # 使用高级索引选择专家输出：expert_outputs_stack[expert_idx, batch_idx, seq_idx, :]
        selected_outputs = expert_outputs_stack[expert_indices_flat, batch_indices, seq_indices]  # [B*S*K, C]
        selected_outputs = selected_outputs.view(B, S, K, hidden_dim)  # [B, S, K, C]
        
        # 应用权重并求和
        routing_weights_expanded = routing_weights.unsqueeze(-1)  # [B, S, K, 1]
        weighted_outputs = selected_outputs * routing_weights_expanded  # [B, S, K, C]
        combined_output = weighted_outputs.sum(dim=2)  # [B, S, C]
        
        return combined_output
    
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
    
    def deploy(self):
        """部署模式"""
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
