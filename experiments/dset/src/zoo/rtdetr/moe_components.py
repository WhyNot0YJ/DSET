"""Adaptive Expert Components for DSET - Vectorized High-Performance Version"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

class AdaptiveRouter(nn.Module):
    """自适应路由器 - 向量化实现 (用于 Decoder 中的 Token MoE)"""
    
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [N, D]
        router_logits = self.gate(x)
        router_probs = F.softmax(router_logits, dim=-1)
        expert_weights, expert_indices = torch.topk(router_probs, self.top_k, dim=-1)
        expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + 1e-9)
        return expert_weights, expert_indices, router_logits

class SpecialistNetwork(nn.Module):
    """专家网络 - 标准MLP"""
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1, activation: str = 'relu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        if activation == 'relu': self.activation = nn.ReLU()
        elif activation == 'gelu': self.activation = nn.GELU()
        elif activation == 'silu': self.activation = nn.SiLU()
        else: self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class PatchLevelRouter(nn.Module):
    """Patch级别路由器 - 向量化优化版 (用于 Encoder)"""
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2, patch_size: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.patch_size = patch_size
        # 使用 1x1 卷积实现快速路由
        self.gate_conv = nn.Conv2d(hidden_dim, num_experts, kernel_size=patch_size, stride=patch_size, bias=False)
        nn.init.normal_(self.gate_conv.weight, mean=0.0, std=0.01)
    
    def forward(self, x: torch.Tensor, spatial_shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B, N, C] 输入的 Tokens
        spatial_shape: (H, W) 空间形状
        """
        if x.dim() == 3:
            B, N, C = x.shape
            H, W = spatial_shape
            x_img = x.transpose(1, 2).reshape(B, C, H, W)
        else:
            x_img = x
            B, C, H, W = x.shape

        # 1. 卷积计算 Logits [B, E, H_patch, W_patch]
        router_logits_map = self.gate_conv(x_img) 
        router_logits = router_logits_map.permute(0, 2, 3, 1).flatten(1, 2)
        
        # 2. Top-K 选择
        router_probs = F.softmax(router_logits, dim=-1)
        expert_weights, expert_indices = torch.topk(router_probs, self.top_k, dim=-1)
        expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + 1e-9)
        
        return expert_weights, expert_indices, router_logits

class PatchMoELayer(nn.Module):
    """Patch-MoE layer - 全向量化极速版 (核心修复点)"""
    
    def __init__(self, d_model: int, dim_feedforward: int, num_experts: int = 4, 
                 top_k: int = 2, dropout: float = 0.1, activation: str = 'gelu', patch_size: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.patch_size = patch_size
        
        self.router = PatchLevelRouter(d_model, num_experts, top_k, patch_size)
        self.experts = nn.ModuleList([
            SpecialistNetwork(d_model, dim_feedforward, dropout, activation)
            for _ in range(num_experts)
        ])
        
        self.router_logits_cache = None
        self.expert_indices_cache = None
    
    def forward(self, x: torch.Tensor, spatial_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Args: x: [B, N, C]
        """
        B, N, C = x.shape
        H, W = spatial_shape
        
        # 1. 路由决策 (已向量化)
        # expert_weights: [B, num_patches, K], expert_indices: [B, num_patches, K]
        expert_weights, expert_indices, router_logits = self.router(x, spatial_shape)

        self.router_logits_cache = router_logits
        self.expert_indices_cache = expert_indices

        # 2. 准备数据 for 计算 (核心：将 Patch 路由结果广播回 Token 层面)
        
        # Patch area size (1x1 = 1)
        patch_area = self.patch_size * self.patch_size
        
        # 扩展路由结果到 Token 维度: [B, num_patches, K] -> [Total_Tokens, K]
        expert_indices_token = expert_indices.unsqueeze(2).expand(-1, -1, patch_area, -1)
        expert_weights_token = expert_weights.unsqueeze(2).expand(-1, -1, patch_area, -1)
        
        flat_indices = expert_indices_token.reshape(-1, self.top_k)
        flat_weights = expert_weights_token.reshape(-1, self.top_k)

        # 准备 Token 输入 flat_x
        if self.patch_size == 1:
            # 极速路径：Token=Patch，直接展平
            flat_x = x.reshape(-1, C)
        else:
            # 慢速路径：Patch > 1，需要 Unfold 还原 Patch 顺序
            x_img = x.transpose(1, 2).reshape(B, C, H, W)
            x_patches = F.unfold(x_img, kernel_size=self.patch_size, stride=self.patch_size)
            x_patches = x_patches.reshape(B, C, -1, patch_area).permute(0, 2, 3, 1)
            flat_x = x_patches.reshape(-1, C)

        # 3. 核心向量化 MoE 计算
        final_output = torch.zeros_like(flat_x)
        
        # 循环专家数量 (4次)，利用 GPU 的并行能力
        # 这一步是关键，它避免了 19200 次的同步
        for i, expert in enumerate(self.experts):
            for k in range(self.top_k):
                idx_k = flat_indices[:, k]
                weight_k = flat_weights[:, k]
                
                mask = (idx_k == i)
                if mask.any():
                    selected_tokens = flat_x[mask]
                    expert_out = expert(selected_tokens)
                    # 加权累加回输出
                    final_output[mask] += expert_out * weight_k[mask].unsqueeze(-1)

        # 4. 恢复形状
        if self.patch_size == 1:
             output = final_output.reshape(B, N, C)
        else:
             # 如果 patch_size > 1，使用 F.fold 还原
             # 为了避免复杂代码，如果只跑 S5 Only (patch_size=1)，这部分不会被执行
             num_patches_h = (H + self.patch_size - 1) // self.patch_size
             num_patches_w = (W + self.patch_size - 1) // self.patch_size
             output_reshaped = final_output.reshape(B, num_patches_h * num_patches_w, patch_area, C).permute(0, 3, 1, 2)
             output_patches = output_reshaped.reshape(B, C * patch_area, num_patches_h * num_patches_w)
             output = F.fold(output_patches, output_size=(H, W), kernel_size=self.patch_size, stride=self.patch_size)
             output = output.transpose(1, 2).reshape(B, N, C) 

        return output