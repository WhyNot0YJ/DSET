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

# =========================================================================
# 【新增】 AdaptiveExpertLayer - 用于 Decoder FFN
# (之前代码中缺失，导致 Decoder 无法导入)
# =========================================================================
class AdaptiveExpertLayer(nn.Module):
    """自适应专家层 - 动态融合多个专家网络的智能FFN层 (用于 Decoder Token MoE)"""
    
    def __init__(self, d_model: int, dim_feedforward: int, num_experts: int = 6, 
                 top_k: int = 2, dropout: float = 0.1, activation: str = 'relu'):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 自适应路由器
        self.router = AdaptiveRouter(d_model, num_experts, top_k)
        
        # 专家网络组
        self.experts = nn.ModuleList([
            SpecialistNetwork(d_model, dim_feedforward, dropout, activation)
            for _ in range(num_experts)
        ])
        
        self.router_logits_cache = None
        self.expert_indices_cache = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        reshape_needed = len(x.shape) == 3
        
        if reshape_needed:
            batch_size, seq_len, d_model = x.shape
            x = x.reshape(-1, d_model)  # [N, D]
        
        # 路由决策
        expert_weights, expert_indices, router_logits = self.router(x)  # [N, K], [N, K], [N, E]
        
        # 缓存用于负载均衡损失
        self.router_logits_cache = router_logits
        self.expert_indices_cache = expert_indices
        
        output = torch.zeros_like(x)
        
        # 向量化计算 MoE
        for i, expert in enumerate(self.experts):
            for k in range(self.top_k):
                idx_k = expert_indices[:, k]
                weight_k = expert_weights[:, k]
                
                mask = (idx_k == i)
                if mask.any():
                    selected_tokens = x[mask]
                    expert_out = expert(selected_tokens)
                    output[mask] += expert_out * weight_k[mask].unsqueeze(-1)
        
        # 恢复形状
        if reshape_needed:
            output = output.reshape(original_shape)
        
        return output
# =========================================================================


class PatchLevelRouter(nn.Module):
    """Patch级别路由器 - 向量化优化版 (用于 Encoder)"""
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2, patch_size: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.patch_size = patch_size
        self.gate_conv = nn.Conv2d(hidden_dim, num_experts, kernel_size=patch_size, stride=patch_size, bias=False)
        nn.init.normal_(self.gate_conv.weight, mean=0.0, std=0.01)
    
    def forward(self, x: torch.Tensor, spatial_shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            B, N, C = x.shape
            H, W = spatial_shape
            x_img = x.transpose(1, 2).reshape(B, C, H, W)
        else:
            x_img = x
            B, C, H, W = x.shape

        router_logits_map = self.gate_conv(x_img) 
        router_logits = router_logits_map.permute(0, 2, 3, 1).flatten(1, 2)
        
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
        B, N, C = x.shape
        H, W = spatial_shape
        
        expert_weights, expert_indices, router_logits = self.router(x, spatial_shape)

        self.router_logits_cache = router_logits
        self.expert_indices_cache = expert_indices

        patch_area = self.patch_size * self.patch_size
        expert_indices_token = expert_indices.unsqueeze(2).expand(-1, -1, patch_area, -1)
        expert_weights_token = expert_weights.unsqueeze(2).expand(-1, -1, patch_area, -1)
        
        flat_indices = expert_indices_token.reshape(-1, self.top_k)
        flat_weights = expert_weights_token.reshape(-1, self.top_k)

        if self.patch_size == 1:
            flat_x = x.reshape(-1, C)
        else:
            x_img = x.transpose(1, 2).reshape(B, C, H, W)
            x_patches = F.unfold(x_img, kernel_size=self.patch_size, stride=self.patch_size)
            x_patches = x_patches.reshape(B, C, -1, patch_area).permute(0, 2, 3, 1)
            flat_x = x_patches.reshape(-1, C)

        final_output = torch.zeros_like(flat_x)
        
        for i, expert in enumerate(self.experts):
            for k in range(self.top_k):
                idx_k = flat_indices[:, k]
                weight_k = flat_weights[:, k]
                
                mask = (idx_k == i)
                if mask.any():
                    selected_tokens = flat_x[mask]
                    expert_out = expert(selected_tokens)
                    final_output[mask] += expert_out * weight_k[mask].unsqueeze(-1)

        if self.patch_size == 1:
             output = final_output.reshape(B, N, C)
        else:
             num_patches_h = (H + self.patch_size - 1) // self.patch_size
             num_patches_w = (W + self.patch_size - 1) // self.patch_size
             output_reshaped = final_output.reshape(B, num_patches_h * num_patches_w, patch_area, C).permute(0, 3, 1, 2)
             output_patches = output_reshaped.reshape(B, C * patch_area, num_patches_h * num_patches_w)
             output = F.fold(output_patches, output_size=(H, W), kernel_size=self.patch_size, stride=self.patch_size)
             output = output.transpose(1, 2).reshape(B, N, C) 

        return output

# =========================================================================
# 【新增】负载均衡损失函数 (用于 Decoder 和 Encoder 损失计算)
# =========================================================================

def compute_patch_moe_balance_loss(router_logits_list: List[torch.Tensor],
                                   num_experts: int,
                                   expert_indices_list: List[torch.Tensor] = None) -> torch.Tensor:
    """Compute Encoder Patch-MoE balance loss (Variance-based, encouraging uniform usage)."""
    if len(router_logits_list) == 0: return torch.tensor(0.0)
    total_loss = 0.0
    for i, logits in enumerate(router_logits_list):
        probs = F.softmax(logits, dim=-1)
        N = logits.shape[0]
        expert_counts = probs.sum(dim=0)
        expert_ratio = expert_counts / N
        uniform_ratio = 1.0 / num_experts
        loss = torch.sum((expert_ratio - uniform_ratio) ** 2) * (num_experts)
        total_loss += loss
    return total_loss / len(router_logits_list)

def compute_patch_moe_entropy_loss(router_logits_list: List[torch.Tensor]) -> torch.Tensor:
    """Compute Encoder Patch-MoE entropy loss (Maximizing entropy)."""
    if len(router_logits_list) == 0: return torch.tensor(0.0)
    total_entropy = 0.0
    for logits in router_logits_list:
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-8)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        total_entropy += entropy
    return -total_entropy / len(router_logits_list)

def compute_expert_balance_loss(router_logits_list: List[torch.Tensor], 
                                num_experts: int,
                                expert_indices_list: List[torch.Tensor] = None) -> torch.Tensor:
    """Compute Decoder/AdaptiveExpert balance loss (Standard Switch Transformer style)."""
    if len(router_logits_list) == 0: return torch.tensor(0.0)
    
    total_loss = 0.0
    num_layers = 0
    
    for i, logits in enumerate(router_logits_list):
        if logits is None: continue
        
        probs = F.softmax(logits, dim=-1)
        expert_probs = probs.mean(dim=0)
        
        if expert_indices_list is not None and i < len(expert_indices_list) and expert_indices_list[i] is not None:
            indices = expert_indices_list[i]
            expert_usage = torch.zeros(num_experts, device=logits.device)
            for expert_id in range(num_experts):
                mask = (indices == expert_id).any(dim=-1)
                expert_usage[expert_id] = mask.float().mean()
        else:
            expert_usage = expert_probs
        
        # Loss = num_experts * sum(f_i * P_i)
        loss = num_experts * torch.sum(expert_usage * expert_probs)
        
        total_loss += loss
        num_layers += 1
    
    return total_loss / num_layers if num_layers > 0 else torch.tensor(0.0)