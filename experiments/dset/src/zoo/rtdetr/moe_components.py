"""Universal Token-Level MoE Components for DSET - Unified Implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

class SpecialistNetwork(nn.Module):
    """专家网络 - 标准MLP"""
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1, activation: str = 'relu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        if activation == 'relu': 
            self.activation = nn.ReLU()
        elif activation == 'gelu': 
            self.activation = nn.GELU()
        elif activation == 'silu': 
            self.activation = nn.SiLU()
        else: 
            self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class MoELayer(nn.Module):
    """Universal Token-Level MoE Layer - 统一的Token级别MoE层
    
    替代原有的 PatchMoELayer 和 AdaptiveExpertLayer，统一为Token级别的MoE实现。
    输入: [B, N, C] (N可以是动态的剪枝后的数量)
    输出: [B, N, C] (保持输入形状)
    """
    
    def __init__(self, d_model: int, dim_feedforward: int, num_experts: int = 4, 
                 top_k: int = 2, dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        self.router = nn.Linear(d_model, num_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        
        self.experts = nn.ModuleList([
            SpecialistNetwork(d_model, dim_feedforward, dropout, activation)
            for _ in range(num_experts)
        ])
        
        self.router_logits_cache = None
        self.expert_indices_cache = None
    
    def forward(self, x: torch.Tensor, spatial_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] Token特征，N可以是动态的（剪枝后）
            spatial_shape: (H, W) 空间形状（可选，仅作为元数据，不再用于计算）
        
        Returns:
            output: [B, N, C] 处理后的Token特征
        """
        B, N, C = x.shape
        
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        expert_weights, expert_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + 1e-9)
        
        self.router_logits_cache = router_logits.reshape(-1, self.num_experts)
        self.expert_indices_cache = expert_indices.reshape(-1, self.top_k)
        
        x_flat = x.reshape(-1, C)
        output_flat = torch.zeros_like(x_flat)
        expert_weights_flat = expert_weights.reshape(-1, self.top_k)
        expert_indices_flat = expert_indices.reshape(-1, self.top_k)
        
        for expert_id in range(self.num_experts):
            for k in range(self.top_k):
                mask = (expert_indices_flat[:, k] == expert_id)
                
                if mask.any():
                    selected_tokens = x_flat[mask]
                    selected_weights = expert_weights_flat[mask, k:k+1]
                    expert_out = self.experts[expert_id](selected_tokens)
                    output_flat[mask] += expert_out * selected_weights
        
        output = output_flat.reshape(B, N, C)
        
        return output

# =========================================================================
# 统一的 MoE 负载均衡损失函数
# =========================================================================

def compute_moe_balance_loss(router_logits_list: List[torch.Tensor], 
                             num_experts: int,
                             expert_indices_list: List[torch.Tensor] = None) -> torch.Tensor:
    """
    统一的 MoE 负载均衡损失函数，同时支持 Encoder 和 Decoder。
    
    使用 Switch Transformer 风格的负载均衡损失：
    Loss = num_experts * sum(f_i * P_i)
    
    其中：
    - f_i: 实际路由到专家 i 的 token 比例 (expert_usage)
    - P_i: 专家 i 的平均 router 概率 (expert_probs)
    
    Args:
        router_logits_list: List of router logits tensors, each of shape [N, num_experts]
        num_experts: Number of experts
        expert_indices_list: Optional list of expert indices tensors for computing actual usage
    
    Returns:
        Average balance loss across all layers
    """
    if len(router_logits_list) == 0: 
        return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    total_loss = 0.0
    num_layers = 0
    
    for i, logits in enumerate(router_logits_list):
        if logits is None or logits.numel() == 0:
            continue
        
        probs = F.softmax(logits, dim=-1)
        expert_probs = probs.mean(dim=0)
        
        if expert_indices_list is not None and i < len(expert_indices_list) and expert_indices_list[i] is not None:
            indices = expert_indices_list[i]
            flat_indices = indices.view(-1)
            usage_counts = torch.bincount(flat_indices, minlength=num_experts).float()
            total_dispatches = flat_indices.size(0)
            expert_usage = usage_counts / total_dispatches
        else:
            expert_usage = expert_probs
        
        loss = num_experts * torch.sum(expert_usage * expert_probs)
        
        total_loss += loss
        num_layers += 1
    
    return total_loss / num_layers if num_layers > 0 else torch.tensor(0.0)

