"""Universal Token-Level MoE Components for MOE-RTDETR - Unified Implementation"""

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
        
        # 权重初始化
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        if self.linear1.bias is not None:
            nn.init.constant_(self.linear1.bias, 0)
        if self.linear2.bias is not None:
            nn.init.constant_(self.linear2.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class MoELayer(nn.Module):
    """Universal Token-Level MoE Layer - 统一的Token级别MoE层
    
    替代原有的 AdaptiveExpertLayer，统一为Token级别的MoE实现。
    输入: [B, N, C] (N可以是动态的)
    输出: [B, N, C] (保持输入形状)
    """
    
    def __init__(self, d_model: int, dim_feedforward: int, num_experts: int = 6, 
                 top_k: int = 2, dropout: float = 0.1, activation: str = 'relu'):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        # Token-Level Router: 使用 Linear 投影
        self.router = nn.Linear(d_model, num_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
        
        # 专家网络组
        self.experts = nn.ModuleList([
            SpecialistNetwork(d_model, dim_feedforward, dropout, activation)
            for _ in range(num_experts)
        ])
        
        # 缓存用于负载均衡损失
        self.router_logits_cache = None
        self.expert_indices_cache = None
    
    def forward(self, x: torch.Tensor, spatial_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] Token特征，N可以是动态的
            spatial_shape: (H, W) 空间形状（可选，仅作为元数据，不再用于计算）
        
        Returns:
            output: [B, N, C] 处理后的Token特征
        """
        B, N, C = x.shape
        
        # 路由决策：Token-Level
        # x: [B, N, C] -> router_logits: [B, N, num_experts]
        router_logits = self.router(x)  # [B, N, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        expert_weights, expert_indices = torch.topk(router_probs, self.top_k, dim=-1)  # [B, N, K]
        
        # 归一化权重
        expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + 1e-9)
        
        # 缓存用于损失计算
        self.router_logits_cache = router_logits.reshape(-1, self.num_experts)  # [B*N, num_experts]
        self.expert_indices_cache = expert_indices.reshape(-1, self.top_k)  # [B*N, K]
        
        # Vectorized Top-K Masking 机制
        x_flat = x.reshape(-1, C)  # [B*N, C]
        output_flat = torch.zeros_like(x_flat)  # [B*N, C]
        
        expert_weights_flat = expert_weights.reshape(-1, self.top_k)  # [B*N, K]
        expert_indices_flat = expert_indices.reshape(-1, self.top_k)  # [B*N, K]
        
        # 对每个专家进行向量化计算
        for expert_id in range(self.num_experts):
            for k in range(self.top_k):
                # 找到分配给当前专家的token
                mask = (expert_indices_flat[:, k] == expert_id)  # [B*N]
                
                if mask.any():
                    # 提取对应的token和权重
                    selected_tokens = x_flat[mask]  # [M, C]
                    selected_weights = expert_weights_flat[mask, k:k+1]  # [M, 1]
                    
                    # 专家处理
                    expert_out = self.experts[expert_id](selected_tokens)  # [M, C]
                    
                    # 加权累加
                    output_flat[mask] += expert_out * selected_weights
        
        # 恢复形状
        output = output_flat.reshape(B, N, C)
        
        return output

# =========================================================================
# 负载均衡损失函数
# =========================================================================

def compute_expert_balance_loss(router_logits_list: List[torch.Tensor], 
                                num_experts: int,
                                expert_indices_list: List[torch.Tensor] = None) -> torch.Tensor:
    """Compute Decoder/AdaptiveExpert balance loss (Standard Switch Transformer style)."""
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

# =========================================================================
# 向后兼容：保留 AdaptiveExpertLayer 作为 MoELayer 的别名
# =========================================================================
AdaptiveExpertLayer = MoELayer

# =========================================================================
# 向后兼容：保留 AdaptiveRouter（虽然不再直接使用，但可能被其他代码引用）
# =========================================================================
class AdaptiveRouter(nn.Module):
    """自适应路由器 - 保留用于向后兼容"""
    
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [N, D] 或 [B*N, D]
        router_logits = self.gate(x)
        router_probs = F.softmax(router_logits, dim=-1)
        expert_weights, expert_indices = torch.topk(router_probs, self.top_k, dim=-1)
        expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + 1e-9)
        return expert_weights, expert_indices, router_logits
