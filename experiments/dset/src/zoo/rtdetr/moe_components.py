"""Universal Token-Level MoE Components for DSET - Vectorized Implementation"""

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
    """Universal Token-Level MoE Layer - 向量化实现（无 for 循环）
    
    替代原有的 PatchMoELayer 和 AdaptiveExpertLayer，统一为Token级别的MoE实现。
    输入: [B, N, C] (N可以是动态的剪枝后的数量)
    输出: [B, N, C] (保持输入形状)
    
    关键优化：使用批量矩阵运算替代循环，充分利用 GPU 并行计算能力。
    """
    
    def __init__(self, d_model: int, dim_feedforward: int, num_experts: int = 4, 
                 top_k: int = 2, dropout: float = 0.1, activation: str = 'gelu', 
                 noise_std: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.noise_std = noise_std
        self.dropout_rate = dropout
        
        # Router
        self.router = nn.Linear(d_model, num_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        
        # 向量化专家权重：每个专家的两层 MLP
        # 第一层：d_model -> dim_feedforward，为每个专家分别存储
        self.expert_w1 = nn.Parameter(torch.empty(num_experts, dim_feedforward, d_model))
        self.expert_b1 = nn.Parameter(torch.zeros(num_experts, dim_feedforward))
        
        # 第二层：dim_feedforward -> d_model，为每个专家分别存储
        self.expert_w2 = nn.Parameter(torch.empty(num_experts, d_model, dim_feedforward))
        self.expert_b2 = nn.Parameter(torch.zeros(num_experts, d_model))
        
        # 初始化专家权重
        for i in range(num_experts):
            nn.init.xavier_uniform_(self.expert_w1[i])
            nn.init.xavier_uniform_(self.expert_w2[i])
        
        self.dropout = nn.Dropout(dropout)
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        self.router_logits_cache = None
        self.expert_indices_cache = None
    
    def forward(self, x: torch.Tensor, spatial_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Memory-Efficient Token-Grouping MoE - Scales with Token Count, not Weight-Token Product.
        Optimized to prevent OOM on large batches.
        
        Args:
            x: [B, N, C] Token features
        Returns:
            output: [B, N, C]
        """
        B, N, C = x.shape
        E = self.num_experts
        K = self.top_k
        
        # 1. Router Logic
        router_logits = self.router(x)  # [B, N, E]
        router_probs = F.softmax(router_logits, dim=-1)  # [B, N, E]
        expert_weights, expert_indices = torch.topk(router_probs, K, dim=-1)  # [B, N, K]
        
        # Renormalize expert weights
        expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + 1e-9)
        
        # Cache for loss (Flattened for convenience)
        self.router_logits_cache = router_logits.view(-1, E)
        self.expert_indices_cache = expert_indices.view(-1, K)

        x_flat = x.view(-1, C)
        out_flat = torch.zeros_like(x_flat)
        
        # Flatten expert mapping for efficient indexing
        flat_expert_indices = expert_indices.view(-1, K)
        flat_expert_weights = expert_weights.view(-1, K)
        
        # 2. Key Fix: Loop over experts and process assigned tokens as a group
        # This keeps memory usage dependent only on active tokens, not the full weight expansion.
        for i in range(E):
            # Find tokens assigned to current expert i
            token_indices, slot_indices = torch.where(flat_expert_indices == i)
            if token_indices.numel() == 0:
                continue
                
            # Extract only the tokens that need this expert
            temp_x = x_flat[token_indices]
            
            # Compute: Single expert pass for the gathered group of tokens
            # Using F.linear with the pre-stored expert parameters
            h = F.linear(temp_x, self.expert_w1[i], self.expert_b1[i])
            h = self.activation(h)
            h = self.dropout(h)
            temp_out = F.linear(h, self.expert_w2[i], self.expert_b2[i])
            
            # Apply routing weights and accumulate back to output
            weights = flat_expert_weights[token_indices, slot_indices].unsqueeze(-1)
            out_flat.index_add_(0, token_indices, temp_out * weights)
            
        return out_flat.view(B, N, C)


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
