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
                 top_k: int = 2, dropout: float = 0.1, activation: str = 'relu', 
                 noise_std: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.noise_std = noise_std
        
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
        Truly Sparse & Vectorized MoE Forward - Optimized for High-End GPUs (RTX 5090)
        
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
        if self.training:
            router_logits = router_logits + torch.randn_like(router_logits) * self.noise_std
        
        router_probs = F.softmax(router_logits, dim=-1)  # [B, N, E]
        expert_weights, expert_indices = torch.topk(router_probs, K, dim=-1)  # [B, N, K]
        expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + 1e-9)
        
        # Cache for loss
        self.router_logits_cache = router_logits.reshape(-1, E)
        self.expert_indices_cache = expert_indices.reshape(-1, K)

        # 2. Sparse Execution via Token Grouping (Optimized to minimize kernel launches)
        # Flatten x to [L, C] where L = B*N
        x_flat = x.view(-1, C)
        output_flat = torch.zeros_like(x_flat)
        
        expert_indices_flat = expert_indices.view(-1)  # [L*K]
        expert_weights_flat = expert_weights.view(-1)  # [L*K]
        
        # Pre-calculate token indices to avoid repeating interleave in loop
        token_indices = torch.arange(B * N, device=x.device).repeat_interleave(K)
        
        # We iterate over experts once.
        # This eliminates the nested loop (for expert_id and for k) and uses efficient indexing.
        for expert_id in range(E):
            # Find tokens assigned to this expert across all Top-K positions
            idx_mask = (expert_indices_flat == expert_id)
            if not idx_mask.any():
                continue
            
            # Extract tokens and their corresponding weights
            selected_token_ids = token_indices[idx_mask]
            selected_weights = expert_weights_flat[idx_mask].view(-1, 1)
            
            # index_select is generally faster than boolean masking for large GPUs
            current_x = torch.index_select(x_flat, 0, selected_token_ids)
            
            # Expert MLP Execution (using the nn.Module instance from ModuleList)
            expert_out = self.experts[expert_id](current_x)
            
            # Scatter addition: output[selected_token_ids] += out * weights
            # Use index_add_ for safe accumulation
            output_flat.index_add_(0, selected_token_ids, expert_out * selected_weights)
            
        return output_flat.view(B, N, C)

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

