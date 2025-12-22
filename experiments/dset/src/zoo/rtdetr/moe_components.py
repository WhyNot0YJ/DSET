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
        向量化 MoE 前向传播 - 无 for 循环
        
        Args:
            x: [B, N, C] Token特征，N可以是动态的（剪枝后）
            spatial_shape: (H, W) 空间形状（可选，仅作为元数据，不再用于计算）
        
        Returns:
            output: [B, N, C] 处理后的Token特征
        """
        B, N, C = x.shape
        E = self.num_experts
        K = self.top_k
        
        # 1. 路由计算
        router_logits = self.router(x)  # [B, N, E]
        
        # Noisy Gating: 在训练时添加噪声以提升探索和负载均衡
        if self.training:
            router_logits = router_logits + torch.randn_like(router_logits) * self.noise_std
        
        router_probs = F.softmax(router_logits, dim=-1)  # [B, N, E]
        expert_weights, expert_indices = torch.topk(router_probs, K, dim=-1)  # [B, N, K]
        
        # 归一化权重
        expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + 1e-9)  # [B, N, K]
        
        # 缓存用于负载均衡损失
        self.router_logits_cache = router_logits.reshape(-1, E)
        self.expert_indices_cache = expert_indices.reshape(-1, K)
        
        # 2. 向量化专家计算
        # 计算所有专家的输出，使用 einsum 实现批量矩阵乘法
        
        # 第一层 MLP：x @ W1^T + b1
        # x: [B, N, C], W1: [E, D_ff, C] -> hidden: [B, N, E, D_ff]
        hidden = torch.einsum('bnc,edc->bned', x, self.expert_w1)  # [B, N, E, D_ff]
        hidden = hidden + self.expert_b1  # broadcast [E, D_ff] -> [B, N, E, D_ff]
        
        # 激活 + Dropout
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        
        # 第二层 MLP：hidden @ W2^T + b2
        # hidden: [B, N, E, D_ff], W2: [E, C, D_ff] -> expert_outputs: [B, N, E, C]
        expert_outputs = torch.einsum('bned,ecd->bnec', hidden, self.expert_w2)  # [B, N, E, C]
        expert_outputs = expert_outputs + self.expert_b2  # broadcast [E, C] -> [B, N, E, C]
        
        # 3. 根据路由索引选择并加权
        # expert_indices: [B, N, K] - 每个 token 选择的 top-K 专家索引
        # expert_weights: [B, N, K] - 对应的权重
        
        # 使用 gather 选择被选中专家的输出
        # 扩展 expert_indices 以匹配 expert_outputs 的形状
        expert_indices_expanded = expert_indices.unsqueeze(-1).expand(-1, -1, -1, C)  # [B, N, K, C]
        
        # 从 expert_outputs 中选择：[B, N, E, C] -> [B, N, K, C]
        selected_outputs = torch.gather(expert_outputs, dim=2, index=expert_indices_expanded)  # [B, N, K, C]
        
        # 加权求和
        expert_weights_expanded = expert_weights.unsqueeze(-1)  # [B, N, K, 1]
        output = (selected_outputs * expert_weights_expanded).sum(dim=2)  # [B, N, C]
        
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
