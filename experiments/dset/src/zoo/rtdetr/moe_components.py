"""Universal Token-Level MoE Components for DSET - Vectorized Implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

class MoELayer(nn.Module):
    """Universal Token-Level MoE Layer - 向量化实现（无 for 循环）
    
    替代原有的 PatchMoELayer 和 AdaptiveExpertLayer，统一为Token级别的MoE实现。
    输入: [B, N, C] (N可以是动态的剪枝后的数量)
    输出: [B, N, C] (保持输入形状)
    
    关键优化：使用批量矩阵运算替代循环，充分利用 GPU 并行计算能力。
    """
    
    def __init__(self, d_model: int, dim_feedforward: int, num_experts: int = 4, 
                 top_k: int = 2, dropout: float = 0.1, activation: str = 'gelu', 
                 noise_std: float = 0.1, router_init_std: float = 0.05):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.noise_std = noise_std
        self.dropout_rate = dropout
        
        # [修复] 提高初始化 std 并支持配置化
        self.router = nn.Linear(d_model, num_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=router_init_std)
        
        # 向量化专家权重：每个专家的两层 MLP
        self.expert_w1 = nn.Parameter(torch.empty(num_experts, dim_feedforward, d_model))
        self.expert_b1 = nn.Parameter(torch.zeros(num_experts, dim_feedforward))
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
        
        # [修复] 缓存改为列表，以支持共享层多次 forward 的记录
        self.router_logits_cache = []
        self.expert_indices_cache = []

    def reset_cache(self):
        """用于在共享层模式下，每个 Batch 开始前清空记录"""
        self.router_logits_cache = []
        self.expert_indices_cache = []
    
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
        # [修复] Noisy Top-K: 仅在训练阶段加入探索噪声
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise
        router_probs = F.softmax(router_logits, dim=-1)  # [B, N, E]
        expert_weights, expert_indices = torch.topk(router_probs, K, dim=-1)  # [B, N, K]
        
        # Renormalize expert weights
        expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + 1e-9)
        
        # [修复] 列表化存储，防止共享层覆盖
        self.router_logits_cache.append(router_logits.view(-1, E))
        self.expert_indices_cache.append(expert_indices.view(-1, K))

        x_flat = x.view(-1, C)
        out_flat = torch.zeros_like(x_flat)
        
        # Flatten expert mapping for efficient indexing
        flat_expert_indices = expert_indices.view(-1, K)
        flat_expert_weights = expert_weights.view(-1, K)
        
        # 2. Key Fix: Loop over experts and process assigned tokens as a group
        for i in range(E):
            # Find tokens assigned to current expert i
            token_indices, slot_indices = torch.where(flat_expert_indices == i)
            if token_indices.numel() == 0:
                continue
                
            # Extract only the tokens that need this expert
            temp_x = x_flat[token_indices]
            
            # Compute: Single expert pass
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
                             expert_indices_list: List[torch.Tensor] = None,
                             top_k: int = 2) -> torch.Tensor:
    """
    [改进版] MoE 负载均衡损失 - 引入实际负载计算，修复统计盲区。
    
    使用公式: Loss = E · Σ(P_mean · f_mean)
    其中:
        P_mean: 原始 Softmax 概率的均值（软概率）
        f_mean: 实际负载代理（基于 Top-K 权重的聚合，保持可微性）
    
    Args:
        router_logits_list: Router logits 列表，每个元素形状为 [N, E]
        num_experts: 专家数量
        expert_indices_list: 专家索引列表（可选，用于未来扩展）
        top_k: Top-K 路由的 K 值（默认 2）
    
    Returns:
        负载均衡损失标量
    """
    if not router_logits_list: 
         return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
        
    total_loss = 0.0
    for logits in router_logits_list:
        # logits: [N, E], N 是 token 数量，E 是专家数量
        
        # 1. 计算原始 Softmax 概率（软概率）
        probs = F.softmax(logits, dim=-1)  # [N, E]
        
        # 2. P_mean: 每个专家的平均软概率
        P_mean = probs.mean(dim=0)  # [E]
        
        # 3. f_mean: 实际负载代理（基于 Top-K 权重的聚合，保持可微性）
        # 计算 Top-K 权重
        expert_weights, expert_indices = torch.topk(probs, top_k, dim=-1)  # [N, K], [N, K]
        
        # 归一化 Top-K 权重
        expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + 1e-9)  # [N, K]
        
        # 将 Top-K 权重聚合到每个专家上，得到实际负载（向量化实现）
        N = expert_indices.shape[0]
        device = logits.device
        
        # 使用 one-hot 编码将权重分配到对应的专家（向量化）
        # 将 [N, K] 的索引和权重展平为 [N*K]
        flat_indices = expert_indices.flatten()  # [N*K]
        flat_weights = expert_weights.flatten()  # [N*K]
        
        # 使用 scatter_add 一次性累加所有权重
        f_mean = torch.zeros(num_experts, device=device, dtype=logits.dtype)  # [E]
        f_mean.scatter_add_(0, flat_indices, flat_weights)  # [E]
        # f_mean 现在表示每个专家接收到的总权重（总和 = N，因为每个token的top-k权重总和为1）
        
        # 归一化实际负载（使其与 P_mean 在同一尺度，都是概率分布）
        f_mean = f_mean / (f_mean.sum() + 1e-9)  # 归一化到总和为 1，与 P_mean 对齐
        
        # 4. 计算负载均衡损失: Loss = E · Σ(P_mean · f_mean)
        # 这个公式能强迫"软概率"去对齐"硬计数"
        loss = num_experts * torch.sum(P_mean * f_mean)
        total_loss += loss
        
    return total_loss / len(router_logits_list)
