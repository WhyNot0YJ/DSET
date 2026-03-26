"""Universal Token-Level MoE Components for MOE-RTDETR - Aligned with CaS_DETR."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

class MoELayer(nn.Module):
    """Universal Token-Level MoE Layer - vectorized implementation aligned to CaS_DETR."""

    def __init__(self, d_model: int, dim_feedforward: int, num_experts: int = 6,
                 top_k: int = 2, dropout: float = 0.1, activation: str = 'gelu',
                 noise_std: float = 0.1, router_init_std: float = 0.02):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.noise_std = noise_std
        self.dropout_rate = dropout

        # Router
        self.router = nn.Linear(d_model, num_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=router_init_std)

        # Vectorized expert weights (same shape/layout as CaS_DETR)
        self.expert_w1 = nn.Parameter(torch.empty(num_experts, dim_feedforward, d_model))
        self.expert_b1 = nn.Parameter(torch.zeros(num_experts, dim_feedforward))
        self.expert_w2 = nn.Parameter(torch.empty(num_experts, d_model, dim_feedforward))
        self.expert_b2 = nn.Parameter(torch.zeros(num_experts, d_model))
        for i in range(num_experts):
            nn.init.xavier_uniform_(self.expert_w1[i])
            nn.init.xavier_uniform_(self.expert_w2[i])

        self.dropout = nn.Dropout(dropout)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()

        # Shared-layer safe caches (same as CaS_DETR)
        self.router_logits_cache = []
        self.expert_indices_cache = []

    def reset_cache(self):
        self.router_logits_cache = []
        self.expert_indices_cache = []
    
    def forward(self, x: torch.Tensor, spatial_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Memory-Efficient Token-Grouping MoE - Scales with Token Count, not Weight-Token Product.
        Optimized to prevent OOM on large batches.

        Same layout as CaS_DETR: list caches for shared-layer safety; balance loss in decoder via
        compute_moe_balance_loss(router_logits_cache).
        """
        B, N, C = x.shape
        E = self.num_experts
        K = self.top_k

        # 1. Router Logic
        router_logits = self.router(x)  # [B, N, E]
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise
        router_probs = F.softmax(router_logits, dim=-1)  # [B, N, E]
        expert_weights, expert_indices = torch.topk(router_probs, K, dim=-1)  # [B, N, K]

        expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + 1e-9)

        self.router_logits_cache.append(router_logits.view(-1, E))
        self.expert_indices_cache.append(expert_indices.view(-1, K))

        x_flat = x.view(-1, C)
        out_flat = torch.zeros_like(x_flat)

        flat_expert_indices = expert_indices.view(-1, K)
        flat_expert_weights = expert_weights.view(-1, K)

        for i in range(E):
            token_indices, slot_indices = torch.where(flat_expert_indices == i)
            if token_indices.numel() == 0:
                continue

            temp_x = x_flat[token_indices]

            h = F.linear(temp_x, self.expert_w1[i], self.expert_b1[i])
            h = self.activation(h)
            h = self.dropout(h)
            temp_out = F.linear(h, self.expert_w2[i], self.expert_b2[i])

            weights = flat_expert_weights[token_indices, slot_indices].unsqueeze(-1)
            out_flat.index_add_(0, token_indices, temp_out * weights)

        return out_flat.view(B, N, C)


# =========================================================================
# 统一的 MoE 负载均衡损失函数（与 CaS_DETR 对齐）
# =========================================================================

def compute_moe_balance_loss(router_logits_list: List[torch.Tensor],
                             num_experts: int,
                             expert_indices_list: List[torch.Tensor] = None,
                             top_k: int = 2) -> torch.Tensor:
    """
    Switch Transformer 风格的 MoE 负载均衡损失。

    使用公式: Loss = E × Σ(f_i × P_i)
    其中:
        f_i: 实际路由到专家 i 的 token 比例 (通过 bincount 统计)，或退化为 expert_probs
        P_i: 专家 i 的平均 router 概率
    """
    if not router_logits_list:
        return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')

    total_loss = 0.0
    num_layers = 0

    for i, logits in enumerate(router_logits_list):
        if logits is None or logits.numel() == 0:
            continue

        probs = F.softmax(logits.float(), dim=-1)
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

