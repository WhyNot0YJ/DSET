"""Adaptive Expert Components for RT-DETR

自适应专家组件：用于Decoder Layer的FFN层
基于Switch Transformer和VisionMoE的设计理念
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class AdaptiveRouter(nn.Module):
    """自适应路由器 - 智能选择Top-K专家处理每个token。"""
    
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2):
        """初始化路由器。
        
        Args:
            hidden_dim: 输入特征维度
            num_experts: 专家数量
            top_k: 选择前K个专家
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        # 简单的线性路由器（Switch Transformer风格）
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        
        # 初始化为均匀分布
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """路由器前向传播。
        
        Args:
            x: [batch_size * seq_len, hidden_dim]
        
        Returns:
            Tuple:
                - expert_weights: [N, top_k] 专家权重
                - expert_indices: [N, top_k] 专家索引
                - router_logits: [N, num_experts] 原始logits（用于负载均衡损失）
        """
        # 计算路由logits
        router_logits = self.gate(x)  # [N, E]
        
        # Softmax + Top-K
        router_probs = F.softmax(router_logits, dim=-1)  # [N, E]
        expert_weights, expert_indices = torch.topk(router_probs, self.top_k, dim=-1)  # [N, K]
        
        # 重新归一化（确保权重和为1）
        expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + 1e-9)
        
        return expert_weights, expert_indices, router_logits


class SpecialistNetwork(nn.Module):
    """专家网络 - 基于标准两层MLP的领域专家。"""
    
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1, activation: str = 'relu'):
        """初始化专家网络。
        
        Args:
            d_model: 输入/输出维度
            dim_feedforward: FFN中间层维度
            dropout: Dropout比率
            activation: 激活函数（'relu', 'gelu', 'silu'）
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            # 默认relu
            self.activation = nn.ReLU()
        
        # 权重初始化
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        if self.linear1.bias is not None:
            nn.init.constant_(self.linear1.bias, 0)
        if self.linear2.bias is not None:
            nn.init.constant_(self.linear2.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。
        
        Args:
            x: [N, d_model]
        
        Returns:
            [N, d_model]
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class AdaptiveExpertLayer(nn.Module):
    """自适应专家层 - 动态融合多个专家网络的智能FFN层。
    
    用于替换Decoder Layer中的标准FFN，实现细粒度的专家混合。
    """
    
    def __init__(self, d_model: int, dim_feedforward: int, num_experts: int = 6, 
                 top_k: int = 2, dropout: float = 0.1, activation: str = 'relu'):
        """初始化自适应专家层。
        
        Args:
            d_model: 输入/输出维度
            dim_feedforward: FFN中间层维度
            num_experts: 专家数量
            top_k: 每次激活的专家数
            dropout: Dropout比率
            activation: 激活函数
        """
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
        
        # 用于收集router logits和expert_indices（计算负载均衡损失）
        self.router_logits_cache = None
        self.expert_indices_cache = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。
        
        Args:
            x: [batch_size, seq_len, d_model] 或 [batch_size * seq_len, d_model]
        
        Returns:
            output: same shape as input
        """
        # 保存原始形状
        original_shape = x.shape
        reshape_needed = len(x.shape) == 3
        
        if reshape_needed:
            batch_size, seq_len, d_model = x.shape
            x = x.reshape(-1, d_model)  # [B*L, D]
        
        # 路由决策
        expert_weights, expert_indices, router_logits = self.router(x)  # [N, K], [N, K], [N, E]
        
        # 缓存router logits和expert_indices用于负载均衡损失（不detach，需要梯度）
        self.router_logits_cache = router_logits
        self.expert_indices_cache = expert_indices
        
        # 初始化输出
        output = torch.zeros_like(x)  # [N, D]
        
        # 对每个专家执行计算（只计算Top-K专家，节省计算）
        # 使用unique找出所有被选中的专家
        unique_experts = torch.unique(expert_indices)
        
        for expert_id in unique_experts:
            expert_id = int(expert_id.item())
            
            # 找到选择了这个专家的所有token
            expert_mask = (expert_indices == expert_id).any(dim=-1)  # [N]
            if not expert_mask.any():
                continue
                
            expert_tokens = x[expert_mask]  # [N_expert, D]
            
            # 专家处理
            expert_output = self.experts[expert_id](expert_tokens)  # [N_expert, D]
            
            # 获取这个专家对应的权重
            expert_weight_mask = (expert_indices == expert_id)  # [N, K]
            expert_weight = torch.zeros(x.shape[0], device=x.device)  # [N]
            for k in range(self.top_k):
                mask_k = expert_weight_mask[:, k]  # [N]
                expert_weight[mask_k] += expert_weights[mask_k, k]
            
            # 加权累加
            output[expert_mask] += expert_output * expert_weight[expert_mask].unsqueeze(-1)
        
        # 恢复形状
        if reshape_needed:
            output = output.reshape(batch_size, seq_len, d_model)
        
        return output


def compute_expert_balance_loss(router_logits_list: List[torch.Tensor], 
                                num_experts: int,
                                expert_indices_list: List[torch.Tensor] = None) -> torch.Tensor:
    """计算专家负载均衡损失。
    
    确保各个专家被均匀激活，避免某些专家过载或闲置。
    使用标准的Switch Transformer负载均衡损失：
        loss = num_experts * sum(f_i * P_i)
    其中：
        - f_i: 实际路由到专家i的token比例（基于top-k选择）
        - P_i: 所有token对专家i的平均路由概率（softmax后的概率）
    
    这个损失鼓励实际使用分布和概率分布保持平衡。
    
    Args:
        router_logits_list: List of [N, num_experts] 每层的路由logits
        num_experts: 专家数量
        expert_indices_list: List of [N, top_k] 每层的专家索引（用于计算实际使用频率）
    
    Returns:
        load_balance_loss: 标量损失
    """
    if len(router_logits_list) == 0:
        return torch.tensor(0.0)
    
    total_loss = 0.0
    num_layers = 0
    
    for i, logits in enumerate(router_logits_list):
        if logits is None:
            continue
        
        # 计算softmax概率
        probs = F.softmax(logits, dim=-1)  # [N, E]
        
        # P_i: 每个专家的平均路由概率
        expert_probs = probs.mean(dim=0)  # [E]
        
        # f_i: 每个专家的实际使用频率（基于实际选择的token）
        if expert_indices_list is not None and i < len(expert_indices_list) and expert_indices_list[i] is not None:
            indices = expert_indices_list[i]  # [N, top_k]
            # 统计每个专家被实际选择的频率
            expert_usage = torch.zeros(num_experts, device=logits.device)
            for expert_id in range(num_experts):
                # 统计选择该专家的token数量（只要在top_k中就算被使用）
                mask = (indices == expert_id).any(dim=-1)  # [N]
                expert_usage[expert_id] = mask.float().mean()
        else:
            # 如果没有提供expert_indices，使用概率作为近似（向后兼容）
            expert_usage = expert_probs
        
        # 标准Switch Transformer负载均衡损失：
        # loss = num_experts * sum(f_i * P_i)
        # 当所有专家均匀使用时（f_i = P_i = 1/E），loss = 1
        # 当某些专家被过度使用时（f_i和P_i都大），loss会增大
        loss = num_experts * torch.sum(expert_usage * expert_probs)
        
        total_loss += loss
        num_layers += 1
    
    return total_loss / num_layers if num_layers > 0 else torch.tensor(0.0)

