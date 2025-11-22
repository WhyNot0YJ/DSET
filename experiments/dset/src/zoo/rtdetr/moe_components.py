"""Adaptive Expert Components for DSET (Dual-Sparse Expert Transformer)

自适应专家组件：用于Encoder和Decoder Layer的FFN层
基于Switch Transformer和VisionMoE的设计理念

支持：
- Decoder中的细粒度MoE（已有）
- Encoder中的Patch-MoE（新增）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


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
            
            if expert_id < 0 or expert_id >= self.num_experts:
                continue  
            
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


def compute_patch_moe_balance_loss(router_logits_list: List[torch.Tensor],
                                   num_experts: int,
                                   expert_indices_list: List[torch.Tensor] = None) -> torch.Tensor:
    """计算Patch-MoE专用的负载均衡损失（Patch-level）。
    
    使用Patch-MoE的特殊设计：
    L_balance = (1/E) * sum_{e=1 to E} ( (N_e / N_patch) - (1/E) )^2
    
    其中：
    - E: 专家数量
    - N_e: 第e个专家被选中的patch数
    - N_patch: 所有patch数
    
    Args:
        router_logits_list: List of [B*N_patches, num_experts] 每层的路由logits
        num_experts: 专家数量
        expert_indices_list: List of [B*N_patches, top_k] 每层的专家索引
    
    Returns:
        balance_loss: 标量损失
    """
    if len(router_logits_list) == 0:
        return torch.tensor(0.0)
    
    total_loss = 0.0
    num_layers = 0
    
    for i, logits in enumerate(router_logits_list):
        if logits is None:
            continue
        
        N_patch = logits.shape[0]  # patch总数
        E = num_experts
        
        # 计算每个专家被选中的patch数 N_e
        if expert_indices_list is not None and i < len(expert_indices_list) and expert_indices_list[i] is not None:
            indices = expert_indices_list[i]  # [N_patch, top_k]
            # 统计每个专家被选中的patch数（只要在top_k中就算被选中）
            expert_counts = torch.zeros(E, device=logits.device)
            for expert_id in range(E):
                # 统计选择该专家的patch数量
                mask = (indices == expert_id).any(dim=-1)  # [N_patch]
                expert_counts[expert_id] = mask.float().sum()
        else:
            # 如果没有提供expert_indices，使用概率作为近似
            probs = F.softmax(logits, dim=-1)  # [N_patch, E]
            expert_counts = probs.sum(dim=0) * N_patch  # [E] 期望的patch数
        
        # Patch-MoE负载均衡损失：
        # L_balance = (1/E) * sum_{e=1 to E} ( (N_e / N_patch) - (1/E) )^2
        expert_ratio = expert_counts / N_patch  # [E] 每个专家被选中的比例
        uniform_ratio = 1.0 / E  # 均匀分布应该是 1/E
        balance_loss = (1.0 / E) * torch.sum((expert_ratio - uniform_ratio) ** 2)
        
        total_loss += balance_loss
        num_layers += 1
    
    return total_loss / num_layers if num_layers > 0 else torch.tensor(0.0)


def compute_patch_moe_entropy_loss(router_logits_list: List[torch.Tensor]) -> torch.Tensor:
    """计算Patch-MoE的熵正则项损失。
    
    防止gating机制变成硬分配，鼓励更平滑的gating分布：
    L_entropy = - (1/N_patch) * sum_{i=1 to N_patch} sum_{e=1 to E} p_{i,e} * log(p_{i,e})
    
    其中：
    - N_patch: patch总数
    - E: 专家数量
    - p_{i,e}: patch i 被路由到专家 e 的概率
    
    Args:
        router_logits_list: List of [B*N_patches, num_experts] 每层的路由logits
    
    Returns:
        entropy_loss: 标量损失（负熵，越小越好，所以需要最小化）
    """
    if len(router_logits_list) == 0:
        return torch.tensor(0.0)
    
    total_entropy = 0.0
    num_layers = 0
    
    for logits in router_logits_list:
        if logits is None:
            continue
        
        # 计算softmax概率
        probs = F.softmax(logits, dim=-1)  # [N_patch, E]
        
        # 计算熵：-sum(p * log(p))
        # 添加小值避免log(0)
        log_probs = torch.log(probs + 1e-8)  # [N_patch, E]
        entropy = -(probs * log_probs).sum(dim=-1)  # [N_patch] 每个patch的熵
        
        # 平均熵：L_entropy = - (1/N_patch) * sum(entropy)
        # 注意：这里返回负熵，因为我们要最大化熵（鼓励平滑分布）
        # 损失越小越好，所以返回 -entropy，这样最小化损失就是最大化熵
        avg_entropy = entropy.mean()  # 平均熵（越大越好）
        total_entropy += avg_entropy
        num_layers += 1
    
    # 返回负熵：损失 = -平均熵
    # 这样最小化损失就是最大化熵（鼓励平滑的gating分布）
    return -total_entropy / num_layers if num_layers > 0 else torch.tensor(0.0)


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


# ==================== Encoder Patch-MoE 组件 ====================

class PatchLevelRouter(nn.Module):
    """Patch级别路由器 - 真正的Patch-MoE路由器
    
    核心思想：每个patch（局部区域）统一选择专家，而不是每个token独立选择。
    这样可以：
    1. 减少路由计算量（从N个token减少到N_patches个patch）
    2. 保持局部性（patch内的所有tokens共享路由决策）
    3. 更适合视觉任务的归纳偏置
    """
    
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2, patch_size: int = 4):
        """初始化Patch级别路由器。
        
        Args:
            hidden_dim: 输入特征维度
            num_experts: 专家数量
            top_k: 选择前K个专家
            patch_size: patch大小（默认4x4，即每个patch包含16个tokens）
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.patch_size = patch_size
        
        # Patch级别的gate：对每个patch的特征进行池化后路由
        # 使用AdaptiveAvgPool2d + Linear实现patch级别的路由
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 对patch内的所有tokens进行平均池化
            nn.Flatten(),
            nn.Linear(hidden_dim, num_experts, bias=False)
        )
        
        # 初始化为均匀分布
        nn.init.normal_(self.gate[-1].weight, mean=0.0, std=0.01)
    
    def forward(self, x: torch.Tensor, spatial_shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Patch级别路由前向传播。
        
        Args:
            x: [B, H, W, C] 或 [B, N, C] (N=H*W) patch特征
            spatial_shape: (H, W) 空间形状
        
        Returns:
            Tuple:
                - expert_weights: [B*N_patches, top_k] 每个patch的专家权重
                - expert_indices: [B*N_patches, top_k] 每个patch的专家索引
                - router_logits: [B*N_patches, num_experts] 原始logits（用于负载均衡损失）
        """
        H, W = spatial_shape
        
        # 处理输入形状
        if len(x.shape) == 4:  # [B, H, W, C]
            B, H_in, W_in, C = x.shape
            # 转换为 [B, C, H, W] 用于卷积操作
            x_2d = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        elif len(x.shape) == 3:  # [B, N, C] where N = H*W
            B, N, C = x.shape
            H_in = W_in = int(N ** 0.5)  # 假设是正方形
            if H_in * W_in != N:
                raise ValueError(f"Cannot infer spatial shape from N={N}. Please provide spatial_shape.")
            x_2d = x.reshape(B, H_in, W_in, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        
        # 将特征图划分成patches
        patch_h = min(self.patch_size, H_in)
        patch_w = min(self.patch_size, W_in)
        
        # 计算patch数量
        num_patches_h = (H_in + patch_h - 1) // patch_h
        num_patches_w = (W_in + patch_w - 1) // patch_w
        num_patches = num_patches_h * num_patches_w
        
        # 使用unfold提取patches
        # 如果H或W不能被patch_size整除，需要padding
        pad_h = (num_patches_h * patch_h - H_in) % patch_h
        pad_w = (num_patches_w * patch_w - W_in) % patch_w
        if pad_h > 0 or pad_w > 0:
            x_2d = F.pad(x_2d, (0, pad_w, 0, pad_h))  # pad right and bottom
        
        # Unfold提取patches: [B, C, num_patches_h, patch_h, num_patches_w, patch_w]
        patches = x_2d.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)  # [B, C, num_patches_h, num_patches_w, patch_h, patch_w]
        patches = patches.contiguous().view(B, C, num_patches, patch_h, patch_w)  # [B, C, num_patches, patch_h, patch_w]
        
        # 对每个patch进行路由决策
        router_logits_list = []
        for p_idx in range(num_patches):
            patch = patches[:, :, p_idx, :, :]  # [B, C, patch_h, patch_w]
            # 使用gate对patch进行池化和路由
            logits = self.gate(patch)  # [B, num_experts]
            router_logits_list.append(logits)
        
        # 合并所有patches的路由logits
        router_logits = torch.stack(router_logits_list, dim=1)  # [B, num_patches, num_experts]
        router_logits = router_logits.reshape(B * num_patches, self.num_experts)  # [B*num_patches, num_experts]
        
        # Softmax + Top-K
        router_probs = F.softmax(router_logits, dim=-1)  # [B*num_patches, num_experts]
        expert_weights, expert_indices = torch.topk(router_probs, self.top_k, dim=-1)  # [B*num_patches, top_k]
        
        # 重新归一化（确保权重和为1）
        expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + 1e-9)
        
        return expert_weights, expert_indices, router_logits


class PatchMoELayer(nn.Module):
    """Patch-MoE层 - 用于Encoder的FFN层
    
    与Decoder MoE的区别：
    - 处理的是空间patch tokens（2D结构）
    - 可以考虑局部patch的相关性
    - 更注重空间特征的建模
    """
    
    def __init__(self, 
                 d_model: int, 
                 dim_feedforward: int, 
                 num_experts: int = 4, 
                 top_k: int = 2, 
                 dropout: float = 0.1, 
                 activation: str = 'gelu',
                 patch_size: int = 4):
        """初始化Patch-MoE层
        
        Args:
            d_model: 输入/输出维度
            dim_feedforward: FFN中间层维度
            num_experts: 专家数量（Encoder通常用较少的专家，如4个）
            top_k: 每次激活的专家数
            dropout: Dropout比率
            activation: 激活函数
            patch_size: patch大小（默认4x4）
        """
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.num_experts = num_experts
        self.top_k = top_k
        self.patch_size = patch_size
        
        # Patch级别路由器（真正的Patch-MoE）
        self.router = PatchLevelRouter(d_model, num_experts, top_k, patch_size)
        
        # 专家网络组
        self.experts = nn.ModuleList([
            SpecialistNetwork(d_model, dim_feedforward, dropout, activation)
            for _ in range(num_experts)
        ])
        
        # 用于收集router logits和expert_indices（计算负载均衡损失）
        self.router_logits_cache = None
        self.expert_indices_cache = None
    
    def forward(self, x: torch.Tensor, spatial_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """前向传播 - 真正的Patch-MoE实现
        
        核心思想：将输入划分成patches，每个patch统一选择专家，patch内的所有tokens共享路由决策。
        
        Args:
            x: [B, N, C] 或 [B, H, W, C] patch tokens
            spatial_shape: (H, W) 空间形状，如果x是[B, N, C]需要提供
        
        Returns:
            output: same shape as input
        """
        # 保存原始形状并提取空间信息
        original_shape = x.shape
        if len(x.shape) == 4:  # [B, H, W, C]
            B, H, W, C = x.shape
            spatial_shape = (H, W)
            x_2d = x  # [B, H, W, C]
            x_flat = x.reshape(B, H * W, C)  # [B, N, C]
        elif len(x.shape) == 3:  # [B, N, C]
            B, N, C = x.shape
            if spatial_shape is None:
                # 尝试从N推断（假设是正方形）
                H = W = int(N ** 0.5)
                if H * W != N:
                    raise ValueError(f"Cannot infer spatial shape from N={N}. Please provide spatial_shape.")
                spatial_shape = (H, W)
            else:
                H, W = spatial_shape
                if H * W != N:
                    raise ValueError(f"spatial_shape {spatial_shape} does not match N={N}")
            x_2d = x.reshape(B, H, W, C)  # [B, H, W, C]
            x_flat = x  # [B, N, C]
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        
        # 使用Patch级别路由器进行路由决策
        expert_weights, expert_indices, router_logits = self.router(x_2d, spatial_shape)  # [B*N_patches, K], [B*N_patches, K], [B*N_patches, E]
        
        # 缓存router logits和expert_indices用于负载均衡损失
        self.router_logits_cache = router_logits
        self.expert_indices_cache = expert_indices
        
        # 将特征图划分成patches（与router中的逻辑一致）
        patch_h = min(self.patch_size, H)
        patch_w = min(self.patch_size, W)
        num_patches_h = (H + patch_h - 1) // patch_h
        num_patches_w = (W + patch_w - 1) // patch_w
        num_patches = num_patches_h * num_patches_w
        
        # 转换为 [B, C, H, W] 用于unfold
        x_2d_conv = x_2d.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Padding（如果需要）
        pad_h = (num_patches_h * patch_h - H) % patch_h
        pad_w = (num_patches_w * patch_w - W) % patch_w
        if pad_h > 0 or pad_w > 0:
            x_2d_conv = F.pad(x_2d_conv, (0, pad_w, 0, pad_h))
        
        # Unfold提取patches
        patches = x_2d_conv.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)  # [B, C, num_patches_h, num_patches_w, patch_h, patch_w]
        patches = patches.contiguous().view(B, C, num_patches, patch_h, patch_w)  # [B, C, num_patches, patch_h, patch_w]
        
        # 重塑expert_weights和expert_indices以便与patches对应
        expert_weights = expert_weights.reshape(B, num_patches, self.top_k)  # [B, num_patches, top_k]
        expert_indices = expert_indices.reshape(B, num_patches, self.top_k)  # [B, num_patches, top_k]
        
        # 初始化输出
        output_patches = torch.zeros_like(patches)  # [B, C, num_patches, patch_h, patch_w]
        
        # 对每个patch进行处理
        for p_idx in range(num_patches):
            patch = patches[:, :, p_idx, :, :]  # [B, C, patch_h, patch_w]
            patch_tokens = patch.permute(0, 2, 3, 1).reshape(B, patch_h * patch_w, C)  # [B, patch_h*patch_w, C]
            patch_tokens_flat = patch_tokens.reshape(-1, C)  # [B*patch_h*patch_w, C]
            
            # 获取这个patch的路由决策
            patch_expert_weights = expert_weights[:, p_idx, :]  # [B, top_k]
            patch_expert_indices = expert_indices[:, p_idx, :]  # [B, top_k]
            
            # 初始化这个patch的输出
            patch_output = torch.zeros_like(patch_tokens_flat)  # [B*patch_h*patch_w, C]
            
            # 对每个选中的专家进行处理
            for k in range(self.top_k):
                expert_id = patch_expert_indices[:, k]  # [B] 每个batch选择哪个专家
                expert_weight = patch_expert_weights[:, k]  # [B] 每个batch的权重
                
                # 对每个batch分别处理（因为每个batch可能选择不同的专家）
                for b in range(B):
                    eid = int(expert_id[b].item())
                    weight = expert_weight[b].item()
                    

                    if eid < 0 or eid >= self.num_experts:
                        continue 
                    
                    # 这个batch的patch tokens
                    batch_patch_tokens = patch_tokens_flat[b * patch_h * patch_w:(b + 1) * patch_h * patch_w]  # [patch_h*patch_w, C]
                    
                    # 专家处理
                    expert_output = self.experts[eid](batch_patch_tokens)  # [patch_h*patch_w, C]
                    
                    # 加权累加
                    patch_output[b * patch_h * patch_w:(b + 1) * patch_h * patch_w] += expert_output * weight
            
            # 恢复patch形状
            patch_output_2d = patch_output.reshape(B, patch_h, patch_w, C).permute(0, 3, 1, 2)  # [B, C, patch_h, patch_w]
            output_patches[:, :, p_idx, :, :] = patch_output_2d
        
        # 将patches重新组合成完整的特征图
        # 首先reshape回 [B, C, num_patches_h, num_patches_w, patch_h, patch_w]
        output_patches_reshaped = output_patches.reshape(B, C, num_patches_h, num_patches_w, patch_h, patch_w)
        
        # 使用fold操作重新组合（fold是unfold的逆操作）
        # 但PyTorch没有直接的fold，我们需要手动组合
        output_2d = torch.zeros(B, C, num_patches_h * patch_h, num_patches_w * patch_w, device=x.device, dtype=x.dtype)
        for h_idx in range(num_patches_h):
            for w_idx in range(num_patches_w):
                p_idx = h_idx * num_patches_w + w_idx
                h_start = h_idx * patch_h
                h_end = h_start + patch_h
                w_start = w_idx * patch_w
                w_end = w_start + patch_w
                output_2d[:, :, h_start:h_end, w_start:w_end] = output_patches_reshaped[:, :, h_idx, w_idx, :, :]
        
        # 裁剪到原始大小（如果有padding）
        output_2d = output_2d[:, :, :H, :W]  # [B, C, H, W]
        
        # 转换回原始形状
        output_2d = output_2d.permute(0, 2, 3, 1)  # [B, H, W, C]
        if len(original_shape) == 3:
            output = output_2d.reshape(B, H * W, C)  # [B, N, C]
        else:
            output = output_2d  # [B, H, W, C]
        
        return output

