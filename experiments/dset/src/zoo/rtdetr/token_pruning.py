"""Token Pruning Module for DSET (Dual-Sparse Expert Transformer)

实现可学习的Token重要性预测器，用于在Encoder输入前进行Token剪枝。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class LearnableImportancePredictor(nn.Module):
    """可学习的Token重要性预测器
    
    通过一个轻量级MLP预测每个token的重要性分数，用于Token Pruning。
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        """初始化重要性预测器
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度（默认128，保持轻量级）
            dropout: Dropout比率
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 轻量级MLP：input -> hidden -> 1
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        # 初始化权重
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """前向传播：预测token重要性分数
        
        Args:
            tokens: [B, N, C] token特征
        
        Returns:
            importance_scores: [B, N] 重要性分数（未归一化）
        """
        # MLP预测重要性
        x = self.fc1(tokens)  # [B, N, hidden_dim]
        x = self.activation(x)
        x = self.dropout(x)
        importance_scores = self.fc2(x).squeeze(-1)  # [B, N]
        
        return importance_scores


class TokenPruner(nn.Module):
    """Token剪枝模块
    
    基于可学习的重要性预测器，在Encoder输入前剪枝冗余tokens。
    支持：
    - 自适应剪枝比例（基于重要性阈值）
    - 固定比例剪枝（保留top-k%）
    - 渐进式剪枝（训练过程中逐步启用）
    """
    
    def __init__(self, 
                 input_dim: int,
                 keep_ratio: float = 0.7,
                 adaptive: bool = True,
                 min_tokens: int = 100,
                 warmup_epochs: int = 10,
                 prune_in_eval: bool = True):
        """初始化Token Pruner
        
        Args:
            input_dim: 输入特征维度
            keep_ratio: 保留token的比例（0.5-0.7推荐）
            adaptive: 是否使用自适应剪枝
            min_tokens: 最少保留的token数量
            warmup_epochs: 剪枝warmup的epoch数（渐进式启用）
            prune_in_eval: 验证/测试时是否也进行剪枝（推荐True以保持训练/推理一致）
        """
        super().__init__()
        self.input_dim = input_dim
        self.keep_ratio = keep_ratio
        self.adaptive = adaptive
        self.min_tokens = min_tokens
        self.warmup_epochs = warmup_epochs
        self.prune_in_eval = prune_in_eval
        
        # 可学习的重要性预测器
        self.importance_predictor = LearnableImportancePredictor(input_dim)
        
        # 训练状态
        self.current_epoch = 0
        self.pruning_enabled = False
    
    def set_epoch(self, epoch: int):
        """设置当前epoch（用于渐进式剪枝）"""
        self.current_epoch = epoch
        # Warmup期间逐渐启用剪枝
        if epoch >= self.warmup_epochs:
            self.pruning_enabled = True
    
    def get_current_keep_ratio(self) -> float:
        """获取当前的保留比例（渐进式调整）"""
        if not self.pruning_enabled or self.current_epoch < self.warmup_epochs:
            return 1.0  # Warmup期间不剪枝
        
        # 渐进式调整：从1.0逐渐降到target keep_ratio
        progress = min(1.0, (self.current_epoch - self.warmup_epochs) / max(1, self.warmup_epochs))
        current_ratio = 1.0 - progress * (1.0 - self.keep_ratio)
        return current_ratio
    
    def forward(self, 
                tokens: torch.Tensor, 
                spatial_shape: Tuple[int, int],
                return_indices: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """Token剪枝前向传播
        
        Args:
            tokens: [B, N, C] 输入token特征（flatten后的）
            spatial_shape: (H, W) 原始空间形状
            return_indices: 是否返回保留的token索引
        
        Returns:
            pruned_tokens: [B, N', C] 剪枝后的tokens
            kept_indices: [B, N'] 保留的token索引（如果return_indices=True）
            info: dict 包含剪枝统计信息
        """
        batch_size, num_tokens, channels = tokens.shape
        
        # 检查是否应该进行剪枝
        should_prune = self.pruning_enabled and (self.training or self.prune_in_eval)
        
        # 如果不应该剪枝，直接返回
        if not should_prune:
            info = {
                'pruning_ratio': 0.0,
                'num_kept': num_tokens,
                'num_pruned': 0
            }
            kept_indices = None if not return_indices else torch.arange(
                num_tokens, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
            return tokens, kept_indices, info
        
        # 预测token重要性
        importance_scores = self.importance_predictor(tokens)  # [B, N]
        
        # 获取当前保留比例
        current_keep_ratio = self.get_current_keep_ratio()
        num_keep = max(self.min_tokens, int(num_tokens * current_keep_ratio))
        
        if self.adaptive:
            # 自适应剪枝：基于重要性阈值
            # 使用softmax转换为概率分布
            importance_probs = F.softmax(importance_scores, dim=-1)
            
            # 选择top-k个重要的tokens
            _, top_indices = torch.topk(importance_probs, num_keep, dim=-1)  # [B, num_keep]
        else:
            # 固定比例剪枝：直接选择top-k
            _, top_indices = torch.topk(importance_scores, num_keep, dim=-1)  # [B, num_keep]
        
        # 对索引排序（保持空间顺序）
        top_indices_sorted, _ = torch.sort(top_indices, dim=-1)
        
        # 收集保留的tokens
        batch_indices = torch.arange(batch_size, device=tokens.device).unsqueeze(1).expand(-1, num_keep)
        pruned_tokens = tokens[batch_indices, top_indices_sorted]  # [B, num_keep, C]
        
        # 统计信息
        info = {
            'pruning_ratio': 1.0 - (num_keep / num_tokens),
            'num_kept': num_keep,
            'num_pruned': num_tokens - num_keep,
            'importance_scores': importance_scores.detach()
        }
        
        kept_indices = top_indices_sorted if return_indices else None
        
        return pruned_tokens, kept_indices, info
    
    def compute_pruning_loss(self, info: dict) -> torch.Tensor:
        """计算剪枝相关的辅助损失（可选）
        
        鼓励模型学习有效的剪枝策略：
        - 稀疏性损失：鼓励重要性分数的概率分布稀疏（少数token高分，多数token低分）
        - 对比度损失：鼓励高重要性token和低重要性token之间有明显差异
        
        设计原理：
        - 使用softmax后的概率分布计算熵，低熵 = 稀疏分布
        - 不直接惩罚scores的绝对值，而是鼓励分布集中在少数token上
        
        Args:
            info: forward()返回的统计信息字典，需包含'importance_scores'
        
        Returns:
            pruning_loss: 标量损失
        """
        if 'importance_scores' not in info or not self.pruning_enabled:
            return torch.tensor(0.0)
        
        importance_scores = info['importance_scores']  # [B, N]
        
        # 转换为概率分布
        importance_probs = F.softmax(importance_scores, dim=-1)  # [B, N]
        
        # 1. 熵损失：鼓励稀疏的概率分布（低熵 = 概率集中在少数token上）
        # H(p) = -sum(p * log(p))，我们希望最小化熵
        entropy = -(importance_probs * torch.log(importance_probs + 1e-8)).sum(dim=-1).mean()
        # 归一化熵到[0, 1]范围，使其在不同token数量下可比
        num_tokens = importance_scores.shape[1]
        max_entropy = torch.log(torch.tensor(num_tokens, dtype=torch.float32))
        normalized_entropy = entropy / max_entropy
        # 熵损失：鼓励低熵（稀疏分布）
        sparsity_loss = normalized_entropy
        
        # 2. 对比度损失：鼓励top-k和其他token之间有明显差异
        # 计算保留比例对应的top-k个token的平均分数
        k = max(1, int(num_tokens * self.keep_ratio))
        top_scores, _ = torch.topk(importance_scores, k, dim=-1)  # [B, k]
        bottom_scores = importance_scores  # [B, N]
        # 创建mask排除top-k
        top_k_mask = torch.zeros_like(importance_scores, dtype=torch.bool)
        top_k_indices = torch.topk(importance_scores, k, dim=-1)[1]
        top_k_mask.scatter_(1, top_k_indices, True)
        bottom_scores_masked = importance_scores.masked_fill(top_k_mask, float('-inf'))
        # 计算bottom部分的均值（排除top-k）
        if bottom_scores_masked.shape[1] > k:
            bottom_mean = bottom_scores_masked[bottom_scores_masked != float('-inf')].mean()
            top_mean = top_scores.mean()
            # 对比度损失：鼓励top-k均值远大于bottom均值
            # 使用负的差值作为损失（差值越大，损失越小）
            contrast_loss = -torch.relu(top_mean - bottom_mean - 1.0)  # 至少相差1.0
        else:
            contrast_loss = torch.tensor(0.0, device=importance_scores.device)
        
        # 组合损失：主要依赖熵损失，对比度作为辅助
        pruning_loss = sparsity_loss + 0.1 * contrast_loss
        
        return pruning_loss


class SpatialTokenPruner(TokenPruner):
    """空间感知的Token剪枝器
    
    在Token Pruner基础上，考虑空间位置信息：
    - 保留关键区域的tokens（如中心区域、边缘区域）
    - 考虑多尺度特征的空间分布
    """
    
    def __init__(self, 
                 input_dim: int,
                 keep_ratio: float = 0.7,
                 adaptive: bool = True,
                 min_tokens: int = 100,
                 warmup_epochs: int = 10,
                 spatial_prior: str = 'none'):
        """初始化空间感知Token Pruner
        
        Args:
            spatial_prior: 空间先验类型 ('none', 'center', 'edge')
        """
        super().__init__(input_dim, keep_ratio, adaptive, min_tokens, warmup_epochs)
        self.spatial_prior = spatial_prior
    
    def get_spatial_mask(self, 
                        spatial_shape: Tuple[int, int], 
                        device: torch.device) -> torch.Tensor:
        """生成空间先验mask
        
        Args:
            spatial_shape: (H, W)
            device: torch设备
        
        Returns:
            spatial_mask: [H*W] 空间权重mask
        """
        H, W = spatial_shape
        
        if self.spatial_prior == 'none':
            return torch.ones(H * W, device=device)
        
        elif self.spatial_prior == 'center':
            # 中心区域权重更高
            y_coords = torch.arange(H, device=device).float() / (H - 1) - 0.5
            x_coords = torch.arange(W, device=device).float() / (W - 1) - 0.5
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            # 高斯权重（中心权重高）
            spatial_mask = torch.exp(-(xx**2 + yy**2) / 0.5)
            return spatial_mask.flatten()
        
        elif self.spatial_prior == 'edge':
            # 边缘区域权重更高
            y_coords = torch.arange(H, device=device).float() / (H - 1) - 0.5
            x_coords = torch.arange(W, device=device).float() / (W - 1) - 0.5
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            # 边缘权重（边缘权重高）
            spatial_mask = 1.0 - torch.exp(-(xx**2 + yy**2) / 0.5)
            return spatial_mask.flatten()
        
        else:
            return torch.ones(H * W, device=device)
    
    def forward(self, 
                tokens: torch.Tensor, 
                spatial_shape: Tuple[int, int],
                return_indices: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """前向传播（带空间先验）"""
        batch_size, num_tokens, channels = tokens.shape
        
        # 如果未启用剪枝，直接返回
        if not self.pruning_enabled or not self.training:
            info = {
                'pruning_ratio': 0.0,
                'num_kept': num_tokens,
                'num_pruned': 0
            }
            kept_indices = None if not return_indices else torch.arange(
                num_tokens, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
            return tokens, kept_indices, info
        
        # 预测token重要性
        importance_scores = self.importance_predictor(tokens)  # [B, N]
        
        # 应用空间先验mask
        if self.spatial_prior != 'none':
            spatial_mask = self.get_spatial_mask(spatial_shape, tokens.device)  # [N]
            importance_scores = importance_scores * spatial_mask.unsqueeze(0)
        
        # 获取当前保留比例
        current_keep_ratio = self.get_current_keep_ratio()
        num_keep = max(self.min_tokens, int(num_tokens * current_keep_ratio))
        
        # 选择top-k个重要的tokens
        _, top_indices = torch.topk(importance_scores, num_keep, dim=-1)  # [B, num_keep]
        
        # 对索引排序（保持空间顺序）
        top_indices_sorted, _ = torch.sort(top_indices, dim=-1)
        
        # 收集保留的tokens
        batch_indices = torch.arange(batch_size, device=tokens.device).unsqueeze(1).expand(-1, num_keep)
        pruned_tokens = tokens[batch_indices, top_indices_sorted]  # [B, num_keep, C]
        
        # 统计信息
        info = {
            'pruning_ratio': 1.0 - (num_keep / num_tokens),
            'num_kept': num_keep,
            'num_pruned': num_tokens - num_keep,
            'importance_scores': importance_scores.detach()
        }
        
        kept_indices = top_indices_sorted if return_indices else None
        
        return pruned_tokens, kept_indices, info

