"""Token Pruning Module for DSET - 可学习的Token重要性预测器"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class LearnableImportancePredictor(nn.Module):
    """可学习的Token重要性预测器（轻量级MLP）"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            dropout: Dropout比率
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, N, C] token特征
        
        Returns:
            importance_scores: [B, N] 重要性分数
        """
        x = self.fc1(tokens)
        x = self.activation(x)
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)


class TokenPruner(nn.Module):
    """Token剪枝模块，支持自适应和渐进式剪枝"""
    
    def __init__(self, 
                 input_dim: int,
                 keep_ratio: float = 0.7,
                 adaptive: bool = True,
                 min_tokens: int = 100,
                 warmup_epochs: int = 10,
                 prune_in_eval: bool = True):
        """
        Args:
            input_dim: 输入特征维度
            keep_ratio: 保留比例（0.5-0.7）
            adaptive: 是否自适应剪枝
            min_tokens: 最少保留token数
            warmup_epochs: warmup epoch数
            prune_in_eval: 验证时是否剪枝
        """
        super().__init__()
        self.input_dim = input_dim
        self.keep_ratio = keep_ratio
        self.adaptive = adaptive
        self.min_tokens = min_tokens
        self.warmup_epochs = warmup_epochs
        self.prune_in_eval = prune_in_eval
        
        self.importance_predictor = LearnableImportancePredictor(input_dim)
        self.current_epoch = 0
        self.pruning_enabled = False
    
    def set_epoch(self, epoch: int):
        """设置当前epoch"""
        self.current_epoch = epoch
        if epoch >= self.warmup_epochs:
            self.pruning_enabled = True
    
    def get_current_keep_ratio(self) -> float:
        """获取当前保留比例（渐进式调整）"""
        if not self.pruning_enabled or self.current_epoch < self.warmup_epochs:
            return 1.0
        # 当 epoch >= warmup_epochs 时开始剪枝，progress 从 1/(warmup_epochs+1) 开始
        # 这样在 epoch = warmup_epochs 时就有一定的剪枝比例
        progress = min(1.0, (self.current_epoch - self.warmup_epochs + 1) / max(1, self.warmup_epochs))
        return 1.0 - progress * (1.0 - self.keep_ratio)
    
    def forward(self, 
                tokens: torch.Tensor, 
                spatial_shape: Tuple[int, int],
                return_indices: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """
        Args:
            tokens: [B, N, C] token特征
            spatial_shape: (H, W) 空间形状
            return_indices: 是否返回保留索引
        
        Returns:
            pruned_tokens: [B, N', C] 剪枝后tokens
            kept_indices: [B, N'] 保留索引
            info: dict 统计信息
        """
        batch_size, num_tokens, channels = tokens.shape
        
        should_prune = self.pruning_enabled and (self.training or self.prune_in_eval)
        
        if not should_prune:
            info = {
                'pruning_ratio': 0.0,
                'num_kept': num_tokens,
                'num_pruned': 0
            }
            kept_indices = None if not return_indices else torch.arange(
                num_tokens, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
            return tokens, kept_indices, info
        
        # 当tokens数量太少时，跳过剪枝（避免不必要的计算，且剪枝收益很小）
        # 如果tokens总数小于等于min_tokens，直接跳过剪枝，返回所有tokens
        if num_tokens <= self.min_tokens:
            info = {
                'pruning_ratio': 0.0,
                'num_kept': num_tokens,
                'num_pruned': 0
            }
            kept_indices = None if not return_indices else torch.arange(
                num_tokens, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
            return tokens, kept_indices, info
        
        importance_scores = self.importance_predictor(tokens)
        current_keep_ratio = self.get_current_keep_ratio()
        num_keep = max(self.min_tokens, int(num_tokens * current_keep_ratio))
        # 确保 num_keep 不超过 num_tokens，避免 torch.topk 报错
        num_keep = min(num_keep, num_tokens)
        
        if self.adaptive:
            importance_probs = F.softmax(importance_scores, dim=-1)
            _, top_indices = torch.topk(importance_probs, num_keep, dim=-1)
        else:
            _, top_indices = torch.topk(importance_scores, num_keep, dim=-1)
        
        top_indices_sorted, _ = torch.sort(top_indices, dim=-1)
        batch_indices = torch.arange(batch_size, device=tokens.device).unsqueeze(1).expand(-1, num_keep)
        pruned_tokens = tokens[batch_indices, top_indices_sorted]
        
        info = {
            'pruning_ratio': 1.0 - (num_keep / num_tokens),
            'num_kept': num_keep,
            'num_pruned': num_tokens - num_keep,
            'importance_scores': importance_scores.detach()
        }
        
        kept_indices = top_indices_sorted if return_indices else None
        return pruned_tokens, kept_indices, info
    
    def compute_pruning_loss(self, info: dict) -> torch.Tensor:
        """计算剪枝损失（稀疏性损失 + 对比度损失）"""
        if 'importance_scores' not in info or not self.pruning_enabled:
            return torch.tensor(0.0)
        
        importance_scores = info['importance_scores']
        importance_probs = F.softmax(importance_scores, dim=-1)
        
        entropy = -(importance_probs * torch.log(importance_probs + 1e-8)).sum(dim=-1).mean()
        num_tokens = importance_scores.shape[1]
        max_entropy = torch.log(torch.tensor(num_tokens, dtype=torch.float32))
        sparsity_loss = entropy / max_entropy
        
        k = max(1, int(num_tokens * self.keep_ratio))
        top_scores, _ = torch.topk(importance_scores, k, dim=-1)
        top_k_mask = torch.zeros_like(importance_scores, dtype=torch.bool)
        top_k_indices = torch.topk(importance_scores, k, dim=-1)[1]
        top_k_mask.scatter_(1, top_k_indices, True)
        bottom_scores_masked = importance_scores.masked_fill(top_k_mask, float('-inf'))
        
        if bottom_scores_masked.shape[1] > k:
            bottom_mean = bottom_scores_masked[bottom_scores_masked != float('-inf')].mean()
            top_mean = top_scores.mean()
            contrast_loss = -torch.relu(top_mean - bottom_mean - 1.0)
        else:
            contrast_loss = torch.tensor(0.0, device=importance_scores.device)
        
        return sparsity_loss + 0.1 * contrast_loss


class SpatialTokenPruner(TokenPruner):
    """空间感知Token剪枝器，考虑空间位置信息"""
    
    def __init__(self, 
                 input_dim: int,
                 keep_ratio: float = 0.7,
                 adaptive: bool = True,
                 min_tokens: int = 100,
                 warmup_epochs: int = 10,
                 spatial_prior: str = 'none'):
        """
        Args:
            spatial_prior: 空间先验类型 ('none', 'center', 'edge')
        """
        super().__init__(input_dim, keep_ratio, adaptive, min_tokens, warmup_epochs)
        self.spatial_prior = spatial_prior
    
    def get_spatial_mask(self, 
                        spatial_shape: Tuple[int, int], 
                        device: torch.device) -> torch.Tensor:
        """
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
            y_coords = torch.arange(H, device=device).float() / (H - 1) - 0.5
            x_coords = torch.arange(W, device=device).float() / (W - 1) - 0.5
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            spatial_mask = torch.exp(-(xx**2 + yy**2) / 0.5)
            return spatial_mask.flatten()
        elif self.spatial_prior == 'edge':
            y_coords = torch.arange(H, device=device).float() / (H - 1) - 0.5
            x_coords = torch.arange(W, device=device).float() / (W - 1) - 0.5
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
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
        
        if not self.pruning_enabled or not self.training:
            info = {
                'pruning_ratio': 0.0,
                'num_kept': num_tokens,
                'num_pruned': 0
            }
            kept_indices = None if not return_indices else torch.arange(
                num_tokens, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
            return tokens, kept_indices, info
        
        # 当tokens数量太少时，跳过剪枝（避免不必要的计算，且剪枝收益很小）
        # 如果tokens总数小于等于min_tokens，直接跳过剪枝，返回所有tokens
        if num_tokens <= self.min_tokens:
            info = {
                'pruning_ratio': 0.0,
                'num_kept': num_tokens,
                'num_pruned': 0
            }
            kept_indices = None if not return_indices else torch.arange(
                num_tokens, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
            return tokens, kept_indices, info
        
        importance_scores = self.importance_predictor(tokens)
        
        if self.spatial_prior != 'none':
            spatial_mask = self.get_spatial_mask(spatial_shape, tokens.device)
            importance_scores = importance_scores * spatial_mask.unsqueeze(0)
        
        current_keep_ratio = self.get_current_keep_ratio()
        num_keep = max(self.min_tokens, int(num_tokens * current_keep_ratio))
        # 确保 num_keep 不超过 num_tokens，避免 torch.topk 报错
        num_keep = min(num_keep, num_tokens)
        
        _, top_indices = torch.topk(importance_scores, num_keep, dim=-1)
        top_indices_sorted, _ = torch.sort(top_indices, dim=-1)
        batch_indices = torch.arange(batch_size, device=tokens.device).unsqueeze(1).expand(-1, num_keep)
        pruned_tokens = tokens[batch_indices, top_indices_sorted]
        
        info = {
            'pruning_ratio': 1.0 - (num_keep / num_tokens),
            'num_kept': num_keep,
            'num_pruned': num_tokens - num_keep,
            'importance_scores': importance_scores.detach()
        }
        
        kept_indices = top_indices_sorted if return_indices else None
        return pruned_tokens, kept_indices, info

