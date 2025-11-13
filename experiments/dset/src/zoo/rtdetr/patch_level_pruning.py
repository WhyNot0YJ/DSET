"""Patch-level Pruning Module for DSET

实现Patch级别的剪枝，与Patch-MoE完全兼容。
核心思想：先划分patches，再在patch级别进行剪枝，保持规则的2D结构。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .token_pruning import LearnableImportancePredictor


class PatchLevelPruner(nn.Module):
    """Patch级别的剪枝器 - 与Patch-MoE兼容
    
    核心思想：
    1. 先将tokens划分成patches（与Patch-MoE的patch_size一致）
    2. 对每个patch计算重要性（patch内所有tokens的平均重要性）
    3. 基于patch重要性进行剪枝（保留top-k patches）
    4. 剪枝后的tokens仍然保持规则的2D结构，与Patch-MoE完全兼容
    
    优势：
    - 保持patch完整性，与Patch-MoE天然兼容
    - 真正实现计算节省（剪枝整个patches）
    - 不破坏空间结构
    """
    
    def __init__(self, 
                 input_dim: int,
                 patch_size: int = 8,
                 keep_ratio: float = 0.7,
                 adaptive: bool = True,
                 min_patches: int = 10,
                 warmup_epochs: int = 10,
                 prune_in_eval: bool = True):
        """初始化Patch级别剪枝器
        
        Args:
            input_dim: 输入特征维度
            patch_size: patch大小（必须与Patch-MoE的patch_size一致）
            keep_ratio: 保留patch的比例（0.5-0.7推荐）
            adaptive: 是否使用自适应剪枝
            min_patches: 最少保留的patch数量
            warmup_epochs: 剪枝warmup的epoch数
            prune_in_eval: 验证/测试时是否也进行剪枝
        """
        super().__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.keep_ratio = keep_ratio
        self.adaptive = adaptive
        self.min_patches = min_patches
        self.warmup_epochs = warmup_epochs
        self.prune_in_eval = prune_in_eval
        
        # 可学习的重要性预测器（用于预测token重要性）
        self.importance_predictor = LearnableImportancePredictor(input_dim)
        
        # 训练状态
        self.current_epoch = 0
        self.pruning_enabled = False
    
    def set_epoch(self, epoch: int):
        """设置当前epoch（用于渐进式剪枝）"""
        self.current_epoch = epoch
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
        """Patch级别剪枝前向传播
        
        Args:
            tokens: [B, N, C] 输入token特征，其中 N = H*W
            spatial_shape: (H, W) 原始空间形状
            return_indices: 是否返回保留的token索引
        
        Returns:
            pruned_tokens: [B, N', C] 剪枝后的tokens（N' = H'*W'，保持规则2D结构）
            kept_indices: [B, N'] 保留的token索引（如果return_indices=True）
            info: dict 包含剪枝统计信息
        """
        batch_size, num_tokens, channels = tokens.shape
        H, W = spatial_shape
        
        # 验证输入形状
        if H * W != num_tokens:
            raise ValueError(f"spatial_shape {spatial_shape} does not match num_tokens {num_tokens}")
        
        # 检查是否应该进行剪枝
        should_prune = self.pruning_enabled and (self.training or self.prune_in_eval)
        
        # 如果不应该剪枝，直接返回
        if not should_prune:
            info = {
                'pruning_ratio': 0.0,
                'num_kept_tokens': num_tokens,
                'num_pruned_tokens': 0,
                'num_patches': (H // self.patch_size) * (W // self.patch_size),
                'num_kept_patches': (H // self.patch_size) * (W // self.patch_size),
                'new_spatial_shape': (H, W)
            }
            kept_indices = None if not return_indices else torch.arange(
                num_tokens, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
            return tokens, kept_indices, info
        
        # 1. 将tokens reshape为2D: [B, H, W, C]
        tokens_2d = tokens.reshape(batch_size, H, W, channels)
        
        # 2. 计算patch数量
        patch_h = min(self.patch_size, H)
        patch_w = min(self.patch_size, W)
        num_patches_h = (H + patch_h - 1) // patch_h
        num_patches_w = (W + patch_w - 1) // patch_w
        num_patches = num_patches_h * num_patches_w
        
        # 3. Padding（如果需要）
        pad_h = (num_patches_h * patch_h - H) % patch_h
        pad_w = (num_patches_w * patch_w - W) % patch_w
        if pad_h > 0 or pad_w > 0:
            # 转换为 [B, C, H, W] 进行padding
            tokens_2d_padded = tokens_2d.permute(0, 3, 1, 2)  # [B, C, H, W]
            tokens_2d_padded = F.pad(tokens_2d_padded, (0, pad_w, 0, pad_h))  # pad right and bottom
            tokens_2d_padded = tokens_2d_padded.permute(0, 2, 3, 1)  # [B, H', W', C]
            H_padded, W_padded = tokens_2d_padded.shape[1], tokens_2d_padded.shape[2]
        else:
            tokens_2d_padded = tokens_2d
            H_padded, W_padded = H, W
        
        # 4. 使用unfold提取patches: [B, C, num_patches_h, patch_h, num_patches_w, patch_w]
        tokens_2d_conv = tokens_2d_padded.permute(0, 3, 1, 2)  # [B, C, H', W']
        patches = tokens_2d_conv.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)  # [B, C, num_patches_h, num_patches_w, patch_h, patch_w]
        patches = patches.contiguous().view(batch_size, channels, num_patches, patch_h, patch_w)  # [B, C, num_patches, patch_h, patch_w]
        
        # 5. 对每个patch计算重要性
        # 将每个patch reshape为 [B, num_patches, patch_h*patch_w, C]
        patches_flat = patches.permute(0, 2, 3, 4, 1).contiguous()  # [B, num_patches, patch_h, patch_w, C]
        patches_flat = patches_flat.reshape(batch_size, num_patches, patch_h * patch_w, channels)  # [B, num_patches, patch_h*patch_w, C]
        
        # 对每个patch内的tokens预测重要性
        patch_importance_list = []
        for p_idx in range(num_patches):
            patch_tokens = patches_flat[:, p_idx, :, :]  # [B, patch_h*patch_w, C]
            # 预测patch内每个token的重要性
            token_importance = self.importance_predictor(patch_tokens)  # [B, patch_h*patch_w]
            # patch的重要性 = patch内所有tokens的平均重要性
            patch_importance = token_importance.mean(dim=-1)  # [B]
            patch_importance_list.append(patch_importance)
        
        patch_importance_scores = torch.stack(patch_importance_list, dim=1)  # [B, num_patches]
        
        # 6. 基于patch重要性进行剪枝
        current_keep_ratio = self.get_current_keep_ratio()
        num_keep_patches = max(self.min_patches, int(num_patches * current_keep_ratio))
        
        # 选择top-k个重要的patches
        _, top_patch_indices = torch.topk(patch_importance_scores, num_keep_patches, dim=-1)  # [B, num_keep_patches]
        # 对索引排序（保持空间顺序）
        top_patch_indices_sorted, _ = torch.sort(top_patch_indices, dim=-1)  # [B, num_keep_patches]
        
        # 7. 收集保留的patches并重新组合
        # 为了保持规则结构，我们按行排列patches
        num_keep_patches_h = int((num_keep_patches + num_patches_w - 1) // num_patches_w)  # 向上取整
        num_keep_patches_w = min(num_keep_patches, num_patches_w)
        
        # 重新排列patches为规则网格
        kept_patches_list = []
        for b in range(batch_size):
            batch_patches = patches[b, :, :, :, :]  # [C, num_patches, patch_h, patch_w]
            kept_patch_indices = top_patch_indices_sorted[b]  # [num_keep_patches]
            kept_patches = batch_patches[:, kept_patch_indices, :, :]  # [C, num_keep_patches, patch_h, patch_w]
            kept_patches_list.append(kept_patches)
        
        kept_patches = torch.stack(kept_patches_list, dim=0)  # [B, C, num_keep_patches, patch_h, patch_w]
        
        # 8. 将保留的patches重新组合成2D特征图
        H_new = num_keep_patches_h * patch_h
        W_new = num_keep_patches_w * patch_w
        
        # 如果patches数量不足，用零填充
        if num_keep_patches < num_keep_patches_h * num_keep_patches_w:
            pad_patches = num_keep_patches_h * num_keep_patches_w - num_keep_patches
            zero_patches = torch.zeros(batch_size, channels, pad_patches, patch_h, patch_w, 
                                      device=kept_patches.device, dtype=kept_patches.dtype)
            kept_patches = torch.cat([kept_patches, zero_patches], dim=2)  # [B, C, num_keep_patches_h*num_keep_patches_w, patch_h, patch_w]
        
        # 重新reshape为规则网格
        kept_patches_reshaped = kept_patches.reshape(batch_size, channels, num_keep_patches_h, num_keep_patches_w, patch_h, patch_w)
        
        # 组合成完整特征图
        output_2d = torch.zeros(batch_size, channels, H_new, W_new, 
                               device=tokens.device, dtype=tokens.dtype)
        
        for h_idx in range(num_keep_patches_h):
            for w_idx in range(num_keep_patches_w):
                p_idx = h_idx * num_keep_patches_w + w_idx
                if p_idx < num_keep_patches:
                    h_start = h_idx * patch_h
                    h_end = h_start + patch_h
                    w_start = w_idx * patch_w
                    w_end = w_start + patch_w
                    output_2d[:, :, h_start:h_end, w_start:w_end] = kept_patches_reshaped[:, :, h_idx, w_idx, :, :]
        
        # 9. 转换回 [B, H'*W', C]
        output_2d = output_2d.permute(0, 2, 3, 1)  # [B, H', W', C]
        pruned_tokens = output_2d.reshape(batch_size, H_new * W_new, channels)  # [B, H'*W', C]
        
        # 10. 构建kept_indices（如果需要）
        if return_indices:
            # 计算保留的token在原grid中的位置
            kept_token_indices_list = []
            for b in range(batch_size):
                batch_kept_patches = top_patch_indices_sorted[b]  # [num_keep_patches]
                batch_indices = []
                for p_idx in batch_kept_patches:
                    p_h = int(p_idx.item()) // num_patches_w
                    p_w = int(p_idx.item()) % num_patches_w
                    # 计算这个patch在原grid中的token位置
                    for ph in range(patch_h):
                        for pw in range(patch_w):
                            orig_h = p_h * patch_h + ph
                            orig_w = p_w * patch_w + pw
                            if orig_h < H_padded and orig_w < W_padded:
                                orig_idx = orig_h * W_padded + orig_w
                                batch_indices.append(orig_idx)
                kept_token_indices_list.append(torch.tensor(batch_indices, device=tokens.device))
            
            # Padding到相同长度（用-1标记无效位置）
            max_len = max(len(indices) for indices in kept_token_indices_list)
            kept_indices = torch.full((batch_size, max_len), -1, device=tokens.device, dtype=torch.long)
            for b, indices in enumerate(kept_token_indices_list):
                kept_indices[b, :len(indices)] = indices
        else:
            kept_indices = None
        
        # 11. 统计信息
        num_kept_tokens = H_new * W_new
        info = {
            'pruning_ratio': 1.0 - (num_keep_patches / num_patches),
            'num_kept_tokens': num_kept_tokens,
            'num_pruned_tokens': num_tokens - num_kept_tokens,
            'num_patches': num_patches,
            'num_kept_patches': num_keep_patches,
            'num_pruned_patches': num_patches - num_keep_patches,
            'patch_importance_scores': patch_importance_scores.detach(),
            'new_spatial_shape': (H_new, W_new)
        }
        
        return pruned_tokens, kept_indices, info
    
    def compute_pruning_loss(self, info: dict) -> torch.Tensor:
        """计算Patch级别剪枝损失
        
        使用与TokenPruner相同的损失函数（基于重要性分数的熵）
        """
        if 'patch_importance_scores' not in info or not self.pruning_enabled:
            if 'patch_importance_scores' in info and info['patch_importance_scores'] is not None:
                return torch.tensor(0.0, device=info['patch_importance_scores'].device)
            return torch.tensor(0.0)
        
        patch_importance_scores = info['patch_importance_scores']  # [B, num_patches]
        
        # 计算每个patch的重要性概率（softmax）
        patch_probs = F.softmax(patch_importance_scores, dim=-1)  # [B, num_patches]
        
        # 熵损失：鼓励patch重要性分布更均匀（避免所有重要性集中在少数patches）
        entropy = -(patch_probs * torch.log(patch_probs + 1e-8)).sum(dim=-1).mean()
        num_patches = patch_importance_scores.shape[-1]
        max_entropy = torch.log(torch.tensor(num_patches, dtype=torch.float32, device=patch_importance_scores.device))
        normalized_entropy = entropy / max_entropy
        
        # 稀疏性损失：鼓励模型学习有效的patch重要性
        sparsity_loss = 1.0 - normalized_entropy
        
        return sparsity_loss

