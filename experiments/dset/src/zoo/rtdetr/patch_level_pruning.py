"""Patch-level Pruning Module for DSET - 与Patch-MoE兼容的patch级别剪枝"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .token_pruning import LearnableImportancePredictor


class PatchLevelPruner(nn.Module):
    """Patch级别剪枝器，与Patch-MoE兼容，保持规则2D结构"""
    
    def __init__(self, 
                 input_dim: int,
                 patch_size: int = 4,
                 keep_ratio: float = 0.7,
                 adaptive: bool = True,
                 min_patches: int = 10,
                 warmup_epochs: int = 10,
                 prune_in_eval: bool = True):
        """
        Args:
            input_dim: Input feature dimension
            patch_size: Patch size (must match Patch-MoE)
            keep_ratio: Retention ratio (0.5-0.7)
            adaptive: Whether to use adaptive pruning
            min_patches: Minimum patches to keep
            warmup_epochs: Warmup epochs
            prune_in_eval: Whether to prune during evaluation
        """
        super().__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.keep_ratio = keep_ratio
        self.adaptive = adaptive
        self.min_patches = min_patches
        self.warmup_epochs = warmup_epochs
        self.prune_in_eval = prune_in_eval
        
        self.importance_predictor = LearnableImportancePredictor(input_dim)
        self.current_epoch = 0
        self.pruning_enabled = False
    
    def set_epoch(self, epoch: int):
        """设置当前epoch"""
        self.current_epoch = epoch
        # 注意：epoch 从 0 开始，所以当 epoch = warmup_epochs 时应该启用剪枝
        # 例如：warmup_epochs=10，epoch=10 时应该启用（第11个epoch）
        if epoch >= self.warmup_epochs:
            self.pruning_enabled = True
        else:
            self.pruning_enabled = False
    
    def get_current_keep_ratio(self) -> float:
        """获取当前保留比例（渐进式调整）"""
        if not self.pruning_enabled:
            return 1.0
        if self.current_epoch < self.warmup_epochs:
            return 1.0
        # Start pruning when epoch >= warmup_epochs
        progress = min(1.0, (self.current_epoch - self.warmup_epochs + 1) / max(1, self.warmup_epochs))
        keep_ratio = 1.0 - progress * (1.0 - self.keep_ratio)
        return keep_ratio
    
    def forward(self, 
                tokens: torch.Tensor, 
                spatial_shape: Tuple[int, int],
                return_indices: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """
        Args:
            tokens: [B, N, C] token特征，N = H*W
            spatial_shape: (H, W) 空间形状
            return_indices: 是否返回保留索引
        
        Returns:
            pruned_tokens: [B, N', C] 剪枝后tokens
            kept_indices: [B, N'] 保留索引
            info: dict 统计信息
        """
        batch_size, num_tokens, channels = tokens.shape
        H, W = spatial_shape
        
        if H * W != num_tokens:
            raise ValueError(f"spatial_shape {spatial_shape} does not match num_tokens {num_tokens}")
        
        should_prune = self.pruning_enabled and (self.training or self.prune_in_eval)
        
        # 即使不执行剪枝，也计算 importance_scores 用于 loss 计算（训练时）
        should_compute_scores = self.training  # 训练时总是计算 scores，即使 pruning_enabled=False
        
        if not should_prune and not should_compute_scores:
            info = {
                'pruning_ratio': 0.0,
                'num_kept_tokens': num_tokens,
                'num_pruned_tokens': 0,
                'num_patches': (H // self.patch_size) * (W // self.patch_size),
                'num_kept_patches': (H // self.patch_size) * (W // self.patch_size),
                'new_spatial_shape': (H, W),
                'original_spatial_shape': (H, W)
            }
            kept_indices = None if not return_indices else torch.arange(
                num_tokens, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
            return tokens, kept_indices, info
        
        tokens_2d = tokens.reshape(batch_size, H, W, channels)
        
        patch_h = min(self.patch_size, H)
        patch_w = min(self.patch_size, W)
        num_patches_h = (H + patch_h - 1) // patch_h
        num_patches_w = (W + patch_w - 1) // patch_w
        num_patches = num_patches_h * num_patches_w
        
        pad_h = num_patches_h * patch_h - H
        pad_w = num_patches_w * patch_w - W
        if pad_h > 0 or pad_w > 0:
            tokens_2d_padded = tokens_2d.permute(0, 3, 1, 2)
            tokens_2d_padded = F.pad(tokens_2d_padded, (0, pad_w, 0, pad_h))
            tokens_2d_padded = tokens_2d_padded.permute(0, 2, 3, 1)
            H_padded, W_padded = tokens_2d_padded.shape[1], tokens_2d_padded.shape[2]
        else:
            tokens_2d_padded = tokens_2d
            H_padded, W_padded = H, W
        
        # 当patches数量太少时，直接跳过剪枝（避免不必要的计算）
        # 但如果需要计算 scores（训练时），仍然计算
        if num_patches <= self.min_patches:
            if should_compute_scores:
                # 计算 importance_scores 用于 loss（即使不执行剪枝）
                # ✅ 向量化修复：批量计算所有patches的评分，避免900次GPU调用
                tokens_2d_conv = tokens_2d_padded.permute(0, 3, 1, 2)
                patches = tokens_2d_conv.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
                patches = patches.contiguous().view(batch_size, channels, num_patches, patch_h, patch_w)
                patches_flat = patches.permute(0, 2, 3, 4, 1).contiguous()
                # patches_flat: [B, num_patches, patch_area, C]
                patches_flat = patches_flat.reshape(batch_size, num_patches, patch_h * patch_w, channels)
                
                # 向量化：将所有patches展平为 [B*num_patches, patch_area, C]
                patches_batch = patches_flat.view(batch_size * num_patches, patch_h * patch_w, channels)
                
                # 一次性计算所有patches的token重要性分数
                token_importance = self.importance_predictor(patches_batch)  # [B*num_patches, patch_area]
                
                # 计算每个patch的平均分数，然后重塑回 [B, num_patches]
                patch_importance_scores = token_importance.mean(dim=-1).view(batch_size, num_patches)
            else:
                patch_importance_scores = None
            
            info = {
                'pruning_ratio': 0.0,
                'num_kept_tokens': num_tokens,
                'num_pruned_tokens': 0,
                'num_patches': num_patches,
                'num_kept_patches': num_patches,
                'new_spatial_shape': (H, W),
                'original_spatial_shape': (H, W)
            }
            if patch_importance_scores is not None:
                info['patch_importance_scores'] = patch_importance_scores
            kept_indices = None if not return_indices else torch.arange(
                num_tokens, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
            return tokens, kept_indices, info
        
        # 计算保留的patches数量
        current_keep_ratio = self.get_current_keep_ratio()
        num_keep_patches_by_ratio = int(num_patches * current_keep_ratio)
        num_keep_patches = max(self.min_patches, num_keep_patches_by_ratio)
        # 确保 num_keep_patches 不超过 num_patches，避免 torch.topk 报错
        num_keep_patches = min(num_keep_patches, num_patches)
        
        tokens_2d_conv = tokens_2d_padded.permute(0, 3, 1, 2)
        patches = tokens_2d_conv.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
        patches = patches.contiguous().view(batch_size, channels, num_patches, patch_h, patch_w)
        
        patches_flat = patches.permute(0, 2, 3, 4, 1).contiguous()
        # patches_flat: [B, num_patches, patch_area, C]
        patches_flat = patches_flat.reshape(batch_size, num_patches, patch_h * patch_w, channels)
        
        # ✅ 向量化修复：批量计算所有patches的评分，避免900次GPU调用
        # 将所有patches展平为 [B*num_patches, patch_area, C]
        patches_batch = patches_flat.view(batch_size * num_patches, patch_h * patch_w, channels)
        
        # 一次性计算所有patches的token重要性分数
        token_importance = self.importance_predictor(patches_batch)  # [B*num_patches, patch_area]
        
        # 计算每个patch的平均分数，然后重塑回 [B, num_patches]
        patch_importance_scores = token_importance.mean(dim=-1).view(batch_size, num_patches)
        
        # 如果不执行剪枝，但需要计算 scores（用于 loss），直接返回原始 tokens 和 scores
        if not should_prune:
            info = {
                'pruning_ratio': 0.0,
                'num_kept_tokens': num_tokens,
                'num_pruned_tokens': 0,
                'num_patches': num_patches,
                'num_kept_patches': num_patches,
                'num_pruned_patches': 0,
                'patch_importance_scores': patch_importance_scores,  # 添加 scores 用于 loss 计算
                'new_spatial_shape': (H, W),
                'original_spatial_shape': (H, W)
            }
            kept_indices = None if not return_indices else torch.arange(
                num_tokens, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
            return tokens, kept_indices, info
        
        # 执行剪枝
        _, top_patch_indices = torch.topk(patch_importance_scores, num_keep_patches, dim=-1)
        top_patch_indices_sorted, _ = torch.sort(top_patch_indices, dim=-1)
        
        # ✅ 向量化修复：使用高级索引批量gather，避免batch循环
        # patches: [B, C, num_patches, patch_h, patch_w]
        # top_patch_indices_sorted: [B, num_keep_patches]
        # 使用 gather 或高级索引选择patches
        batch_indices = torch.arange(batch_size, device=patches.device).view(batch_size, 1).expand(batch_size, num_keep_patches)
        # 使用高级索引：patches[batch_indices, :, top_patch_indices_sorted, :, :]
        kept_patches = patches[batch_indices, :, top_patch_indices_sorted, :, :]  # [B, C, num_keep_patches, patch_h, patch_w]
        
        num_keep_patches_h = int((num_keep_patches + num_patches_w - 1) // num_patches_w)
        num_keep_patches_w = min(num_keep_patches, num_patches_w)
        if num_keep_patches_h * num_keep_patches_w < num_keep_patches:
            num_keep_patches_h += 1
        
        if num_keep_patches < num_keep_patches_h * num_keep_patches_w:
            pad_patches = num_keep_patches_h * num_keep_patches_w - num_keep_patches
            zero_patches = torch.zeros(batch_size, channels, pad_patches, patch_h, patch_w, 
                                      device=kept_patches.device, dtype=kept_patches.dtype)
            kept_patches = torch.cat([kept_patches, zero_patches], dim=2)
        
        kept_patches_reshaped = kept_patches.reshape(batch_size, channels, num_keep_patches_h, num_keep_patches_w, patch_h, patch_w)
        
        H_pruned = num_keep_patches_h * patch_h
        W_pruned = num_keep_patches_w * patch_w
        
        # ✅ 向量化修复：使用permute + reshape替代双重循环填充
        # kept_patches_reshaped: [B, C, num_keep_patches_h, num_keep_patches_w, patch_h, patch_w]
        # 目标：pruned_2d: [B, C, H_pruned, W_pruned]
        # 方法：permute + reshape
        pruned_2d = kept_patches_reshaped.permute(0, 1, 2, 4, 3, 5).contiguous()  # [B, C, num_keep_patches_h, patch_h, num_keep_patches_w, patch_w]
        pruned_2d = pruned_2d.view(batch_size, channels, H_pruned, W_pruned)  # [B, C, H_pruned, W_pruned]
        
        pruned_tokens = pruned_2d.permute(0, 2, 3, 1).reshape(batch_size, H_pruned * W_pruned, channels)
        
        if return_indices:
            # ✅ 向量化修复：使用向量化索引生成，避免4重循环
            # top_patch_indices_sorted: [B, num_keep_patches]
            # 计算每个patch对应的 (p_h, p_w)
            p_h_indices = top_patch_indices_sorted // num_patches_w  # [B, num_keep_patches]
            p_w_indices = top_patch_indices_sorted % num_patches_w   # [B, num_keep_patches]
            
            # 生成patch内所有token的偏移量
            ph_offsets = torch.arange(patch_h, device=tokens.device).view(1, 1, patch_h, 1)  # [1, 1, patch_h, 1]
            pw_offsets = torch.arange(patch_w, device=tokens.device).view(1, 1, 1, patch_w)  # [1, 1, 1, patch_w]
            
            # 扩展维度以便广播
            p_h_expanded = p_h_indices.unsqueeze(-1).unsqueeze(-1)  # [B, num_keep_patches, 1, 1]
            p_w_expanded = p_w_indices.unsqueeze(-1).unsqueeze(-1)  # [B, num_keep_patches, 1, 1]
            
            # 计算所有token的原始坐标
            orig_h_all = p_h_expanded * patch_h + ph_offsets  # [B, num_keep_patches, patch_h, 1]
            orig_w_all = p_w_expanded * patch_w + pw_offsets  # [B, num_keep_patches, 1, patch_w]
            
            # 广播到完整形状
            orig_h_all = orig_h_all.expand(-1, -1, -1, patch_w)  # [B, num_keep_patches, patch_h, patch_w]
            orig_w_all = orig_w_all.expand(-1, -1, patch_h, -1)  # [B, num_keep_patches, patch_h, patch_w]
            
            # 计算原始索引
            orig_idx_all = orig_h_all * W_padded + orig_w_all  # [B, num_keep_patches, patch_h, patch_w]
            
            # 创建mask：只保留有效的token（在原始H, W范围内）
            valid_mask = (orig_h_all < H) & (orig_w_all < W)  # [B, num_keep_patches, patch_h, patch_w]
            
            # 展平并过滤有效索引
            orig_idx_flat = orig_idx_all.reshape(batch_size, num_keep_patches * patch_h * patch_w)  # [B, num_keep_patches*patch_h*patch_w]
            valid_mask_flat = valid_mask.reshape(batch_size, num_keep_patches * patch_h * patch_w)  # [B, num_keep_patches*patch_h*patch_w]
            
            # 为每个batch收集有效索引
            kept_indices_list = []
            max_len = 0
            for b in range(batch_size):
                valid_indices = orig_idx_flat[b][valid_mask_flat[b]]  # [num_valid]
                kept_indices_list.append(valid_indices)
                max_len = max(max_len, len(valid_indices))
            
            if max_len > 0:
                kept_indices = torch.full((batch_size, max_len), -1, device=tokens.device, dtype=torch.long)
                for b, indices in enumerate(kept_indices_list):
                    if len(indices) > 0:
                        kept_indices[b, :len(indices)] = indices
            else:
                kept_indices = None
        else:
            kept_indices = None
        
        num_kept_tokens = H_pruned * W_pruned
        info = {
            'pruning_ratio': 1.0 - (num_keep_patches / num_patches),
            'num_kept_tokens': num_kept_tokens,
            'num_pruned_tokens': num_tokens - num_kept_tokens,
            'num_patches': num_patches,
            'num_kept_patches': num_keep_patches,
            'num_pruned_patches': num_patches - num_keep_patches,
            'patch_importance_scores': patch_importance_scores,
            'new_spatial_shape': (H_pruned, W_pruned),
            'original_spatial_shape': (H, W)
        }
        
        return pruned_tokens, kept_indices, info
    
    def compute_pruning_loss(self, info: dict) -> torch.Tensor:
        """计算剪枝损失（基于重要性分数的熵）"""
        # 只要有 patch_importance_scores 就计算 loss（即使 pruning_enabled=False，用于 warmup 期间训练）
        if 'patch_importance_scores' not in info or info.get('patch_importance_scores') is None:
            return torch.tensor(0.0, requires_grad=False)
        
        patch_importance_scores = info['patch_importance_scores']
        patch_probs = F.softmax(patch_importance_scores, dim=-1)
        entropy = -(patch_probs * torch.log(patch_probs + 1e-8)).sum(dim=-1).mean()
        num_patches = patch_importance_scores.shape[-1]
        max_entropy = torch.log(torch.tensor(num_patches, dtype=torch.float32, device=patch_importance_scores.device))
        normalized_entropy = entropy / max_entropy
        return 1.0 - normalized_entropy

