"""Patch-level Pruning Module for DSET"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict


class LearnableImportancePredictor(nn.Module):
    """Token重要性预测器（轻量级MLP）"""
    
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


class PatchLevelPruner(nn.Module):
    """Patch级别剪枝器，支持CASS监督学习"""
    
    def __init__(self, 
                 input_dim: int,
                 patch_size: int = 4,
                 keep_ratio: float = 0.7,
                 adaptive: bool = True,
                 min_patches: int = 10,
                 warmup_epochs: int = 10,
                 prune_in_eval: bool = True,
                 # CASS parameters
                 use_cass: bool = False,
                 cass_expansion_ratio: float = 0.3,
                 cass_min_size: float = 1.0,
                 cass_decay_type: str = 'gaussian',
                 use_subpixel_offset: bool = True,
                 use_focal_loss: bool = True,
                 cass_focal_alpha: float = 2.0,
                 cass_focal_beta: float = 4.0):
        """
        Args:
            input_dim: Input feature dimension
            patch_size: Patch size (must match Patch-MoE)
            keep_ratio: Retention ratio (0.5-0.7)
            adaptive: Whether to use adaptive pruning
            min_patches: Minimum patches to keep
            warmup_epochs: Warmup epochs
            prune_in_eval: Whether to prune during evaluation
            use_cass: Whether to use Context-Aware Soft Supervision
            cass_expansion_ratio: Expansion ratio for context band (0.2-0.3)
            cass_min_size: Minimum box size on feature map (pixels)
            cass_decay_type: Decay type for context band ('gaussian' or 'linear')
            use_subpixel_offset: Whether to use sub-pixel offset compensation
            use_focal_loss: Whether to use Focal Loss instead of MSE
            cass_focal_alpha: Focal Loss alpha parameter
            cass_focal_beta: Focal Loss beta parameter
        """
        super().__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.keep_ratio = keep_ratio
        self.adaptive = adaptive
        self.min_patches = min_patches
        self.warmup_epochs = warmup_epochs
        self.prune_in_eval = prune_in_eval
        
        # CASS parameters
        self.use_cass = use_cass
        self.cass_expansion_ratio = cass_expansion_ratio
        self.cass_min_size = cass_min_size
        self.cass_decay_type = cass_decay_type
        self.use_subpixel_offset = use_subpixel_offset
        self.use_focal_loss = use_focal_loss
        self.cass_focal_alpha = cass_focal_alpha
        self.cass_focal_beta = cass_focal_beta
        
        self.importance_predictor = LearnableImportancePredictor(input_dim)
        self.current_epoch = 0
        self.pruning_enabled = False
    
    def set_epoch(self, epoch: int):
        """设置当前epoch"""
        self.current_epoch = epoch
        self.pruning_enabled = epoch >= self.warmup_epochs
    
    def get_current_keep_ratio(self) -> float:
        """获取当前保留比例"""
        if not self.pruning_enabled or self.current_epoch < self.warmup_epochs:
            return 1.0
        progress = min(1.0, (self.current_epoch - self.warmup_epochs + 1) / max(1, self.warmup_epochs))
        return 1.0 - progress * (1.0 - self.keep_ratio)
    
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
        
        # Warmup阶段跳过所有剪枝计算
        if self.training and self.current_epoch < self.warmup_epochs:
            patch_h = min(self.patch_size, H)
            patch_w = min(self.patch_size, W)
            num_patches_h = (H + patch_h - 1) // patch_h
            num_patches_w = (W + patch_w - 1) // patch_w
            num_patches_total = num_patches_h * num_patches_w
            
            info = {
                'pruning_ratio': 0.0,
                'num_kept_tokens': num_tokens,
                'num_pruned_tokens': 0,
                'num_patches': num_patches_total,
                'num_kept_patches': num_patches_total,
                'patch_importance_scores': None,
                'new_spatial_shape': (H, W),
                'original_spatial_shape': (H, W)
            }
            kept_indices = None if not return_indices else torch.arange(
                num_tokens, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
            return tokens, kept_indices, info
        
        should_prune = self.pruning_enabled and (self.training or self.prune_in_eval)
        should_compute_scores = self.training
        
        if not should_prune and not should_compute_scores:
            patch_h = min(self.patch_size, H)
            patch_w = min(self.patch_size, W)
            num_patches_h = (H + patch_h - 1) // patch_h
            num_patches_w = (W + patch_w - 1) // patch_w
            num_patches_total = num_patches_h * num_patches_w
            
            info = {
                'pruning_ratio': 0.0,
                'num_kept_tokens': num_tokens,
                'num_pruned_tokens': 0,
                'num_patches': num_patches_total,
                'num_kept_patches': num_patches_total,
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
        
        if num_patches <= self.min_patches:
            if should_compute_scores:
                tokens_2d_conv = tokens_2d_padded.permute(0, 3, 1, 2)
                patches = tokens_2d_conv.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
                patches = patches.contiguous().view(batch_size, channels, num_patches, patch_h, patch_w)
                patches_flat = patches.permute(0, 2, 3, 4, 1).contiguous()
                patches_flat = patches_flat.reshape(batch_size, num_patches, patch_h * patch_w, channels)
                patches_batch = patches_flat.view(batch_size * num_patches, patch_h * patch_w, channels)
                token_importance = self.importance_predictor(patches_batch)
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
        
        current_keep_ratio = self.get_current_keep_ratio()
        num_keep_patches_by_ratio = int(num_patches * current_keep_ratio)
        num_keep_patches = min(max(self.min_patches, num_keep_patches_by_ratio), num_patches)
        
        tokens_2d_conv = tokens_2d_padded.permute(0, 3, 1, 2)
        patches = tokens_2d_conv.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
        patches = patches.contiguous().view(batch_size, channels, num_patches, patch_h, patch_w)
        patches_flat = patches.permute(0, 2, 3, 4, 1).contiguous()
        patches_flat = patches_flat.reshape(batch_size, num_patches, patch_h * patch_w, channels)
        patches_batch = patches_flat.view(batch_size * num_patches, patch_h * patch_w, channels)
        token_importance = self.importance_predictor(patches_batch)
        patch_importance_scores = token_importance.mean(dim=-1).view(batch_size, num_patches)
        
        if not should_prune:
            info = {
                'pruning_ratio': 0.0,
                'num_kept_tokens': num_tokens,
                'num_pruned_tokens': 0,
                'num_patches': num_patches,
                'num_kept_patches': num_patches,
                'num_pruned_patches': 0,
                'patch_importance_scores': patch_importance_scores,
                'new_spatial_shape': (H, W),
                'original_spatial_shape': (H, W)
            }
            kept_indices = None if not return_indices else torch.arange(
                num_tokens, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
            return tokens, kept_indices, info
        
        _, top_patch_indices = torch.topk(patch_importance_scores, num_keep_patches, dim=-1)
        top_patch_indices_sorted, _ = torch.sort(top_patch_indices, dim=-1)
        
        top_patch_indices_expanded = top_patch_indices_sorted.unsqueeze(-1).unsqueeze(-1)
        top_patch_indices_expanded = top_patch_indices_expanded.expand(-1, -1, patch_h * patch_w, channels)
        kept_patches_flat = patches_flat.gather(dim=1, index=top_patch_indices_expanded)
        pruned_tokens = kept_patches_flat.reshape(batch_size, num_keep_patches * patch_h * patch_w, channels)
        num_kept_tokens_actual = num_keep_patches * patch_h * patch_w
        
        if return_indices:
            p_h_indices = top_patch_indices_sorted // num_patches_w
            p_w_indices = top_patch_indices_sorted % num_patches_w
            ph_offsets = torch.arange(patch_h, device=tokens.device).view(1, 1, patch_h, 1)
            pw_offsets = torch.arange(patch_w, device=tokens.device).view(1, 1, 1, patch_w)
            p_h_expanded = p_h_indices.unsqueeze(-1).unsqueeze(-1)
            p_w_expanded = p_w_indices.unsqueeze(-1).unsqueeze(-1)
            orig_h_all = p_h_expanded * patch_h + ph_offsets
            orig_w_all = p_w_expanded * patch_w + pw_offsets
            orig_h_all = orig_h_all.expand(-1, -1, -1, patch_w)
            orig_w_all = orig_w_all.expand(-1, -1, patch_h, -1)
            orig_idx_all = orig_h_all * W_padded + orig_w_all
            valid_mask = (orig_h_all < H) & (orig_w_all < W)
            orig_idx_flat = orig_idx_all.reshape(batch_size, -1)
            valid_mask_flat = valid_mask.reshape(batch_size, -1)
            
            # 向量化处理：完全消除 Python 循环
            # 1. 确定每个 batch 的最大有效 token 数
            valid_counts = valid_mask_flat.sum(dim=1)
            max_valid_count = int(valid_counts.max().item()) if valid_counts.numel() > 0 and valid_counts.max() > 0 else 0
            
            if max_valid_count > 0:
                # 2. 构造 [B, max_valid_count] 的索引矩阵
                # 使用 topk 配合 mask 可以快速提取有效索引并保持相对顺序
                # 我们给无效位置一个非常大的负数，然后取前 max_valid_count 个
                score_for_selection = torch.where(valid_mask_flat, 
                                                 torch.arange(orig_idx_flat.shape[1], device=tokens.device).float(),
                                                 torch.tensor(-1e9, device=tokens.device))
                _, selection_indices = torch.topk(score_for_selection, max_valid_count, dim=1, largest=True)
                # 重新排序以保持原始顺序
                selection_indices, _ = torch.sort(selection_indices, dim=1)
                
                # 3. 提取索引并应用 mask
                kept_indices = torch.gather(orig_idx_flat, 1, selection_indices)
                batch_valid_mask = torch.gather(valid_mask_flat, 1, selection_indices)
                kept_indices = torch.where(batch_valid_mask, kept_indices, torch.tensor(-1, device=tokens.device))
            else:
                kept_indices = None
        else:
            kept_indices = None
        
        num_kept_tokens = num_kept_tokens_actual
        H_pruned_approx = int((num_kept_tokens + W - 1) // W) if W > 0 else num_kept_tokens
        W_pruned_approx = W if H_pruned_approx * W >= num_kept_tokens else num_kept_tokens
        
        info = {
            'pruning_ratio': 1.0 - (num_keep_patches / num_patches),
            'num_kept_tokens': num_kept_tokens,
            'num_pruned_tokens': num_tokens - num_kept_tokens,
            'num_patches': num_patches,
            'num_kept_patches': num_keep_patches,
            'num_pruned_patches': num_patches - num_keep_patches,
            'patch_importance_scores': patch_importance_scores,
            'new_spatial_shape': (H_pruned_approx, W_pruned_approx),
            'original_spatial_shape': (H, W)
        }
        
        return pruned_tokens, kept_indices, info
    
    def compute_pruning_loss(self, info: dict) -> torch.Tensor:
        """计算剪枝损失（基于重要性分数的熵）"""
        if 'patch_importance_scores' not in info or info.get('patch_importance_scores') is None:
            return torch.tensor(0.0, requires_grad=False)
        
        patch_importance_scores = info['patch_importance_scores']
        patch_probs = F.softmax(patch_importance_scores, dim=-1)
        entropy = -(patch_probs * torch.log(patch_probs + 1e-8)).sum(dim=-1).mean()
        num_patches = patch_importance_scores.shape[-1]
        max_entropy = torch.log(torch.tensor(num_patches, dtype=torch.float32, device=patch_importance_scores.device))
        normalized_entropy = entropy / max_entropy
        return 1.0 - normalized_entropy

    def generate_soft_target_mask(
        self,
        gt_bboxes: List[torch.Tensor],
        feat_shape: Tuple[int, int],
        img_shape: Tuple[int, int],
        device: torch.device,
        expansion_ratio: Optional[float] = None,
        min_size: Optional[float] = None,
        decay_type: Optional[str] = None
    ) -> torch.Tensor:
        """
        Generates a Gaussian-weighted dilated mask for CASS supervision.
        
        Args:
            gt_bboxes: List of tensors [N, 4] in (x1, y1, x2, y2) format, original image scale.
                       Each tensor corresponds to one image in the batch.
            feat_shape: (h, w) of the S5 feature map.
            img_shape: (H, W) of the original image.
            device: Target device for the mask tensor.
            expansion_ratio: Override for self.cass_expansion_ratio
            min_size: Override for self.cass_min_size
            decay_type: Override for self.cass_decay_type ('gaussian' or 'linear')
        
        Returns:
            target_mask: Tensor [B, num_patches] with values 0.0 to 1.0.
                         Matches the shape of patch_importance_scores.
        """
        B = len(gt_bboxes)
        h, w = feat_shape
        ImgH, ImgW = img_shape
        
        expansion_ratio = expansion_ratio if expansion_ratio is not None else self.cass_expansion_ratio
        min_size = min_size if min_size is not None else self.cass_min_size
        decay_type = decay_type if decay_type is not None else self.cass_decay_type
        
        stride_h = ImgH / h
        stride_w = ImgW / w
        patch_h = min(self.patch_size, h)
        patch_w = min(self.patch_size, w)
        num_patches_h = (h + patch_h - 1) // patch_h
        num_patches_w = (w + patch_w - 1) // patch_w
        
        target_mask_2d = torch.zeros((B, num_patches_h, num_patches_w), device=device, dtype=torch.float32)
        patch_center_y = (torch.arange(num_patches_h, device=device, dtype=torch.float32) + 0.5) * patch_h
        patch_center_x = (torch.arange(num_patches_w, device=device, dtype=torch.float32) + 0.5) * patch_w
        yy, xx = torch.meshgrid(patch_center_y, patch_center_x, indexing='ij')
        yy = yy.unsqueeze(0)
        xx = xx.unsqueeze(0)
        
        for b_idx in range(B):
            bboxes = gt_bboxes[b_idx]
            if bboxes is None or len(bboxes) == 0:
                continue
            
            if bboxes.dim() == 1:
                bboxes = bboxes.unsqueeze(0)
            
            N = bboxes.shape[0]
            bboxes_feat = bboxes.clone().float().to(device)
            bboxes_feat[:, 0] = bboxes[:, 0] / stride_w
            bboxes_feat[:, 1] = bboxes[:, 1] / stride_h
            bboxes_feat[:, 2] = bboxes[:, 2] / stride_w
            bboxes_feat[:, 3] = bboxes[:, 3] / stride_h
            
            x1 = bboxes_feat[:, 0]
            y1 = bboxes_feat[:, 1]
            x2 = bboxes_feat[:, 2]
            y2 = bboxes_feat[:, 3]
            box_w = x2 - x1
            box_h = y2 - y1
            
            needs_expand_w = box_w < min_size
            needs_expand_h = box_h < min_size
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            x1 = torch.where(needs_expand_w, center_x - min_size / 2, x1)
            x2 = torch.where(needs_expand_w, center_x + min_size / 2, x2)
            y1 = torch.where(needs_expand_h, center_y - min_size / 2, y1)
            y2 = torch.where(needs_expand_h, center_y + min_size / 2, y2)
            box_w = x2 - x1
            box_h = y2 - y1
            
            expand_w = box_w * expansion_ratio
            expand_h = box_h * expansion_ratio
            x1_dilated = x1 - expand_w
            y1_dilated = y1 - expand_h
            x2_dilated = x2 + expand_w
            y2_dilated = y2 + expand_h
            
            x1_core = torch.clamp(x1, 0, w - 1e-6)
            y1_core = torch.clamp(y1, 0, h - 1e-6)
            x2_core = torch.clamp(x2, 1e-6, w)
            y2_core = torch.clamp(y2, 1e-6, h)
            x1_dilated = torch.clamp(x1_dilated, 0, w - 1e-6)
            y1_dilated = torch.clamp(y1_dilated, 0, h - 1e-6)
            x2_dilated = torch.clamp(x2_dilated, 1e-6, w)
            y2_dilated = torch.clamp(y2_dilated, 1e-6, h)
            
            x1_core = x1_core.view(N, 1, 1)
            y1_core = y1_core.view(N, 1, 1)
            x2_core = x2_core.view(N, 1, 1)
            y2_core = y2_core.view(N, 1, 1)
            x1_dilated = x1_dilated.view(N, 1, 1)
            y1_dilated = y1_dilated.view(N, 1, 1)
            x2_dilated = x2_dilated.view(N, 1, 1)
            y2_dilated = y2_dilated.view(N, 1, 1)
            expand_w = expand_w.view(N, 1, 1)
            expand_h = expand_h.view(N, 1, 1)
            
            # Reshape center and box dimensions for broadcasting
            center_x_view = center_x.view(N, 1, 1)
            center_y_view = center_y.view(N, 1, 1)
            box_w_view = box_w.view(N, 1, 1)
            box_h_view = box_h.view(N, 1, 1)
            
            in_core = (xx >= x1_core) & (xx <= x2_core) & (yy >= y1_core) & (yy <= y2_core)
            in_dilated = (xx >= x1_dilated) & (xx <= x2_dilated) & (yy >= y1_dilated) & (yy <= y2_dilated)
            in_context = in_dilated & ~in_core
            
            # Sub-pixel offset compensation
            if self.use_subpixel_offset:
                # Use true floating-point center to compute distance
                dist_x_abs = torch.abs(xx - center_x_view)
                dist_y_abs = torch.abs(yy - center_y_view)
                # Compute distance to boundary
                dist_x = torch.maximum(dist_x_abs - box_w_view / 2, torch.zeros_like(dist_x_abs))
                dist_y = torch.maximum(dist_y_abs - box_h_view / 2, torch.zeros_like(dist_y_abs))
            else:
                # Original implementation: use boundary distance
                dist_x = torch.where(xx < x1_core, x1_core - xx,
                            torch.where(xx > x2_core, xx - x2_core, torch.zeros_like(xx)))
                dist_y = torch.where(yy < y1_core, y1_core - yy,
                            torch.where(yy > y2_core, yy - y2_core, torch.zeros_like(yy)))
            
            # Keep circular symmetric Gaussian
            dist_to_core = torch.sqrt(dist_x**2 + dist_y**2)
            # Fix: Ensure expand_dist has a minimum value to prevent extremely small radius
            # for thin objects (e.g., traffic cones) which would cause zero decay values
            expand_dist = torch.sqrt(expand_w**2 + expand_h**2)
            expand_dist = torch.clamp(expand_dist, min=1.0)  # Ensure minimum radius of 1 pixel
            normalized_dist = dist_to_core / expand_dist
            
            if decay_type == 'gaussian':
                sigma = 0.5
                decay_values = torch.exp(-normalized_dist**2 / (2 * sigma**2))
                decay_values = torch.where(normalized_dist < 1.0, decay_values, torch.zeros_like(decay_values))
            else:
                decay_values = torch.clamp(1.0 - normalized_dist, 0.0, 1.0)
            
            box_masks = in_core.float()
            context_contribution = decay_values * in_context.float()
            box_masks = torch.maximum(box_masks, context_contribution)
            merged_mask, _ = torch.max(box_masks, dim=0)
            target_mask_2d[b_idx] = merged_mask
        
        target_mask = target_mask_2d.view(B, -1)
        
        return target_mask
    
    def _compute_focal_loss(
        self,
        pred_scores: torch.Tensor,
        target_mask: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Modified Focal Loss (reference: CornerNet/CenterNet)
        
        Formula:
        - Positive samples: (1 - p)^alpha * log(p) * y
        - Negative samples: p^beta * log(1 - p) * (1 - y)
        
        where p = sigmoid(pred_scores), y = target_mask
        
        Args:
            pred_scores: Predicted importance logits [B, num_patches] (before sigmoid)
            target_mask: Target soft mask [B, num_patches] with values 0.0 to 1.0
            reduction: Loss reduction method ('mean', 'sum', 'none')
        
        Returns:
            loss: Scalar loss tensor (if reduction != 'none')
        """
        pred_probs = torch.sigmoid(pred_scores)
        
        # 预计算常用中间值
        one_minus_p = 1 - pred_probs
        log_p = torch.log(pred_probs + 1e-8)
        log_one_minus_p = torch.log(one_minus_p + 1e-8)
        
        # 优化的幂运算：避免使用 torch.pow()
        # alpha=2.0: (1-p)^2 = (1-p) * (1-p)
        # beta=4.0: p^4 = (p*p) * (p*p)
        if self.cass_focal_alpha == 2.0:
            focal_weight_pos = one_minus_p * one_minus_p
        else:
            focal_weight_pos = torch.pow(one_minus_p, self.cass_focal_alpha)
        
        if self.cass_focal_beta == 4.0:
            p_sq = pred_probs * pred_probs
            focal_weight_neg = p_sq * p_sq
        elif self.cass_focal_beta == 2.0:
            focal_weight_neg = pred_probs * pred_probs
        else:
            focal_weight_neg = torch.pow(pred_probs, self.cass_focal_beta)
        
        # Positive sample loss: (1 - p)^alpha * log(p) * y
        pos_loss = -focal_weight_pos * log_p * target_mask
        
        # Negative sample loss: p^beta * log(1 - p) * (1 - y)
        neg_loss = -focal_weight_neg * log_one_minus_p * (1 - target_mask)
        
        loss = pos_loss + neg_loss
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def compute_cass_loss(
        self,
        pred_scores: torch.Tensor,
        target_mask: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Computes the CASS supervision loss using MSE or Focal Loss.
        
        Args:
            pred_scores: Predicted importance logits [B, num_patches] (before sigmoid)
            target_mask: Target soft mask [B, num_patches] with values 0.0 to 1.0
            reduction: Loss reduction method ('mean', 'sum', 'none')
        
        Returns:
            loss: Scalar loss tensor (if reduction != 'none')
        """
        if self.use_focal_loss:
            return self._compute_focal_loss(pred_scores, target_mask, reduction)
        else:
            # Original MSE implementation
            pred_probs = torch.sigmoid(pred_scores)
            loss = F.mse_loss(pred_probs, target_mask, reduction=reduction)
            return loss
    
    def _debug_visualize_mask(self, target_mask: torch.Tensor, pred_scores: torch.Tensor):
        """Debug helper: Visualize the target mask and predicted scores as heatmaps."""
        import os
        import time
        import numpy as np
        
        try:
            import cv2
        except ImportError:
            print("[CASS Debug] cv2 not available, skipping visualization")
            return
        
        debug_dir = "./debug_cass_vis"
        os.makedirs(debug_dir, exist_ok=True)
        
        target_np = target_mask[0].detach().cpu().numpy()
        pred_np = torch.sigmoid(pred_scores[0]).detach().cpu().numpy()
        num_patches = target_np.shape[0]
        
        side = int(np.sqrt(num_patches))
        if side * side == num_patches:
            grid_h, grid_w = side, side
        else:
            for h in range(int(np.sqrt(num_patches)), 0, -1):
                if num_patches % h == 0:
                    grid_h = h
                    grid_w = num_patches // h
                    break
            else:
                grid_h = 1
                grid_w = num_patches
        
        target_2d = target_np.reshape(grid_h, grid_w)
        pred_2d = pred_np.reshape(grid_h, grid_w)
        target_vis = (target_2d * 255).astype(np.uint8)
        pred_vis = (pred_2d * 255).astype(np.uint8)
        
        target_heatmap = cv2.applyColorMap(target_vis, cv2.COLORMAP_JET)
        pred_heatmap = cv2.applyColorMap(pred_vis, cv2.COLORMAP_JET)
        
        scale = 10
        target_heatmap = cv2.resize(target_heatmap, (grid_w * scale, grid_h * scale), 
                                     interpolation=cv2.INTER_NEAREST)
        pred_heatmap = cv2.resize(pred_heatmap, (grid_w * scale, grid_h * scale), 
                                   interpolation=cv2.INTER_NEAREST)
        
        combined = np.hstack([target_heatmap, pred_heatmap])
        cv2.putText(combined, "Target Mask (GT)", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Pred Scores (sigmoid)", (grid_w * scale + 10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        timestamp = int(time.time() * 1000) % 1000000
        filename = f"{debug_dir}/cass_mask_{timestamp}.png"
        cv2.imwrite(filename, combined)
        
        print(f"[CASS Debug] Saved: {filename}")
        print(f"  Grid shape: {grid_h}x{grid_w} = {num_patches} patches")
        print(f"  Target mask: min={target_np.min():.3f}, max={target_np.max():.3f}, "
              f"mean={target_np.mean():.3f}, nonzero={np.sum(target_np > 0)}")
        print(f"  Pred scores: min={pred_np.min():.3f}, max={pred_np.max():.3f}, "
              f"mean={pred_np.mean():.3f}")
    
    def compute_cass_loss_from_info(
        self,
        info: Dict,
        gt_bboxes: List[torch.Tensor],
        feat_shape: Tuple[int, int],
        img_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Convenience method to compute CASS loss directly from pruner info dict.
        
        Args:
            info: Info dict from forward() containing 'patch_importance_scores'
            gt_bboxes: List of ground truth boxes [N, 4] in (x1, y1, x2, y2) format
            feat_shape: (h, w) of the feature map
            img_shape: (H, W) of the original image
        
        Returns:
            loss: CASS supervision loss
        """
        if 'patch_importance_scores' not in info or info['patch_importance_scores'] is None:
            return torch.tensor(0.0, requires_grad=False)
        
        pred_scores = info['patch_importance_scores']
        device = pred_scores.device
        target_mask = self.generate_soft_target_mask(
            gt_bboxes=gt_bboxes,
            feat_shape=feat_shape,
            img_shape=img_shape,
            device=device
        )
        loss = self.compute_cass_loss(pred_scores, target_mask)
        
        return loss

