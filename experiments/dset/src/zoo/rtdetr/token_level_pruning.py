"""Token-level Pruning Module for DSET"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict


class LinearImportancePredictor(nn.Module):
    """
    Token重要性预测器（极简Linear版本）
    输出维度严格对应特征图的 Token 序列 [B, H*W]
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            tokens: [B, N, C] 输入的 Token 序列，其中 N = H*W
            H, W: 特征图的原始高度和宽度（用于验证，不参与计算）
        
        Returns:
            scores: [B, N] Token重要性分数（logits，不应用Sigmoid）
                   严格对应特征图的每个格子（Token）
        """
        # tokens: [B, N, C] where N = H*W
        x = self.fc1(tokens)  # [B, N, hidden_dim]
        x = self.activation(x)
        x = self.dropout(x)
        scores = self.fc2(x).squeeze(-1)  # [B, N] = [B, H*W]
        return scores


class TokenLevelPruner(nn.Module):
    """Token级别剪枝器，支持CASS监督学习（每个token独立处理）"""
    
    def __init__(self, 
                 input_dim: int,
                 keep_ratio: float = 0.7,
                 adaptive: bool = True,
                 min_tokens: int = 10,
                 warmup_epochs: int = 10,
                 prune_in_eval: bool = True,
                 # CASS parameters
                 use_cass: bool = False,
                 cass_expansion_ratio: float = 0.3,
                 cass_min_size: float = 1.0,
                 cass_decay_type: str = 'gaussian',
                 use_subpixel_offset: bool = True,
                 cass_loss_type: str = 'vfl',  # 'focal' or 'vfl'
                 cass_focal_alpha: float = 0.75,
                 cass_focal_beta: float = 2.0):
        """
        Args:
            input_dim: Input feature dimension
            keep_ratio: Retention ratio (0.5-0.7)
            adaptive: Whether to use adaptive pruning
            min_tokens: Minimum tokens to keep
            warmup_epochs: Warmup epochs
            prune_in_eval: Whether to prune during evaluation
            use_cass: Whether to use Context-Aware Soft Supervision
            cass_expansion_ratio: Expansion ratio for context band (0.2-0.8)
            cass_min_size: Minimum box size on feature map (pixels)
            cass_decay_type: Decay type for context band ('gaussian' or 'linear')
            use_subpixel_offset: Whether to use sub-pixel offset compensation
            cass_loss_type: Loss type ('focal' for Focal Loss, 'vfl' for Varifocal Loss)
            cass_focal_alpha: Focal/VFL alpha parameter (positive sample weight)
            cass_focal_beta: Focal/VFL beta/gamma parameter (hard example mining strength)
        """
        super().__init__()
        self.input_dim = input_dim
        self.keep_ratio = keep_ratio
        self.adaptive = adaptive
        self.min_tokens = min_tokens
        self.warmup_epochs = warmup_epochs
        self.prune_in_eval = prune_in_eval
        
        # CASS parameters
        self.use_cass = use_cass
        self.cass_expansion_ratio = cass_expansion_ratio
        self.cass_min_size = cass_min_size
        self.cass_decay_type = cass_decay_type
        self.use_subpixel_offset = use_subpixel_offset
        self.cass_loss_type = cass_loss_type
        self.cass_focal_alpha = cass_focal_alpha
        self.cass_focal_beta = cass_focal_beta
        
        # 只使用 Linear 预测器
        self.importance_predictor = LinearImportancePredictor(input_dim)
        
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
        
        # --- 核心修复：即使在 Warmup 期间，训练时也计算分数 ---
        # 这样 CASS Loss 才能在 Pruning 真正开始前就训练 Predictor
        token_importance_scores = self.importance_predictor(tokens, H, W)  # [B, N]

        # 如果处于 Pruning 的 Warmup 阶段
        if self.training and self.current_epoch < self.warmup_epochs:
            info = {
                'pruning_ratio': 0.0,
                'num_kept_tokens': num_tokens,
                'num_pruned_tokens': 0,
                'num_tokens': num_tokens,
                'num_kept_tokens_info': num_tokens,
                'token_importance_scores': token_importance_scores, # 必须传出分数！
                'new_spatial_shape': (H, W),
                'original_spatial_shape': (H, W)
            }
            # 返回全部索引，不执行剪枝
            kept_indices = None if not return_indices else torch.arange(
                num_tokens, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
            return tokens, kept_indices, info
        
        # --- 原有的正常剪枝逻辑 (Warmup 结束后执行) ---
        should_prune = self.pruning_enabled and (self.training or self.prune_in_eval)
        should_compute_scores = self.training
        
        if not should_prune and not should_compute_scores:
            info = {
                'pruning_ratio': 0.0,
                'num_kept_tokens': num_tokens,
                'num_pruned_tokens': 0,
                'num_tokens': num_tokens,
                'num_kept_tokens_info': num_tokens,
                'token_importance_scores': token_importance_scores, 
                'new_spatial_shape': (H, W),
                'original_spatial_shape': (H, W)
            }
            kept_indices = None if not return_indices else torch.arange(
                num_tokens, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
            return tokens, kept_indices, info
        
        if num_tokens <= self.min_tokens:
            info = {
                'pruning_ratio': 0.0,
                'num_kept_tokens': num_tokens,
                'num_pruned_tokens': 0,
                'num_tokens': num_tokens,
                'num_kept_tokens_info': num_tokens,
                'token_importance_scores': token_importance_scores,
                'new_spatial_shape': (H, W),
                'original_spatial_shape': (H, W)
            }
            kept_indices = None if not return_indices else torch.arange(
                num_tokens, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
            return tokens, kept_indices, info
        
        current_keep_ratio = self.get_current_keep_ratio()
        num_keep_tokens_by_ratio = int(num_tokens * current_keep_ratio)
        num_keep_tokens = min(max(self.min_tokens, num_keep_tokens_by_ratio), num_tokens)
        
        if not should_prune:
            info = {
                'pruning_ratio': 0.0,
                'num_kept_tokens': num_tokens,
                'num_pruned_tokens': 0,
                'num_tokens': num_tokens,
                'num_kept_tokens_info': num_tokens,
                'token_importance_scores': token_importance_scores,
                'new_spatial_shape': (H, W),
                'original_spatial_shape': (H, W)
            }
            kept_indices = None if not return_indices else torch.arange(
                num_tokens, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
            return tokens, kept_indices, info
        
        # 选择top-k tokens
        _, top_token_indices = torch.topk(token_importance_scores, num_keep_tokens, dim=-1)
        top_token_indices_sorted, _ = torch.sort(top_token_indices, dim=-1)
        
        # 收集保留的tokens
        batch_indices = torch.arange(batch_size, device=tokens.device).unsqueeze(1).expand(-1, num_keep_tokens)
        pruned_tokens = tokens[batch_indices, top_token_indices_sorted]
        
        if return_indices:
            kept_indices = top_token_indices_sorted
        else:
            kept_indices = None
        
        # 计算新的空间形状（近似）
        H_pruned_approx = int((num_keep_tokens + W - 1) // W) if W > 0 else num_keep_tokens
        W_pruned_approx = W if H_pruned_approx * W >= num_keep_tokens else num_keep_tokens
        
        info = {
            'pruning_ratio': 1.0 - (num_keep_tokens / num_tokens),
            'num_kept_tokens': num_keep_tokens,
            'num_pruned_tokens': num_tokens - num_keep_tokens,
            'num_tokens': num_tokens,
            'num_kept_tokens_info': num_keep_tokens,
            'token_importance_scores': token_importance_scores,
            'new_spatial_shape': (H_pruned_approx, W_pruned_approx),
            'original_spatial_shape': (H, W)
        }
        
        return pruned_tokens, kept_indices, info
    
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
            feat_shape: (h, w) of the feature map.
            img_shape: (H, W) of the original image.
            device: Target device for the mask tensor.
            expansion_ratio: Override for self.cass_expansion_ratio
            min_size: Override for self.cass_min_size
            decay_type: Override for self.cass_decay_type ('gaussian' or 'linear')
        
        Returns:
            target_mask: Tensor [B, num_tokens] with values 0.0 to 1.0.
                         Matches the shape of token_importance_scores.
        """
        B = len(gt_bboxes)
        h, w = feat_shape
        ImgH, ImgW = img_shape
        
        expansion_ratio = expansion_ratio if expansion_ratio is not None else self.cass_expansion_ratio
        min_size = min_size if min_size is not None else self.cass_min_size
        decay_type = decay_type if decay_type is not None else self.cass_decay_type
        
        stride_h = ImgH / h
        stride_w = ImgW / w
        num_tokens = h * w
        
        target_mask_2d = torch.zeros((B, h, w), device=device, dtype=torch.float32)
        
        # 修正：加入 +0.5 偏移，确保与 patch_size=1 时的中心点逻辑完全一致
        # 原版逻辑：patch_center_y = (torch.arange(num_patches_h) + 0.5) * patch_h
        # 当 patch_h = 1 时，坐标序列是 [0.5, 1.5, 2.5, ...]
        # 这 0.5 个像素的特征图偏移，对应到原图上就是 stride * 0.5 像素的物理位移
        # 对于小目标（如交通锥，约 10-15 像素）来说，这个偏移很重要
        token_y = (torch.arange(h, device=device, dtype=torch.float32) + 0.5)
        token_x = (torch.arange(w, device=device, dtype=torch.float32) + 0.5)
        
        # 使用 indexing='ij' 保持与原版网格一致
        yy, xx = torch.meshgrid(token_y, token_x, indexing='ij')
        yy = yy.unsqueeze(0)  # [1, h, w]
        xx = xx.unsqueeze(0)  # [1, h, w]
        
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
            # Add epsilon to denominator for numerical stability
            epsilon = 1e-6
            normalized_dist = dist_to_core / (expand_dist + epsilon)
            
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
    
    def generate_all_object_masks(
        self,
        gt_bboxes: List[torch.Tensor],
        feat_shape: Tuple[int, int],
        img_shape: Tuple[int, int],
        device: torch.device,
        expansion_ratio: Optional[float] = None,
        min_size: Optional[float] = None,
        decay_type: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized version: Generate individual masks for all objects in batch.
        
        This method generates masks for each object separately, avoiding repeated
        meshgrid computations. It's optimized for performance when computing
        object-level normalized CASS loss.
        
        Args:
            gt_bboxes: List of tensors [N, 4] in (x1, y1, x2, y2) format, original image scale.
                       Each tensor corresponds to one image in the batch.
            feat_shape: (h, w) of the feature map.
            img_shape: (H, W) of the original image.
            device: Target device for the mask tensor.
            expansion_ratio: Override for self.cass_expansion_ratio
            min_size: Override for self.cass_min_size
            decay_type: Override for self.cass_decay_type ('gaussian' or 'linear')
        
        Returns:
            all_obj_masks: Tensor [Total_Objects, num_tokens] with values 0.0 to 1.0.
                          Each row corresponds to one object's mask.
            batch_indices: Tensor [Total_Objects] indicating which batch index each object belongs to.
        """
        B = len(gt_bboxes)
        h, w = feat_shape
        ImgH, ImgW = img_shape
        
        expansion_ratio = expansion_ratio if expansion_ratio is not None else self.cass_expansion_ratio
        min_size = min_size if min_size is not None else self.cass_min_size
        decay_type = decay_type if decay_type is not None else self.cass_decay_type
        
        stride_h = ImgH / h
        stride_w = ImgW / w
        num_tokens = h * w
        
        # Collect all objects and their batch indices
        all_bboxes_list = []
        batch_indices_list = []
        
        for b_idx in range(B):
            bboxes = gt_bboxes[b_idx]
            if bboxes is None or len(bboxes) == 0:
                continue
            
            if bboxes.dim() == 1:
                bboxes = bboxes.unsqueeze(0)
            
            for i in range(len(bboxes)):
                all_bboxes_list.append(bboxes[i:i+1])  # Keep as [1, 4] for compatibility
                batch_indices_list.append(b_idx)
        
        if len(all_bboxes_list) == 0:
            # Return empty tensors with correct shape
            return torch.zeros((0, num_tokens), device=device, dtype=torch.float32), \
                   torch.zeros((0,), device=device, dtype=torch.long)
        
        Total_Objects = len(all_bboxes_list)
        batch_indices = torch.tensor(batch_indices_list, device=device, dtype=torch.long)
        
        # Generate meshgrid once (shared across all objects)
        token_y = (torch.arange(h, device=device, dtype=torch.float32) + 0.5)
        token_x = (torch.arange(w, device=device, dtype=torch.float32) + 0.5)
        yy, xx = torch.meshgrid(token_y, token_x, indexing='ij')
        yy = yy.unsqueeze(0)  # [1, h, w]
        xx = xx.unsqueeze(0)  # [1, h, w]
        
        # Stack all bboxes into a single tensor [Total_Objects, 4]
        all_bboxes = torch.cat(all_bboxes_list, dim=0).float().to(device)  # [Total_Objects, 4]
        
        # Convert to feature map coordinates
        bboxes_feat = all_bboxes.clone()
        bboxes_feat[:, 0] = bboxes_feat[:, 0] / stride_w
        bboxes_feat[:, 1] = bboxes_feat[:, 1] / stride_h
        bboxes_feat[:, 2] = bboxes_feat[:, 2] / stride_w
        bboxes_feat[:, 3] = bboxes_feat[:, 3] / stride_h
        
        x1 = bboxes_feat[:, 0]  # [Total_Objects]
        y1 = bboxes_feat[:, 1]
        x2 = bboxes_feat[:, 2]
        y2 = bboxes_feat[:, 3]
        box_w = x2 - x1
        box_h = y2 - y1
        
        # Apply min_size expansion
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
        
        # Compute dilated boxes
        expand_w = box_w * expansion_ratio
        expand_h = box_h * expansion_ratio
        x1_dilated = x1 - expand_w
        y1_dilated = y1 - expand_h
        x2_dilated = x2 + expand_w
        y2_dilated = y2 + expand_h
        
        # Clamp coordinates
        x1_core = torch.clamp(x1, 0, w - 1e-6)
        y1_core = torch.clamp(y1, 0, h - 1e-6)
        x2_core = torch.clamp(x2, 1e-6, w)
        y2_core = torch.clamp(y2, 1e-6, h)
        x1_dilated = torch.clamp(x1_dilated, 0, w - 1e-6)
        y1_dilated = torch.clamp(y1_dilated, 0, h - 1e-6)
        x2_dilated = torch.clamp(x2_dilated, 1e-6, w)
        y2_dilated = torch.clamp(y2_dilated, 1e-6, h)
        
        # Reshape for broadcasting: [Total_Objects, 1, 1]
        x1_core = x1_core.view(Total_Objects, 1, 1)
        y1_core = y1_core.view(Total_Objects, 1, 1)
        x2_core = x2_core.view(Total_Objects, 1, 1)
        y2_core = y2_core.view(Total_Objects, 1, 1)
        x1_dilated = x1_dilated.view(Total_Objects, 1, 1)
        y1_dilated = y1_dilated.view(Total_Objects, 1, 1)
        x2_dilated = x2_dilated.view(Total_Objects, 1, 1)
        y2_dilated = y2_dilated.view(Total_Objects, 1, 1)
        expand_w = expand_w.view(Total_Objects, 1, 1)
        expand_h = expand_h.view(Total_Objects, 1, 1)
        center_x_view = center_x.view(Total_Objects, 1, 1)
        center_y_view = center_y.view(Total_Objects, 1, 1)
        box_w_view = box_w.view(Total_Objects, 1, 1)
        box_h_view = box_h.view(Total_Objects, 1, 1)
        
        # Vectorized mask computation: [Total_Objects, h, w]
        in_core = (xx >= x1_core) & (xx <= x2_core) & (yy >= y1_core) & (yy <= y2_core)
        in_dilated = (xx >= x1_dilated) & (xx <= x2_dilated) & (yy >= y1_dilated) & (yy <= y2_dilated)
        in_context = in_dilated & ~in_core
        
        # Sub-pixel offset compensation
        if self.use_subpixel_offset:
            dist_x_abs = torch.abs(xx - center_x_view)
            dist_y_abs = torch.abs(yy - center_y_view)
            dist_x = torch.maximum(dist_x_abs - box_w_view / 2, torch.zeros_like(dist_x_abs))
            dist_y = torch.maximum(dist_y_abs - box_h_view / 2, torch.zeros_like(dist_y_abs))
        else:
            dist_x = torch.where(xx < x1_core, x1_core - xx,
                        torch.where(xx > x2_core, xx - x2_core, torch.zeros_like(xx)))
            dist_y = torch.where(yy < y1_core, y1_core - yy,
                        torch.where(yy > y2_core, yy - y2_core, torch.zeros_like(yy)))
        
        # Compute decay values
        dist_to_core = torch.sqrt(dist_x**2 + dist_y**2)
        expand_dist = torch.sqrt(expand_w**2 + expand_h**2)
        expand_dist = torch.clamp(expand_dist, min=1.0)
        # Add epsilon to denominator for numerical stability
        epsilon = 1e-6
        normalized_dist = dist_to_core / (expand_dist + epsilon)
        
        if decay_type == 'gaussian':
            sigma = 0.5
            decay_values = torch.exp(-normalized_dist**2 / (2 * sigma**2))
            decay_values = torch.where(normalized_dist < 1.0, decay_values, torch.zeros_like(decay_values))
        else:
            decay_values = torch.clamp(1.0 - normalized_dist, 0.0, 1.0)
        
        # Combine core and context masks
        box_masks = in_core.float()
        context_contribution = decay_values * in_context.float()
        box_masks = torch.maximum(box_masks, context_contribution)
        
        # Reshape to [Total_Objects, num_tokens]
        all_obj_masks = box_masks.view(Total_Objects, -1)
        
        return all_obj_masks, batch_indices
    
    def _compute_focal_loss(
        self,
        pred_scores: torch.Tensor,
        target_mask: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Token-level Focal Loss for CASS supervision.
        
        Formula:
        - For positive samples (y > 0): Loss = -α * (1-p)^γ * log(p)
        - For negative samples (y = 0): Loss = -α * p^γ * log(1-p)
        
        where p = sigmoid(pred_scores), y = target_mask, α = cass_focal_alpha, γ = cass_focal_beta
        
        Args:
            pred_scores: Predicted importance logits [B, num_tokens] (before sigmoid)
            target_mask: Target soft mask [B, num_tokens] with values 0.0 to 1.0
            reduction: Loss reduction method ('mean', 'sum', 'none')
        
        Returns:
            loss: Token-level loss tensor [B, num_tokens] (if reduction='none') or scalar
        """
        # Numerical safety: Clamp logits to prevent extreme gradients in FP16 training
        # Tightened clamp range: [-8, 8] ensures p won't be too small, preventing log(p) explosion
        pred_scores = torch.clamp(pred_scores, min=-8.0, max=8.0)
        
        # Use PyTorch's built-in binary_cross_entropy_with_logits for numerical stability
        # It internally optimizes log(sigmoid) computation, extremely stable
        bce_loss = F.binary_cross_entropy_with_logits(pred_scores, target_mask, reduction='none')
        
        # Compute probability p = sigmoid(pred_scores)
        p = torch.sigmoid(pred_scores)
        
        # Compute target probability: p_t = p * y + (1-p) * (1-y)
        p_t = p * target_mask + (1.0 - p) * (1.0 - target_mask)
        
        # Compute focal weight: α * (1 - p_t)^γ
        alpha = self.cass_focal_alpha
        gamma = self.cass_focal_beta
        
        # Optimize power computation for common values
        if gamma == 2.0:
            focal_weight = alpha * (1.0 - p_t) * (1.0 - p_t)
        elif gamma == 1.0:
            focal_weight = alpha * (1.0 - p_t)
        else:
            focal_weight = alpha * torch.pow(1.0 - p_t, gamma)
        
        # Focal Loss: focal_weight * bce_loss
        loss = focal_weight * bce_loss
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def _compute_vfl(
        self,
        pred_scores: torch.Tensor,
        target_mask: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Token-level Varifocal Loss (VFL) for CASS supervision.
        
        Formula:
        - For positive samples (y > 0): Loss = -y * [y * log(p) + (1-y) * log(1-p)]
        - For negative samples (y = 0): Loss = -α * p^γ * log(1-p)
        
        where p = sigmoid(pred_scores), y = target_mask, α = cass_focal_alpha, γ = cass_focal_beta
        
        Args:
            pred_scores: Predicted importance logits [B, num_tokens] (before sigmoid)
            target_mask: Target soft mask [B, num_tokens] with values 0.0 to 1.0
            reduction: Loss reduction method ('mean', 'sum', 'none')
        
        Returns:
            loss: Token-level loss tensor [B, num_tokens] (if reduction='none') or scalar
        """
        # Numerical safety: Clamp logits to prevent extreme gradients in FP16 training
        # Tightened clamp range: [-8, 8] ensures p won't be too small, preventing log(p) explosion
        pred_scores = torch.clamp(pred_scores, min=-8.0, max=8.0)
        pred_probs = torch.sigmoid(pred_scores)
        
        # Add epsilon for numerical stability in log computation
        epsilon = 1e-8
        log_p = torch.log(pred_probs + epsilon)
        log_one_minus_p = torch.log(1.0 - pred_probs + epsilon)
        
        # Identify positive and negative samples
        is_positive = target_mask > 0
        
        # For positive samples: Loss = -y * [y * log(p) + (1-y) * log(1-p)]
        pos_loss = -target_mask * (target_mask * log_p + (1.0 - target_mask) * log_one_minus_p)
        
        # For negative samples: Loss = -α * p^γ * log(1-p)
        alpha = self.cass_focal_alpha
        gamma = self.cass_focal_beta
        
        # Optimize power computation for common values
        if gamma == 2.0:
            p_gamma = pred_probs * pred_probs
        elif gamma == 1.0:
            p_gamma = pred_probs
        else:
            p_gamma = torch.pow(pred_probs, gamma)
        
        neg_loss = -alpha * p_gamma * log_one_minus_p
        
        # Combine losses: use positive loss where y > 0, negative loss where y = 0
        loss = torch.where(is_positive, pos_loss, neg_loss)
        
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
        Computes Token-level CASS supervision loss (Focal Loss or Varifocal Loss).
        
        Args:
            pred_scores: Predicted importance logits [B, num_tokens] (before sigmoid)
            target_mask: Target soft mask [B, num_tokens] with values 0.0 to 1.0
            reduction: Loss reduction method ('mean', 'sum', 'none')
        
        Returns:
            loss: Token-level loss tensor [B, num_tokens] (if reduction='none') or scalar
        """
        # Numerical safety: Clamp logits to prevent extreme gradients in FP16 training
        # Tightened clamp range: [-8, 8] ensures p won't be too small, preventing log(p) explosion
        pred_scores = torch.clamp(pred_scores, min=-8.0, max=8.0)
        
        if self.cass_loss_type == 'focal':
            loss = self._compute_focal_loss(pred_scores, target_mask, reduction)
        elif self.cass_loss_type == 'vfl':
            loss = self._compute_vfl(pred_scores, target_mask, reduction)
        else:
            raise ValueError(f"Unsupported cass_loss_type: {self.cass_loss_type}. Must be 'focal' or 'vfl'")
        
        # Final safety check: replace NaN with zeros
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        
        return loss
    
    def compute_cass_loss_from_info(
        self,
        info: Dict,
        gt_bboxes: List[torch.Tensor],
        feat_shape: Tuple[int, int],
        img_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Compute Token-level dense CASS supervision loss.
        
        Token-level Dense Supervision Strategy:
        - Directly compute loss for each token (grid cell) in the feature map
        - No object-level aggregation or pooling
        - Element-wise loss: pred_scores [B, H*W] vs target_mask [B, H*W]
        
        Args:
            info: Info dict from forward() containing 'token_importance_scores'
            gt_bboxes: List of ground truth boxes [N, 4] in (x1, y1, x2, y2) format
            feat_shape: (h, w) of the feature map
            img_shape: (H, W) of the original image
        
        Returns:
            loss: Token-level dense CASS supervision loss (scalar)
        """
        if 'token_importance_scores' not in info or info['token_importance_scores'] is None:
            return torch.tensor(0.0, device=info.get('device', torch.device('cpu')), requires_grad=False)
        
        pred_scores = info['token_importance_scores']  # [B, H*W]
        device = pred_scores.device
        
        # Generate dense target mask: [B, H*W]
        # This mask aggregates all objects in each image, creating a soft supervision signal
        target_mask = self.generate_soft_target_mask(
            gt_bboxes=gt_bboxes,
            feat_shape=feat_shape,
            img_shape=img_shape,
            device=device
        )  # [B, num_tokens] = [B, H*W]
        
        # Token-level dense loss: Element-wise computation
        # pred_scores: [B, H*W], target_mask: [B, H*W]
        # Each token gets its own loss value, no aggregation
        loss = self.compute_cass_loss(
            pred_scores,
            target_mask,
            reduction='mean'  # Average over all tokens in batch
        )
        
        return loss
