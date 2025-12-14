"""Patch-level Pruning Module for DSET - 与Patch-MoE兼容的patch级别剪枝"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
from .token_pruning import LearnableImportancePredictor


class PatchLevelPruner(nn.Module):
    """Patch级别剪枝器，与Patch-MoE兼容，保持规则2D结构
    
    支持Context-Aware Soft Supervision (CASS) 机制进行显式监督学习。
    """
    
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
                 cass_decay_type: str = 'gaussian'):
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
        
        # ✅ 终极优化：直接输出1D序列，跳过2D重建（避免F.fold等慢操作）
        # patches_flat: [B, num_patches, patch_h*patch_w, C]
        # 使用 gather 在 dim=1 (num_patches维度) 上选择patches
        top_patch_indices_expanded = top_patch_indices_sorted.unsqueeze(-1).unsqueeze(-1)  # [B, num_keep_patches, 1, 1]
        top_patch_indices_expanded = top_patch_indices_expanded.expand(-1, -1, patch_h * patch_w, channels)  # [B, num_keep_patches, patch_h*patch_w, C]
        kept_patches_flat = patches_flat.gather(dim=1, index=top_patch_indices_expanded)  # [B, num_keep_patches, patch_h*patch_w, C]
        
        # 直接reshape为1D序列，无需重建2D图像
        pruned_tokens = kept_patches_flat.reshape(batch_size, num_keep_patches * patch_h * patch_w, channels)  # [B, num_keep_patches*patch_h*patch_w, C]
        
        # 计算实际保留的token数量（考虑padding）
        num_kept_tokens_actual = num_keep_patches * patch_h * patch_w
        
        if return_indices:
            # ✅ 完全向量化索引生成：消除所有Python循环
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
            
            # ✅ 优化索引生成：最小化循环（只循环batch维度，循环内纯向量化操作）
            # 展平
            orig_idx_flat = orig_idx_all.reshape(batch_size, -1)  # [B, num_keep_patches*patch_h*patch_w]
            valid_mask_flat = valid_mask.reshape(batch_size, -1)  # [B, num_keep_patches*patch_h*patch_w]
            
            # 计算每个batch的有效token数（向量化）
            valid_counts = valid_mask_flat.sum(dim=1)  # [B]
            max_valid_count = int(valid_counts.max().item()) if valid_counts.numel() > 0 and valid_counts.max() > 0 else 0
            
            if max_valid_count > 0:
                # 创建输出tensor，用-1填充无效位置
                kept_indices = torch.full((batch_size, max_valid_count), -1, device=tokens.device, dtype=torch.long)
                
                # ✅ 最小化循环：只循环batch维度（不可避免，因为每个batch的有效token数不同）
                # 但循环内是纯向量化操作，无.item()调用，无Python列表操作
                for b in range(batch_size):
                    batch_valid_mask = valid_mask_flat[b]  # [num_keep_patches*patch_h*patch_w]
                    if batch_valid_mask.any():  # 向量化判断
                        batch_indices_flat = orig_idx_flat[b]  # [num_keep_patches*patch_h*patch_w]
                        valid_indices = batch_indices_flat[batch_valid_mask]  # [num_valid] - 纯向量化masked_select
                        num_valid = valid_indices.shape[0]  # 无.item()，直接取shape[0]
                        if num_valid > 0:
                            kept_indices[b, :num_valid] = valid_indices  # 向量化赋值
            else:
                kept_indices = None
        else:
            kept_indices = None
        
        num_kept_tokens = num_kept_tokens_actual
        # 计算近似的2D形状（用于兼容性，实际输出是1D序列）
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
            'new_spatial_shape': (H_pruned_approx, W_pruned_approx),  # 近似2D形状（实际输出是1D序列）
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

    # ==================== CASS (Context-Aware Soft Supervision) ====================
    
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
        
        # Use instance params or overrides
        expansion_ratio = expansion_ratio if expansion_ratio is not None else self.cass_expansion_ratio
        min_size = min_size if min_size is not None else self.cass_min_size
        decay_type = decay_type if decay_type is not None else self.cass_decay_type
        
        # Calculate stride for coordinate mapping
        stride_h = ImgH / h
        stride_w = ImgW / w
        
        # Calculate patch grid dimensions
        patch_h = min(self.patch_size, h)
        patch_w = min(self.patch_size, w)
        num_patches_h = (h + patch_h - 1) // patch_h
        num_patches_w = (w + patch_w - 1) // patch_w
        num_patches = num_patches_h * num_patches_w
        
        # Initialize mask at patch level [B, num_patches_h, num_patches_w]
        target_mask_2d = torch.zeros((B, num_patches_h, num_patches_w), device=device, dtype=torch.float32)
        
        # Create coordinate grid for patch centers (vectorized)
        # Patch center coordinates in feature map space
        patch_center_y = (torch.arange(num_patches_h, device=device, dtype=torch.float32) + 0.5) * patch_h
        patch_center_x = (torch.arange(num_patches_w, device=device, dtype=torch.float32) + 0.5) * patch_w
        
        # Create meshgrid for patch centers [num_patches_h, num_patches_w]
        yy, xx = torch.meshgrid(patch_center_y, patch_center_x, indexing='ij')
        
        for b_idx in range(B):
            bboxes = gt_bboxes[b_idx]
            if bboxes is None or len(bboxes) == 0:
                continue
            
            # Ensure bboxes is 2D [N, 4]
            if bboxes.dim() == 1:
                bboxes = bboxes.unsqueeze(0)
            
            # Map boxes to feature map scale
            # bboxes: [N, 4] in (x1, y1, x2, y2) format
            bboxes_feat = bboxes.clone().float().to(device)
            bboxes_feat[:, 0] = bboxes[:, 0] / stride_w  # x1
            bboxes_feat[:, 1] = bboxes[:, 1] / stride_h  # y1
            bboxes_feat[:, 2] = bboxes[:, 2] / stride_w  # x2
            bboxes_feat[:, 3] = bboxes[:, 3] / stride_h  # y2
            
            # Process each box
            for box_idx in range(bboxes_feat.shape[0]):
                x1, y1, x2, y2 = bboxes_feat[box_idx]
                
                # Calculate box dimensions and enforce minimum size
                box_w = x2 - x1
                box_h = y2 - y1
                
                # Enforce minimum size (CRITICAL for small objects on S5)
                if box_w < min_size:
                    center_x = (x1 + x2) / 2
                    x1 = center_x - min_size / 2
                    x2 = center_x + min_size / 2
                    box_w = min_size
                
                if box_h < min_size:
                    center_y = (y1 + y2) / 2
                    y1 = center_y - min_size / 2
                    y2 = center_y + min_size / 2
                    box_h = min_size
                
                # Calculate dilated box (context region)
                expand_w = box_w * expansion_ratio
                expand_h = box_h * expansion_ratio
                
                x1_dilated = x1 - expand_w
                y1_dilated = y1 - expand_h
                x2_dilated = x2 + expand_w
                y2_dilated = y2 + expand_h
                
                # Clamp to feature map bounds
                x1_core = torch.clamp(x1, 0, w - 1e-6)
                y1_core = torch.clamp(y1, 0, h - 1e-6)
                x2_core = torch.clamp(x2, 1e-6, w)
                y2_core = torch.clamp(y2, 1e-6, h)
                
                x1_dilated = torch.clamp(x1_dilated, 0, w - 1e-6)
                y1_dilated = torch.clamp(y1_dilated, 0, h - 1e-6)
                x2_dilated = torch.clamp(x2_dilated, 1e-6, w)
                y2_dilated = torch.clamp(y2_dilated, 1e-6, h)
                
                # Calculate mask values for each patch
                # Core region mask: patches whose centers are inside the core box
                in_core = (xx >= x1_core) & (xx <= x2_core) & (yy >= y1_core) & (yy <= y2_core)
                
                # Dilated region mask: patches in dilated box but not in core
                in_dilated = (xx >= x1_dilated) & (xx <= x2_dilated) & (yy >= y1_dilated) & (yy <= y2_dilated)
                in_context = in_dilated & ~in_core
                
                # Calculate distance-based decay for context region
                if in_context.any():
                    # Distance from patch center to core box edge (normalized)
                    # For each patch, compute signed distance to core box
                    dist_left = x1_core - xx
                    dist_right = xx - x2_core
                    dist_top = y1_core - yy
                    dist_bottom = yy - y2_core
                    
                    # Distance to core box (positive = outside core)
                    dist_x = torch.maximum(dist_left, dist_right)
                    dist_y = torch.maximum(dist_top, dist_bottom)
                    dist_to_core = torch.sqrt(torch.clamp(dist_x, min=0)**2 + torch.clamp(dist_y, min=0)**2)
                    
                    # Normalize by expansion distance
                    # Note: expand_w/expand_h may be tensors or floats depending on min_size path
                    expand_w_val = expand_w.item() if isinstance(expand_w, torch.Tensor) else float(expand_w)
                    expand_h_val = expand_h.item() if isinstance(expand_h, torch.Tensor) else float(expand_h)
                    expand_dist = math.sqrt(expand_w_val**2 + expand_h_val**2) + 1e-6
                    normalized_dist = dist_to_core / expand_dist
                    
                    # Apply decay function
                    if decay_type == 'gaussian':
                        # Gaussian decay: exp(-d^2 / (2 * sigma^2)), sigma = 0.5
                        sigma = 0.5
                        decay_values = torch.exp(-normalized_dist**2 / (2 * sigma**2))
                    else:  # linear
                        # Linear decay: 1 - d (clamped to [0, 1])
                        decay_values = torch.clamp(1.0 - normalized_dist, 0.0, 1.0)
                    
                    # Apply context values
                    context_values = decay_values * in_context.float()
                    target_mask_2d[b_idx] = torch.maximum(target_mask_2d[b_idx], context_values)
                
                # Apply core region (value = 1.0)
                core_values = in_core.float()
                target_mask_2d[b_idx] = torch.maximum(target_mask_2d[b_idx], core_values)
        
        # Flatten to [B, num_patches] to match patch_importance_scores shape
        target_mask = target_mask_2d.view(B, -1)
        
        return target_mask
    
    def compute_cass_loss(
        self,
        pred_scores: torch.Tensor,
        target_mask: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Computes the CASS supervision loss using MSE.
        
        Args:
            pred_scores: Predicted importance logits [B, num_patches] (before sigmoid)
            target_mask: Target soft mask [B, num_patches] with values 0.0 to 1.0
            reduction: Loss reduction method ('mean', 'sum', 'none')
        
        Returns:
            loss: Scalar loss tensor (if reduction != 'none')
        """
        # ============ DEBUG: Visualize target_mask (TEMPORARY) ============
        # Uncomment the following block to enable debug visualization:
        # if self.training:
        #     import random
        #     if random.random() < 0.05:  # 5% probability
        #         self._debug_visualize_mask(target_mask, pred_scores)
        # ===================================================================
        
        # Apply sigmoid to convert logits to probabilities
        pred_probs = torch.sigmoid(pred_scores)
        
        # Compute MSE loss
        loss = F.mse_loss(pred_probs, target_mask, reduction=reduction)
        
        return loss
    
    def _debug_visualize_mask(self, target_mask: torch.Tensor, pred_scores: torch.Tensor):
        """
        Debug helper: Visualize the target mask and predicted scores as heatmaps.
        Saves to ./debug_cass_vis/ folder.
        """
        import os
        import time
        import numpy as np
        
        try:
            import cv2
        except ImportError:
            print("[CASS Debug] cv2 not available, skipping visualization")
            return
        
        # Create output directory
        debug_dir = "./debug_cass_vis"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Get first sample from batch
        target_np = target_mask[0].detach().cpu().numpy()  # [num_patches]
        pred_np = torch.sigmoid(pred_scores[0]).detach().cpu().numpy()  # [num_patches]
        
        # Reshape to 2D grid if possible
        num_patches = target_np.shape[0]
        
        # Try to find a reasonable 2D shape
        # Assume roughly square or use patch grid dimensions
        patch_h = min(self.patch_size, 23)  # Typical S5 height for 736 input
        patch_w = min(self.patch_size, 40)  # Typical S5 width for 1280 input
        
        # Calculate grid dimensions based on patch_size
        # For patch_size=1, num_patches = h * w directly
        side = int(np.sqrt(num_patches))
        if side * side == num_patches:
            grid_h, grid_w = side, side
        else:
            # Try common aspect ratios (16:9 ≈ 1.78, typical for driving data)
            # For S5 with 1280x736 input: 40x23 = 920 patches (if patch_size=1)
            # Try to find factors
            for h in range(int(np.sqrt(num_patches)), 0, -1):
                if num_patches % h == 0:
                    grid_h = h
                    grid_w = num_patches // h
                    break
            else:
                grid_h = 1
                grid_w = num_patches
        
        # Reshape to 2D
        target_2d = target_np.reshape(grid_h, grid_w)
        pred_2d = pred_np.reshape(grid_h, grid_w)
        
        # Scale to 0-255 for visualization
        target_vis = (target_2d * 255).astype(np.uint8)
        pred_vis = (pred_2d * 255).astype(np.uint8)
        
        # Apply colormap (JET: blue=0, red=1)
        target_heatmap = cv2.applyColorMap(target_vis, cv2.COLORMAP_JET)
        pred_heatmap = cv2.applyColorMap(pred_vis, cv2.COLORMAP_JET)
        
        # Resize for better visibility (scale up by 10x)
        scale = 10
        target_heatmap = cv2.resize(target_heatmap, (grid_w * scale, grid_h * scale), 
                                     interpolation=cv2.INTER_NEAREST)
        pred_heatmap = cv2.resize(pred_heatmap, (grid_w * scale, grid_h * scale), 
                                   interpolation=cv2.INTER_NEAREST)
        
        # Create side-by-side comparison
        combined = np.hstack([target_heatmap, pred_heatmap])
        
        # Add labels
        cv2.putText(combined, "Target Mask (GT)", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Pred Scores (sigmoid)", (grid_w * scale + 10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save with timestamp
        timestamp = int(time.time() * 1000) % 1000000
        filename = f"{debug_dir}/cass_mask_{timestamp}.png"
        cv2.imwrite(filename, combined)
        
        # Print debug info
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
        
        # Generate soft target mask
        target_mask = self.generate_soft_target_mask(
            gt_bboxes=gt_bboxes,
            feat_shape=feat_shape,
            img_shape=img_shape,
            device=device
        )
        
        # Compute CASS loss
        loss = self.compute_cass_loss(pred_scores, target_mask)
        
        return loss

