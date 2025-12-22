"""
CASS Loss 改进实现
包含以下优化：
1. 各向异性高斯分布 (Anisotropic Gaussian)
2. 自适应温度系数 (Adaptive Temperature)
3. 子像素偏移补偿 (Sub-pixel Offset)
4. Focal Loss 替代 MSE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict


class ImprovedPatchLevelPrunerCASS:
    """
    改进的 CASS Loss 实现
    可以直接替换 patch_level_pruning.py 中的对应方法
    """
    
    def __init__(self,
                 # 原有参数
                 cass_expansion_ratio: float = 0.3,
                 cass_min_size: float = 1.0,
                 cass_decay_type: str = 'gaussian',
                 patch_size: int = 4,
                 # 新增优化参数
                 use_anisotropic_gaussian: bool = True,
                 use_adaptive_temperature: bool = True,
                 use_subpixel_offset: bool = True,
                 use_focal_loss: bool = True,
                 # 自适应温度参数
                 base_sigma: float = 0.5,
                 small_target_gain: float = 1.5,
                 small_target_threshold: float = 4.0,
                 # Focal Loss 参数
                 focal_alpha: float = 2.0,
                 focal_beta: float = 4.0):
        """
        Args:
            cass_expansion_ratio: 上下文扩展比例
            cass_min_size: 最小box尺寸
            cass_decay_type: 衰减类型 ('gaussian' or 'linear')
            patch_size: Patch大小
            use_anisotropic_gaussian: 是否使用各向异性高斯
            use_adaptive_temperature: 是否使用自适应温度
            use_subpixel_offset: 是否使用子像素偏移补偿
            use_focal_loss: 是否使用Focal Loss
            base_sigma: 基础高斯sigma值
            small_target_gain: 小目标增益系数
            small_target_threshold: 小目标阈值（像素）
            focal_alpha: Focal Loss alpha参数
            focal_beta: Focal Loss beta参数
        """
        self.cass_expansion_ratio = cass_expansion_ratio
        self.cass_min_size = cass_min_size
        self.cass_decay_type = cass_decay_type
        self.patch_size = patch_size
        
        self.use_anisotropic_gaussian = use_anisotropic_gaussian
        self.use_adaptive_temperature = use_adaptive_temperature
        self.use_subpixel_offset = use_subpixel_offset
        self.use_focal_loss = use_focal_loss
        
        self.base_sigma = base_sigma
        self.small_target_gain = small_target_gain
        self.small_target_threshold = small_target_threshold
        
        self.focal_alpha = focal_alpha
        self.focal_beta = focal_beta
    
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
        改进的软目标掩码生成函数
        
        优化点：
        1. 各向异性高斯分布（针对长条状物体）
        2. 自适应温度系数（增强小目标信号）
        3. 子像素偏移补偿（提高定位精度）
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
        
        # 计算patch中心坐标
        patch_center_y = (torch.arange(num_patches_h, device=device, dtype=torch.float32) + 0.5) * patch_h
        patch_center_x = (torch.arange(num_patches_w, device=device, dtype=torch.float32) + 0.5) * patch_w
        yy, xx = torch.meshgrid(patch_center_y, patch_center_x, indexing='ij')
        yy = yy.unsqueeze(0)  # [1, H_p, W_p]
        xx = xx.unsqueeze(0)  # [1, H_p, W_p]
        
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
            
            # 确保最小尺寸
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
            
            # 计算扩展区域
            expand_w = box_w * expansion_ratio
            expand_h = box_h * expansion_ratio
            x1_dilated = x1 - expand_w
            y1_dilated = y1 - expand_h
            x2_dilated = x2 + expand_w
            y2_dilated = y2 + expand_h
            
            # 裁剪到特征图范围
            x1_core = torch.clamp(x1, 0, w - 1e-6)
            y1_core = torch.clamp(y1, 0, h - 1e-6)
            x2_core = torch.clamp(x2, 1e-6, w)
            y2_core = torch.clamp(y2, 1e-6, h)
            x1_dilated = torch.clamp(x1_dilated, 0, w - 1e-6)
            y1_dilated = torch.clamp(y1_dilated, 0, h - 1e-6)
            x2_dilated = torch.clamp(x2_dilated, 1e-6, w)
            y2_dilated = torch.clamp(y2_dilated, 1e-6, h)
            
            # 重塑为 [N, 1, 1] 用于广播
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
            box_w = box_w.view(N, 1, 1)
            box_h = box_h.view(N, 1, 1)
            center_x = center_x.view(N, 1, 1)
            center_y = center_y.view(N, 1, 1)
            
            # 计算核心区域和扩展区域
            in_core = (xx >= x1_core) & (xx <= x2_core) & (yy >= y1_core) & (yy <= y2_core)
            in_dilated = (xx >= x1_dilated) & (xx <= x2_dilated) & (yy >= y1_dilated) & (yy <= y2_dilated)
            in_context = in_dilated & ~in_core
            
            # ========== 优化1: 子像素偏移补偿 ==========
            if self.use_subpixel_offset:
                # 使用bbox真实中心（而非patch中心）计算距离
                dist_x = xx - center_x  # [N, 1, 1] -> [N, H_p, W_p]
                dist_y = yy - center_y
                dist_x_abs = torch.abs(dist_x)
                dist_y_abs = torch.abs(dist_y)
                
                # 对于core区域内的点，距离为0
                # 对于core区域外的点，计算到边界的距离
                dist_x_core = torch.maximum(dist_x_abs - box_w / 2, torch.zeros_like(dist_x_abs))
                dist_y_core = torch.maximum(dist_y_abs - box_h / 2, torch.zeros_like(dist_y_abs))
            else:
                # 原始实现：使用到边界的距离
                dist_x = torch.where(xx < x1_core, x1_core - xx,
                            torch.where(xx > x2_core, xx - x2_core, torch.zeros_like(xx)))
                dist_y = torch.where(yy < y1_core, y1_core - yy,
                            torch.where(yy > y2_core, yy - y2_core, torch.zeros_like(yy)))
                dist_x_core = dist_x
                dist_y_core = dist_y
            
            # ========== 优化2: 各向异性高斯分布 ==========
            if self.use_anisotropic_gaussian:
                # 分别归一化 x 和 y 方向的距离
                expand_w_clamped = torch.clamp(expand_w, min=1.0)
                expand_h_clamped = torch.clamp(expand_h, min=1.0)
                normalized_dist_x = dist_x_core / expand_w_clamped
                normalized_dist_y = dist_y_core / expand_h_clamped
                
                # 计算到核心区域边界的归一化距离（用于判断是否在context band内）
                # 对于context区域，需要确保 normalized_dist < 1.0
                max_normalized_dist = torch.maximum(normalized_dist_x, normalized_dist_y)
            else:
                # 原始实现：圆对称高斯
                dist_to_core = torch.sqrt(dist_x_core**2 + dist_y_core**2)
                expand_dist = torch.sqrt(expand_w**2 + expand_h**2)
                expand_dist = torch.clamp(expand_dist, min=1.0)
                max_normalized_dist = dist_to_core / expand_dist
            
            # ========== 优化3: 自适应温度系数 ==========
            if self.use_adaptive_temperature and decay_type == 'gaussian':
                # 计算目标大小
                target_size = torch.sqrt(box_w.squeeze() * box_h.squeeze())  # [N]
                is_small = target_size < self.small_target_threshold
                
                # 小目标使用更高的sigma（更慢的衰减）
                sigma = torch.where(
                    is_small,
                    self.base_sigma * self.small_target_gain,
                    self.base_sigma
                )
                sigma = sigma.view(N, 1, 1)  # [N, 1, 1]
            else:
                sigma = self.base_sigma
            
            # 计算衰减值
            if decay_type == 'gaussian':
                if self.use_anisotropic_gaussian:
                    # 各向异性高斯
                    decay_values = torch.exp(
                        -normalized_dist_x**2 / (2 * sigma**2) 
                        - normalized_dist_y**2 / (2 * sigma**2)
                    )
                else:
                    # 圆对称高斯
                    decay_values = torch.exp(-max_normalized_dist**2 / (2 * sigma**2))
                
                # 只保留在context band内的衰减值
                decay_values = torch.where(
                    max_normalized_dist < 1.0,
                    decay_values,
                    torch.zeros_like(decay_values)
                )
            else:
                # Linear decay
                decay_values = torch.clamp(1.0 - max_normalized_dist, 0.0, 1.0)
            
            # 合并所有box的mask（使用max策略）
            box_masks = in_core.float()
            context_contribution = decay_values * in_context.float()
            box_masks = torch.maximum(box_masks, context_contribution)
            merged_mask, _ = torch.max(box_masks, dim=0)  # Max over all boxes
            target_mask_2d[b_idx] = merged_mask
        
        target_mask = target_mask_2d.view(B, -1)
        return target_mask
    
    def compute_cass_loss(
        self,
        pred_scores: torch.Tensor,
        target_mask: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        改进的CASS Loss计算函数
        
        优化点：
        1. Focal Loss 替代 MSE（解决样本不平衡问题）
        """
        if self.use_focal_loss:
            return self._compute_focal_loss(pred_scores, target_mask, reduction)
        else:
            # 原始MSE实现
            pred_probs = torch.sigmoid(pred_scores)
            loss = F.mse_loss(pred_probs, target_mask, reduction=reduction)
            return loss
    
    def _compute_focal_loss(
        self,
        pred_scores: torch.Tensor,
        target_mask: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Modified Focal Loss (参考 CornerNet/CenterNet)
        
        公式：
        - 正样本: (1 - p)^alpha * log(p) * y
        - 负样本: p^beta * log(1 - p) * (1 - y)
        
        其中 p = sigmoid(pred_scores), y = target_mask
        """
        pred_probs = torch.sigmoid(pred_scores)
        
        # 正样本损失: (1 - p)^alpha * log(p) * y
        pos_loss = -torch.pow(1 - pred_probs, self.focal_alpha) * \
                   torch.log(pred_probs + 1e-8) * target_mask
        
        # 负样本损失: p^beta * log(1 - p) * (1 - y)
        neg_loss = -torch.pow(pred_probs, self.focal_beta) * \
                   torch.log(1 - pred_probs + 1e-8) * (1 - target_mask)
        
        loss = pos_loss + neg_loss
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ========== 使用示例 ==========

def example_usage():
    """展示如何使用改进的CASS Loss"""
    
    # 创建改进的CASS实例
    improved_cass = ImprovedPatchLevelPrunerCASS(
        cass_expansion_ratio=0.3,
        cass_min_size=1.0,
        cass_decay_type='gaussian',
        patch_size=4,
        # 启用所有优化
        use_anisotropic_gaussian=True,
        use_adaptive_temperature=True,
        use_subpixel_offset=True,
        use_focal_loss=True,
        # 超参数
        base_sigma=0.5,
        small_target_gain=1.5,
        small_target_threshold=4.0,
        focal_alpha=2.0,
        focal_beta=4.0
    )
    
    # 模拟输入
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    feat_shape = (20, 20)  # 特征图尺寸
    img_shape = (640, 640)  # 原始图像尺寸
    
    # 模拟GT bboxes (x1, y1, x2, y2)
    gt_bboxes = [
        torch.tensor([[100, 100, 120, 150], [200, 200, 250, 220]], device=device),  # 长条状 + 正方形
        torch.tensor([[300, 300, 310, 350]], device=device)  # 细长条
    ]
    
    # 生成目标掩码
    target_mask = improved_cass.generate_soft_target_mask(
        gt_bboxes=gt_bboxes,
        feat_shape=feat_shape,
        img_shape=img_shape,
        device=device
    )
    
    # 模拟预测分数
    num_patches = target_mask.shape[1]
    pred_scores = torch.randn(batch_size, num_patches, device=device)
    
    # 计算损失
    loss = improved_cass.compute_cass_loss(pred_scores, target_mask)
    
    print(f"Target mask shape: {target_mask.shape}")
    print(f"Pred scores shape: {pred_scores.shape}")
    print(f"CASS Loss: {loss.item():.4f}")


if __name__ == '__main__':
    example_usage()

