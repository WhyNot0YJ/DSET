#!/usr/bin/env python3
"""Visualize Token Pruning Sparsity for DSET Paper Teaser Figure

This script generates visualization heatmaps showing token pruning sparsity patterns
for DSET (Dual-Sparse Expert Transformer) compared to RT-DETR baseline.

Usage Examples:

1. Standard resolution (1280, default):
   python visualize_sparsity.py \
       --image /path/to/image.jpg \
       --config config.yaml \
       --checkpoint checkpoint.pth \
       --target_size 1280

2. Full-resolution input (1920x1080):
   python visualize_sparsity.py \
       --image /path/to/1920x1080_image.jpg \
       --config config.yaml \
       --checkpoint checkpoint.pth \
       --target_size 1920

Note: Input dimensions will be automatically padded to multiples of stride (32).
      For 1920x1080 input, height will be padded to 1088 (1080 -> 1088, +8 rows).
"""

import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

# Setup paths
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root.parent) not in sys.path:
    sys.path.insert(0, str(project_root.parent))

from batch_inference import load_model, preprocess_image


def visualize_token_pruning(model, image_path, device='cuda', output_dir=None, patch_size=1, stride=32, target_size=1280):
    """
    Visualize token pruning importance scores.
    
    Args:
        model: DSETRTDETR model instance
        image_path: Path to input image
        device: Device to run inference on
        output_dir: Directory to save visualizations
        patch_size: Patch size used in token pruning
        stride: Feature map stride (default: 32)
        target_size: Target size for longest edge (default: 1280). 
                     Use 1920 for full-resolution input (e.g., 1920x1080)
    """
    # 1. Preprocessing
    print(f"Loading image: {image_path}")
    try:
        orig_image_bgr = cv2.imread(str(image_path))
        if orig_image_bgr is None:
            raise ValueError(f"Failed to read image: {image_path}")
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")
    
    orig_h, orig_w = orig_image_bgr.shape[:2]
    print(f"Original image size: {orig_w} x {orig_h}")
    print(f"Target size (longest edge): {target_size}")
    
    # Preprocess
    img_tensor, _, meta = preprocess_image(str(image_path), target_size=target_size)
    img_tensor = img_tensor.to(device)
    
    # Get dimensions for alignment
    orig_h, orig_w = meta['orig_size'][0].tolist()  # 原始尺寸
    padded_h, padded_w = meta['padded_h'], meta['padded_w']  # Pad 后的输入尺寸
    scale = meta['scale']
    
    # Pad 之前的有效区域尺寸 (Valid Area)
    valid_h = int(round(orig_h * scale))
    valid_w = int(round(orig_w * scale))
    
    # Print preprocessing info
    print(f"Resized size (before padding): {valid_w} x {valid_h}")
    print(f"Padded size (after padding to {stride} multiple): {padded_w} x {padded_h}")
    print(f"Padding: +{padded_w - valid_w} width, +{padded_h - valid_h} height")
    print(f"Scale factor: {scale:.4f}")
    
    # 2. Model Inference
    print("Running model inference...")
    model.eval()
    
    # Enable pruning during inference by setting epoch >= warmup_epochs
    # This ensures pruning_enabled=True in PatchLevelPruner
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'set_epoch'):
        # Set epoch to a value >= warmup_epochs to enable pruning
        # Use a large epoch number to ensure full pruning (not gradual)
        model.encoder.set_epoch(100)  # Large enough to bypass warmup
        print("  ✓ Enabled token pruning for visualization")
    
    with torch.no_grad():
        outputs = model(img_tensor)
    
    # 3. Extract Scores
    if 'encoder_info' not in outputs:
        raise ValueError("Model output missing 'encoder_info'.")
    
    encoder_info = outputs.get('encoder_info', {})
    importance_scores_list = encoder_info.get('importance_scores_list', [])
    token_pruning_ratios = encoder_info.get('token_pruning_ratios', [])
    
    if len(importance_scores_list) == 0:
        raise ValueError("No importance scores found.")
    
    # Print pruning statistics
    if token_pruning_ratios:
        pruning_ratio = token_pruning_ratios[0]
        print(f"Token pruning ratio: {pruning_ratio:.2%} ({(1-pruning_ratio)*100:.1f}% tokens pruned)")
        if pruning_ratio == 0.0:
            print("  ⚠ WARNING: Pruning ratio is 0.0! Pruning may not be enabled.")
        else:
            print(f"  ✓ Pruning is active: {pruning_ratio*100:.1f}% tokens kept")
    
    importance_scores = importance_scores_list[0] # [B, num_patches]
    importance_scores = importance_scores.cpu().numpy()
    if importance_scores.ndim == 2:
        importance_scores = importance_scores[0]
        
    print(f"Importance scores shape: {importance_scores.shape}")
    
    # 4. Reshape Logic
    feat_h = padded_h // stride
    feat_w = padded_w // stride
    
    num_patches_h = (feat_h + patch_size - 1) // patch_size
    num_patches_w = (feat_w + patch_size - 1) // patch_size
    expected_num = num_patches_h * num_patches_w
    
    print(f"Feature map size: {feat_w} x {feat_h} (stride={stride})")
    print(f"Patch grid size: {num_patches_w} x {num_patches_h} (patch_size={patch_size})")
    print(f"Total patches: {expected_num}")
    
    # Pad/Crop scores if mismatch
    if len(importance_scores) != expected_num:
        if len(importance_scores) < expected_num:
            importance_scores = np.pad(importance_scores, (0, expected_num - len(importance_scores)), 
                                     mode='constant', constant_values=importance_scores.min())
        else:
            importance_scores = importance_scores[:expected_num]
            
    scores_2d = importance_scores.reshape(num_patches_h, num_patches_w)
    
    # 5. Generate Visualizations (Applying Alignment Fix)
    
    # (A) DSET
    print("Generating DSET visualization...")
    dset_vis = create_heatmap_visualization(
        scores_2d, orig_image_bgr, 
        valid_h, valid_w,      # 有效区域 (Resize后, Pad前)
        padded_h, padded_w,    # Pad后的总尺寸 (用于计算比例)
        title="DSET (Ours)"
    )
    
    # (B) RT-DETR (Baseline) - Dense
    print("Generating RT-DETR baseline visualization...")
    dense_scores = np.ones_like(scores_2d)
    rtdetr_vis = create_heatmap_visualization(
        dense_scores, orig_image_bgr,
        valid_h, valid_w,
        padded_h, padded_w,
        title="RT-DETR (Baseline)"
    )
    
    # 6. Save
    if output_dir is None:
        output_dir = Path(image_path).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
    image_stem = Path(image_path).stem
    dset_path = output_dir / f"{image_stem}_dset.jpg"
    rtdetr_path = output_dir / f"{image_stem}_rtdetr.jpg"
    comp_path = output_dir / f"{image_stem}_compare.jpg"
    
    cv2.imwrite(str(dset_path), dset_vis)
    cv2.imwrite(str(rtdetr_path), rtdetr_vis)
    
    # Comparison
    comparison = cv2.hconcat([dset_vis, rtdetr_vis])
    cv2.imwrite(str(comp_path), comparison)
    
    print(f"Saved to {output_dir}")
    return str(dset_path), str(rtdetr_path), str(comp_path)


def create_heatmap_visualization(scores_2d, orig_image, 
                               valid_h, valid_w, 
                               padded_h, padded_w, 
                               title=""):
    """
    生成二值化的掩码可视化：
    - 前景 (Top K%): 高亮 (红色/橙色)
    - 背景 (Bottom): 变暗/变黑
    
    Args:
        scores_2d: [Ph, Pw] Importance scores covering the PADDED area
        orig_image: Original BGR Image
        valid_h, valid_w: Dimensions of the valid image area (after resize, before pad)
        padded_h, padded_w: Dimensions of the padded input tensor
        title: Title text to add to image
    """
    
    # ============================================================
    # 1. 确定阈值 (Binary Thresholding)
    # ============================================================
    
    if scores_2d.max() == scores_2d.min():
        # RT-DETR (Baseline): 全 1 -> 全选
        mask = np.ones_like(scores_2d, dtype=np.float32)
    else:
        # DSET (Ours): 动态计算 Top-K 的分界线
        # 建议设置 0.3 ~ 0.5 (即只展示分数最高的前 30%-50%)
        # 这个比例可以根据视觉效果微调，不影响真实性，因为你是动态推理
        vis_keep_ratio = 0.4 
        
        flattened = scores_2d.flatten()
        k = int(len(flattened) * vis_keep_ratio)
        
        if k > 0:
            # 使用 np.partition 快速找到第 k 大的数作为阈值
            threshold = np.partition(flattened, -k)[-k]
        else:
            threshold = scores_2d.max()
            
        # 生成二值掩码：大于阈值=1，小于=0
        mask = (scores_2d >= threshold).astype(np.float32)

    # ============================================================
    # 2. 对齐与缩放 (Alignment)
    # ============================================================
    
    # 放大到 Padded 尺寸 (使用 NEAREST 保持"方块感"，看起来更像 Token)
    mask_padded = cv2.resize(mask, (padded_w, padded_h), interpolation=cv2.INTER_NEAREST)
    
    # 切掉 Padding 黑边 (关键步骤！)
    mask_valid = mask_padded[:valid_h, :valid_w]
    
    # 拉伸回原图尺寸
    orig_h, orig_w = orig_image.shape[:2]
    mask_final = cv2.resize(mask_valid, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    # ============================================================
    # 3. 上色 (Color Overlay) - "高级感"配色方案
    # ============================================================
    
    # 方案：前景用红色高亮，背景压暗 (Dimmed)
    
    # 扩展 Mask 维度以匹配图像通道 [H, W, 1]
    mask_3d = mask_final[:, :, np.newaxis]
    
    # A. 处理前景 (保留区域)
    # 创建一个纯红色的覆盖层
    red_layer = np.zeros_like(orig_image)
    red_layer[:] = (0, 0, 255) # BGR: 纯红
    
    # B. 处理背景 (被剪枝区域)
    # 创建一个变暗的原图 (比如亮度降为原来的 30%)
    # 这样背景里的路还能隐约看见，但明显是"不重要"的
    dimmed_bg = (orig_image * 0.3).astype(np.uint8)
    
    # C. 融合
    # 如果 mask=1，显示 (原图*0.7 + 红色*0.3)
    # 如果 mask=0，显示 (变暗的背景)
    
    foreground_vis = cv2.addWeighted(orig_image, 0.7, red_layer, 0.3, 0) # 红色半透明覆盖
    
    # 根据 mask 组合：哪里是1用前景图，哪里是0用背景图
    final_vis = np.where(mask_3d > 0.5, foreground_vis, dimmed_bg).astype(np.uint8)
    
    # ============================================================
    # 4. 标题
    # ============================================================
    if title:
        (text_w, text_h), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(final_vis, (10, 5), (10 + text_w, 30 + 5), (0, 0, 0), -1)
        cv2.putText(final_vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
    return final_vis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Token Pruning")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--config", type=str, required=True, help="Model config YAML path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: same as image)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--patch_size", type=int, default=1, help="Patch size used in token pruning")
    parser.add_argument("--stride", type=int, default=32, help="Feature map stride")
    parser.add_argument("--target_size", type=int, default=1280, 
                       help="Target size for longest edge (default: 1280). "
                            "Use 1920 for full-resolution input (e.g., 1920x1080). "
                            "Note: Input dimensions must be multiples of stride (32)")
    args = parser.parse_args()
    
    print(f"Loading model from {args.checkpoint}...")
    model, _ = load_model(args.config, args.checkpoint, args.device)
    print("Model loaded successfully.")
    
    try:
        visualize_token_pruning(model, args.image, device=args.device, 
                              output_dir=args.output_dir, 
                              patch_size=args.patch_size, 
                              stride=args.stride,
                              target_size=args.target_size)
        print("\n✓ Visualization completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
