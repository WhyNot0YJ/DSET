#!/usr/bin/env python3
"""
DSET Visualization Tool
Supports multiple visualization modes for Token Pruning / Sparsity.

Modes:
1. --mode teaser (Default): 
   Generates the paper's Teaser Figure (Figure 1).
   - RT-DETR: Dense Paradigm (Uniform Orange).
   - DSET: Sparse Paradigm (Blue=Filtered/Background, Red=Focus/Foreground).
   - Uses actual binary masks captured via Forward Hook.

2. --mode heatmap:
   Visualizes the continuous Importance Scores as a heatmap.
   - Warmer colors (Red) = Higher Importance.
   - Cooler colors (Blue) = Lower Importance.
   - Useful for analyzing what the model considers "important" before pruning.
"""

import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root.parent) not in sys.path:
    sys.path.insert(0, str(project_root.parent))

# Import from batch_inference
try:
    from batch_inference import load_model, preprocess_image
except ImportError:
    from experiments.dset.batch_inference import load_model, preprocess_image


class PruningHook:
    """
    Hook to capture the pruning mask AND importance scores from the model's encoder.
    """
    def __init__(self):
        self.mask = None
        self.spatial_shape = None
        self.kept_indices = None
        self.scores = None

    def __call__(self, module, inputs, outputs):
        """
        Hook function to intercept forward pass.
        outputs: (pruned_tokens, kept_indices, info)
        """
        # Unpack outputs from TokenLevelPruner
        # Note: HybridEncoder calls it with return_indices=True
        if isinstance(outputs, tuple):
            _, kept_indices, info = outputs
        else:
            return

        # Capture Importance Scores
        if 'token_importance_scores' in info:
            self.scores = info['token_importance_scores']

        # Get original feature map dimensions from info
        if 'original_spatial_shape' in info:
            H, W = info['original_spatial_shape']
        else:
            return

        # Initialize binary mask (0 = Pruned/Background)
        # Flattened size: H * W
        total_tokens = H * W
        mask_flat = np.zeros(total_tokens, dtype=np.float32)

        if kept_indices is not None:
            # DSET Sparse Mode
            # Get indices for the first image in batch
            indices = kept_indices[0].cpu().numpy()
            # Handle -1 padding
            indices = indices[indices >= 0]
            # Set kept locations to 1 (Foreground)
            mask_flat[indices] = 1.0
        else:
            # No pruning (Dense Mode or Warmup) - Set all to 1
            mask_flat[:] = 1.0

        # Reshape to 2D feature map
        self.mask = mask_flat.reshape(H, W)
        self.spatial_shape = (H, W)
        self.kept_indices = kept_indices


def apply_mask_overlay(image, mask, color, alpha):
    """
    Apply a colored overlay mask to an image.
    image: BGR image
    mask: Binary mask (0 or 1) of same size as image (or to be resized)
    color: BGR tuple (e.g., (0, 0, 255) for Red)
    alpha: Transparency (0.0 - 1.0)
    """
    H, W = image.shape[:2]
    # Resize mask to match image size (Nearest Neighbor to keep blocky look for tokens)
    mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    
    # Create colored overlay
    overlay = np.zeros_like(image)
    overlay[:] = color
    
    # Mask for where to apply the color
    mask_indices = mask_resized > 0.5
    
    # Apply alpha blending only on masked areas
    output = image.copy()
    roi = output[mask_indices]
    
    if roi.size > 0:
        colored_roi = cv2.addWeighted(roi, 1 - alpha, overlay[mask_indices], alpha, 0)
        output[mask_indices] = colored_roi
    
    return output


def visualize_heatmap(image, scores, alpha=0.6):
    """
    Overlay a continuous heatmap on the image.
    scores: 2D float array (importance scores)
    """
    H, W = image.shape[:2]
    
    # Normalize scores to 0-255
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    scores_uint8 = (scores_norm * 255).astype(np.uint8)
    
    # Resize to image size
    heatmap_resized = cv2.resize(scores_uint8, (W, H), interpolation=cv2.INTER_NEAREST)
    
    # Apply colormap (JET is standard for heatmaps)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    # Overlay
    output = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    return output


def run_visualization(model, image_path, device='cuda', output_dir=None, target_size=1280, mode='teaser'):
    """
    Main visualization routine.
    """
    # 1. Load and Preprocess Image
    print(f"Loading image: {image_path}")
    orig_image = cv2.imread(str(image_path))
    if orig_image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    img_tensor, _, meta = preprocess_image(str(image_path), target_size=target_size)
    img_tensor = img_tensor.to(device)
    
    # Dimensions for aligning mask back to original image (crop padding)
    orig_h, orig_w = meta['orig_size'][0].tolist()
    padded_h, padded_w = meta['padded_h'], meta['padded_w']

    # 2. Register Hook & Run Inference
    hook_handle = None
    pruning_hook = PruningHook()
    
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'token_pruners'):
        if len(model.encoder.token_pruners) > 0:
            # Hook the first pruner layer
            hook_handle = model.encoder.token_pruners[0].register_forward_hook(pruning_hook)
        else:
            print("⚠ No token pruners found.")
    
    print("Running inference...")
    model.eval()
    
    # Force enable pruning (bypass warmup) for visualization
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'set_epoch'):
        model.encoder.set_epoch(100)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        
    if hook_handle:
        hook_handle.remove()

    # 3. Process Mask/Scores
    
    # (A) Get Binary Mask from Hook (Actual Pruning Decision)
    feature_mask = pruning_hook.mask
    if feature_mask is None:
        print("⚠ Failed to capture mask via hook. Falling back to all-ones.")
        feature_mask = np.ones((padded_h // 32, padded_w // 32), dtype=np.float32)

    # Get feature map dimensions from hook
    if pruning_hook.spatial_shape:
        h_feat, w_feat = pruning_hook.spatial_shape
    else:
        # Fallback: calculate from padded dimensions (stride=32)
        h_feat = padded_h // 32
        w_feat = padded_w // 32

    # Align Map to Image: Use train.py's physical alignment strategy
    def align_map_to_image(map_2d, h_feat, w_feat, H_tensor, W_tensor, orig_h, orig_w):
        """
        Align feature map to original image using physical space calibration.
        
        Strategy (matching train.py):
        1. Calculate valid region size in feature map coordinate system
        2. Crop padding at feature map level
        3. Single resize from feature map to original image
        
        Args:
            map_2d: Feature map of shape (h_feat, w_feat)
            h_feat, w_feat: Feature map dimensions
            H_tensor, W_tensor: Padded tensor dimensions (same as padded_h, padded_w)
            orig_h, orig_w: Original image dimensions
        
        Returns:
            Aligned map of shape (orig_h, orig_w)
        """
        # Step 1: Calculate valid region size in feature map coordinate system
        # Formula from train.py: valid_h_feat = round(orig_h × (h_feat / H_tensor))
        valid_h_feat = int(round(orig_h * (h_feat / H_tensor)))
        valid_w_feat = int(round(orig_w * (w_feat / W_tensor)))
        
        # Step 2: Crop padding at feature map level
        map_valid = map_2d[:valid_h_feat, :valid_w_feat]
        
        # Step 3: Single resize from feature map to original image
        # Use INTER_NEAREST to maintain blocky token appearance
        map_final = cv2.resize(map_valid, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        return map_final

    feature_mask_final = align_map_to_image(feature_mask, h_feat, w_feat, padded_h, padded_w, orig_h, orig_w)

    # (B) Get Importance Scores (UPDATED LOGIC)
    importance_scores_2d = None
    
    # Priority 1: Get directly from Hook (Most reliable)
    if pruning_hook.scores is not None:
        scores = pruning_hook.scores.cpu().numpy()
        if scores.ndim == 2: scores = scores[0] # Take batch 0
        if pruning_hook.spatial_shape:
            H, W = pruning_hook.spatial_shape
            if len(scores) == H * W:
                importance_scores_2d = scores.reshape(H, W)
    
    # Priority 2: Fallback to existing logic (Encoder Info)
    if importance_scores_2d is None:
        # 尝试从 outputs 中提取 encoder_info (Training mode 返回 tuple)
        if isinstance(outputs, tuple) and len(outputs) == 2:
             _, encoder_info_out = outputs
             if 'importance_scores_list' in encoder_info_out:
                 scores_list = encoder_info_out['importance_scores_list']
                 if scores_list:
                    scores = scores_list[0].cpu().numpy()
                    if scores.ndim == 2: scores = scores[0]
                    if pruning_hook.spatial_shape:
                        H, W = pruning_hook.spatial_shape
                        if len(scores) == H * W:
                            importance_scores_2d = scores.reshape(H, W)
        
        # 尝试从 tensor 属性中提取 (Inference mode hack)
        elif isinstance(outputs, list) and len(outputs) > 0 and hasattr(outputs[0], 'encoder_info'):
            encoder_info_out = getattr(outputs[0], 'encoder_info')
            if 'importance_scores_list' in encoder_info_out:
                 scores_list = encoder_info_out['importance_scores_list']
                 if scores_list:
                    scores = scores_list[0].cpu().numpy()
                    if scores.ndim == 2: scores = scores[0]
                    if pruning_hook.spatial_shape:
                        H, W = pruning_hook.spatial_shape
                        if len(scores) == H * W:
                            importance_scores_2d = scores.reshape(H, W)
        
        # 原有逻辑：直接检查 'encoder_info' key (如果 outputs 是 dict)
        elif isinstance(outputs, dict) and 'encoder_info' in outputs:
            scores_list = outputs['encoder_info'].get('importance_scores_list', [])
            if scores_list:
                scores = scores_list[0].cpu().numpy()
                if scores.ndim == 2: scores = scores[0] # Take batch 0
                
                # Reshape scores to 2D
                if pruning_hook.spatial_shape:
                    H, W = pruning_hook.spatial_shape
                    if len(scores) == H * W:
                        importance_scores_2d = scores.reshape(H, W)

    # 4. Generate Visualization based on Mode
    
    if output_dir is None:
        output_dir = Path(image_path).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    stem = Path(image_path).stem
    
    if mode == 'teaser':
        print("Generating Teaser Figure (Binary Masks)...")
        # --- Figure (a): RT-DETR (Dense) ---
        orange_color = (0, 128, 255) # BGR
        full_mask = np.ones((orig_h, orig_w), dtype=np.float32)
        fig_a = apply_mask_overlay(orig_image.copy(), full_mask, orange_color, alpha=0.3)
        cv2.putText(fig_a, "RT-DETR: Dense Compute", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(fig_a, "RT-DETR: Dense Compute", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1)

        # --- Figure (b): DSET (Sparse) ---
        blue_color = (255, 0, 0)
        red_color = (0, 0, 255)
        bg_mask = 1.0 - feature_mask_final
        
        # Apply Background (Blue)
        fig_b = apply_mask_overlay(orig_image.copy(), bg_mask, blue_color, alpha=0.3)
        # Apply Foreground (Red)
        fig_b = apply_mask_overlay(fig_b, feature_mask_final, red_color, alpha=0.5)
        
        cv2.putText(fig_b, "DSET (Ours): Sparse Focus", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(fig_b, "DSET (Ours): Sparse Focus", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1)
        
        # Save
        path_a = output_dir / f"{stem}_teaser_a_rtdetr.jpg"
        path_b = output_dir / f"{stem}_teaser_b_dset.jpg"
        path_concat = output_dir / f"{stem}_teaser_combined.jpg"
        
        cv2.imwrite(str(path_a), fig_a)
        cv2.imwrite(str(path_b), fig_b)
        combined = cv2.hconcat([fig_a, fig_b])
        cv2.imwrite(str(path_concat), combined)
        print(f"Saved: {path_concat}")

    elif mode == 'heatmap':
        print("Generating Importance Heatmap...")
        if importance_scores_2d is None:
            print("Error: No importance scores found for heatmap mode.")
            return

        # Get feature map dimensions for alignment
        if pruning_hook.spatial_shape:
            h_feat, w_feat = pruning_hook.spatial_shape
        else:
            h_feat = padded_h // 32
            w_feat = padded_w // 32

        scores_final = align_map_to_image(importance_scores_2d, h_feat, w_feat, padded_h, padded_w, orig_h, orig_w)
        fig_heatmap = visualize_heatmap(orig_image.copy(), scores_final)
        
        cv2.putText(fig_heatmap, "Token Importance Heatmap", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        path_heatmap = output_dir / f"{stem}_heatmap.jpg"
        cv2.imwrite(str(path_heatmap), fig_heatmap)
        print(f"Saved: {path_heatmap}")
        
    else:
        print(f"Unknown mode: {mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSET Sparsity Visualization")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--config", type=str, required=True, help="Model config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--mode", type=str, default="teaser", choices=["teaser", "heatmap"], 
                        help="Visualization mode: 'teaser' (binary mask, 3-color) or 'heatmap' (continuous scores)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--target_size", type=int, default=1280, help="Inference size")
    
    args = parser.parse_args()
    
    print(f"Initializing model for mode: {args.mode}...")
    model, _ = load_model(args.config, args.checkpoint, args.device)
    
    run_visualization(model, args.image, device=args.device, output_dir=args.output_dir, 
                     target_size=args.target_size, mode=args.mode)
