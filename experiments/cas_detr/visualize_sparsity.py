#!/usr/bin/env python3
"""
Cas_DETR Visualization Tool
Supports multiple visualization modes for Token Pruning / Sparsity.

Modes:
1. --mode teaser (Default): 
   Generates the paper's Teaser Figure (Figure 1).
   - RT-DETR: Dense Paradigm (Uniform Orange).
   - Cas_DETR: Sparse Paradigm (Blue=Filtered/Background, Red=Focus/Foreground).
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
    from experiments.cas_detr.batch_inference import load_model, preprocess_image


class PruningHook:
    """
    Hook to capture the pruning mask AND importance scores from the model's encoder.
    
    NOTE: HybridEncoder calls shared_token_pruner with spatial_shape=None because
    the input is a multi-scale concatenated sequence [S4_tokens, S5_tokens].
    Therefore we only capture raw scores and kept_indices here; the spatial
    shape (S4) is computed externally from padded tensor dimensions.
    """
    def __init__(self):
        self.kept_indices = None
        self.scores = None

    def __call__(self, module, inputs, outputs):
        """
        Hook function to intercept forward pass.
        outputs: (pruned_tokens, kept_indices, info)
        """
        if isinstance(outputs, tuple):
            _, kept_indices, info = outputs
        else:
            return

        # Capture raw importance scores (full multi-scale sequence: [S4_tokens, S5_tokens])
        if 'token_importance_scores' in info:
            self.scores = info['token_importance_scores']

        # Capture kept_indices (global indices into the concatenated sequence)
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
    
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'shared_token_pruner'):
        if model.encoder.shared_token_pruner is not None:
            # Hook the shared pruner layer
            hook_handle = model.encoder.shared_token_pruner.register_forward_hook(pruning_hook)
        else:
            print("⚠ No token pruners found.")
    
    print("Running inference...")
    model.eval()
    
    # Pruning is always enabled from epoch 0, but we still call set_epoch for interface compatibility
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'set_epoch'):
        model.encoder.set_epoch(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        
    if hook_handle:
        hook_handle.remove()

    # 3. Process Mask/Scores
    #
    # Compute S4 spatial shape from padded tensor dimensions.
    # use_encoder_idx=[1,2] → S4 (stride 16) and S5 (stride 32).
    # We only visualize the S4 portion (higher resolution, first segment in
    # the concatenated sequence).
    h_s4, w_s4 = padded_h // 16, padded_w // 16
    num_tokens_s4 = h_s4 * w_s4
    h_feat, w_feat = h_s4, w_s4

    # (A) Build Binary Mask from Hook's kept_indices (S4 region only)
    if pruning_hook.kept_indices is not None:
        kept_indices = pruning_hook.kept_indices[0].cpu().numpy()  # batch 0
        mask_flat = np.zeros(num_tokens_s4, dtype=np.float32)
        indices = kept_indices[(kept_indices >= 0) & (kept_indices < num_tokens_s4)]
        mask_flat[indices] = 1.0
        feature_mask = mask_flat.reshape(h_s4, w_s4)
        print(f"  ✓ Captured pruning mask: {indices.shape[0]}/{num_tokens_s4} S4 tokens kept")
    else:
        print("⚠ No kept_indices captured. Using all-ones mask (no pruning).")
        feature_mask = np.ones((h_s4, w_s4), dtype=np.float32)

    # Align Map to Image: Use train.py's physical alignment strategy
    def align_map_to_image(map_2d, h_feat, w_feat, H_tensor, W_tensor, orig_h, orig_w, normalize_before_resize=False):
        """
        Align feature map to original image using physical space calibration.
        
        Strategy (matching train.py):
        1. Calculate valid region size in feature map coordinate system
        2. Crop padding at feature map level
        3. (Optional) Normalize on cropped valid region (matching train.py exactly)
        4. Single resize from feature map to original image
        
        Args:
            map_2d: Feature map of shape (h_feat, w_feat)
            h_feat, w_feat: Feature map dimensions
            H_tensor, W_tensor: Padded tensor dimensions (same as padded_h, padded_w)
            orig_h, orig_w: Original image dimensions
            normalize_before_resize: If True, normalize on cropped region before resize (matches train.py exactly)
        
        Returns:
            Aligned map of shape (orig_h, orig_w), values in [0, 1] if normalize_before_resize=True
        """
        # Step 1: Calculate valid region size in feature map coordinate system
        # Formula from train.py: valid_h_feat = round(orig_h × (h_feat / H_tensor))
        valid_h_feat = int(round(orig_h * (h_feat / H_tensor)))
        valid_w_feat = int(round(orig_w * (w_feat / W_tensor)))
        
        # Step 2: Crop padding at feature map level
        map_valid = map_2d[:valid_h_feat, :valid_w_feat]
        
        # Step 3: (Optional) Normalize on cropped valid region (matching train.py exactly)
        # train.py does: s_norm = (s_valid - s_valid.min()) / (s_valid.max() - s_valid.min() + 1e-8)
        if normalize_before_resize:
            map_min = map_valid.min()
            map_max = map_valid.max()
            if map_max - map_min > 1e-8:
                map_valid = (map_valid - map_min) / (map_max - map_min + 1e-8)
            else:
                # All values are the same, set to 0
                map_valid = np.zeros_like(map_valid)
        
        # Step 4: Single resize from feature map to original image
        # Use INTER_CUBIC for smooth upsampling
        map_final = cv2.resize(map_valid, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        
        return map_final

    feature_mask_final = align_map_to_image(feature_mask, h_feat, w_feat, padded_h, padded_w, orig_h, orig_w)

    # (B) Get Importance Scores — reshape S4 portion to 2D
    importance_scores_2d = None

    def _extract_s4_scores(raw_scores):
        """Extract S4 portion from raw scores and reshape to 2D."""
        if isinstance(raw_scores, torch.Tensor):
            s = raw_scores.detach().cpu().numpy()
        else:
            s = np.asarray(raw_scores)
        if s.ndim >= 2:
            s = s[0]  # batch 0
        if len(s) >= num_tokens_s4:
            return s[:num_tokens_s4].reshape(h_s4, w_s4)
        return None

    # Priority 1: Hook scores (most reliable)
    if pruning_hook.scores is not None:
        importance_scores_2d = _extract_s4_scores(pruning_hook.scores)

    # Priority 2: Fallback — try to get scores from model outputs
    if importance_scores_2d is None:
        def _try_extract_from_encoder_info(encoder_info_out):
            # Option A: layer_wise_heatmaps (already reshaped per-level by encoder)
            heatmaps = encoder_info_out.get('layer_wise_heatmaps', [])
            if heatmaps:
                # Pick the largest spatial heatmap (S4)
                hm = max(heatmaps, key=lambda t: t.shape[-2] * t.shape[-1])
                return hm[0, 0].detach().cpu().numpy()
            # Option B: importance_scores_list
            scores_list = encoder_info_out.get('importance_scores_list', [])
            if scores_list:
                return _extract_s4_scores(scores_list[0])
            return None

        if isinstance(outputs, tuple) and len(outputs) == 2:
            _, encoder_info_out = outputs
            importance_scores_2d = _try_extract_from_encoder_info(encoder_info_out)
        elif isinstance(outputs, dict) and 'encoder_info' in outputs:
            importance_scores_2d = _try_extract_from_encoder_info(outputs['encoder_info'])

    # 4. Generate Visualization based on Mode
    
    if output_dir is None:
        output_dir = Path(image_path).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    stem = Path(image_path).stem

    # ── Shared layout constants & helpers (used by both modes) ──────────────
    HEADER_H  = 64
    DIVIDER_W = 4
    C_BG      = (240, 240, 240)   # light gray bar background
    C_BLACK   = (0, 0, 0)
    _FONT     = cv2.FONT_HERSHEY_DUPLEX

    def put_text_centered(img, text, cx, cy, scale, color, thickness=2):
        """Draw horizontally-centred text at (cx, cy)."""
        (tw, th), _ = cv2.getTextSize(text, _FONT, scale, thickness)
        cv2.putText(img, text, (cx - tw // 2, cy + th // 2), _FONT, scale, color, thickness, cv2.LINE_AA)

    def make_dashed_divider(height):
        """White dashed vertical line on a light-gray background."""
        div = np.full((height, DIVIDER_W, 3), C_BG, dtype=np.uint8)
        cx = DIVIDER_W // 2
        dash_len, gap_len = 14, 8
        y = gap_len
        while y < height:
            end = min(y + dash_len, height)
            cv2.line(div, (cx, y), (cx, end), (255, 255, 255), 1)
            y += dash_len + gap_len
        return div

    if mode == 'teaser':
        print("Generating Teaser Figure (Paper-style two-column comparison)...")

        # Colours (BGR) for image overlays
        C_ORANGE = (15, 55, 140)   # dark burnt-orange / rust (暗沉橙)
        C_BLUE   = (220,  80,  20)
        C_RED    = (30,   30, 220)

        # ── Build the two image panels ───────────────────────────────────────
        # (a) Dense: dark muted tint — conveys O(N²) computational burden
        full_mask = np.ones((orig_h, orig_w), dtype=np.float32)
        fig_a = apply_mask_overlay(orig_image.copy(), full_mask, C_ORANGE, alpha=0.55)

        # (b) Sparse: blue background + red foreground
        bg_mask  = 1.0 - feature_mask_final
        fig_b = apply_mask_overlay(orig_image.copy(), bg_mask,          C_BLUE,   alpha=0.35)
        fig_b = apply_mask_overlay(fig_b,             feature_mask_final, C_RED,  alpha=0.50)

        W, H = orig_w, orig_h   # panel dimensions

        # ── Sub-label: (a) / (b) in bottom-left of each panel ───────────────
        # (removed per user request)

        # ── Assemble: [header] / [fig_a | divider | fig_b] ──────────────────
        total_w = W * 2 + DIVIDER_W

        # Header bar
        header = np.full((HEADER_H, total_w, 3), C_BG, dtype=np.uint8)
        put_text_centered(header, "Dense Paradigm",         W // 2,                       HEADER_H // 2, 1.0, C_BLACK)
        put_text_centered(header, "Sparse Paradigm (Ours)", W + DIVIDER_W // 2 + W // 2, HEADER_H // 2, 1.0, C_BLACK)

        # Image row with dashed divider
        divider = make_dashed_divider(H)
        img_row = cv2.hconcat([fig_a, divider, fig_b])

        # Stack vertically (no footer)
        combined = cv2.vconcat([header, img_row])

        # ── Save ────────────────────────────────────────────────────────────
        path_a      = output_dir / f"{stem}_teaser_a_dense.jpg"
        path_b      = output_dir / f"{stem}_teaser_b_sparse.jpg"
        path_concat = output_dir / f"{stem}_teaser_combined.jpg"

        cv2.imwrite(str(path_a), fig_a)
        cv2.imwrite(str(path_b), fig_b)
        cv2.imwrite(str(path_concat), combined)
        print(f"Saved: {path_concat}")


    elif mode == 'heatmap':
        print("Generating Smooth Importance Heatmap...")
        if importance_scores_2d is None:
            print("Error: No importance scores found for heatmap mode.")
            return

        # Apply sigmoid to convert logits to probabilities
        s_2d = torch.sigmoid(
            torch.from_numpy(importance_scores_2d.astype(np.float32))
        ).numpy()

        # Align to original image (crop padding + resize)
        s_final = align_map_to_image(
            s_2d,
            h_feat,
            w_feat,
            padded_h,
            padded_w,
            orig_h,
            orig_w,
            normalize_before_resize=True,
        )

        # Gaussian blur for smooth appearance
        sigma = orig_w / 40.0
        k_size = int(sigma * 3) * 2 + 1
        s_smooth = cv2.GaussianBlur(s_final, (k_size, k_size), sigmaX=sigma, sigmaY=sigma)

        # Re-normalize
        s_min, s_max = s_smooth.min(), s_smooth.max()
        if s_max - s_min > 1e-8:
            s_smooth = (s_smooth - s_min) / (s_max - s_min + 1e-8)
        else:
            s_smooth = np.zeros_like(s_smooth)

        # Apply JET colormap and blend
        heatmap = cv2.applyColorMap((s_smooth * 255).astype(np.uint8), cv2.COLORMAP_JET)
        res = cv2.addWeighted(orig_image, 0.5, heatmap, 0.5, 0)

        # ── Three outputs matching teaser layout ─────────────────────────────
        C_ORANGE = (15, 55, 140)   # dark burnt-orange / rust (暗沉橙)
        full_mask = np.ones((orig_h, orig_w), dtype=np.float32)
        fig_orig = apply_mask_overlay(orig_image.copy(), full_mask, C_ORANGE, alpha=0.55)
        fig_heat = res.copy()
        W, H = orig_w, orig_h

        # ── Assemble: [header] / [fig_orig | divider | fig_heat] ────────────
        total_w = W * 2 + DIVIDER_W
        header  = np.full((HEADER_H, total_w, 3), C_BG, dtype=np.uint8)
        put_text_centered(header, "Dense Paradigm",         W // 2,                       HEADER_H // 2, 1.0, C_BLACK)
        put_text_centered(header, "Sparse Paradigm (Ours)", W + DIVIDER_W // 2 + W // 2, HEADER_H // 2, 1.0, C_BLACK)

        divider  = make_dashed_divider(H)
        img_row  = cv2.hconcat([fig_orig, divider, fig_heat])

        combined = cv2.vconcat([header, img_row])

        # ── Save three images ────────────────────────────────────────────────
        path_a      = output_dir / f"{stem}_heatmap_a_original.jpg"
        path_b      = output_dir / f"{stem}_heatmap_b_heatmap.jpg"
        path_concat = output_dir / f"{stem}_heatmap_combined.jpg"

        cv2.imwrite(str(path_a), fig_orig)
        cv2.imwrite(str(path_b), fig_heat)
        cv2.imwrite(str(path_concat), combined)
        print(f"Saved: {path_concat}")

    else:
        print(f"Unknown mode: {mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cas_DETR Sparsity Visualization")
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
