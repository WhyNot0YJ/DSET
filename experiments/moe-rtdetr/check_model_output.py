#!/usr/bin/env python3
"""检查模型输出格式和坐标转换"""

import sys
import yaml
import torch
import cv2
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))

# 从train.py导入模型类
try:
    from train import AdaptiveExpertRTDETR
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("train", project_root / "train.py")
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    AdaptiveExpertRTDETR = train_module.AdaptiveExpertRTDETR

from src.nn.postprocessor.detr_postprocessor import DetDETRPostProcessor
from src.nn.postprocessor.box_revert import BoxProcessFormat

def check_model_output(image_path, config_path, checkpoint_path, device='cuda'):
    """检查模型输出格式"""
    print("="*60)
    print("检查模型输出格式")
    print("="*60)
    
    # 加载配置和模型
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    encoder_config = config['model']['encoder']
    model = AdaptiveExpertRTDETR(
        config_name=config['model'].get('config_name', 'A'),
        hidden_dim=config['model']['hidden_dim'],
        num_queries=config['model']['num_queries'],
        top_k=config['model']['top_k'],
        backbone_type=config['model']['backbone'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        encoder_in_channels=encoder_config['in_channels'],
        encoder_expansion=encoder_config['expansion'],
        num_experts=config['model'].get('num_experts', None)
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 验证时使用 self.ema.module，所以推理时也应该使用EMA权重
    if 'ema_state_dict' in checkpoint:
        print("  使用EMA模型权重（与验证时一致）")
        ema_state_dict = checkpoint['ema_state_dict']
        # EMA的state_dict格式是 {'module': {...}, 'updates': ...}
        if isinstance(ema_state_dict, dict) and 'module' in ema_state_dict:
            state_dict = ema_state_dict['module']
        else:
            # 兼容旧格式
            state_dict = ema_state_dict
    elif 'model_state_dict' in checkpoint:
        print("  使用普通模型权重（未找到EMA权重）")
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(device)
    
    # 预处理图像（与训练时一致）
    image_bgr = cv2.imread(str(image_path))
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]
    
    target_size = 640
    scale = min(target_size / orig_h, target_size / orig_w)
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)
    new_h = ((new_h + 31) // 32) * 32
    new_w = ((new_w + 31) // 32) * 32
    new_h = min(new_h, target_size)
    new_w = min(new_w, target_size)
    
    image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)
    resized_image = torch.nn.functional.interpolate(
        image_tensor.unsqueeze(0), 
        size=(new_h, new_w), 
        mode='bilinear', 
        align_corners=False,
        antialias=False
    ).squeeze(0)
    
    padded_image = torch.zeros(3, target_size, target_size, dtype=resized_image.dtype)
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    padded_image[:, pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized_image
    img_tensor = (padded_image / 255.0).unsqueeze(0).to(device)
    
    print(f"\n图像预处理:")
    print(f"  原始尺寸: {orig_w}x{orig_h}")
    print(f"  缩放后: {new_w}x{new_h} (scale={scale:.4f})")
    print(f"  Padding: pad_w={pad_w}, pad_h={pad_h}")
    print(f"  输入tensor形状: {img_tensor.shape}")
    print(f"  输入tensor范围: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
    
    # 模型前向传播
    with torch.no_grad():
        outputs = model(img_tensor)
    
    print(f"\n模型输出:")
    print(f"  pred_logits形状: {outputs['pred_logits'].shape}")
    print(f"  pred_boxes形状: {outputs['pred_boxes'].shape}")
    print(f"  pred_boxes范围: [{outputs['pred_boxes'].min():.4f}, {outputs['pred_boxes'].max():.4f}]")
    print(f"  pred_boxes前5个: {outputs['pred_boxes'][0, :5, :]}")
    
    # 检查logits
    pred_logits = outputs['pred_logits'][0]  # [Q, C]
    pred_scores_sigmoid = torch.sigmoid(pred_logits)
    max_scores, pred_classes = torch.max(pred_scores_sigmoid, dim=-1)
    
    print(f"\n置信度分析:")
    print(f"  使用sigmoid后:")
    print(f"    最大置信度: {max_scores.max().item():.4f}")
    print(f"    平均置信度: {max_scores.mean().item():.4f}")
    print(f"    置信度>0.1的数量: {(max_scores > 0.1).sum().item()}")
    print(f"    置信度>0.3的数量: {(max_scores > 0.3).sum().item()}")
    print(f"    前10个最高置信度: {torch.sort(max_scores, descending=True)[0][:10].tolist()}")
    
    # 检查boxes格式
    pred_boxes = outputs['pred_boxes'][0]  # [Q, 4]
    print(f"\nBoxes格式分析:")
    print(f"  Boxes范围: [{pred_boxes.min():.4f}, {pred_boxes.max():.4f}]")
    if pred_boxes.max() <= 1.0:
        print(f"  ✓ Boxes是归一化的 (0-1范围)")
        print(f"  前5个boxes (cxcywh格式):")
        for i in range(min(5, len(pred_boxes))):
            cx, cy, w, h = pred_boxes[i].tolist()
            print(f"    [{i}] cx={cx:.4f}, cy={cy:.4f}, w={w:.4f}, h={h:.4f}")
            # 转换为640x640像素坐标
            x1 = (cx - w/2) * target_size
            y1 = (cy - h/2) * target_size
            x2 = (cx + w/2) * target_size
            y2 = (cy + h/2) * target_size
            print(f"         -> xyxy (640x640): [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    else:
        print(f"  ⚠️ Boxes不是归一化的")
    
    # 测试后处理器
    print(f"\n后处理器测试:")
    postprocessor = DetDETRPostProcessor(
        num_classes=6,
        use_focal_loss=True,
        num_top_queries=300,
        box_process_format=BoxProcessFormat.RESIZE
    )
    
    eval_sizes = torch.tensor([[target_size, target_size]], device=device)
    results = postprocessor(outputs, eval_sizes=eval_sizes)
    result = results[0]
    
    print(f"  后处理器输出:")
    print(f"    labels形状: {result['labels'].shape}")
    print(f"    boxes形状: {result['boxes'].shape}")
    print(f"    scores形状: {result['scores'].shape}")
    print(f"    boxes范围: [{result['boxes'].min():.1f}, {result['boxes'].max():.1f}]")
    print(f"    前5个boxes (xyxy格式, 640x640):")
    for i in range(min(5, len(result['boxes']))):
        x1, y1, x2, y2 = result['boxes'][i].tolist()
        score = result['scores'][i].item()
        label = result['labels'][i].item()
        print(f"      [{i}] label={label}, score={score:.4f}, box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    
    return outputs, results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    check_model_output(args.image, args.config, args.checkpoint, args.device)

