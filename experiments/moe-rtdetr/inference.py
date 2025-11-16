#!/usr/bin/env python3
"""MOE-RTDETR 推理脚本 - 在图像上绘制预测框"""

import sys
import argparse
import yaml
import torch
import cv2
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))

# 从train.py导入模型类（因为AdaptiveExpertRTDETR定义在那里）
try:
    from train import AdaptiveExpertRTDETR
except ImportError:
    # 如果直接导入失败，尝试从当前目录导入
    import importlib.util
    spec = importlib.util.spec_from_file_location("train", project_root / "train.py")
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    AdaptiveExpertRTDETR = train_module.AdaptiveExpertRTDETR
from src.nn.postprocessor.detr_postprocessor import DetDETRPostProcessor
from src.nn.postprocessor.box_revert import box_revert, BoxProcessFormat

# 类别名称（与数据集 dairv2x_detection.py 保持一致，10类）
CLASS_NAMES = [
    "Car", "Truck", "Van", "Bus", "Pedestrian", 
    "Cyclist", "Tricyclist", "Motorcyclist", "Barrowlist", "Trafficcone"
]
COLORS = [
    (255, 0, 0),      # Car - 红色
    (0, 255, 0),      # Truck - 绿色
    (255, 128, 0),    # Van - 橙色
    (0, 0, 255),      # Bus - 蓝色
    (255, 255, 0),    # Pedestrian - 黄色
    (255, 0, 255),    # Cyclist - 品红
    (128, 0, 255),    # Tricyclist - 紫色
    (0, 255, 255),    # Motorcyclist - 青色
    (255, 192, 203),  # Barrowlist - 粉色
    (128, 128, 128),  # Trafficcone - 灰色
]


def load_model(config_path: str, checkpoint_path: str, device: str = "cuda"):
    """加载模型和权重"""
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建模型
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
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 验证时使用 self.ema.module，所以推理时也应该使用EMA权重
    if 'ema_state_dict' in checkpoint:
        # EMA模型通常性能更好
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
    
    # 创建后处理器（使用RESIZE模式，然后手动处理padding和缩放）
    postprocessor = DetDETRPostProcessor(
        num_classes=6,
        use_focal_loss=True,
        num_top_queries=300,
        box_process_format=BoxProcessFormat.RESIZE
    )
    
    return model, postprocessor


def preprocess_image(image_path: str, target_size: int = 640):
    """预处理图像 - 与训练时保持一致"""
    # 读取图像（BGR格式）
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # ⚠️ 关键修复：转换为RGB格式（训练时使用RGB）
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    orig_h, orig_w = image.shape[:2]
    
    # ⚠️ 关键修复：确保尺寸是32的倍数（与训练时一致）
    # 计算缩放比例（保持宽高比）
    scale = min(target_size / orig_h, target_size / orig_w)
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)
    
    # 确保缩放后的尺寸是32的倍数（backbone需要）
    new_h = ((new_h + 31) // 32) * 32
    new_w = ((new_w + 31) // 32) * 32
    
    # 确保不超过目标尺寸
    new_h = min(new_h, target_size)
    new_w = min(new_w, target_size)
    
    # ⚠️ 关键修复：使用torch的interpolate（与训练时一致）
    # 转换为tensor进行缩放
    image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)  # HWC -> CHW
    resized_image = torch.nn.functional.interpolate(
        image_tensor.unsqueeze(0), 
        size=(new_h, new_w), 
        mode='bilinear', 
        align_corners=False,
        antialias=False
    ).squeeze(0)
    
    # 创建填充后的图像
    padded_image = torch.zeros(3, target_size, target_size, dtype=resized_image.dtype)
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    padded_image[:, pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized_image
    
    # 归一化到[0, 1]
    img_tensor = padded_image / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # [1, 3, H, W]
    
    # 保存原始尺寸和padding信息用于后处理
    meta = {
        'orig_size': torch.tensor([[orig_h, orig_w]]),  # [h, w] format
        'pad_h': pad_h,
        'pad_w': pad_w,
        'scale': scale,
        'new_h': new_h,
        'new_w': new_w
    }
    
    # 返回原始BGR图像用于绘制（cv2使用BGR）
    return img_tensor, image_bgr, meta


def postprocess_outputs(outputs, postprocessor, meta, conf_threshold=0.3, target_size=640, device='cuda'):
    """后处理模型输出"""
    # 获取模型输出的设备
    if isinstance(outputs, dict) and 'pred_logits' in outputs:
        output_device = outputs['pred_logits'].device
    else:
        output_device = torch.device(device)
    
    # 获取原始尺寸和resize后的尺寸 [w, h] 格式
    orig_h, orig_w = meta['orig_size'][0].tolist()
    new_w = meta.get('new_w', int(orig_w * meta['scale']))
    new_h = meta.get('new_h', int(orig_h * meta['scale']))
    
    # 准备后处理参数（使用RESIZE模式，只传递eval_sizes，得到640x640坐标）
    # 然后手动处理padding和缩放
    eval_sizes = torch.tensor([[target_size, target_size]], device=output_device)  # [1, 2] format: [w, h]
    
    # 后处理：归一化boxes -> 乘以eval_sizes -> 640x640坐标（包含padding区域）
    results = postprocessor(
        outputs, 
        eval_sizes=eval_sizes
    )
    
    # 手动处理坐标转换：640x640（含padding）-> resize后图像 -> 原始图像
    result = results[0]
    boxes_640 = result['boxes']  # [num_queries, 4] in xyxy format, 640x640 coordinates
    
    # 减去padding，得到resize后图像的坐标
    pad_w = meta['pad_w']
    pad_h = meta['pad_h']
    boxes_resized = boxes_640.clone()
    boxes_resized[:, [0, 2]] -= pad_w  # x coordinates
    boxes_resized[:, [1, 3]] -= pad_h  # y coordinates
    
    # 裁剪到resize后图像的有效区域 [0, 0, new_w, new_h]
    boxes_resized[:, 0] = torch.clamp(boxes_resized[:, 0], 0, new_w)
    boxes_resized[:, 1] = torch.clamp(boxes_resized[:, 1], 0, new_h)
    boxes_resized[:, 2] = torch.clamp(boxes_resized[:, 2], 0, new_w)
    boxes_resized[:, 3] = torch.clamp(boxes_resized[:, 3], 0, new_h)
    
    # 缩放回原始图像尺寸
    scale_w = orig_w / new_w
    scale_h = orig_h / new_h
    boxes_resized[:, [0, 2]] *= scale_w  # x coordinates
    boxes_resized[:, [1, 3]] *= scale_h  # y coordinates
    
    # 更新结果
    result['boxes'] = boxes_resized
    results = [result]
    
    # 提取结果
    result = results[0]  # batch_size=1
    labels = result['labels'].cpu().numpy()
    boxes = result['boxes'].cpu().numpy()
    scores = result['scores'].cpu().numpy()
    
    # 过滤低置信度
    mask = scores >= conf_threshold
    labels = labels[mask]
    boxes = boxes[mask]
    scores = scores[mask]
    
    return labels, boxes, scores


def draw_boxes(image, labels, boxes, scores):
    """在图像上绘制预测框"""
    if len(labels) == 0:
        return image
    
    for label, box, score in zip(labels, boxes, scores):
        x1, y1, x2, y2 = box.astype(int)
        
        # 确保坐标在图像范围内
        x1 = max(0, min(x1, image.shape[1] - 1))
        y1 = max(0, min(y1, image.shape[0] - 1))
        x2 = max(0, min(x2, image.shape[1] - 1))
        y2 = max(0, min(y2, image.shape[0] - 1))
        
        # 绘制边界框
        color = COLORS[label]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签和置信度
        class_name = CLASS_NAMES[label]
        label_text = f"{class_name}: {score:.2f}"
        
        # 计算文本位置
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_y = max(text_h + 4, y1)
        cv2.rectangle(image, (x1, text_y - text_h - 4), (x1 + text_w, text_y), color, -1)
        cv2.putText(image, label_text, (x1, text_y - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image


def inference(image_path: str, config_path: str, checkpoint_path: str, 
              output_path: str = None, conf_threshold: float = 0.3, 
              device: str = "cuda"):
    """执行推理"""
    # 加载模型
    print(f"加载模型: {checkpoint_path}")
    model, postprocessor = load_model(config_path, checkpoint_path, device)
    
    # 预处理图像
    print(f"处理图像: {image_path}")
    img_tensor, orig_image, meta = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)
    
    # 推理
    print("执行推理...")
    with torch.no_grad():
        outputs = model(img_tensor)
    
    # 后处理
    print("后处理...")
    labels, boxes, scores = postprocess_outputs(outputs, postprocessor, meta, conf_threshold, target_size=640, device=device)
    
    # 绘制结果
    result_image = draw_boxes(orig_image.copy(), labels, boxes, scores)
    
    # 保存结果
    if output_path is None:
        output_path = str(Path(image_path).with_suffix('.result.jpg'))
    
    cv2.imwrite(output_path, result_image)
    print(f"结果已保存到: {output_path}")
    print(f"检测到 {len(labels)} 个目标")
    
    return result_image, labels, boxes, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MOE-RTDETR 推理脚本")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型checkpoint路径")
    parser.add_argument("--output", type=str, default=None, help="输出图像路径")
    parser.add_argument("--conf", type=float, default=0.3, help="置信度阈值")
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda/cpu)")
    
    args = parser.parse_args()
    
    inference(
        args.image,
        args.config,
        args.checkpoint,
        args.output,
        args.conf,
        args.device
    )

