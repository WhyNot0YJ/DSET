#!/usr/bin/env python3
"""RT-DETR 批量推理脚本 - 处理整个图像目录"""

import sys
import argparse
import yaml
import torch
import cv2
import numpy as np
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    class SimpleProgress:
        def __init__(self, iterable, desc=""):
            self.iterable = iterable
            self.desc = desc
            self.total = len(iterable)
            self.current = 0
            print(f"{desc}: 开始处理 {self.total} 个文件...")
        
        def __iter__(self):
            for item in self.iterable:
                self.current += 1
                if self.current % 10 == 0 or self.current == self.total:
                    print(f"  进度: {self.current}/{self.total} ({100*self.current/self.total:.1f}%)")
                yield item
    
    def tqdm(iterable, desc=""):
        return SimpleProgress(iterable, desc) if not HAS_TQDM else iterable

# 添加项目路径（仅在直接运行时执行，导入时不执行）
def _setup_paths():
    """设置项目路径（延迟导入，避免循环导入）"""
    project_root = Path(__file__).parent.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(project_root.parent) not in sys.path:
        sys.path.insert(0, str(project_root.parent))

# 延迟导入，避免在导入时就执行路径设置
def _import_modules():
    """延迟导入模块"""
    _setup_paths()
    from train import RTDETRTrainer, create_backbone
    from src.nn.postprocessor.detr_postprocessor import DetDETRPostProcessor
    from src.nn.postprocessor.box_revert import box_revert, BoxProcessFormat
    return RTDETRTrainer, create_backbone, DetDETRPostProcessor, BoxProcessFormat

# 类别名称
CLASS_NAMES = ["Car", "Truck", "Bus", "Van", "Pedestrian", "Cyclist", "Motorcyclist"]
COLORS = [
    (255, 0, 0),    # Car - 红色
    (0, 255, 0),    # Truck - 绿色
    (0, 0, 255),    # Bus - 蓝色
    (255, 128, 0),  # Van - 橙色
    (255, 255, 0),  # Pedestrian - 黄色
    (255, 0, 255),  # Cyclist - 品红
    (0, 255, 255),  # Motorcyclist - 青色
]


def load_model(config_path: str, checkpoint_path: str, device: str = "cuda"):
    """加载模型和权重"""
    RTDETRTrainer, _, DetDETRPostProcessor, BoxProcessFormat = _import_modules()
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建训练器以构建模型
    trainer = RTDETRTrainer(config)
    model = trainer.create_model()
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 验证时使用 self.ema.module，所以推理时也应该使用EMA权重
    if 'ema' in checkpoint and 'module' in checkpoint['ema']:
        print("  使用EMA模型权重（与验证时一致）")
        state_dict = checkpoint['ema']['module']
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
        num_classes=7,
        use_focal_loss=True,
        num_top_queries=300,
        box_process_format=BoxProcessFormat.RESIZE
    )
    
    return model, postprocessor


def preprocess_image(image_path: str, target_size: int = 640):
    """预处理图像 - 与训练时保持一致"""
    # 读取图像（BGR格式）
    image_bgr = cv2.imread(str(image_path))
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
    
    # ⚠️ 关键修复：训练时图像未归一化！
    img_tensor = padded_image  # 保持[0, 255]范围，与训练时一致
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


def postprocess_outputs(outputs, postprocessor, meta, conf_threshold=0.3, target_size=640, device='cuda', verbose=False):
    """后处理模型输出
    
    Args:
        outputs: 模型输出
        postprocessor: 后处理器
        meta: 包含原始尺寸、padding等信息的字典
        conf_threshold: 置信度阈值
        target_size: 目标尺寸（默认640）
        device: 设备
        verbose: 是否打印调试信息（默认False，避免训练时输出过多）
    
    Returns:
        labels, boxes, scores: 检测结果
    """
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
    
    # 调试信息：检查坐标范围和置信度（仅在verbose模式下打印）
    if verbose and len(boxes) > 0:
        print(f"  检测到 {len(boxes)} 个候选框，坐标范围: x=[{boxes[:, 0].min():.1f}, {boxes[:, 2].max():.1f}], y=[{boxes[:, 1].min():.1f}, {boxes[:, 3].max():.1f}], 原始图像尺寸: {orig_w}x{orig_h}")
        print(f"  置信度范围: [{scores.min():.4f}, {scores.max():.4f}], 阈值: {conf_threshold:.4f}, 最大10个置信度: {np.sort(scores)[-10:][::-1]}")
    
    # 过滤低置信度
    mask = scores >= conf_threshold
    labels = labels[mask]
    boxes = boxes[mask]
    scores = scores[mask]
    
    if verbose:
        if len(labels) > 0:
            print(f"  置信度过滤后: {len(labels)} 个目标，坐标范围: x=[{boxes[:, 0].min():.1f}, {boxes[:, 2].max():.1f}], y=[{boxes[:, 1].min():.1f}, {boxes[:, 3].max():.1f}]")
        elif len(boxes) > 0:
            print(f"  警告: 所有 {len(boxes)} 个候选框都被置信度阈值 {conf_threshold:.4f} 过滤掉了")
    
    return labels, boxes, scores


def draw_boxes(image, labels, boxes, scores, class_names=None, colors=None):
    """在图像上绘制预测框
    
    Args:
        image: BGR格式的图像（numpy数组）
        labels: 类别标签数组
        boxes: 边界框数组 [N, 4] (x1, y1, x2, y2)
        scores: 置信度数组
        class_names: 类别名称列表（默认使用全局CLASS_NAMES）
        colors: 颜色列表（默认使用全局COLORS）
    
    Returns:
        绘制了检测框的图像
    """
    if len(labels) == 0:
        return image
    
    if class_names is None:
        class_names = CLASS_NAMES
    if colors is None:
        colors = COLORS
    
    for label, box, score in zip(labels, boxes, scores):
        x1, y1, x2, y2 = box.astype(int)
        
        # 确保坐标在图像范围内
        x1 = max(0, min(x1, image.shape[1] - 1))
        y1 = max(0, min(y1, image.shape[0] - 1))
        x2 = max(0, min(x2, image.shape[1] - 1))
        y2 = max(0, min(y2, image.shape[0] - 1))
        
        # 绘制边界框
        color = colors[label]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签和置信度
        class_name = class_names[label]
        label_text = f"{class_name}: {score:.2f}"
        
        # 计算文本位置
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_y = max(text_h + 4, y1)
        cv2.rectangle(image, (x1, text_y - text_h - 4), (x1 + text_w, text_y), color, -1)
        cv2.putText(image, label_text, (x1, text_y - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image


def inference_from_preprocessed_image(image_tensor, model, postprocessor, orig_image_path, 
                                     conf_threshold=0.3, target_size=640, device='cuda', 
                                     class_names=None, colors=None, verbose=False):
    """从已预处理的图像tensor进行推理（用于训练时）
    
    Args:
        image_tensor: 已预处理的图像tensor [1, 3, H, W]
        model: 模型
        postprocessor: 后处理器
        orig_image_path: 原始图像路径（用于读取原始图像和构建meta）
        conf_threshold: 置信度阈值
        target_size: 目标尺寸
        device: 设备
        class_names: 类别名称列表
        colors: 颜色列表
        verbose: 是否打印调试信息
    
    Returns:
        result_image: 绘制了检测框的图像（BGR格式），如果没有检测结果则返回None
    """
    # 读取原始图像
    orig_image_bgr = cv2.imread(str(orig_image_path))
    if orig_image_bgr is None:
        return None
    
    orig_h, orig_w = orig_image_bgr.shape[:2]
    
    # 构建meta字典（与preprocess_image返回格式一致）
    scale = min(target_size / orig_h, target_size / orig_w)
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)
    new_h = ((new_h + 31) // 32) * 32
    new_w = ((new_w + 31) // 32) * 32
    new_h = min(new_h, target_size)
    new_w = min(new_w, target_size)
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    
    meta = {
        'orig_size': torch.tensor([[orig_h, orig_w]]),  # [1, 2] format: [h, w]
        'pad_h': pad_h,
        'pad_w': pad_w,
        'scale': scale,
        'new_h': new_h,
        'new_w': new_w
    }
    
    # 推理
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # 后处理
    labels, boxes, scores = postprocess_outputs(
        outputs, postprocessor, meta, conf_threshold, target_size, device, verbose=verbose
    )
    
    if len(labels) == 0:
        return None
    
    # 绘制结果
    result_image = draw_boxes(orig_image_bgr.copy(), labels, boxes, scores, class_names, colors)
    return result_image


def process_single_image(image_path: Path, model, postprocessor, output_dir: Path, 
                        conf_threshold: float, device: str, target_size: int = 640):
    """处理单张图像"""
    try:
        # 预处理图像
        img_tensor, orig_image, meta = preprocess_image(str(image_path), target_size)
        img_tensor = img_tensor.to(device)
        
        # 推理
        with torch.no_grad():
            outputs = model(img_tensor)
        
        # 后处理（verbose=True 用于批量推理时显示信息）
        labels, boxes, scores = postprocess_outputs(
            outputs, postprocessor, meta, conf_threshold, target_size, device, verbose=True
        )
        
        # 绘制结果
        result_image = draw_boxes(orig_image.copy(), labels, boxes, scores)
        
        # 保存结果
        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), result_image)
        
        return len(labels), True, None
    except Exception as e:
        return 0, False, str(e)


def batch_inference(image_dir: str, config_path: str, checkpoint_path: str, 
                   output_dir: str = None, conf_threshold: float = 0.3, 
                   device: str = "cuda", max_images: int = None,
                   image_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp')):
    """批量推理"""
    # 加载模型
    print(f"加载模型: {checkpoint_path}")
    model, postprocessor = load_model(config_path, checkpoint_path, device)
    print("✓ 模型加载完成")
    
    # 设置输入和输出目录
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise ValueError(f"图像目录不存在: {image_dir}")
    
    if output_dir is None:
        output_dir = image_dir.parent / f"{image_dir.name}_results"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 获取所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_dir.glob(f"*{ext}")))
        image_files.extend(list(image_dir.glob(f"*{ext.upper()}")))
    
    if len(image_files) == 0:
        print(f"警告: 在 {image_dir} 中未找到图像文件")
        return
    
    # 限制处理数量
    total_images = len(image_files)
    if max_images is not None and max_images > 0:
        image_files = image_files[:max_images]
        print(f"找到 {total_images} 张图像，将处理前 {len(image_files)} 张")
    else:
        print(f"找到 {len(image_files)} 张图像")
    
    # 批量处理
    total_detections = 0
    success_count = 0
    failed_images = []
    
    for image_path in tqdm(image_files, desc="处理图像"):
        num_detections, success, error = process_single_image(
            image_path, model, postprocessor, output_dir, 
            conf_threshold, device
        )
        
        if success:
            total_detections += num_detections
            success_count += 1
        else:
            failed_images.append((image_path.name, error))
    
    # 打印统计信息
    print("\n" + "="*50)
    print("处理完成!")
    print(f"成功处理: {success_count}/{len(image_files)} 张图像")
    print(f"总检测数: {total_detections} 个目标")
    if failed_images:
        print(f"失败: {len(failed_images)} 张图像")
        for img_name, error in failed_images[:5]:  # 只显示前5个错误
            print(f"  - {img_name}: {error}")
        if len(failed_images) > 5:
            print(f"  ... 还有 {len(failed_images) - 5} 个错误")
    print(f"结果保存在: {output_dir}")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RT-DETR 批量推理脚本")
    parser.add_argument("--image_dir", type=str, required=True, 
                       help="输入图像目录路径")
    parser.add_argument("--config", type=str, required=True, 
                       help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="模型checkpoint路径")
    parser.add_argument("--output_dir", type=str, default=None, 
                       help="输出图像目录路径（默认：输入目录_results）")
    parser.add_argument("--conf", type=float, default=0.3, 
                       help="置信度阈值")
    parser.add_argument("--device", type=str, default="cuda", 
                       help="设备 (cuda/cpu)")
    parser.add_argument("--max_images", type=int, default=None,
                       help="最大处理图像数量（默认：处理所有图像）")
    
    args = parser.parse_args()
    
    batch_inference(
        args.image_dir,
        args.config,
        args.checkpoint,
        args.output_dir,
        args.conf,
        args.device,
        args.max_images
    )

