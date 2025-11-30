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

# 类别名称（8类）
CLASS_NAMES = [
    "Car", "Truck", "Van", "Bus", "Pedestrian", 
    "Cyclist", "Motorcyclist", "Trafficcone"
]
COLORS = [
    (255, 0, 0),      # Car - 红色
    (0, 255, 0),      # Truck - 绿色
    (255, 128, 0),    # Van - 橙色
    (0, 0, 255),      # Bus - 蓝色
    (255, 255, 0),    # Pedestrian - 黄色
    (255, 0, 255),    # Cyclist - 品红
    (0, 255, 255),    # Motorcyclist - 青色
    (128, 128, 128),  # Trafficcone - 灰色
]


def load_model(config_path: str, checkpoint_path: str, device: str = "cuda"):
    """加载模型和权重"""
    RTDETRTrainer, _, DetDETRPostProcessor, BoxProcessFormat = _import_modules()
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建训练器以构建模型
    trainer = RTDETRTrainer(config)
    
    # 创建一个简单的logger（推理时不需要日志，只需要模型能创建）
    if trainer.logger is None:
        class SimpleLogger:
            def info(self, msg): pass  # 什么都不做
        trainer.logger = SimpleLogger()
    
    model = trainer.create_model()
    
    # 加载checkpoint
    # 兼容 PyTorch 2.6+ (weights_only=True by default)
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        # 旧版本 PyTorch 不支持 weights_only 参数
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
    
    # 从配置文件读取 num_queries（模型实际生成的查询数量）
    num_queries = config.get('model', {}).get('num_queries', 300)
    
    # 创建后处理器（使用RESIZE模式，然后手动处理padding和缩放）
    # num_top_queries 应该使用配置中的 num_queries，因为模型只生成了这么多查询
    postprocessor = DetDETRPostProcessor(
        num_classes=8,
        use_focal_loss=True,
        num_top_queries=num_queries,  # 使用配置文件中的 num_queries
        box_process_format=BoxProcessFormat.RESIZE
    )
    
    return model, postprocessor


def preprocess_image(image_path: str, target_size: int = 1280):
    """
    预处理图像 - 严格对齐 Phase 2 验证集逻辑
    逻辑：Resize(short=720, max=1280) -> Normalize -> Top-Left Pad stride 32
    """
    # 1. 读取图像
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # BGR -> RGB
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]
    
    # 2. 智能缩放 (Rectangular Resize)
    # 目标：短边720，长边限制1280 (对应 target_size)
    # 计算缩放比例
    im_size_min = min(orig_h, orig_w)
    im_size_max = max(orig_h, orig_w)
    
    # 这里的逻辑复刻 T.Resize(size=720, max_size=1280)
    scale = 720 / float(im_size_min)
    
    # 如果缩放后长边超过 1280，则按长边缩放
    if round(scale * im_size_max) > target_size:
        scale = target_size / float(im_size_max)
    
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    
    # Bilinear 插值
    image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) # HWC->CHW
    image_tensor = torch.nn.functional.interpolate(
        image_tensor.unsqueeze(0), 
        size=(new_h, new_w), 
        mode='bilinear', 
        align_corners=False
    ).squeeze(0)
    
    # 3. 归一化 (Normalize)
    image_tensor = image_tensor / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    # 4. 左上角对齐填充 (Top-Left Padding to Stride 32)
    stride = 32
    # 向上取整到 32 的倍数
    padded_h = int(np.ceil(new_h / stride) * stride)
    padded_w = int(np.ceil(new_w / stride) * stride)
    
    # 创建画布 (填充 0，即黑边)
    # 注意：虽然 Normalize 后 0 不是黑色，但 RT-DETR 对 Padding 值不敏感，0 即可
    padded_image = torch.zeros(3, padded_h, padded_w, dtype=torch.float32)
    padded_image[:, :new_h, :new_w] = image_tensor  # 👈 关键：贴在左上角！
    
    # 添加 Batch 维度
    img_input = padded_image.unsqueeze(0) # [1, 3, H, W]
    
    # 构建 Meta 信息 (用于还原坐标)
    meta = {
        'orig_size': torch.tensor([[orig_h, orig_w]]),
        'scale': scale,        # 缩放比例
        'padded_h': padded_h,  # 输入网络的实际高
        'padded_w': padded_w,  # 输入网络的实际宽
        # 下面这两个其实不需要了，因为是 0，但为了兼容性保留
        'pad_h': 0, 
        'pad_w': 0
    }
    
    return img_input, image_bgr, meta


def postprocess_outputs(outputs, postprocessor, meta, conf_threshold=0.3, target_size=None, device='cuda', verbose=False):
    """后处理模型输出"""
    # 获取模型输出的设备
    if isinstance(outputs, dict) and 'pred_logits' in outputs:
        output_device = outputs['pred_logits'].device
    else:
        output_device = torch.device(device)

    # 1. 告诉 PostProcessor 画布有多大 (padded_w, padded_h)
    target_sizes = torch.tensor([[meta['padded_h'], meta['padded_w']]], device=output_device)
    
    # 2. 获取归一化还原后的坐标 (在 Padded Image 上的绝对坐标)
    # DetDETRPostProcessor 默认使用 orig_sizes 将 0-1 映射回像素
    # 这里我们要它映射回 "padded_image" 的像素坐标
    # 注意：必须使用关键字参数 orig_sizes，因为 DetDETRPostProcessor.forward 只接受 outputs 作为位置参数
    results = postprocessor(outputs, orig_sizes=target_sizes) 
    result = results[0]
    
    labels = result['labels'].cpu().numpy()
    boxes = result['boxes'].cpu().numpy() # [x1, y1, x2, y2]
    scores = result['scores'].cpu().numpy()
    
    # 3. 映射回原图
    # 因为是左上角对齐，原点 (0,0) 没变，所以只需要除以缩放比例 scale
    scale = meta['scale']
    
    boxes /= scale  # ✅ 核心修正：直接除以比例，无需减 padding
    
    # 4. 裁剪边界 (防止超出原图)
    orig_h, orig_w = meta['orig_size'][0].tolist()
    boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)
    
    # 5. 过滤低置信度
    mask = scores >= conf_threshold
    labels = labels[mask]
    boxes = boxes[mask]
    scores = scores[mask]
    
    # 调试信息：检查坐标范围和置信度（仅在verbose模式下打印）
    if verbose and len(boxes) > 0:
        print(f"  检测到 {len(boxes)} 个候选框")
        print(f"  置信度范围: [{scores.min():.4f}, {scores.max():.4f}], 阈值: {conf_threshold:.4f}")
    
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
    
    # 复用 preprocess_image 中的逻辑计算 meta
    # 注意：这里为了简单，重新计算一遍 scale 等信息
    # 实际应该从外部传入 meta，或者提取公共函数
    
    im_size_min = min(orig_h, orig_w)
    im_size_max = max(orig_h, orig_w)
    scale = 720 / float(im_size_min)
    if round(scale * im_size_max) > target_size:
        scale = target_size / float(im_size_max)
        
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    
    stride = 32
    padded_h = int(np.ceil(new_h / stride) * stride)
    padded_w = int(np.ceil(new_w / stride) * stride)
    
    pad_h = 0
    pad_w = 0
    
    meta = {
        'orig_size': torch.tensor([[orig_h, orig_w]]),
        'scale': scale,
        'padded_h': padded_h,
        'padded_w': padded_w,
        'pad_h': pad_h,
        'pad_w': pad_w
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
                        conf_threshold: float, device: str, target_size: int = 1280):
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
                   target_size: int = 1280,
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
    print(f"推理尺寸 (Max Size): {target_size}")
    
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
            conf_threshold, device, target_size
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
    parser.add_argument("--target_size", type=int, default=1280,
                       help="推理图像尺寸（长边限制，默认1280）")
    
    args = parser.parse_args()
    
    batch_inference(
        args.image_dir,
        args.config,
        args.checkpoint,
        args.output_dir,
        args.conf,
        args.device,
        args.max_images,
        args.target_size
    )
