#!/usr/bin/env python3
"""RT-DETR 批量推理脚本 - 处理整个图像目录"""

import sys
from pathlib import Path
_cas_detr_root = Path(__file__).resolve().parent
if str(_cas_detr_root) not in sys.path:
    sys.path.insert(0, str(_cas_detr_root))

import argparse
import yaml
import torch
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
import torch.nn.functional as F_nn
import json

from src.data.transforms.letterbox_geom import (
    compute_letterbox_layout,
    build_letterbox_meta_for_postprocess,
)


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
    from train import CaS_DETRRTDETR
    from src.nn.postprocessor.detr_postprocessor import DetDETRPostProcessor
    from src.nn.postprocessor.box_revert import BoxProcessFormat
    return CaS_DETRRTDETR, DetDETRPostProcessor, BoxProcessFormat

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
    CaS_DETRRTDETR, DetDETRPostProcessor, BoxProcessFormat = _import_modules()
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model_cfg = config.get("model", {})
    encoder_cfg = model_cfg.get("encoder", {})
    ablation_cfg = model_cfg.get("ablation", {})
    cas_detr_cfg = model_cfg.get("cas_detr", {})
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    enable_cas_predictor = bool(ablation_cfg.get("enable_cas_predictor", True))
    enable_decoder_moe = bool(ablation_cfg.get("enable_decoder_moe", True))
    token_keep_ratio = cas_detr_cfg.get(
        "token_keep_ratio", 1.0 if not enable_cas_predictor else 0.7
    )

    model = CaS_DETRRTDETR(
        hidden_dim=model_cfg["hidden_dim"],
        decoder_hidden_dim=model_cfg.get("decoder_hidden_dim"),
        num_queries=model_cfg["num_queries"],
        top_k=model_cfg.get("top_k", 1),
        backbone_type=model_cfg["backbone"],
        num_decoder_layers=model_cfg["num_decoder_layers"],
        encoder_in_channels=encoder_cfg["in_channels"],
        encoder_expansion=encoder_cfg["expansion"],
        num_experts=model_cfg.get("num_experts", 6),
        num_encoder_layers=encoder_cfg.get("num_encoder_layers", 1),
        use_encoder_idx=encoder_cfg.get("use_encoder_idx", [1, 2]),
        token_keep_ratio=token_keep_ratio,
        enable_cas_predictor=enable_cas_predictor,
        enable_decoder_moe=enable_decoder_moe,
        decoder_moe_balance_weight=training_cfg.get("decoder_moe_balance_weight"),
        moe_balance_warmup_epochs=training_cfg.get("moe_balance_warmup_epochs", 0),
        use_cass=cas_detr_cfg.get("use_cass", False),
        cass_loss_weight=cas_detr_cfg.get("cass_loss_weight", 0.2),
        cass_expansion_ratio=cas_detr_cfg.get("cass_expansion_ratio", 0.3),
        cass_min_size=cas_detr_cfg.get("cass_min_size", 1.0),
        cass_loss_type=cas_detr_cfg.get("cass_loss_type", "vfl"),
        cass_focal_alpha=cas_detr_cfg.get("cass_focal_alpha", 0.75),
        cass_focal_beta=cas_detr_cfg.get("cass_focal_beta", 2.0),
        moe_noise_std=model_cfg.get("moe_noise_std", 0.1),
        num_classes=int(data_cfg.get("num_classes", len(CLASS_NAMES))),
    )
    
    # 加载checkpoint
    # 兼容 PyTorch 2.6+ (weights_only=True by default)
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        # 旧版本 PyTorch 不支持 weights_only 参数
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # [FIX] 增强的权重加载逻辑
    if 'ema_state_dict' in checkpoint:
        print("  ✓ 检测到 'ema_state_dict'，加载 EMA 权重")
        state_dict = checkpoint['ema_state_dict']
        # EMA state dict 可能包含 'module' 键 (如果使用 ModelEMA 类保存)
        if isinstance(state_dict, dict) and 'module' in state_dict:
            state_dict = state_dict['module']
    elif 'ema' in checkpoint and 'module' in checkpoint['ema']:
        print("  ✓ 检测到 'ema.module'，加载 EMA 权重")
        state_dict = checkpoint['ema']['module']
    elif 'model_state_dict' in checkpoint:
        print("  ⚠ 未找到 EMA，加载普通 'model_state_dict' 权重")
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(device)
    
    # 从配置文件读取 num_queries（模型实际生成的查询数量）
    num_queries = config.get('model', {}).get('num_queries', 100)
    
    # 创建后处理器（使用RESIZE模式，然后手动处理padding和缩放）
    # num_top_queries 应该使用配置中的 num_queries，因为模型只生成了这么多查询
    postprocessor = DetDETRPostProcessor(
        num_classes=8,
        use_focal_loss=True,
        num_top_queries=num_queries,  # 使用配置文件中的 num_queries
        box_process_format=BoxProcessFormat.RESIZE
    )
    
    return model, postprocessor


def inference_from_preprocessed_image(img_tensor, model, postprocessor, orig_image_path,
                                      conf_threshold=0.3, target_size=640, device='cuda',
                                      class_names=None, colors=None, verbose=False,
                                      target_dict=None):
    """
    供 Trainer 调用的推理接口。
    img_tensor: [1, 3, H, W] 已按与验证 DataLoader 相同方式归一化。
    target_dict: 可选，来自 collate 的 target，用于精确 letterbox meta（推荐）。
    """
    orig_image = cv2.imread(str(orig_image_path))
    if orig_image is None:
        return None

    orig_h, orig_w = orig_image.shape[:2]
    input_h, input_w = int(img_tensor.shape[-2]), int(img_tensor.shape[-1])

    if target_dict is not None:
        meta = build_letterbox_meta_for_postprocess(target_dict, input_h, input_w)
    else:
        L = compute_letterbox_layout(orig_w, orig_h, max(input_h, input_w))
        meta = {
            'orig_size': torch.tensor([[orig_h, orig_w]]),
            'scale': L['scale'],
            'pad_left': float(L['pad_left']),
            'pad_top': float(L['pad_top']),
            'new_w': int(L['new_w']),
            'new_h': int(L['new_h']),
            'padded_h': input_h,
            'padded_w': input_w,
            'letterbox_uniform': True,
        }
    
    # 推理
    with torch.no_grad():
        outputs = model(img_tensor)
        
    # 后处理
    labels, boxes, scores = postprocess_outputs(
        outputs, postprocessor, meta, conf_threshold, target_size, device, verbose=verbose
    )
    
    # [NEW] 尝试加载并绘制 GT (如果能找到对应json)
    # 预测图
    pred_image = draw_boxes(orig_image.copy(), labels, boxes, scores, class_names, colors)
    cv2.putText(pred_image, "Prediction", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # GT图
    gt_path = get_gt_annotation_path(Path(orig_image_path))
    if gt_path and class_names:
        gt_labels, gt_boxes, gt_scores = load_gt_boxes(gt_path, class_names)
        gt_image = draw_boxes(orig_image.copy(), gt_labels, gt_boxes, gt_scores, class_names, colors)
        cv2.putText(gt_image, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # 拼接
        result_image = cv2.hconcat([pred_image, gt_image])
    else:
        result_image = pred_image
        
    return result_image


def preprocess_image(
    image_path: str,
    target_size: int = 640,
    letterbox_fill: int = 0,
):
    """
    预处理图像：与训练时 ``build_square_input_transform`` 一致（letterbox）、ImageNet 归一化。
    """
    try:
        image_pil = Image.open(str(image_path)).convert("RGB")
    except Exception as e:
        raise ValueError(f"无法读取图像: {image_path}, 错误: {e}")

    orig_w, orig_h = image_pil.size
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image_bgr_vis = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    L = compute_letterbox_layout(orig_w, orig_h, target_size)
    resized_pil = image_pil.resize((L["new_w"], L["new_h"]), resample=Image.BILINEAR)
    image_tensor = T.functional.to_tensor(resized_pil)
    fill_v = float(letterbox_fill) / 255.0
    padded = F_nn.pad(
        image_tensor,
        (L["pad_left"], L["pad_right"], L["pad_top"], L["pad_bottom"]),
        mode="constant",
        value=fill_v,
    )
    image_tensor = (padded - mean) / std
    img_input = image_tensor.unsqueeze(0)

    meta = {
        "orig_size": torch.tensor([[orig_h, orig_w]]),
        "scale": L["scale"],
        "pad_left": float(L["pad_left"]),
        "pad_top": float(L["pad_top"]),
        "new_w": int(L["new_w"]),
        "new_h": int(L["new_h"]),
        "padded_h": int(L["padded_h"]),
        "padded_w": int(L["padded_w"]),
        "letterbox_uniform": True,
    }
    return img_input, image_bgr_vis, meta


def postprocess_outputs(outputs, postprocessor, meta, conf_threshold=0.3, target_size=None, device='cuda', verbose=False):
    """后处理模型输出"""
    # 获取模型输出的设备
    if isinstance(outputs, dict) and 'pred_logits' in outputs:
        output_device = outputs['pred_logits'].device
    else:
        output_device = torch.device(device)
    
    # 1. 告诉 PostProcessor 画布有多大 (padded_w, padded_h)
    # [FIX] 调换顺序，使用 [padded_w, padded_h] 以匹配 box_revert 的期望
    target_sizes = torch.tensor([[meta['padded_w'], meta['padded_h']]], device=output_device)
    
    # 2. 获取归一化还原后的坐标 (在 Padded Image 上的绝对坐标)
    # DetDETRPostProcessor 默认使用 orig_sizes 将 0-1 映射回像素
    # 这里我们要它映射回 "padded_image" 的像素坐标
    # 注意：必须使用关键字参数 orig_sizes，因为 DetDETRPostProcessor.forward 只接受 outputs 作为位置参数
    results = postprocessor(outputs, orig_sizes=target_sizes) 
    result = results[0]
    
    labels = result['labels'].cpu().numpy()
    boxes = result['boxes'].cpu().numpy() # [x1, y1, x2, y2]
    scores = result['scores'].cpu().numpy()
    
    # 3. 映射回原图（letterbox：先减居中 pad，再除以统一 scale）
    if meta.get('letterbox_uniform', True):
        pad_left = float(meta.get('pad_left', 0.0))
        pad_top = float(meta.get('pad_top', 0.0))
        scale = float(meta.get('scale', 1.0))
        boxes[:, [0, 2]] -= pad_left
        boxes[:, [1, 3]] -= pad_top
        boxes[:, [0, 2]] /= scale
        boxes[:, [1, 3]] /= scale
    elif 'scale_h' in meta and 'scale_w' in meta:
        boxes[:, [0, 2]] /= meta['scale_w']
        boxes[:, [1, 3]] /= meta['scale_h']
    else:
        scale = float(meta.get('scale', 1.0))
        boxes /= scale
    
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
    # 确保image是连续的内存块 (避免 hconcat 报错)
    if not image.flags['C_CONTIGUOUS']:
        image = np.ascontiguousarray(image)

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


def get_gt_annotation_path(image_path: Path):
    """尝试根据图像路径推断标注文件路径"""
    # 假设结构: .../image/xxxxx.jpg -> .../annotations/camera/xxxxx.json
    try:
        # 策略1: 检查父目录是否是 image, 且有 annotations 兄弟目录
        if image_path.parent.name == 'image':
            data_root = image_path.parent.parent
            # DAIR-V2X 标准结构
            json_path = data_root / 'annotations' / 'camera' / f"{image_path.stem}.json"
            if json_path.exists():
                return json_path
            
            # 或者是单路侧数据集 (infrastructure-side) 可能的变体
            json_path = data_root / 'infrastructure-side' / 'annotations' / 'camera' / f"{image_path.stem}.json"
            if json_path.exists():
                return json_path
        
        # 策略2: 检查是否有同名的 json 文件 (flat 结构)
        json_path = image_path.with_suffix('.json')
        if json_path.exists():
            return json_path
            
    except Exception:
        pass
    return None


def load_gt_boxes(json_path, class_names_list):
    """加载GT框"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    boxes = []
    labels = []
    scores = [] # GT confidence = 1.0
    
    # 建立 name -> id 映射
    name_to_id = {name: i for i, name in enumerate(class_names_list)}
    
    # 特殊映射 (参考 dairv2x_detection.py)
    merge_map = {"Barrowlist": "Cyclist"}
    ignore_classes = ["PedestrianIgnore", "CarIgnore", "OtherIgnore", "Unknown_movable", "Unknown_unmovable"]
    
    # 只有当data是列表时才遍历（DAIR-V2X格式）
    if isinstance(data, list):
        for ann in data:
            if 'type' not in ann or '2d_box' not in ann:
                continue
                
            cat_name = ann['type']
            
            # 处理映射
            if cat_name in merge_map:
                cat_name = merge_map[cat_name]
                
            if cat_name in ignore_classes:
                continue
                
            if cat_name not in name_to_id:
                continue
                
            class_id = name_to_id[cat_name]
            
            bbox = ann['2d_box']
            x1, y1, x2, y2 = float(bbox['xmin']), float(bbox['ymin']), float(bbox['xmax']), float(bbox['ymax'])
            
            boxes.append([x1, y1, x2, y2])
            labels.append(class_id)
            scores.append(1.0)
            
    return np.array(labels), np.array(boxes), np.array(scores)


def process_single_image(image_path: Path, model, postprocessor, output_dir: Path, 
                        conf_threshold: float, device: str, target_size: int = 640,
                        letterbox_fill: int = 0):
    """处理单张图像"""
    try:
        # 预处理图像
        img_tensor, orig_image, meta = preprocess_image(
            str(image_path), target_size, letterbox_fill=letterbox_fill
        )
        img_tensor = img_tensor.to(device)
        
        # 推理
        with torch.no_grad():
            outputs = model(img_tensor)
        
        # 后处理（verbose=True 用于批量推理时显示信息）
        labels, boxes, scores = postprocess_outputs(
            outputs, postprocessor, meta, conf_threshold, target_size, device, verbose=True
        )
        
        # [NEW] 绘制预测结果并尝试加载 GT 拼接对比
        
        # 1. 绘制预测图
        pred_image = draw_boxes(orig_image.copy(), labels, boxes, scores)
        # 添加标题
        cv2.putText(pred_image, "Prediction", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # 2. 尝试加载 GT
        gt_path = get_gt_annotation_path(image_path)
        if gt_path:
            gt_labels, gt_boxes, gt_scores = load_gt_boxes(gt_path, CLASS_NAMES)
            gt_image = draw_boxes(orig_image.copy(), gt_labels, gt_boxes, gt_scores)
            cv2.putText(gt_image, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 拼接图片 (左右)
            final_image = cv2.hconcat([pred_image, gt_image])
        else:
            final_image = pred_image
        
        # 保存结果
        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), final_image)
        
        return len(labels), True, None
    except Exception as e:
        return 0, False, str(e)


def batch_inference(image_dir: str, config_path: str, checkpoint_path: str, 
                   output_dir: str = None, conf_threshold: float = 0.3, 
                   device: str = "cuda", max_images: int = None,
                   target_size: int = 640,
                   image_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp'),
                   letterbox_fill: int = None):
    """批量推理"""
    if letterbox_fill is None:
        with open(config_path, "r", encoding="utf-8") as f:
            _cfg = yaml.safe_load(f)
        _aug = _cfg.get("augmentation") or {}
        _r = _aug.get("resize") or {}
        letterbox_fill = int(_r.get("letterbox_fill", _aug.get("letterbox_fill", 0)))

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
    print(f"推理尺寸: {target_size}, letterbox")
    
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
            conf_threshold, device, target_size,
            letterbox_fill=letterbox_fill,
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
    parser.add_argument("--target_size", type=int, default=640,
                       help="推理方形边长（默认 640，与 target_size 对齐）")
    parser.add_argument(
        "--letterbox_fill",
        type=int,
        default=None,
        help="letterbox 填充灰度 0-255；默认读取 augmentation.resize.letterbox_fill 或 letterbox_fill",
    )
    
    args = parser.parse_args()
    
    batch_inference(
        args.image_dir,
        args.config,
        args.checkpoint,
        args.output_dir,
        args.conf,
        args.device,
        args.max_images,
        args.target_size,
        letterbox_fill=args.letterbox_fill,
    )
