#!/usr/bin/env python3
"""DSET 批量推理脚本 - 处理整个图像目录"""

import sys
import argparse
import yaml
import torch
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
import json
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
    # [CHANGE] 使用 DSETTrainer
    from train import DSETTrainer, create_backbone
    from src.nn.postprocessor.detr_postprocessor import DetDETRPostProcessor
    from src.nn.postprocessor.box_revert import box_revert, BoxProcessFormat
    return DSETTrainer, create_backbone, DetDETRPostProcessor, BoxProcessFormat

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
    """Load model and weights."""
    # [CHANGE] Import DSETTrainer
    DSETTrainer, _, DetDETRPostProcessor, BoxProcessFormat = _import_modules()
    
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Create trainer to build model
    trainer = DSETTrainer(config)
    
    # Simple logger for inference
    if trainer.logger is None:
        class SimpleLogger:
            def info(self, msg): pass
        trainer.logger = SimpleLogger()
    
    model = trainer._create_model()
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # [FIX] Enhanced weight loading logic
    if 'ema_state_dict' in checkpoint:
        print("  ✓ Found 'ema_state_dict', loading EMA weights")
        state_dict = checkpoint['ema_state_dict']
        if isinstance(state_dict, dict) and 'module' in state_dict:
            state_dict = state_dict['module']
    elif 'ema' in checkpoint and 'module' in checkpoint['ema']:
        print("  ✓ Found 'ema.module', loading EMA weights")
        state_dict = checkpoint['ema']['module']
    elif 'model_state_dict' in checkpoint:
        print("  ⚠ EMA not found, loading 'model_state_dict'")
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(device)
    
    # Enable pruning during inference by setting epoch to a large value (e.g., 100)
    # This ensures pruning is fully active and keep_ratio is at its target value
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'set_epoch'):
        # Get dset config from model section
        dset_config = config.get('model', {}).get('dset', {})
        warmup_epochs = dset_config.get('token_pruning_warmup_epochs', 10)
        
        # [FIX] Set epoch=100 to ensure pruning progress is 1.0 (fully active)
        # This aligns inference with the best model performance during training
        forced_epoch = 100
        model.encoder.set_epoch(forced_epoch)
        print(f"  ✓ Enabled token pruning for inference (epoch={forced_epoch}, warmup={warmup_epochs})")
    
    num_queries = config.get('model', {}).get('num_queries', 100)
    
    postprocessor = DetDETRPostProcessor(
        num_classes=8,
        use_focal_loss=True,
        num_top_queries=num_queries,
        box_process_format=BoxProcessFormat.RESIZE
    )
    
    return model, postprocessor


def inference_from_preprocessed_image(img_tensor, model, postprocessor, orig_image_path, 
                                      conf_threshold=0.3, target_size=640, device='cuda', 
                                      class_names=None, colors=None, verbose=False):
    """Inference interface called by Trainer."""
    orig_image = cv2.imread(str(orig_image_path))
    if orig_image is None:
        return None
    
    orig_h, orig_w = orig_image.shape[:2]
    input_h, input_w = img_tensor.shape[-2:]
    
    im_size_min = min(orig_h, orig_w)
    im_size_max = max(orig_h, orig_w)
    scale = 720 / float(im_size_min)
    if round(scale * im_size_max) > 1280:
        scale = 1280 / float(im_size_max)
        
    meta = {
        'orig_size': torch.tensor([[orig_h, orig_w]]),
        'padded_h': input_h,
        'padded_w': input_w,
        'scale': scale
    }
    
    with torch.no_grad():
        outputs = model(img_tensor)
        
    labels, boxes, scores = postprocess_outputs(
        outputs, postprocessor, meta, conf_threshold, target_size, device, verbose=verbose
    )
    
    pred_image = draw_boxes(orig_image.copy(), labels, boxes, scores, class_names, colors)
    cv2.putText(pred_image, "Prediction", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    gt_path = get_gt_annotation_path(Path(orig_image_path))
    if gt_path and class_names:
        gt_labels, gt_boxes, gt_scores = load_gt_boxes(gt_path, class_names)
        gt_image = draw_boxes(orig_image.copy(), gt_labels, gt_boxes, gt_scores, class_names, colors)
        cv2.putText(gt_image, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        result_image = cv2.hconcat([pred_image, gt_image])
    else:
        result_image = pred_image
        
    return result_image


def preprocess_image(image_path: str, target_size: int = 1280):
    """Preprocess image (PIL version)."""
    try:
        image_pil = Image.open(str(image_path)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to read image: {image_path}, Error: {e}")
    
    orig_w, orig_h = image_pil.size
    im_size_min = min(orig_h, orig_w)
    im_size_max = max(orig_h, orig_w)
    target_short = 720
    scale = target_short / float(im_size_min)
    if round(scale * im_size_max) > target_size:
        scale = target_size / float(im_size_max)
    
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    
    resized_pil = image_pil.resize((new_w, new_h), resample=Image.BILINEAR)
    image_tensor = T.functional.to_tensor(resized_pil) 
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    stride = 32
    padded_h = int(np.ceil(new_h / stride) * stride)
    padded_w = int(np.ceil(new_w / stride) * stride)
    
    padded_image = torch.zeros(3, padded_h, padded_w, dtype=torch.float32)
    padded_image[:, :new_h, :new_w] = image_tensor
    img_input = padded_image.unsqueeze(0)
    
    image_bgr_vis = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    meta = {
        'orig_size': torch.tensor([[orig_h, orig_w]]),
        'scale': scale,
        'padded_h': padded_h,
        'padded_w': padded_w,
        'scale_h': scale,
        'scale_w': scale,
        'pad_h': 0, 'pad_w': 0
    }
    
    return img_input, image_bgr_vis, meta


def postprocess_outputs(outputs, postprocessor, meta, conf_threshold=0.3, target_size=None, device='cuda', verbose=False):
    """后处理模型输出"""
    if isinstance(outputs, dict) and 'pred_logits' in outputs:
        output_device = outputs['pred_logits'].device
    else:
        output_device = torch.device(device)
    
    # [FIX] 使用 [padded_w, padded_h]
    target_sizes = torch.tensor([[meta['padded_w'], meta['padded_h']]], device=output_device)
    
    results = postprocessor(outputs, orig_sizes=target_sizes) 
    result = results[0]
    
    labels = result['labels'].cpu().numpy()
    boxes = result['boxes'].cpu().numpy()
    scores = result['scores'].cpu().numpy()
    
    scale = meta['scale']
    boxes /= scale
    
    orig_h, orig_w = meta['orig_size'][0].tolist()
    boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)
    
    mask = scores >= conf_threshold
    labels = labels[mask]
    boxes = boxes[mask]
    scores = scores[mask]
    
    if verbose and len(boxes) > 0:
        print(f"  检测到 {len(boxes)} 个候选框")
        print(f"  置信度范围: [{scores.min():.4f}, {scores.max():.4f}], 阈值: {conf_threshold:.4f}")
    
    return labels, boxes, scores


def draw_boxes(image, labels, boxes, scores, class_names=None, colors=None):
    """在图像上绘制预测框"""
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
        x1 = max(0, min(x1, image.shape[1] - 1))
        y1 = max(0, min(y1, image.shape[0] - 1))
        x2 = max(0, min(x2, image.shape[1] - 1))
        y2 = max(0, min(y2, image.shape[0] - 1))
        
        color = colors[label]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        class_name = class_names[label]
        label_text = f"{class_name}: {score:.2f}"
        
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_y = max(text_h + 4, y1)
        cv2.rectangle(image, (x1, text_y - text_h - 4), (x1 + text_w, text_y), color, -1)
        cv2.putText(image, label_text, (x1, text_y - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image


def get_gt_annotation_path(image_path: Path):
    """尝试根据图像路径推断标注文件路径"""
    try:
        if image_path.parent.name == 'image':
            data_root = image_path.parent.parent
            json_path = data_root / 'annotations' / 'camera' / f"{image_path.stem}.json"
            if json_path.exists():
                return json_path
            json_path = data_root / 'infrastructure-side' / 'annotations' / 'camera' / f"{image_path.stem}.json"
            if json_path.exists():
                return json_path
        
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
    scores = []
    name_to_id = {name: i for i, name in enumerate(class_names_list)}
    merge_map = {"Barrowlist": "Cyclist"}
    ignore_classes = ["PedestrianIgnore", "CarIgnore", "OtherIgnore", "Unknown_movable", "Unknown_unmovable"]
    
    if isinstance(data, list):
        for ann in data:
            if 'type' not in ann or '2d_box' not in ann:
                continue
            cat_name = ann['type']
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
                        conf_threshold: float, device: str, target_size: int = 1280):
    """处理单张图像"""
    try:
        img_tensor, orig_image, meta = preprocess_image(str(image_path), target_size)
        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
        
        labels, boxes, scores = postprocess_outputs(
            outputs, postprocessor, meta, conf_threshold, target_size, device, verbose=True
        )
        
        # 绘制预测结果 (左)
        pred_image = draw_boxes(orig_image.copy(), labels, boxes, scores)
        cv2.putText(pred_image, "Prediction", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # 尝试加载 GT (右)
        gt_path = get_gt_annotation_path(image_path)
        if gt_path:
            gt_labels, gt_boxes, gt_scores = load_gt_boxes(gt_path, CLASS_NAMES)
            gt_image = draw_boxes(orig_image.copy(), gt_labels, gt_boxes, gt_scores)
            cv2.putText(gt_image, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            final_image = cv2.hconcat([pred_image, gt_image])
        else:
            final_image = pred_image
        
        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), final_image)
        
        return len(labels), True, None
    except Exception as e:
        return 0, False, str(e)


def batch_inference(image_dir: str, config_path: str, checkpoint_path: str, 
                   output_dir: str = None, conf_threshold: float = 0.3, 
                   device: str = "cuda", max_images: int = None,
                   target_size: int = 1280,
                   image_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp')):
    """批量推理"""
    print(f"加载模型: {checkpoint_path}")
    model, postprocessor = load_model(config_path, checkpoint_path, device)
    print("✓ 模型加载完成")
    
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
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_dir.glob(f"*{ext}")))
        image_files.extend(list(image_dir.glob(f"*{ext.upper()}")))
    
    if len(image_files) == 0:
        print(f"警告: 在 {image_dir} 中未找到图像文件")
        return
    
    total_images = len(image_files)
    if max_images is not None and max_images > 0:
        image_files = image_files[:max_images]
        print(f"找到 {total_images} 张图像，将处理前 {len(image_files)} 张")
    else:
        print(f"找到 {len(image_files)} 张图像")
    
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
    
    print("\n" + "="*50)
    print("处理完成!")
    print(f"成功处理: {success_count}/{len(image_files)} 张图像")
    print(f"总检测数: {total_detections} 个目标")
    if failed_images:
        print(f"失败: {len(failed_images)} 张图像")
        for img_name, error in failed_images[:5]:
            print(f"  - {img_name}: {error}")
    print(f"结果保存在: {output_dir}")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSET 批量推理脚本")
    parser.add_argument("--image_dir", type=str, required=True, help="输入图像目录路径")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型checkpoint路径")
    parser.add_argument("--output_dir", type=str, default=None, help="输出图像目录路径")
    parser.add_argument("--conf", type=float, default=0.3, help="置信度阈值")
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda/cpu)")
    parser.add_argument("--max_images", type=int, default=None, help="最大处理图像数量")
    parser.add_argument("--target_size", type=int, default=1280, help="推理图像尺寸")
    
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
