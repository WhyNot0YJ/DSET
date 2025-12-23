#!/usr/bin/env python3
"""
模型性能评估脚本 - 支持 DSET, RT-DETR, Deformable-DETR, YOLOv8, YOLOv10

功能：
1. 自动从 logs/ 目录查找最新的 best_model.pth 或使用指定的检查点
2. 使用 pycocotools 在验证集上运行 COCO 评估
3. 计算模型参数量和 FLOPs
4. 基于验证集样本测量性能（FPS 和延迟）
5. 所有性能测试在 batch_size=1 条件下进行（学术论文标准）

使用方法：
    python generate_benchmark_table.py --model_type dset
    python generate_benchmark_table.py --models_config models.json
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import time
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from io import StringIO
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

# 设置项目根目录
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
project_root = _project_root

# 尝试导入 thop
try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("警告: thop 未安装，将无法计算 FLOPs。请运行: pip install thop")


def _cuda_sync_if_available(device: str):
    """同步 CUDA（如果可用）"""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_checkpoint(checkpoint_path: str) -> dict:
    """加载检查点文件"""
    try:
        return torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location='cpu')


def _extract_state_dict(checkpoint: dict) -> dict:
    """从检查点中提取状态字典"""
    if 'ema_state_dict' in checkpoint:
        state_dict = checkpoint['ema_state_dict']
        if isinstance(state_dict, dict) and 'module' in state_dict:
            state_dict = state_dict['module']
        return state_dict
    elif 'ema' in checkpoint and 'module' in checkpoint['ema']:
        return checkpoint['ema']['module']
    elif 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        return checkpoint['model']
    return checkpoint


def get_model_info(model, input_size: Tuple[int, int, int, int] = (1, 3, 736, 1280), is_yolo: bool = False) -> Tuple[float, float]:
    """计算模型的参数量和 FLOPs"""
    # 计算参数量
    if is_yolo and hasattr(model, 'model'):
        pytorch_model = model.model
    else:
        pytorch_model = model
    total_params = sum(p.numel() for p in pytorch_model.parameters())
    params_m = total_params / 1e6
    
    # 计算 FLOPs
    flops_g = 0.0
    if is_yolo:
        try:
            from copy import deepcopy
            if hasattr(model, 'model'):
                pytorch_model = model.model
                h, w = input_size[2], input_size[3]
                imgsz = [h, w] if h != w else h
                try:
                    from ultralytics.utils.torch_utils import get_flops
                    flops_g = get_flops(pytorch_model, imgsz=imgsz)
                except (ImportError, AttributeError):
                    if HAS_THOP:
                        pytorch_model = pytorch_model.eval()
                        device = next(pytorch_model.parameters()).device
                        dummy_input = torch.randn(input_size).to(device)
                        flops, _ = profile(deepcopy(pytorch_model), inputs=(dummy_input,), verbose=False)
                        flops_g = flops / 1e9
        except Exception as e:
            print(f"  ⚠ YOLO FLOPs 计算失败: {e}")
    elif HAS_THOP:
        try:
            model.eval()
            device = next(model.parameters()).device
            dummy_input = torch.randn(input_size).to(device)
            flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
            flops_g = flops / 1e9
            print(f"  ✓ FLOPs: {flops_g:.2f} G")
        except Exception as e:
            print(f"  ⚠ FLOPs 计算失败: {e}")
    
    return params_m, flops_g


def measure_fps(model, input_size: Tuple[int, int, int, int] = (1, 3, 736, 1280),
                num_iter: int = 100, warmup_iter: int = 50,
                device: str = "cuda", is_yolo: bool = False) -> float:
    """测量模型的 FPS（使用 CUDA Events 精确计时，batch_size=1）"""
    model.eval()
    use_cuda_events = (device == "cuda" and torch.cuda.is_available())
    fps_input_size = (1, input_size[1], input_size[2], input_size[3])
    
    if is_yolo:
        model = model.to(device)
        yolo_size = max(input_size[2], input_size[3])
        dummy_input = torch.rand(1, 3, yolo_size, yolo_size).to(device).clamp(0.0, 1.0)
        
        for _ in range(warmup_iter):
            _ = model(dummy_input, verbose=False)
        
        _cuda_sync_if_available(device)
        
        if use_cuda_events:
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            for _ in range(num_iter):
                _ = model(dummy_input, verbose=False)
            ender.record()
            torch.cuda.synchronize()
            elapsed_time = starter.elapsed_time(ender) / 1000.0
        else:
            start_time = time.time()
            for _ in range(num_iter):
                _ = model(dummy_input, verbose=False)
            _cuda_sync_if_available(device)
            elapsed_time = time.time() - start_time
    else:
        model = model.to(device)
        dummy_input = torch.randn(fps_input_size).to(device)
        
        with torch.no_grad():
            for _ in range(warmup_iter):
                _ = model(dummy_input)
        
        _cuda_sync_if_available(device)
        
        if use_cuda_events:
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            with torch.no_grad():
                for _ in range(num_iter):
                    _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            elapsed_time = starter.elapsed_time(ender) / 1000.0
        else:
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_iter):
                    _ = model(dummy_input)
            _cuda_sync_if_available(device)
            elapsed_time = time.time() - start_time
    
    return num_iter / elapsed_time if elapsed_time > 0 else 0.0


def load_dset_model(config_path: str, checkpoint_path: str, device: str = "cuda"):
    """加载 DSET 模型"""
    try:
        from experiments.dset.train import DSETTrainer
    except ImportError:
        dset_dir = Path(config_path).parent.parent
        if str(dset_dir) not in sys.path:
            sys.path.insert(0, str(dset_dir))
        from train import DSETTrainer
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if 'misc' not in config:
        config['misc'] = {}
    config['misc']['device'] = device
    
    trainer = DSETTrainer(config, config_file_path=str(config_path))
    model = trainer._create_model()
    
    checkpoint = _load_checkpoint(checkpoint_path)
    state_dict = _extract_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # 启用 token pruning - 强制设置 epoch=100 以跨越 warmup 阶段
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'set_epoch'):
        dset_config = config.get('model', {}).get('dset', {})
        warmup_epochs = dset_config.get('token_pruning_warmup_epochs', 10)
        target_keep_ratio = dset_config.get('token_keep_ratio', 1.0)
        
        # 强制 epoch=100 以确保剪枝完全激活（progress=1.0）
        forced_epoch = 100
        model.encoder.set_epoch(forced_epoch)
        
        # 验证剪枝状态
        if hasattr(model.encoder, 'token_pruners') and model.encoder.token_pruners:
            pruner = model.encoder.token_pruners[0]
            actual_keep_ratio = pruner.get_current_keep_ratio() if hasattr(pruner, 'get_current_keep_ratio') else None
            pruning_enabled = pruner.pruning_enabled if hasattr(pruner, 'pruning_enabled') else False
            print(f"  ✓ Token Pruning: epoch={forced_epoch}, warmup={warmup_epochs}")
            print(f"    - pruning_enabled: {pruning_enabled}")
            print(f"    - target_keep_ratio: {target_keep_ratio}")
            print(f"    - actual_keep_ratio: {actual_keep_ratio}")
        else:
            print(f"  ✓ 已启用 token pruning (epoch={forced_epoch})")
    
    return model


def load_rtdetr_model(config_path: str, checkpoint_path: str, device: str = "cuda"):
    """加载 RT-DETRv2 模型"""
    rtdetr_dir = Path(config_path).parent.parent
    if str(rtdetr_dir) not in sys.path:
        sys.path.insert(0, str(rtdetr_dir))
    from train import RTDETRTrainer
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if 'misc' not in config:
        config['misc'] = {}
    config['misc']['device'] = device
    
    trainer = RTDETRTrainer(config)
    if trainer.logger is None:
        class SimpleLogger:
            def info(self, msg): pass
        trainer.logger = SimpleLogger()
    
    model = trainer.create_model()
    checkpoint = _load_checkpoint(checkpoint_path)
    state_dict = _extract_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(device)
    
    return model


def load_deformable_detr_model(checkpoint_path: str, device: str = "cuda", config_path: str = None):
    """加载 Deformable-DETR 模型"""
    try:
        from mmengine.config import Config
        from mmdet.registry import MODELS
    except ImportError:
        raise ImportError("需要安装 mmengine 和 mmdet")

    # 关键：确保 mmdet 的所有模块都已注册到 mmengine Registry，否则会出现
    # "DetDataPreprocessor is not in the mmengine::model registry" 之类的错误。
    try:
        from mmdet.utils import register_all_modules
        try:
            # 新版 mmdet 推荐：同时初始化 default scope
            register_all_modules(init_default_scope=True)
        except TypeError:
            # 兼容旧版签名
            register_all_modules()
    except Exception:
        # 某些环境可能没有该工具函数，但正常 import mmdet 也会触发注册
        try:
            import mmdet  # noqa: F401
        except Exception:
            pass
    
    checkpoint = _load_checkpoint(checkpoint_path)
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    
    cfg = None
    # 优先使用显式提供的 config_path（更稳定、可复现）
    if config_path and os.path.exists(config_path):
        cfg = Config.fromfile(config_path)
    # 回退：尝试从 checkpoint meta 中恢复配置
    elif 'meta' in checkpoint and 'cfg' in checkpoint['meta']:
        meta_cfg = checkpoint['meta']['cfg']
        # 兼容：有些 mmdet/mmengine checkpoint 会把 cfg 保存为 dict；也有很多保存为 str（文件路径或文本内容）
        if isinstance(meta_cfg, dict):
            cfg = Config(meta_cfg)
        elif isinstance(meta_cfg, str):
            # 1) 如果是文件路径
            if os.path.exists(meta_cfg):
                cfg = Config.fromfile(meta_cfg)
            else:
                # 2) 可能是 python config 文本内容
                if hasattr(Config, 'fromstring'):
                    try:
                        cfg = Config.fromstring(meta_cfg, file_format='.py')
                    except TypeError:
                        # 兼容旧版本签名
                        cfg = Config.fromstring(meta_cfg)
                else:
                    # 3) 极端兼容：写入临时文件再 fromfile
                    import tempfile
                    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf-8') as f:
                        f.write(meta_cfg)
                        tmp_cfg_path = f.name
                    cfg = Config.fromfile(tmp_cfg_path)
    
    if cfg is None:
        raise FileNotFoundError("无法找到 Deformable-DETR 配置文件")
    
    model = MODELS.build(cfg.model)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(device)
    
    return model


def _ensure_yolo_checkpoint_path(checkpoint_path: str) -> str:
    """确保 YOLO 检查点路径使用 .pt 后缀"""
    checkpoint_path_obj = Path(checkpoint_path)
    
    if checkpoint_path_obj.suffix.lower() == '.pt':
        return str(checkpoint_path_obj)
    
    if checkpoint_path_obj.suffix.lower() == '.pth':
        pt_path = checkpoint_path_obj.with_suffix('.pt')
        if pt_path.exists():
            return str(pt_path)
        if not checkpoint_path_obj.exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path_obj}")
        
        try:
            pt_path.symlink_to(checkpoint_path_obj)
            return str(pt_path)
        except OSError:
            import shutil
            shutil.copy2(checkpoint_path_obj, pt_path)
            return str(pt_path)
    
    return str(checkpoint_path_obj)


def load_yolov8_model(checkpoint_path: str, device: str = "cuda"):
    """加载 YOLOv8 模型"""
    yolov8_dir = Path(__file__).parent.parent.parent / "experiments" / "yolov8"
    if str(yolov8_dir) not in sys.path:
        sys.path.insert(0, str(yolov8_dir))
    from ultralytics import YOLO
    
    checkpoint_path_pt = _ensure_yolo_checkpoint_path(checkpoint_path)
    model = YOLO(checkpoint_path_pt)
    model.to(device)
    model.eval()
    return model


def load_yolov10_model(checkpoint_path: str, device: str = "cuda"):
    """加载 YOLOv10 模型"""
    yolov10_dir = Path(__file__).parent.parent.parent / "experiments" / "yolov10"
    if str(yolov10_dir) not in sys.path:
        sys.path.insert(0, str(yolov10_dir))
    from ultralytics import YOLO
    
    checkpoint_path_pt = _ensure_yolo_checkpoint_path(checkpoint_path)
    model = YOLO(checkpoint_path_pt)
    model.to(device)
    model.eval()
    return model


def evaluate_yolo_accuracy(model, config_path: str, device: str = "cuda", max_samples: int = 300) -> Dict[str, float]:
    """使用 YOLO model.val() 进行评估"""
    try:
        print(f"  ✓ 使用 YOLO model.val() 进行评估")
        
        results = model.val(
            data=str(config_path),
            device=device,
            verbose=False,
            max_det=300,
            batch=1  # batch_size=1（性能测试标准）
        )
        
        metrics = {'mAP': 0.0, 'AP50': 0.0, 'APS': 0.0,
                   'latency_inference_ms': 0.0, 'latency_postprocess_ms': 0.0,
                   'latency_total_ms': 0.0, 'fps': 0.0}
        
        if hasattr(results, 'box'):
            if hasattr(results.box, 'map'):
                metrics['mAP'] = float(results.box.map)
            if hasattr(results.box, 'map50'):
                metrics['AP50'] = float(results.box.map50)
            if hasattr(results.box, 'maps') and len(results.box.maps) > 1:
                metrics['APS'] = float(results.box.maps[1])
        
        if hasattr(results, 'speed'):
            speed = results.speed
            if isinstance(speed, dict):
                preprocess_ms = float(speed.get('preprocess', 0.0))
                inference_ms = float(speed.get('inference', 0.0))
                postprocess_ms = float(speed.get('postprocess', 0.0))
            elif isinstance(speed, (list, tuple)) and len(speed) >= 3:
                preprocess_ms, inference_ms, postprocess_ms = float(speed[0]), float(speed[1]), float(speed[2])
            else:
                preprocess_ms = inference_ms = postprocess_ms = 0.0
            
            latency_total_ms = preprocess_ms + inference_ms + postprocess_ms
            fps = 1000.0 / latency_total_ms if latency_total_ms > 0 else 0.0
            
            metrics.update({
                'latency_inference_ms': inference_ms,
                'latency_postprocess_ms': postprocess_ms,
                'latency_total_ms': latency_total_ms,
                'fps': fps
            })
            print(f"  ✓ Latency: Inf={inference_ms:.2f}ms, Post={postprocess_ms:.2f}ms, Total={latency_total_ms:.2f}ms")
        
        print(f"  ✓ mAP: {metrics['mAP']:.4f}, AP50: {metrics['AP50']:.4f}, APS: {metrics['APS']:.4f}")
        return metrics
        
    except Exception as e:
        print(f"  ⚠ YOLO 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return {'mAP': 0.0, 'AP50': 0.0, 'APS': 0.0,
                'latency_inference_ms': 0.0, 'latency_postprocess_ms': 0.0,
                'latency_total_ms': 0.0, 'fps': 0.0}


def evaluate_deformable_detr_accuracy(model, config_path: str, device: str = "cuda") -> Dict[str, float]:
    """评估 Deformable-DETR 模型（暂不支持自动评估）"""
    print(f"  ⚠ Deformable-DETR 评估暂不支持，返回默认值")
    return {'mAP': 0.0, 'AP50': 0.0, 'APS': 0.0,
            'latency_inference_ms': 0.0, 'latency_postprocess_ms': 0.0,
            'latency_total_ms': 0.0, 'fps': 0.0}


def _get_outputs_info(outputs: Dict) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """从模型输出中提取 logits 和 boxes
    
    注意：DSET 和 RT-DETR 均使用 Focal Loss，推理时应使用 Sigmoid 激活
    """
    if 'pred_logits' in outputs:
        return outputs['pred_logits'], outputs['pred_boxes'], True  # RT-DETR: sigmoid
    elif 'class_scores' in outputs:
        return outputs['class_scores'], outputs['bboxes'], True  # DSET: sigmoid (Focal Loss)
    return None, None, False


def _postprocess_for_timing(outputs: Dict, img_w: int, img_h: int) -> None:
    """仅执行必要的后处理张量计算（用于性能测试计时）"""
    pred_logits, pred_boxes, use_sigmoid = _get_outputs_info(outputs)
    if pred_logits is None:
        return
    
    for i in range(pred_logits.shape[0]):
        if use_sigmoid:
            pred_scores = torch.sigmoid(pred_logits[i])
        else:
            pred_scores = torch.softmax(pred_logits[i], dim=-1)
        max_scores, pred_classes = torch.max(pred_scores, dim=-1)
        
        valid_boxes_mask = ~torch.all(pred_boxes[i] == 1.0, dim=1)
        valid_indices = torch.where(valid_boxes_mask)[0]
        
        if len(valid_indices) > 0:
            filtered_boxes = pred_boxes[i][valid_indices]
            if filtered_boxes.shape[0] > 0:
                boxes_coco = torch.zeros_like(filtered_boxes)
                if filtered_boxes.max() <= 1.0:
                    boxes_coco[:, 0] = (filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2) * img_w
                    boxes_coco[:, 1] = (filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2) * img_h
                    boxes_coco[:, 2] = filtered_boxes[:, 2] * img_w
                    boxes_coco[:, 3] = filtered_boxes[:, 3] * img_h
                else:
                    boxes_coco = filtered_boxes.clone()
                boxes_coco = torch.clamp(boxes_coco, min=0)


def _collect_predictions_for_coco(outputs: Dict, targets: List[Dict], batch_idx: int,
                                  all_predictions: List, all_targets: List,
                                  img_w: int, img_h: int, batch_size: int) -> None:
    """收集预测结果用于 COCO 评估"""
    pred_logits, pred_boxes, use_sigmoid = _get_outputs_info(outputs)
    if pred_logits is None:
        return
    
    batch_size_actual = pred_logits.shape[0]
    
    for i in range(batch_size_actual):
        if use_sigmoid:
            pred_scores = torch.sigmoid(pred_logits[i])
        else:
            pred_scores = torch.softmax(pred_logits[i], dim=-1)
        max_scores, pred_classes = torch.max(pred_scores, dim=-1)
        
        valid_boxes_mask = ~torch.all(pred_boxes[i] == 1.0, dim=1)
        valid_indices = torch.where(valid_boxes_mask)[0]
        
        if len(valid_indices) > 0:
            filtered_boxes = pred_boxes[i][valid_indices]
            filtered_classes = pred_classes[valid_indices]
            filtered_scores = max_scores[valid_indices]
            
            if filtered_boxes.shape[0] > 0:
                boxes_coco = torch.zeros_like(filtered_boxes)
                if filtered_boxes.max() <= 1.0:
                    boxes_coco[:, 0] = (filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2) * img_w
                    boxes_coco[:, 1] = (filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2) * img_h
                    boxes_coco[:, 2] = filtered_boxes[:, 2] * img_w
                    boxes_coco[:, 3] = filtered_boxes[:, 3] * img_h
                else:
                    boxes_coco = filtered_boxes.clone()
                
                boxes_coco[:, 0] = torch.clamp(boxes_coco[:, 0], 0, img_w)
                boxes_coco[:, 1] = torch.clamp(boxes_coco[:, 1], 0, img_h)
                boxes_coco[:, 2] = torch.clamp(boxes_coco[:, 2], 1, img_w)
                boxes_coco[:, 3] = torch.clamp(boxes_coco[:, 3], 1, img_h)
                
                for j in range(boxes_coco.shape[0]):
                    x, y, w, h = boxes_coco[j].cpu().numpy()
                    all_predictions.append({
                        'image_id': batch_idx * batch_size + i,
                        'category_id': int(filtered_classes[j].item()) + 1,
                        'bbox': [float(x), float(y), float(w), float(h)],
                        'score': float(filtered_scores[j].item())
                    })
        
        # 处理真实标签
        if i < len(targets) and 'labels' in targets[i] and 'boxes' in targets[i]:
            true_labels = targets[i]['labels']
            true_boxes = targets[i]['boxes']
            
            if len(true_labels) > 0:
                max_val = float(true_boxes.max().item()) if true_boxes.numel() > 0 else 0.0
                true_boxes_coco = torch.zeros_like(true_boxes)
                
                if max_val <= 1.0 + 1e-6:
                    true_boxes_coco[:, 0] = (true_boxes[:, 0] - true_boxes[:, 2] / 2) * img_w
                    true_boxes_coco[:, 1] = (true_boxes[:, 1] - true_boxes[:, 3] / 2) * img_h
                    true_boxes_coco[:, 2] = true_boxes[:, 2] * img_w
                    true_boxes_coco[:, 3] = true_boxes[:, 3] * img_h
                else:
                    true_boxes_coco = true_boxes.clone()
                
                true_boxes_coco[:, 0] = torch.clamp(true_boxes_coco[:, 0], 0, img_w)
                true_boxes_coco[:, 1] = torch.clamp(true_boxes_coco[:, 1], 0, img_h)
                true_boxes_coco[:, 2] = torch.clamp(true_boxes_coco[:, 2], 1, img_w)
                true_boxes_coco[:, 3] = torch.clamp(true_boxes_coco[:, 3], 1, img_h)
                
                has_iscrowd = 'iscrowd' in targets[i]
                iscrowd_values = targets[i].get('iscrowd', torch.zeros(len(true_labels), dtype=torch.int64))
                
                for j in range(len(true_labels)):
                    x, y, w, h = true_boxes_coco[j].cpu().numpy()
                    ann_dict = {
                        'id': len(all_targets),
                        'image_id': batch_idx * batch_size + i,
                        'category_id': int(true_labels[j].item()) + 1,
                        'bbox': [float(x), float(y), float(w), float(h)],
                        'area': float(w * h)
                    }
                    if has_iscrowd:
                        ann_dict['iscrowd'] = int(iscrowd_values[j].item())
                    all_targets.append(ann_dict)


def evaluate_accuracy(model, config_path: str, device: str = "cuda", 
                      model_type: str = "dset", max_samples: int = 300) -> Dict[str, float]:
    """使用 pycocotools 在验证集上运行 COCO 评估，并测量性能"""
    try:
        # 导入 Trainer
        if model_type == "dset":
            try:
                from experiments.dset.train import DSETTrainer
            except ImportError:
                dset_dir = Path(config_path).parent.parent
                if str(dset_dir) not in sys.path:
                    sys.path.insert(0, str(dset_dir))
                from train import DSETTrainer
            TrainerClass = DSETTrainer
        elif model_type == "rtdetr":
            rtdetr_dir = Path(config_path).parent.parent
            if str(rtdetr_dir) not in sys.path:
                sys.path.insert(0, str(rtdetr_dir))
            from train import RTDETRTrainer
            TrainerClass = RTDETRTrainer
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 加载配置并强制覆盖所有可能的 batch_size 配置项为 1
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if 'misc' not in config:
            config['misc'] = {}
        config['misc']['device'] = device
        
        # 强制覆盖所有可能的 batch_size 配置键
        if 'dataset' not in config:
            config['dataset'] = {}
        config['dataset']['val_batch_size'] = 1
        
        if 'val' not in config:
            config['val'] = {}
        config['val']['batch_size'] = 1
        
        if 'dataloader' in config and 'test_dataloader' in config['dataloader']:
            config['dataloader']['test_dataloader']['batch_size'] = 1
        
        # 创建 DataLoader
        if model_type == "dset":
            trainer = TrainerClass(config, config_file_path=str(config_path))
            _, val_loader = trainer._create_data_loaders()
        else:
            trainer = TrainerClass(config)
            trainer.model = model
            trainer.criterion = trainer.create_criterion()
            _, val_loader = trainer.create_datasets()
        
        # 强制重构 DataLoader：如果 batch_size 不为 1，则重新包装
        actual_batch_size = val_loader.batch_size if hasattr(val_loader, 'batch_size') else None
        if actual_batch_size != 1:
            print(f"  ⚠ 发现加载器 batch_size={actual_batch_size}，正在强行重构为 1...")
            val_loader = torch.utils.data.DataLoader(
                val_loader.dataset,
                batch_size=1,
                shuffle=False,
                num_workers=val_loader.num_workers if hasattr(val_loader, 'num_workers') else 0,
                pin_memory=val_loader.pin_memory if hasattr(val_loader, 'pin_memory') else False,
                collate_fn=val_loader.collate_fn if hasattr(val_loader, 'collate_fn') else None
            )
        
        dataset_size = len(val_loader.dataset)
        dataloader_length = len(val_loader)
        
        # 断言检查：重构后 batch_size 必须为 1
        final_batch_size = val_loader.batch_size if hasattr(val_loader, 'batch_size') else None
        assert final_batch_size == 1, \
            f"错误: 重构后 DataLoader batch_size={final_batch_size}，仍不为 1。"
        assert dataloader_length == dataset_size, \
            f"错误: DataLoader 长度 ({dataloader_length}) != 数据集大小 ({dataset_size})。"
        
        print(f"  ✓ DataLoader: {dataloader_length}/{dataset_size} (batch_size=1 已确认)")
        
        model.eval()
        model = model.to(device)
        
        # 验证 token pruning 状态
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'token_pruners'):
            if model.encoder.token_pruners:
                pruner = model.encoder.token_pruners[0]
                if hasattr(pruner, 'pruning_enabled'):
                    print(f"  ✓ Token Pruning: {'已激活' if pruner.pruning_enabled else '未激活'}")
        
        all_predictions = []
        all_targets = []
        use_cuda_events = (device == "cuda" and torch.cuda.is_available())
        inference_times = []
        postprocess_times = []
        total_samples = 0
        perf_samples = 0
        limit_samples = (max_samples < 999999)
        shape_printed = False  # 只打印一次维度信息
        
        print(f"  运行推理循环...")
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                B, C, H_tensor, W_tensor = images.shape
                need_timing = limit_samples and (perf_samples < max_samples)
                
                images = images.to(device)
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                           for k, v in t.items()} for t in targets]
                
                # 推理计时（计时器内部保持纯净，无 print/assert）
                if need_timing:
                    if use_cuda_events:
                        torch.cuda.synchronize()
                        inference_starter = torch.cuda.Event(enable_timing=True)
                        inference_ender = torch.cuda.Event(enable_timing=True)
                        inference_starter.record()
                    else:
                        # CPU 模式：使用 perf_counter 获得高精度计时
                        inference_start_time = time.perf_counter()
                
                outputs = model(images, targets)
                
                # 物理维度检查（只打印一次）
                if not shape_printed and isinstance(outputs, dict):
                    shape_printed = True
                    if 'class_scores' in outputs:
                        seq_len = outputs['class_scores'].shape[1]
                        print(f"  ✓ 物理维度检查: class_scores shape = {outputs['class_scores'].shape} (seq_len={seq_len})")
                    elif 'pred_logits' in outputs:
                        seq_len = outputs['pred_logits'].shape[1]
                        print(f"  ✓ 物理维度检查: pred_logits shape = {outputs['pred_logits'].shape} (seq_len={seq_len})")
                
                if need_timing:
                    if use_cuda_events:
                        inference_ender.record()
                        torch.cuda.synchronize()
                        inference_elapsed_ms = inference_starter.elapsed_time(inference_ender)
                    else:
                        # CPU 模式：使用 perf_counter 计算耗时
                        inference_elapsed_ms = (time.perf_counter() - inference_start_time) * 1000.0
                    inference_times.append(inference_elapsed_ms)
                    
                    # 后处理计时
                    if use_cuda_events:
                        postprocess_starter = torch.cuda.Event(enable_timing=True)
                        postprocess_ender = torch.cuda.Event(enable_timing=True)
                        postprocess_starter.record()
                    else:
                        # CPU 模式：使用 perf_counter
                        postprocess_start_time = time.perf_counter()
                
                has_predictions = isinstance(outputs, dict) and (
                    ('class_scores' in outputs and 'bboxes' in outputs) or
                    ('pred_logits' in outputs and 'pred_boxes' in outputs)
                )
                
                if need_timing and has_predictions:
                    _postprocess_for_timing(outputs, W_tensor, H_tensor)
                
                if need_timing:
                    if use_cuda_events:
                        postprocess_ender.record()
                        torch.cuda.synchronize()
                        postprocess_elapsed_ms = postprocess_starter.elapsed_time(postprocess_ender)
                    else:
                        # CPU 模式：使用 perf_counter 计算耗时
                        postprocess_elapsed_ms = (time.perf_counter() - postprocess_start_time) * 1000.0
                    postprocess_times.append(postprocess_elapsed_ms)
                    perf_samples += 1
                
                if has_predictions:
                    _collect_predictions_for_coco(
                        outputs, targets, batch_idx, all_predictions, all_targets,
                        W_tensor, H_tensor, 1
                    )
                
                total_samples += 1
                
                # 进度打印：每 100 个样本打印一次
                if (batch_idx + 1) % 100 == 0:
                    print(f"    进度: {batch_idx + 1}/{dataloader_length}")
        
        print(f"  ✓ 完成: {total_samples} 样本, {len(all_predictions)} 预测框")
        
        # 计算耗时
        if inference_times and postprocess_times:
            avg_inference_ms = sum(inference_times) / len(inference_times)
            avg_postprocess_ms = sum(postprocess_times) / len(postprocess_times)
            latency_total_ms = avg_inference_ms + avg_postprocess_ms
            fps = 1000.0 / latency_total_ms if latency_total_ms > 0 else 0.0
            print(f"  ✓ Latency: Inf={avg_inference_ms:.2f}ms, Post={avg_postprocess_ms:.2f}ms, Total={latency_total_ms:.2f}ms")
        else:
            avg_inference_ms = avg_postprocess_ms = latency_total_ms = fps = 0.0
        
        # COCO 评估
        metrics = _compute_coco_metrics(all_predictions, all_targets, H_tensor, W_tensor)
        metrics.update({
            'latency_inference_ms': avg_inference_ms,
            'latency_postprocess_ms': avg_postprocess_ms,
            'latency_total_ms': latency_total_ms,
            'fps': fps
        })
        
        return metrics
        
    except Exception as e:
        print(f"  ⚠ COCO 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return {'mAP': 0.0, 'AP50': 0.0, 'APS': 0.0,
                'latency_inference_ms': 0.0, 'latency_postprocess_ms': 0.0,
                'latency_total_ms': 0.0, 'fps': 0.0}


def _compute_coco_metrics(predictions: List[Dict], targets: List[Dict],
                          img_h: int = 736, img_w: int = 1280) -> Dict[str, float]:
    """使用 pycocotools 计算 COCO 指标"""
    try:
        if len(predictions) == 0:
            return {'mAP': 0.0, 'AP50': 0.0, 'APS': 0.0}
        
        categories = [
            {'id': 1, 'name': 'Car'}, {'id': 2, 'name': 'Truck'},
            {'id': 3, 'name': 'Van'}, {'id': 4, 'name': 'Bus'},
            {'id': 5, 'name': 'Pedestrian'}, {'id': 6, 'name': 'Cyclist'},
            {'id': 7, 'name': 'Motorcyclist'}, {'id': 8, 'name': 'Trafficcone'}
        ]
        
        image_ids = set(t['image_id'] for t in targets)
        coco_gt = {
            'images': [{'id': img_id, 'width': img_w, 'height': img_h} for img_id in image_ids],
            'annotations': targets,
            'categories': categories,
            'info': {'description': 'DAIR-V2X Dataset', 'version': '1.0', 'year': 2024}
        }
        
        coco_gt_obj = COCO()
        coco_gt_obj.dataset = coco_gt
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            coco_gt_obj.createIndex()
            coco_dt = coco_gt_obj.loadRes(predictions)
        finally:
            sys.stdout = old_stdout
        
        coco_eval = COCOeval(coco_gt_obj, coco_dt, 'bbox')
        
        sys.stdout = StringIO()
        try:
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        finally:
            sys.stdout = old_stdout
        
        stats = coco_eval.stats
        metrics = {
            'mAP': float(stats[0]) if len(stats) > 0 else 0.0,
            'AP50': float(stats[1]) if len(stats) > 1 else 0.0,
            'APS': float(stats[3]) if len(stats) > 3 else 0.0
        }
        
        print(f"  ✓ mAP: {metrics['mAP']:.4f}, AP50: {metrics['AP50']:.4f}, APS: {metrics['APS']:.4f}")
        return metrics
        
    except Exception as e:
        print(f"  ⚠ COCO 指标计算失败: {e}")
        return {'mAP': 0.0, 'AP50': 0.0, 'APS': 0.0}


def _resolve_path(path_str: str, project_root: Path) -> Path:
    """解析路径"""
    if not path_str:
        return None
    path = Path(path_str)
    return path if path.is_absolute() else project_root / path


def _get_yolo_data_path(model, model_config: Dict, project_root: Path) -> Optional[Path]:
    """获取 YOLO 模型的数据集配置文件路径"""
    data_config = model_config.get('data', None)
    if data_config:
        data_path = _resolve_path(data_config, project_root)
        if data_path and data_path.exists():
            return data_path
    
    try:
        if hasattr(model, 'ckpt') and model.ckpt and hasattr(model.ckpt, 'data'):
            data_path = _resolve_path(model.ckpt.data, project_root)
            if data_path and data_path.exists():
                return data_path
    except:
        pass
    
    return None


def evaluate_single_model(model_name: str, model_config: Dict, args, project_root: Path) -> Optional[Dict]:
    """评估单个模型"""
    print("\n" + "=" * 80)
    print(f"评估模型: {model_name}")
    print("=" * 80)
    
    model_type = model_config.get('type', args.model_type)
    config_path_str = model_config.get('config', args.config)
    checkpoint_path_str = model_config.get('checkpoint', None)
    input_size_override = model_config.get('input_size', None)
    
    # 确定 input_size
    if input_size_override is not None:
        input_size = input_size_override
    elif "yolo" in model_type.lower():
        input_size = [1280, 1280]
    else:
        input_size = [736, 1280]
    
    config_path = _resolve_path(config_path_str, project_root) if config_path_str else None
    
    # 处理检查点路径
    if checkpoint_path_str:
        checkpoint_path = _resolve_path(checkpoint_path_str, project_root)
        if not checkpoint_path.exists():
            print(f"⚠ 检查点不存在: {checkpoint_path}")
            return None
    else:
        logs_dir = _resolve_path(args.logs_dir, project_root)
        checkpoint_path = find_latest_best_model(logs_dir, model_type)
        if checkpoint_path is None:
            print(f"⚠ 无法找到检查点")
            return None
    
    print(f"  类型: {model_type}, 输入: {input_size[0]}x{input_size[1]}")
    print(f"  检查点: {checkpoint_path}")
    
    # 加载模型
    is_yolo_model = model_type.startswith("yolov8") or model_type.startswith("yolov10")
    try:
        if model_type == "dset":
            model = load_dset_model(str(config_path), str(checkpoint_path), args.device)
        elif model_type == "rtdetr":
            model = load_rtdetr_model(str(config_path), str(checkpoint_path), args.device)
        elif model_type == "deformable-detr":
            model = load_deformable_detr_model(str(checkpoint_path), args.device, 
                                               config_path=str(config_path) if config_path else None)
        elif model_type.startswith("yolov8"):
            model = load_yolov8_model(str(checkpoint_path), args.device)
        elif model_type.startswith("yolov10"):
            model = load_yolov10_model(str(checkpoint_path), args.device)
        else:
            print(f"  ⚠ 不支持的模型类型: {model_type}")
            return None
        print("  ✓ 模型加载成功")
    except Exception as e:
        print(f"  ⚠ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 计算参数量和 FLOPs
    input_size_tuple = (1, 3, input_size[0], input_size[1])
    params_m, flops_g = get_model_info(model, input_size_tuple, is_yolo=is_yolo_model)
    print(f"  ✓ Params: {params_m:.2f}M, FLOPs: {flops_g:.2f}G")
    
    # 评估
    metrics = {'mAP': 0.0, 'AP50': 0.0, 'APS': 0.0,
               'latency_inference_ms': 0.0, 'latency_postprocess_ms': 0.0,
               'latency_total_ms': 0.0, 'fps': 0.0}
    max_samples = getattr(args, 'max_samples', 300)  # 从参数获取，支持 CPU 模式
    
    if model_type in ["dset", "rtdetr"] and config_path:
        if not args.skip_fps:
            metrics = evaluate_accuracy(model, str(config_path), args.device, 
                                        model_type=model_type, max_samples=max_samples)
        else:
            metrics_partial = evaluate_accuracy(model, str(config_path), args.device,
                                               model_type=model_type, max_samples=999999)
            metrics.update({k: metrics_partial.get(k, 0.0) for k in ['mAP', 'AP50', 'APS']})
    elif model_type == "deformable-detr" and config_path:
        metrics_partial = evaluate_deformable_detr_accuracy(model, str(config_path), args.device)
        metrics.update({k: metrics_partial.get(k, 0.0) for k in ['mAP', 'AP50', 'APS']})
    elif is_yolo_model:
        data_config_path = _get_yolo_data_path(model, model_config, project_root)
        if data_config_path:
            if not args.skip_fps:
                metrics = evaluate_yolo_accuracy(model, str(data_config_path), args.device, max_samples)
            else:
                metrics_partial = evaluate_yolo_accuracy(model, str(data_config_path), args.device, 999999)
                metrics.update({k: metrics_partial.get(k, 0.0) for k in ['mAP', 'AP50', 'APS']})
        else:
            print(f"  ⚠ YOLO 模型需要数据集配置文件")
    
    # 清理显存
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'model_name': model_name,
        'model_type': model_type,
        'params_m': params_m,
        'flops_g': flops_g,
        'fps': metrics.get('fps', 0.0),
        'latency_inference_ms': metrics.get('latency_inference_ms', 0.0),
        'latency_postprocess_ms': metrics.get('latency_postprocess_ms', 0.0),
        'latency_total_ms': metrics.get('latency_total_ms', 0.0),
        'mAP': metrics.get('mAP', 0.0),
        'AP50': metrics.get('AP50', 0.0),
        'APS': metrics.get('APS', 0.0),
        'input_size': f"{input_size[0]}x{input_size[1]}"
    }


def print_summary_table(results: List[Dict], gpu_name: str = "GPU", save_csv: bool = True, max_samples: int = 300):
    """打印结果汇总表格并保存为 CSV"""
    if not results:
        print("\n⚠ 没有评估结果")
        return
    
    print("\n" + "=" * 120)
    print("BATCH EVALUATION SUMMARY".center(120))
    print("=" * 120)
    
    header = f"{'Model':<25} {'Params':<10} {'FLOPs':<10} {'Latency':<12} {'Inference':<10} {'Post':<10} {'FPS':<8} {'mAP':<8} {'AP50':<8} {'APS':<8}"
    print(header)
    print("-" * 120)
    
    csv_rows = [['Model', 'Type', 'Params(M)', 'FLOPs(G)', 'Latency(ms)', 'Inference(ms)', 
                 'Post(ms)', 'FPS', 'mAP', 'AP50', 'APS', 'Input']]
    
    for r in results:
        name = r.get('model_name', 'Unknown')[:24]
        params = f"{r.get('params_m', 0):.2f}" if r.get('params_m', 0) > 0 else "N/A"
        flops = f"{r.get('flops_g', 0):.2f}" if r.get('flops_g', 0) > 0 else "N/A"
        latency = f"{r.get('latency_total_ms', 0):.2f}" if r.get('latency_total_ms', 0) > 0 else "N/A"
        inf = f"{r.get('latency_inference_ms', 0):.2f}" if r.get('latency_inference_ms', 0) > 0 else "N/A"
        post = f"{r.get('latency_postprocess_ms', 0):.2f}" if r.get('latency_postprocess_ms', 0) > 0 else "N/A"
        fps = f"{r.get('fps', 0):.1f}" if r.get('fps', 0) > 0 else "N/A"
        mAP = f"{r.get('mAP', 0):.4f}" if r.get('mAP', 0) > 0 else "N/A"
        ap50 = f"{r.get('AP50', 0):.4f}" if r.get('AP50', 0) > 0 else "N/A"
        aps = f"{r.get('APS', 0):.4f}" if r.get('APS', 0) > 0 else "N/A"
        
        print(f"{name:<25} {params:<10} {flops:<10} {latency:<12} {inf:<10} {post:<10} {fps:<8} {mAP:<8} {ap50:<8} {aps:<8}")
        
        csv_rows.append([name, r.get('model_type', ''), params, flops, latency, inf, post, fps, mAP, ap50, aps, r.get('input_size', '')])
    
    print("-" * 120)
    print(f"Note: Performance on {gpu_name}, batch_size=1, {max_samples} samples")
    print("=" * 120)
    
    if save_csv:
        import csv
        csv_path = Path(project_root) / "benchmark_results.csv"
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerows(csv_rows)
            print(f"\n✓ 结果已保存到: {csv_path}")
        except Exception as e:
            print(f"\n⚠ 保存 CSV 失败: {e}")


def find_latest_best_model(logs_dir: Path, model_type: str = "dset") -> Optional[Path]:
    """在 logs 目录下找到最新的 best_model.pth"""
    if model_type == "deformable-detr":
        best_models = list(logs_dir.rglob('best_*.pth'))
    elif model_type.startswith("yolov"):
        best_models = list(logs_dir.rglob('best_model.pth')) or list(logs_dir.rglob('weights/best.pt'))
    else:
        best_models = list(logs_dir.rglob('best_model.pth'))
    
    if not best_models:
        return None
    
    best_models.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return best_models[0]


def _format_evaluation_results(model_type: str, params_m: float, flops_g: float, fps: float,
                               latency_total_ms: float, latency_inference_ms: float, 
                               latency_postprocess_ms: float, metrics: Dict[str, float],
                               input_resolution: Tuple[int, int], is_yolo: bool = False,
                               gpu_name: str = "GPU") -> None:
    """格式化并输出评估结果"""
    model_names = {
        'dset': 'DSET', 'rtdetr': 'RT-DETRv2', 'deformable-detr': 'Deformable-DETR',
        'yolov8s': 'YOLOv8-s', 'yolov8m': 'YOLOv8-m',
        'yolov10s': 'YOLOv10-s', 'yolov10m': 'YOLOv10-m'
    }
    name = model_names.get(model_type, model_type.upper())
    
    if is_yolo:
        res = f"{max(input_resolution)}x{max(input_resolution)}"
    else:
        res = f"{input_resolution[0]}x{input_resolution[1]}"
    
    print("\n" + "=" * 70)
    print(f"Model: {name} | Input: {res}")
    print("-" * 70)
    print(f"Params: {params_m:.2f}M | FLOPs: {flops_g:.2f}G")
    print(f"Latency ({gpu_name}): {latency_total_ms:.2f}ms (Inf: {latency_inference_ms:.2f}, Post: {latency_postprocess_ms:.2f})")
    print(f"FPS: {fps:.1f}")
    print(f"mAP: {metrics['mAP']:.4f} | AP50: {metrics['AP50']:.4f} | APS: {metrics['APS']:.4f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='生成性能对比表')
    parser.add_argument('--logs_dir', type=str, default='experiments/dset/logs')
    parser.add_argument('--config', type=str, default='experiments/dset/configs/dset4_r18_ratio0.5.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--input_size', type=int, nargs=2, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--fps_iter', type=int, default=300)
    parser.add_argument('--warmup_iter', type=int, default=50)
    parser.add_argument('--skip_fps', action='store_true')
    parser.add_argument('--model_type', type=str, default='dset',
                       choices=['dset', 'rtdetr', 'deformable-detr', 
                               'yolov8s', 'yolov8m', 'yolov10s', 'yolov10m'])
    parser.add_argument('--rtdetr_config', type=str, default=None)
    parser.add_argument('--deformable_work_dir', type=str, default=None)
    parser.add_argument('--deformable_config', type=str, default=None)
    parser.add_argument('--models_config', type=str, default=None)
    parser.add_argument('--cpu_mode', action='store_true',
                       help='使用 CPU 模拟边缘设备推理（自动限制样本量为 50）')
    parser.add_argument('--max_samples', type=int, default=300,
                       help='性能测试的最大样本数（默认 300，CPU 模式自动设为 50）')
    
    args = parser.parse_args()
    
    # CPU 模式处理
    if args.cpu_mode:
        args.device = 'cpu'
        args.max_samples = min(args.max_samples, 50)  # CPU 模式限制样本量
        print("=" * 80)
        print("⚠️  CPU 模拟边缘设备模式")
        print("=" * 80)
        print(f"  • 设备: CPU (模拟边缘计算设备)")
        print(f"  • 样本量: {args.max_samples} (缩减以加快测试)")
        print(f"  • 预期: Token Pruning 将带来显著的倍数级加速")
        print("=" * 80)
        print()
    
    # GPU/CPU 名称
    if args.device == 'cuda' and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    else:
        import platform
        gpu_name = f"CPU ({platform.processor() or 'Unknown'})"
    
    print("=" * 80)
    print("性能对比表生成脚本")
    print("=" * 80)
    
    # 构造配置
    if args.models_config:
        json_config_path = _resolve_path(args.models_config, project_root)
        if not json_config_path.exists():
            print(f"错误: JSON 配置文件不存在: {json_config_path}")
            return
        with open(json_config_path, 'r', encoding='utf-8') as f:
            json_config = json.load(f)
        print(f"✓ 批量评估: {len(json_config)} 个模型\n")
    else:
        if "yolo" in args.model_type.lower():
            input_size = [1280, 1280]
        else:
            input_size = [736, 1280]
        input_size = args.input_size or input_size
        
        single_config = {
            'type': args.model_type,
            'config': args.rtdetr_config or args.deformable_config or args.config,
            'checkpoint': args.checkpoint,
            'input_size': input_size
        }
        json_config = {'single_model': single_config}
        print(f"✓ 单模型评估\n")
    
    # 评估
    all_results = []
    for model_name, model_config in json_config.items():
        if not isinstance(model_config, dict):
            continue
        result = evaluate_single_model(model_name, model_config, args, project_root)
        if result:
            all_results.append(result)
    
    # 输出结果
    if len(all_results) > 1:
        print_summary_table(all_results, gpu_name, save_csv=True, max_samples=args.max_samples)
    elif all_results:
        r = all_results[0]
        _format_evaluation_results(
            r['model_type'], r['params_m'], r['flops_g'], r['fps'],
            r.get('latency_total_ms', 0), r.get('latency_inference_ms', 0),
            r.get('latency_postprocess_ms', 0),
            {'mAP': r['mAP'], 'AP50': r['AP50'], 'APS': r['APS']},
            (int(r['input_size'].split('x')[1]), int(r['input_size'].split('x')[0])),
            r['model_type'].startswith("yolov"), gpu_name
        )


if __name__ == '__main__':
    main()
