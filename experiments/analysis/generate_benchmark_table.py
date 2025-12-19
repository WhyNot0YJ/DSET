#!/usr/bin/env python3
"""
模型性能评估脚本
支持多种模型的性能指标提取和测试

功能：
1. 自动从 logs/ 目录查找最新的 best_model.pth 或使用指定的检查点
2. 从训练日志 (training_history.csv) 提取最佳 mAP, AP50, APS 指标
3. 计算模型参数量（Params）和 FLOPs
4. 测量模型在当前设备上的 FPS（包含 warmup 过程，YOLO 模型包含 NMS）
5. 输出清晰的评估结果清单

支持的模型类型：
- dset: DSET (Dual-Sparse Expert Transformer) 模型
- rtdetr: RT-DETR 模型
- deformable-detr: Deformable-DETR 模型（使用 MMEngine）
- yolov8s: YOLOv8-small 模型
- yolov8m: YOLOv8-medium 模型
- yolov10s: YOLOv10-small 模型
- yolov10m: YOLOv10-medium 模型

使用方法：
    # DSET 模型（默认）
    python generate_benchmark_table.py --model_type dset
    
    # RT-DETR 模型
    python generate_benchmark_table.py --model_type rtdetr --rtdetr_config experiments/rt-detr/configs/xxx.yaml
    
    # Deformable-DETR 模型
    python generate_benchmark_table.py --model_type deformable-detr --checkpoint experiments/deformable-detr/work_dirs/xxx/best_*.pth
    
    # YOLOv8-s 模型
    python generate_benchmark_table.py --model_type yolov8s --logs_dir experiments/yolov8/logs
    
    # YOLOv8-m 模型
    python generate_benchmark_table.py --model_type yolov8m --logs_dir experiments/yolov8/logs
    
    # YOLOv10-s 模型
    python generate_benchmark_table.py --model_type yolov10s --logs_dir experiments/yolov10/logs
    
    # YOLOv10-m 模型
    python generate_benchmark_table.py --model_type yolov10m --logs_dir experiments/yolov10/logs
    
    # 指定检查点路径
    python generate_benchmark_table.py --checkpoint experiments/dset/logs/xxx/best_model.pth
    
    # 跳过 FPS 测试（仅使用已有数据）
    python generate_benchmark_table.py --skip_fps
    
    # 自定义输入尺寸和测试迭代次数
    python generate_benchmark_table.py --input_size 1280 1280 --fps_iter 200

注意：
- 脚本会自动适配不同的列名格式（mAP_s, mAP_small, metrics/mAP50-95(B) 等）
- 如果 mAP 值 > 1.0，会自动归一化（除以 100）为标准格式
- 需要安装 thop 库来计算 FLOPs: pip install thop
- YOLO 模型的 FLOPs 计算包含 NMS 的估算值（基于输入尺寸和典型检测数量）
- YOLO 模型的 FPS 测量包含完整的推理流程（前向传播 + NMS）
- DSET 模型会自动启用 token pruning（设置 epoch >= warmup_epochs）
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import time
import csv
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 尝试导入 thop
try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("警告: thop 未安装，将无法计算 FLOPs。请运行: pip install thop")


def get_model_info(model, input_size: Tuple[int, int, int, int] = (1, 3, 640, 640), is_yolo: bool = False) -> Tuple[float, float]:
    """
    计算模型的参数量和 FLOPs
    
    注意：对于 YOLO 模型，FLOPs 计算包含 NMS 的估算值
    
    Args:
        model: PyTorch 模型或 YOLO 模型
        input_size: 输入尺寸 (batch, channels, height, width)
        is_yolo: 是否为 YOLO 模型（如果是，会计算包含 NMS 的 FLOPs）
    
    Returns:
        (params_in_millions, flops_in_giga)
        - 对于 YOLO 模型，flops_in_giga 包含基础模型 FLOPs + NMS FLOPs 估算
    """
    # 计算参数量
    if is_yolo:
        # YOLO 模型：尝试从 model.model 获取实际的 PyTorch 模型
        if hasattr(model, 'model'):
            pytorch_model = model.model
        else:
            pytorch_model = model
        total_params = sum(p.numel() for p in pytorch_model.parameters())
    else:
        total_params = sum(p.numel() for p in model.parameters())
    
    params_m = total_params / 1e6
    
    # 计算 FLOPs
    flops_g = 0.0
    if is_yolo:
        # YOLO 模型：使用 ultralytics 的方法计算 FLOPs（包含 NMS）
        # 注意：FLOPs 计算使用实际输入尺寸（720x1280），不考虑 Letterbox 填充
        # 因为 FLOPs 反映的是模型本身的计算量，而不是预处理开销
        try:
            # 方法1：尝试使用 ultralytics 内置的 get_flops 函数
            from copy import deepcopy
            if hasattr(model, 'model'):
                pytorch_model = model.model
                # 获取输入尺寸（使用实际输入尺寸，不是填充后的）
                h, w = input_size[2], input_size[3]
                imgsz = [h, w] if h != w else h
                
                # 尝试使用 ultralytics 的 get_flops（如果可用）
                try:
                    from ultralytics.utils.torch_utils import get_flops
                    flops_g = get_flops(pytorch_model, imgsz=imgsz)
                    print(f"  ✓ 使用 ultralytics.get_flops 计算 FLOPs")
                except (ImportError, AttributeError):
                    # 方法2：使用 thop 直接计算
                    if HAS_THOP:
                        pytorch_model = pytorch_model.eval()
                        device = next(pytorch_model.parameters()).device
                        dummy_input = torch.randn(input_size).to(device)
                        flops, _ = profile(deepcopy(pytorch_model), inputs=(dummy_input,), verbose=False)
                        flops_g = flops / 1e9 * 2  # thop 返回的是 MACs，需要 *2 得到 FLOPs
                        print(f"  ✓ 使用 thop 计算基础 FLOPs: {flops_g:.2f} G")
                        
                        # 估算 NMS 的 FLOPs（基于输入尺寸和典型的检测数量）
                        # NMS 主要包括：
                        # 1. IoU 计算：对于 n 个框，需要计算 n*(n-1)/2 次 IoU
                        # 2. 排序操作：O(n log n)
                        # 3. 过滤操作：O(n)
                        # 
                        # 典型的 YOLO 模型在 640x640 输入下会产生约 8400 个候选框（3个尺度）
                        # 经过置信度过滤后，通常剩余 100-1000 个框
                        # 假设平均 300 个框进行 NMS：
                        #   - IoU 计算：300*299/2 * 4 ops ≈ 180K FLOPs ≈ 0.00018 GFLOPs
                        #   - 排序：300*log2(300) ≈ 2400 ops ≈ 0.000002 GFLOPs
                        #   总计约 0.0002 GFLOPs，但实际实现可能有额外开销
                        # 
                        # 我们使用基于输入尺寸的估算：
                        # 基准：640x640 输入，假设 300 个候选框，NMS FLOPs ≈ 0.01 GFLOPs
                        # 缩放：根据输入尺寸比例调整候选框数量
                        base_size = 640
                        scale_factor = (h * w) / (base_size * base_size)
                        # 候选框数量与特征图大小成正比，特征图大小与输入尺寸成正比
                        # 所以候选框数量与输入尺寸的平方成正比
                        nms_flops_estimate = 0.01 * scale_factor  # 基准 0.01 GFLOPs at 640x640
                        flops_g += nms_flops_estimate
                        print(f"  ✓ 估算 NMS FLOPs: {nms_flops_estimate:.4f} G (总 FLOPs: {flops_g:.2f} G)")
                    else:
                        print(f"  ⚠ thop 未安装，无法计算 FLOPs")
            else:
                print(f"  ⚠ 无法获取 YOLO 模型的 PyTorch 模型")
        except Exception as e:
            print(f"  ⚠ YOLO FLOPs 计算失败: {e}")
            import traceback
            traceback.print_exc()
    elif HAS_THOP:
        # 标准 PyTorch 模型（DSET, RT-DETR, Deformable-DETR）
        try:
            model.eval()
            device = next(model.parameters()).device
            dummy_input = torch.randn(input_size).to(device)
            flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
            flops_g = flops / 1e9
            print(f"  ✓ 使用 thop 计算 FLOPs: {flops_g:.2f} G")
        except Exception as e:
            print(f"  ⚠ FLOPs 计算失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        # thop 未安装，无法计算 FLOPs
        print(f"  ⚠ thop 未安装，无法计算 FLOPs（请运行: pip install thop）")
    
    return params_m, flops_g


def measure_fps(model, 
                input_size: Tuple[int, int, int, int] = (1, 3, 640, 640),
                num_iter: int = 100,
                warmup_iter: int = 10,
                device: str = "cuda",
                is_yolo: bool = False) -> float:
    """
    测量模型的 FPS (Frames Per Second)
    
    注意：对于 YOLO 模型，FPS 测量包含完整的推理流程（前向传播 + NMS）
    
    Args:
        model: PyTorch 模型或 YOLO 模型
        input_size: 输入尺寸 (batch, channels, height, width)
        num_iter: 测试迭代次数
        warmup_iter: 预热迭代次数
        device: 设备 ('cuda' 或 'cpu')
        is_yolo: 是否为 YOLO 模型（使用不同的输入格式，且包含 NMS）
    
    Returns:
        FPS 值（对于 YOLO 模型，包含 NMS 的完整推理时间）
    """
    model.eval()
    
    if is_yolo:
        # YOLO 模型使用 numpy/PIL 图像格式
        # 注意：YOLO 模型的 model() 调用已经包含完整的推理流程（前向传播 + NMS）
        # 确保模型在正确的设备上
        model = model.to(device)
        
        import numpy as np
        # YOLO 通常需要正方形输入，使用 Letterbox 填充
        # 为了包含填充开销，使用 1280x1280 进行测速
        # 注意：实际输入可能是 720x1280，但 YOLO 会填充到 1280x1280
        yolo_size = max(input_size[2], input_size[3])  # 使用最大边作为正方形边长
        dummy_image = np.random.randint(0, 255, (yolo_size, yolo_size, 3), dtype=np.uint8)
        
        # Warmup（包含 NMS）
        for _ in range(warmup_iter):
            _ = model(dummy_image, verbose=False)  # 这会执行完整的推理流程，包括 NMS
        
        # 同步（如果是 CUDA）
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 实际测试（包含 NMS）
        start_time = time.time()
        for _ in range(num_iter):
            _ = model(dummy_image, verbose=False)  # 包含 NMS 的完整推理
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
        
        end_time = time.time()
    else:
        # 标准 PyTorch 模型使用 tensor
        model = model.to(device)
        dummy_input = torch.randn(input_size).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iter):
                _ = model(dummy_input)
        
        # 同步（如果是 CUDA）
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 实际测试
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iter):
                _ = model(dummy_input)
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
        
        end_time = time.time()
    
    elapsed_time = end_time - start_time
    fps = num_iter / elapsed_time if elapsed_time > 0 else 0.0
    
    return fps


def load_dset_model(config_path: str, checkpoint_path: str, device: str = "cuda"):
    """
    加载 DSET 模型
    
    Args:
        config_path: 配置文件路径
        checkpoint_path: 检查点路径
        device: 设备
    
    Returns:
        加载的模型
    """
    # 导入必要的模块
    dset_dir = Path(config_path).parent.parent / "dset"
    if str(dset_dir) not in sys.path:
        sys.path.insert(0, str(dset_dir))
    from train import DSETTrainer
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 确保 device 配置正确
    if 'misc' not in config:
        config['misc'] = {}
    config['misc']['device'] = device
    
    # 创建 trainer 以构建模型（需要传入 config_file_path 以正确初始化）
    trainer = DSETTrainer(config, config_file_path=str(config_path))
    
    # 创建模型
    model = trainer._create_model()
    
    # 加载检查点
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 加载权重
    if 'ema_state_dict' in checkpoint:
        state_dict = checkpoint['ema_state_dict']
        if isinstance(state_dict, dict) and 'module' in state_dict:
            state_dict = state_dict['module']
    elif 'ema' in checkpoint and 'module' in checkpoint['ema']:
        state_dict = checkpoint['ema']['module']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    
    # 确保模型处于 eval 模式（重要：确保 CASS 和 token pruning 在推理时正确工作）
    model.eval()
    
    # 启用 token pruning 和 CASS（如果适用）
    # 确保在推理时启用 token pruning，需要设置 epoch >= warmup_epochs
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'set_epoch'):
        dset_config = config.get('model', {}).get('dset', {})
        warmup_epochs = dset_config.get('token_pruning_warmup_epochs', 10)
        # 设置为 warmup_epochs 或更大的值以启用 pruning
        model.encoder.set_epoch(warmup_epochs)
        
        # 检查 CASS 是否启用
        use_cass = dset_config.get('use_cass', False)
        token_keep_ratio = dset_config.get('token_keep_ratio', {})
        if isinstance(token_keep_ratio, dict):
            # 如果是字典格式，获取第一个值
            keep_ratio = list(token_keep_ratio.values())[0] if token_keep_ratio else 1.0
        else:
            keep_ratio = token_keep_ratio if token_keep_ratio else 1.0
        
        print(f"  ✓ 已启用 token pruning (epoch={warmup_epochs}, keep_ratio={keep_ratio})")
        if use_cass:
            print(f"  ✓ CASS 模块已激活")
    
    return model


def load_rtdetr_model(config_path: str, checkpoint_path: str, device: str = "cuda"):
    """
    加载 RT-DETR 模型
    
    Args:
        config_path: 配置文件路径
        checkpoint_path: 检查点路径
        device: 设备
    
    Returns:
        加载的模型
    """
    # 导入必要的模块
    rtdetr_dir = Path(config_path).parent.parent / "rt-detr"
    if str(rtdetr_dir) not in sys.path:
        sys.path.insert(0, str(rtdetr_dir))
    from train import RTDETRTrainer
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 确保 device 配置正确
    if 'misc' not in config:
        config['misc'] = {}
    config['misc']['device'] = device
    
    # 创建 trainer 以构建模型
    trainer = RTDETRTrainer(config)
    
    # 创建一个简单的logger
    if trainer.logger is None:
        class SimpleLogger:
            def info(self, msg): pass
        trainer.logger = SimpleLogger()
    
    # 创建模型
    model = trainer.create_model()
    
    # 加载检查点
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 加载权重
    if 'ema_state_dict' in checkpoint:
        state_dict = checkpoint['ema_state_dict']
        if isinstance(state_dict, dict) and 'module' in state_dict:
            state_dict = state_dict['module']
    elif 'ema' in checkpoint and 'module' in checkpoint['ema']:
        state_dict = checkpoint['ema']['module']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model


def load_deformable_detr_model(checkpoint_path: str, device: str = "cuda", config_path: str = None):
    """
    加载 Deformable-DETR 模型（使用 MMEngine 格式）
    
    Args:
        checkpoint_path: 检查点路径（MMEngine 格式）
        device: 设备
        config_path: 可选的配置文件路径（如果检查点中没有配置）
    
    Returns:
        加载的模型
    """
    try:
        from mmengine.config import Config
        from mmdet.registry import MODELS
    except ImportError:
        raise ImportError("需要安装 mmengine 和 mmdet: pip install mmengine mmdet")
    
    # 加载检查点
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # MMEngine checkpoint 格式：包含 'state_dict' 或 'model'
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 从检查点中提取配置（如果存在）
    cfg = None
    if 'meta' in checkpoint and 'cfg' in checkpoint['meta']:
        cfg_dict = checkpoint['meta']['cfg']
        cfg = Config(cfg_dict)
    elif config_path and os.path.exists(config_path):
        cfg = Config.fromfile(config_path)
    else:
        # 尝试查找默认配置文件
        possible_config_paths = [
            '/root/miniconda3/lib/python3.10/site-packages/mmdet/.mim/configs/deformable_detr/deformable-detr_r50_16xb2-50e_coco.py',
            str(Path(__file__).parent.parent.parent / 'experiments' / 'deformable-detr' / 'config.py'),
        ]
        for config_path in possible_config_paths:
            if os.path.exists(config_path):
                cfg = Config.fromfile(config_path)
                break
        
        if cfg is None:
            # 使用硬编码的配置（ResNet-18 + Deformable DETR）
            # 这需要与 train_deformable_r18.py 中的配置保持一致
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'experiments' / 'deformable-detr'))
            try:
                # 尝试从训练脚本导入配置
                from train_deformable_r18 import main as train_main
                # 创建一个临时配置（简化版）
                cfg = Config(dict(
                    model=dict(
                        type='DeformableDETR',
                        backbone=dict(
                            type='ResNet',
                            depth=18,
                            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')
                        ),
                        neck=dict(
                            type='ChannelMapper',
                            in_channels=[128, 256, 512]
                        ),
                        bbox_head=dict(
                            num_classes=8
                        ),
                        num_queries=100
                    )
                ))
            except:
                raise FileNotFoundError("无法找到 Deformable-DETR 配置文件，请使用 --config 参数指定")
    
    # 确保配置正确（R18 设置）
    if hasattr(cfg, 'model'):
        if hasattr(cfg.model, 'backbone'):
            cfg.model.backbone.depth = 18
            cfg.model.backbone.init_cfg = dict(type='Pretrained', checkpoint='torchvision://resnet18')
        if hasattr(cfg.model, 'neck'):
            cfg.model.neck.in_channels = [128, 256, 512]
        if hasattr(cfg.model, 'bbox_head'):
            cfg.model.bbox_head.num_classes = 8
        if hasattr(cfg.model, 'num_queries'):
            cfg.model.num_queries = 100
    
    # 创建模型
    model = MODELS.build(cfg.model)
    
    # 加载权重
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(device)
    
    return model


def load_yolov8_model(checkpoint_path: str, device: str = "cuda"):
    """
    加载 YOLOv8 模型
    
    Args:
        checkpoint_path: 检查点路径（.pt 或 .pth 文件）
        device: 设备
    
    Returns:
        加载的模型
    """
    try:
        # 导入 ultralytics（从 yolov8 目录）
        yolov8_dir = Path(__file__).parent.parent.parent / "experiments" / "yolov8"
        if str(yolov8_dir) not in sys.path:
            sys.path.insert(0, str(yolov8_dir))
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("需要安装 ultralytics 或确保 yolov8/ultralytics 目录存在")
    
    # 加载模型
    model = YOLO(str(checkpoint_path))
    model.to(device)
    model.eval()
    
    return model


def load_yolov10_model(checkpoint_path: str, device: str = "cuda"):
    """
    加载 YOLOv10 模型
    
    Args:
        checkpoint_path: 检查点路径（.pt 或 .pth 文件）
        device: 设备
    
    Returns:
        加载的模型
    """
    try:
        # 导入 ultralytics（从 yolov10 目录）
        yolov10_dir = Path(__file__).parent.parent.parent / "experiments" / "yolov10"
        if str(yolov10_dir) not in sys.path:
            sys.path.insert(0, str(yolov10_dir))
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("需要安装 ultralytics 或确保 yolov10/ultralytics 目录存在")
    
    # 加载模型
    model = YOLO(str(checkpoint_path))
    model.to(device)
    model.eval()
    
    return model


def parse_training_log(csv_path: str, is_yolo: bool = False) -> Dict[str, float]:
    """
    从 training_history.csv 解析最佳性能指标
    
    Args:
        csv_path: CSV 文件路径
        is_yolo: 是否为 YOLO 模型（使用不同的列名格式）
    
    Returns:
        包含最佳指标的字典（已归一化到 0-1 范围）
    """
    try:
        df = pd.read_csv(csv_path)
        
        # 根据模型类型选择列名（支持多种变体）
        if is_yolo:
            # YOLO 格式：metrics/mAP50-95(B), metrics/mAP50(B)
            map_col = 'metrics/mAP50-95(B)'
            ap50_col = 'metrics/mAP50(B)'
            # 尝试查找 APS 列（可能不存在）
            aps_col = None
            for col in df.columns:
                if 'mAP' in col and ('small' in col.lower() or 's' in col.lower()):
                    aps_col = col
                    break
        else:
            # DETR 格式：支持多种列名变体
            # mAP 列：优先 mAP_0.5_0.95，也支持 mAP_0.5:0.95
            map_col = None
            for col in df.columns:
                if 'mAP' in col and ('0.5' in col or '50' in col) and ('0.95' in col or '95' in col):
                    map_col = col
                    break
            if map_col is None:
                map_col = 'mAP_0.5_0.95'  # 默认
            
            # AP50 列：优先 mAP_0.5，也支持 mAP_50, mAP50, mAP_50 等变体
            ap50_col = None
            # 按优先级尝试匹配
            ap50_candidates = ['mAP_0.5', 'mAP_50', 'mAP50', 'mAP@0.5', 'mAP@50']
            for candidate in ap50_candidates:
                if candidate in df.columns:
                    ap50_col = candidate
                    break
            # 如果精确匹配失败，尝试模糊匹配
            if ap50_col is None:
                for col in df.columns:
                    if ('mAP' in col or 'AP' in col) and ('0.5' in col or '50' in col) and '95' not in col and '75' not in col:
                        ap50_col = col
                        break
            if ap50_col is None:
                ap50_col = 'mAP_0.5'  # 默认
            
            # APS 列：支持 mAP_s, mAP_small, mAP_S 等
            aps_col = None
            for col in df.columns:
                if 'mAP' in col and ('small' in col.lower() or col.endswith('_s') or col == 'mAP_s'):
                    aps_col = col
                    break
            if aps_col is None:
                aps_col = 'mAP_s'  # 默认
        
        # 找到 mAP 最大的行
        if map_col not in df.columns:
            print(f"警告: {csv_path} 中未找到 {map_col} 列")
            print(f"  可用列: {list(df.columns)}")
            return {'mAP': 0.0, 'AP50': 0.0, 'APS': 0.0}
        
        # 过滤掉 mAP 为 0 的行
        valid_df = df[df[map_col] > 0]
        
        if len(valid_df) == 0:
            print(f"警告: {csv_path} 中没有有效的 mAP 数据")
            return {'mAP': 0.0, 'AP50': 0.0, 'APS': 0.0}
        
        # 找到最佳 mAP
        best_idx = valid_df[map_col].idxmax()
        best_row = valid_df.loc[best_idx]
        
        # 提取指标值
        map_val = best_row.get(map_col, 0.0)
        ap50_val = best_row.get(ap50_col, 0.0) if ap50_col in df.columns else 0.0
        aps_val = best_row.get(aps_col, 0.0) if aps_col and aps_col in df.columns else 0.0
        
        # 数值归一化：如果值 > 1.0，可能是百分比格式（如 65.2），需要除以 100
        def normalize_value(val):
            if val > 1.0:
                return val / 100.0
            return val
        
        metrics = {
            'mAP': normalize_value(map_val),
            'AP50': normalize_value(ap50_val),
            'APS': normalize_value(aps_val)
        }
        
        return metrics
    except Exception as e:
        print(f"警告: 解析训练日志失败 {csv_path}: {e}")
        import traceback
        traceback.print_exc()
        return {'mAP': 0.0, 'AP50': 0.0, 'APS': 0.0}


def find_latest_best_model(logs_dir: Path, model_type: str = "dset") -> Optional[Tuple[Path, Path]]:
    """
    在 logs 目录下找到最新的 best_model.pth 和对应的 training_history.csv
    
    Args:
        logs_dir: logs 目录路径
        model_type: 模型类型 ("dset", "rtdetr", "deformable-detr", "yolov8s", "yolov8m", "yolov10s", "yolov10m")
    
    Returns:
        (best_model_path, training_history_path) 或 None
    """
    if model_type == "deformable-detr":
        # Deformable-DETR 使用 work_dirs 目录，检查点格式不同
        work_dirs = list(logs_dir.rglob('work_dirs'))
        if not work_dirs:
            # 尝试直接查找 best_*.pth
            best_models = list(logs_dir.rglob('best_*.pth'))
        else:
            # 在 work_dirs 下查找
            best_models = []
            for work_dir in work_dirs:
                best_models.extend(list(work_dir.rglob('best_*.pth')))
    elif model_type.startswith("yolov8") or model_type.startswith("yolov10"):
        # YOLO 模型：查找 best_model.pth 或 weights/best.pt
        best_models = []
        # 优先查找 best_model.pth（统一命名）
        best_models.extend(list(logs_dir.rglob('best_model.pth')))
        # 如果没有找到，查找 weights/best.pt
        if not best_models:
            best_models.extend(list(logs_dir.rglob('weights/best.pt')))
    else:
        # DSET 和 RT-DETR 使用 best_model.pth
        best_models = list(logs_dir.rglob('best_model.pth'))
    
    if not best_models:
        return None
    
    # 按修改时间排序，取最新的
    best_models.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    best_model_path = best_models[0]
    
    # 找到同目录下的 training_history.csv
    training_history_path = None
    if model_type != "deformable-detr":
        log_dir = best_model_path.parent
        # 如果检查点在 weights/ 子目录，向上查找
        if log_dir.name == "weights":
            log_dir = log_dir.parent
        training_history_path = log_dir / 'training_history.csv'
        
        if not training_history_path.exists():
            print(f"警告: 未找到对应的 training_history.csv: {training_history_path}")
            return (best_model_path, None)
    
    return (best_model_path, training_history_path)




def main():
    parser = argparse.ArgumentParser(description='生成性能对比表')
    parser.add_argument('--logs_dir', type=str, 
                       default='experiments/dset/logs',
                       help='logs 目录路径（相对于项目根目录）')
    parser.add_argument('--config', type=str,
                       default='experiments/dset/configs/dset4_r18_ratio0.5.yaml',
                       help='DSET 配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='指定检查点路径（如果未指定，将自动查找最新的）')
    parser.add_argument('--input_size', type=int, nargs=2, default=[720, 1280],
                       help='输入图像尺寸 [height, width] (默认: 720 1280，对应 1280x720 缩放后尺寸)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda 或 cpu)')
    parser.add_argument('--fps_iter', type=int, default=100,
                       help='FPS 测试迭代次数')
    parser.add_argument('--warmup_iter', type=int, default=10,
                       help='FPS 测试预热迭代次数')
    parser.add_argument('--skip_fps', action='store_true',
                       help='跳过 FPS 测试（仅使用已有数据）')
    parser.add_argument('--model_type', type=str, default='dset',
                       choices=['dset', 'rtdetr', 'deformable-detr', 
                               'yolov8s', 'yolov8m', 'yolov10s', 'yolov10m'],
                       help='模型类型: dset, rtdetr, deformable-detr, yolov8s, yolov8m, yolov10s, yolov10m')
    parser.add_argument('--rtdetr_config', type=str, default=None,
                       help='RT-DETR 配置文件路径（当 model_type=rtdetr 时使用）')
    parser.add_argument('--deformable_work_dir', type=str, default=None,
                       help='Deformable-DETR work_dirs 路径（当 model_type=deformable-detr 时使用）')
    parser.add_argument('--deformable_config', type=str, default=None,
                       help='Deformable-DETR 配置文件路径（可选）')
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    project_root = Path(__file__).parent.parent.parent.resolve()
    logs_dir = project_root / args.logs_dir
    config_path = project_root / args.config
    
    print("=" * 80)
    print("性能对比表生成脚本")
    print("=" * 80)
    
    # 1. 查找或使用指定的检查点
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = project_root / checkpoint_path
        if args.model_type != "deformable-detr":
            training_history_path = checkpoint_path.parent / 'training_history.csv'
        else:
            training_history_path = None
    else:
        print(f"\n查找最新的检查点在: {logs_dir} (模型类型: {args.model_type})")
        result = find_latest_best_model(logs_dir, args.model_type)
        if result is None:
            print(f"错误: 在 {logs_dir} 下未找到检查点")
            return
        checkpoint_path, training_history_path = result
    
    print(f"✓ 找到检查点: {checkpoint_path}")
    if training_history_path and training_history_path.exists():
        print(f"✓ 找到训练日志: {training_history_path}")
    
    # 2. 加载模型并计算参数量
    print(f"\n加载模型: {checkpoint_path}")
    is_yolo_model = args.model_type.startswith("yolov8") or args.model_type.startswith("yolov10")
    try:
        if args.model_type == "dset":
            model = load_dset_model(str(config_path), str(checkpoint_path), args.device)
        elif args.model_type == "rtdetr":
            rtdetr_config = args.rtdetr_config or config_path
            model = load_rtdetr_model(str(rtdetr_config), str(checkpoint_path), args.device)
        elif args.model_type == "deformable-detr":
            deformable_config = args.deformable_config
            model = load_deformable_detr_model(str(checkpoint_path), args.device, config_path=deformable_config)
        elif args.model_type.startswith("yolov8"):
            model = load_yolov8_model(str(checkpoint_path), args.device)
        elif args.model_type.startswith("yolov10"):
            model = load_yolov10_model(str(checkpoint_path), args.device)
        else:
            raise ValueError(f"不支持的模型类型: {args.model_type}")
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"错误: 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 计算参数量和 FLOPs
    # 注意：对于 YOLO 模型，FLOPs 计算使用实际输入尺寸（720x1280）
    # 但 FPS 测速会使用 1280x1280（包含 Letterbox 填充开销）
    input_size = (1, 3, args.input_size[0], args.input_size[1])
    print(f"\n计算模型信息 (输入尺寸: {input_size[2]}x{input_size[3]})...")
    if is_yolo_model:
        print(f"  注意: YOLO 模型 FPS 测速将使用 1280x1280 以包含 Letterbox 填充开销")
    params_m, flops_g = get_model_info(model, input_size, is_yolo=is_yolo_model)
    print(f"✓ 参数量: {params_m:.2f} M")
    if flops_g > 0:
        print(f"✓ FLOPs: {flops_g:.2f} G")
    else:
        print(f"⚠ FLOPs: 未计算（可能需要安装 thop: pip install thop）")
    
    # 4. 测量 FPS
    fps = 0.0
    if not args.skip_fps:
        print(f"\n测量 FPS (迭代 {args.fps_iter} 次, 预热 {args.warmup_iter} 次)...")
        try:
            fps = measure_fps(model, input_size, args.fps_iter, args.warmup_iter, args.device, is_yolo=is_yolo_model)
            print(f"✓ FPS: {fps:.1f}")
        except Exception as e:
            print(f"警告: FPS 测量失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n跳过 FPS 测试")
    
    # 5. 解析训练日志获取 mAP 指标
    metrics = {'mAP': 0.0, 'AP50': 0.0, 'APS': 0.0}
    if args.model_type != "deformable-detr" and training_history_path and training_history_path.exists():
        print(f"\n解析训练日志: {training_history_path}")
        metrics = parse_training_log(str(training_history_path), is_yolo=is_yolo_model)
        print(f"✓ mAP: {metrics['mAP']:.4f}")
        print(f"✓ AP50: {metrics['AP50']:.4f}")
        print(f"✓ APS: {metrics['APS']:.4f}")
    elif args.model_type == "deformable-detr":
        # Deformable-DETR 使用 MMEngine 的日志格式，需要从其他地方读取
        print("\n注意: Deformable-DETR 使用 MMEngine 日志格式，需要手动填入 mAP 指标")
        print("   或者从训练日志文件中提取")
    else:
        print("\n警告: 未找到训练日志，mAP 指标将使用默认值 0.0")
    
    # 6. 输出评估结果
    # 确定实际测试分辨率（YOLO 使用填充后的尺寸）
    if is_yolo_model:
        test_resolution = "1280 x 1280"
        resolution_note = " (YOLO 使用 Letterbox 填充到 1280x1280)"
    else:
        test_resolution = f"{args.input_size[1]} x {args.input_size[0]}"
        resolution_note = ""
    
    print("\n" + "=" * 60)
    print(f"评估结果 ({test_resolution}){resolution_note}")
    print("=" * 60)
    print()
    
    # 根据模型类型显示不同的名称
    model_name_map = {
        'dset': 'Ours (DSET-R18)',
        'rtdetr': 'RT-DETR-R18',
        'deformable-detr': 'Deformable-DETR-R18',
        'yolov8s': 'YOLOv8-s',
        'yolov8m': 'YOLOv8-m',
        'yolov10s': 'YOLOv10-s',
        'yolov10m': 'YOLOv10-m'
    }
    model_display_name = model_name_map.get(args.model_type, args.model_type.upper())
    
    print(f"模型名称: {model_display_name}")
    print()
    print(f"参数量 (Params): {params_m:.2f} M")
    if flops_g > 0:
        print(f"计算量 (FLOPs):  {flops_g:.2f} G")
    else:
        print(f"计算量 (FLOPs):  N/A")
    if fps > 0:
        print(f"推理速度 (FPS):  {fps:.1f}")
    else:
        print(f"推理速度 (FPS):  N/A")
    print()
    print("-" * 60)
    print()
    print(f"mAP (0.5:0.95): {metrics['mAP']:.4f}")
    print(f"AP50:           {metrics['AP50']:.4f}")
    print(f"APS (Small):    {metrics['APS']:.4f}")
    print()
    print("=" * 60)


if __name__ == '__main__':
    main()

