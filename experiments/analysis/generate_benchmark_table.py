#!/usr/bin/env python3
"""
模型性能评估脚本
支持多种模型的性能指标提取和测试

功能：
1. 自动从 logs/ 目录查找最新的 best_model.pth 或使用指定的检查点
2. 使用 pycocotools 在验证集上运行完整的 COCO 评估，提取 mAP, AP50, APS 指标
3. 计算模型参数量（Params）和 FLOPs
4. 测量模型在当前设备上的 FPS（包含 warmup 过程，YOLO 模型包含 NMS）
5. 输出清晰的评估结果清单

支持的模型类型：
- dset: DSET (Dual-Sparse Expert Transformer) 模型
- rtdetr: RT-DETRv2 模型
- deformable-detr: Deformable-DETR 模型（使用 MMEngine）
- yolov8s: YOLOv8-small 模型
- yolov8m: YOLOv8-medium 模型
- yolov10s: YOLOv10-small 模型
- yolov10m: YOLOv10-medium 模型

使用方法：
    # DSET 模型（默认）
    python generate_benchmark_table.py --model_type dset
    
    # RT-DETRv2 模型
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
- 精度指标（mAP [0.5:0.95], AP50, APS）通过 pycocotools 在验证集上实时评测得出，确保数据权威性
- 需要安装 thop 库来计算 FLOPs: pip install thop
- 需要安装 pycocotools 进行 COCO 评估: pip install pycocotools
- FLOPs 计算仅反映模型网络结构的计算量（MACs），不包含 NMS 等后处理操作
- YOLO 模型的 FPS 测量包含完整的推理流程（前向传播 + NMS）
- DSET 模型会自动启用 token pruning（设置 epoch >= warmup_epochs）以确保评估反映剪枝后的状态
- 输入尺寸逻辑：
  * DSET/RT-DETRv2: FLOPs 和 FPS 均使用 736x1280（对应 1280x720 有效分辨率，736 是 32 的倍数）
  * YOLO: FLOPs 使用 736x1280（公平比较），FPS 使用 1280x1280（包含 Letterbox 填充开销）
- FPS 测量使用 CUDA Events 进行精确计时（如果使用 CUDA）
- 输出格式符合学术论文标准，可直接复制使用
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from io import StringIO
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

# 显式设置项目根目录（必须在所有本地导入之前）
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


def get_model_info(model, input_size: Tuple[int, int, int, int] = (1, 3, 736, 1280), is_yolo: bool = False) -> Tuple[float, float]:
    """
    计算模型的参数量和 FLOPs
    
    注意：FLOPs 仅反映模型网络结构的计算量，不包含 NMS 等后处理操作
    
    Args:
        model: PyTorch 模型或 YOLO 模型
        input_size: 输入尺寸 (batch, channels, height, width)，默认 (1, 3, 736, 1280)
        is_yolo: 是否为 YOLO 模型
    
    Returns:
        (params_in_millions, flops_in_giga)
        - flops_in_giga 仅包含模型本身的 FLOPs（MACs），符合主流学术论文统计口径
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
        # YOLO 模型：使用 ultralytics 的方法计算 FLOPs
        # 注意：FLOPs 计算使用实际输入尺寸，不考虑 Letterbox 填充
        # FLOPs 仅反映模型本身的计算量，不包含 NMS 等后处理操作
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
                    print(f"  ✓ 使用 ultralytics.get_flops 计算 FLOPs: {flops_g:.2f} G")
                except (ImportError, AttributeError):
                    # 方法2：使用 thop 直接计算
                    if HAS_THOP:
                        pytorch_model = pytorch_model.eval()
                        device = next(pytorch_model.parameters()).device
                        dummy_input = torch.randn(input_size).to(device)
                        flops, _ = profile(deepcopy(pytorch_model), inputs=(dummy_input,), verbose=False)
                        # thop 返回的是 MACs，在主流学术论文中 MACs = FLOPs（不需要 *2）
                        flops_g = flops / 1e9
                        print(f"  ✓ 使用 thop 计算 FLOPs: {flops_g:.2f} G (MACs)")
                    else:
                        print(f"  ⚠ thop 未安装，无法计算 FLOPs")
            else:
                print(f"  ⚠ 无法获取 YOLO 模型的 PyTorch 模型")
        except Exception as e:
            print(f"  ⚠ YOLO FLOPs 计算失败: {e}")
            import traceback
            traceback.print_exc()
    elif HAS_THOP:
        # 标准 PyTorch 模型（DSET, RT-DETRv2, Deformable-DETR）
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
                input_size: Tuple[int, int, int, int] = (1, 3, 736, 1280),
                num_iter: int = 100,
                warmup_iter: int = 20,
                device: str = "cuda",
                is_yolo: bool = False) -> float:
    """
    测量模型的 FPS (Frames Per Second)
    
    使用 CUDA Events 进行精确计时（如果使用 CUDA），确保获取真实的 GPU 硬件执行耗时。
    对于 YOLO 模型，FPS 测量包含完整的推理流程（前向传播 + NMS）
    
    Args:
        model: PyTorch 模型或 YOLO 模型
        input_size: 输入尺寸 (batch, channels, height, width)，默认 (1, 3, 736, 1280)
        num_iter: 测试迭代次数
        warmup_iter: 预热迭代次数（默认 20，确保模型充分预热）
        device: 设备 ('cuda' 或 'cpu')
        is_yolo: 是否为 YOLO 模型（使用不同的输入格式，且包含 NMS）
    
    Returns:
        FPS 值（对于 YOLO 模型，包含 NMS 的完整推理时间）
    """
    model.eval()
    
    # 使用 CUDA Events 进行精确计时（如果使用 CUDA）
    use_cuda_events = (device == "cuda" and torch.cuda.is_available())
    
    if is_yolo:
        # YOLO 模型使用 numpy/PIL 图像格式
        # 注意：YOLO 模型的 model() 调用已经包含完整的推理流程（前向传播 + NMS）
        # 确保模型在正确的设备上
        model = model.to(device)
        
        import numpy as np
        # YOLO 通常需要正方形输入，使用 Letterbox 填充
        # 为了包含填充开销，使用最大边作为正方形边长进行测速
        # 注意：实际输入可能是 736x1280，但 YOLO 会填充到 1280x1280
        yolo_size = max(input_size[2], input_size[3])  # 使用最大边作为正方形边长
        dummy_image = np.random.randint(0, 255, (yolo_size, yolo_size, 3), dtype=np.uint8)
        
        # Warmup（包含 NMS）
        for _ in range(warmup_iter):
            _ = model(dummy_image, verbose=False)  # 这会执行完整的推理流程，包括 NMS
            if use_cuda_events:
                torch.cuda.synchronize()
        
        # 同步（如果是 CUDA）
        if use_cuda_events:
            torch.cuda.synchronize()
        
        # 实际测试（包含 NMS）
        if use_cuda_events:
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
        else:
            start_time = time.time()
        
        for _ in range(num_iter):
            _ = model(dummy_image, verbose=False)  # 包含 NMS 的完整推理
            if use_cuda_events:
                torch.cuda.synchronize()  # 每轮都同步，确保精确计时
        
        if use_cuda_events:
            ender.record()
            torch.cuda.synchronize()
            elapsed_time = starter.elapsed_time(ender) / 1000.0  # 转换为秒
        else:
            end_time = time.time()
            elapsed_time = end_time - start_time
    else:
        # 标准 PyTorch 模型使用 tensor
        # 显式确保模型和输入都在正确的设备上
        model = model.to(device)
        dummy_input = torch.randn(input_size).to(device)
        # 再次确保设备一致性（防止 thop 计算后的残留状态）
        model = model.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iter):
                _ = model(dummy_input)
                if use_cuda_events:
                    torch.cuda.synchronize()
        
        # 同步（如果是 CUDA）
        if use_cuda_events:
            torch.cuda.synchronize()
        
        # 实际测试
        if use_cuda_events:
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
        else:
            start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iter):
                _ = model(dummy_input)
                if use_cuda_events:
                    torch.cuda.synchronize()  # 每轮都同步，确保精确计时
        
        if use_cuda_events:
            ender.record()
            torch.cuda.synchronize()
            elapsed_time = starter.elapsed_time(ender) / 1000.0  # 转换为秒
        else:
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
    # 导入必要的模块 - 优先使用项目根目录的导入路径
    try:
        # 首先尝试从项目根目录导入（推荐方式）
        from experiments.dset.train import DSETTrainer
    except ImportError:
        # 回退：尝试从 dset 目录直接导入
        # config_path 通常是 experiments/dset/configs/xxx.yaml
        # parent.parent 就是 experiments/dset/，这是包含 train.py 的目录
        dset_dir = Path(config_path).parent.parent
        if str(dset_dir) not in sys.path:
            sys.path.insert(0, str(dset_dir))
        print(f"DEBUG: 正在尝试从 {dset_dir} 加载 DSETTrainer")
        try:
            from train import DSETTrainer
        except ImportError:
            raise ImportError(f"无法导入 DSETTrainer。请确保项目根目录正确设置: {project_root}")
    
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
    # 关键：必须在评估前激活 token pruning，确保精度反映剪枝后的状态
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'set_epoch'):
        dset_config = config.get('model', {}).get('dset', {})
        warmup_epochs = dset_config.get('token_pruning_warmup_epochs', 10)
        # 设置为 warmup_epochs 或更大的值以启用 pruning（使用 10 确保激活）
        model.encoder.set_epoch(max(warmup_epochs, 10))
        
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
    加载 RT-DETRv2 模型
    
    Args:
        config_path: 配置文件路径
        checkpoint_path: 检查点路径
        device: 设备
    
    Returns:
        加载的模型
    """
    # 导入必要的模块
    # config_path 通常是 experiments/rt-detr/configs/xxx.yaml
    # parent.parent 就是 experiments/rt-detr/，这是包含 train.py 的目录
    rtdetr_dir = Path(config_path).parent.parent
    if str(rtdetr_dir) not in sys.path:
        sys.path.insert(0, str(rtdetr_dir))
    print(f"DEBUG: 正在尝试从 {rtdetr_dir} 加载 RTDETRTrainer")
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


def evaluate_accuracy(model, config_path: str, device: str = "cuda", model_type: str = "dset") -> Dict[str, float]:
    """
    使用 pycocotools 在验证集上运行完整的 COCO 评估循环
    
    确保使用与训练相同的图像缩放逻辑（max_size=1280，保持宽高比，结果 1280x720）
    
    Args:
        model: 已加载的模型（必须处于 eval 模式，token pruning 已激活）
        config_path: 配置文件路径
        device: 设备
        model_type: 模型类型 ("dset" 或 "rtdetr")
    
    Returns:
        包含 mAP [0.5:0.95], AP50, APS (Small) 的字典
    """
    try:
        # 根据模型类型选择正确的 trainer
        if model_type == "dset":
            # 导入 DSETTrainer 以创建数据加载器 - 优先使用项目根目录导入
            try:
                from experiments.dset.train import DSETTrainer
            except ImportError:
                # 回退：尝试从 dset 目录直接导入
                # config_path 通常是 experiments/dset/configs/xxx.yaml
                # parent.parent 就是 experiments/dset/，这是包含 train.py 的目录
                dset_dir = Path(config_path).parent.parent
                if str(dset_dir) not in sys.path:
                    sys.path.insert(0, str(dset_dir))
                print(f"DEBUG: 正在尝试从 {dset_dir} 加载 DSETTrainer")
                try:
                    from train import DSETTrainer
                except ImportError:
                    raise ImportError(f"无法导入 DSETTrainer。请确保项目根目录正确设置: {project_root}")
            TrainerClass = DSETTrainer
        elif model_type == "rtdetr":
            # 导入 RTDETRTrainer 以创建数据加载器
            try:
                from experiments.rt_detr.train import RTDETRTrainer
            except ImportError:
                # 回退：尝试从 rt-detr 目录直接导入
                # config_path 通常是 experiments/rt-detr/configs/xxx.yaml
                # parent.parent 就是 experiments/rt-detr/，这是包含 train.py 的目录
                rtdetr_dir = Path(config_path).parent.parent
                if str(rtdetr_dir) not in sys.path:
                    sys.path.insert(0, str(rtdetr_dir))
                print(f"DEBUG: 正在尝试从 {rtdetr_dir} 加载 RTDETRTrainer")
                try:
                    from train import RTDETRTrainer
                except ImportError:
                    raise ImportError(f"无法导入 RTDETRTrainer。请确保项目根目录正确设置: {project_root}")
            TrainerClass = RTDETRTrainer
        else:
            raise ValueError(f"不支持的模型类型: {model_type}，仅支持 'dset' 或 'rtdetr'")
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 确保 device 配置正确
        if 'misc' not in config:
            config['misc'] = {}
        config['misc']['device'] = device
        
        # 创建 trainer 以获取数据加载器（使用与训练相同的配置）
        if model_type == "dset":
            trainer = TrainerClass(config, config_file_path=str(config_path))
        else:
            trainer = TrainerClass(config)
        _, val_loader = trainer._create_data_loaders()
        
        print(f"  ✓ 创建验证集 DataLoader (样本数: {len(val_loader.dataset)})")
        print(f"  ✓ 使用与训练相同的图像缩放逻辑 (max_size=1280, 保持宽高比 → 1280x720)")
        
        # 确保模型处于 eval 模式（token pruning 应在 load_dset_model 中已激活）
        model.eval()
        
        # 验证 token pruning 状态（仅 DSET 模型）
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'set_epoch'):
            if hasattr(model.encoder, 'token_pruners') and model.encoder.token_pruners:
                pruner = model.encoder.token_pruners[0]
                if hasattr(pruner, 'pruning_enabled'):
                    print(f"  ✓ Token Pruning 状态: {'已激活' if pruner.pruning_enabled else '未激活'}")
        
        all_predictions = []
        all_targets = []
        
        print(f"  运行推理循环...")
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                B, C, H_tensor, W_tensor = images.shape
                images = images.to(device)
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                           for k, v in t.items()} for t in targets]
                
                outputs = model(images, targets)
                
                # 收集预测结果
                if isinstance(outputs, dict) and 'class_scores' in outputs and 'bboxes' in outputs:
                    _collect_predictions_for_coco(
                        outputs, targets, batch_idx, all_predictions, all_targets,
                        W_tensor, H_tensor, config.get('training', {}).get('batch_size', 1)
                    )
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"    处理进度: {batch_idx + 1}/{len(val_loader)} batches")
        
        print(f"  ✓ 收集到 {len(all_predictions)} 个预测框, {len(all_targets)} 个真实标注")
        
        # 使用 pycocotools 评估
        metrics = _compute_coco_metrics(all_predictions, all_targets, H_tensor, W_tensor)
        
        return metrics
        
    except Exception as e:
        print(f"  ⚠ COCO 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return {'mAP': 0.0, 'AP50': 0.0, 'APS': 0.0}


def _collect_predictions_for_coco(outputs: Dict, targets: List[Dict], batch_idx: int,
                                  all_predictions: List, all_targets: List,
                                  img_w: int, img_h: int, batch_size: int) -> None:
    """收集预测结果用于 COCO 评估"""
    pred_logits = outputs['class_scores']  # [B, Q, C]
    pred_boxes = outputs['bboxes']  # [B, Q, 4]
    
    batch_size_actual = pred_logits.shape[0]
    
    for i in range(batch_size_actual):
        pred_scores = torch.softmax(pred_logits[i], dim=-1)  # [Q, C]
        max_scores, pred_classes = torch.max(pred_scores, dim=-1)  # [Q]
        
        # 过滤无效框（padding框）
        valid_boxes_mask = ~torch.all(pred_boxes[i] == 1.0, dim=1)
        valid_indices = torch.where(valid_boxes_mask)[0]
        
        if len(valid_indices) > 0:
            filtered_boxes = pred_boxes[i][valid_indices]
            filtered_classes = pred_classes[valid_indices]
            filtered_scores = max_scores[valid_indices]
            
            # 转换为COCO格式 (x, y, w, h)
            if filtered_boxes.shape[0] > 0:
                boxes_coco = torch.zeros_like(filtered_boxes)
                if filtered_boxes.max() <= 1.0:
                    # 归一化坐标 -> 像素坐标
                    boxes_coco[:, 0] = (filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2) * img_w
                    boxes_coco[:, 1] = (filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2) * img_h
                    boxes_coco[:, 2] = filtered_boxes[:, 2] * img_w
                    boxes_coco[:, 3] = filtered_boxes[:, 3] * img_h
                else:
                    boxes_coco = filtered_boxes.clone()
                
                # Clamp坐标
                boxes_coco[:, 0] = torch.clamp(boxes_coco[:, 0], 0, img_w)
                boxes_coco[:, 1] = torch.clamp(boxes_coco[:, 1], 0, img_h)
                boxes_coco[:, 2] = torch.clamp(boxes_coco[:, 2], 1, img_w)
                boxes_coco[:, 3] = torch.clamp(boxes_coco[:, 3], 1, img_h)
                
                # 转换为 (x, y, w, h) 格式
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
                iscrowd_values = targets[i]['iscrowd'] if has_iscrowd else torch.zeros(len(true_labels), dtype=torch.int64)
                
                for j in range(len(true_labels)):
                    x, y, w, h = true_boxes_coco[j].cpu().numpy()
                    ann_dict = {
                        'image_id': batch_idx * batch_size + i,
                        'category_id': int(true_labels[j].item()) + 1,
                        'bbox': [float(x), float(y), float(w), float(h)],
                        'area': float(w * h)
                    }
                    if has_iscrowd:
                        ann_dict['iscrowd'] = int(iscrowd_values[j].item())
                    all_targets.append(ann_dict)


def _compute_coco_metrics(predictions: List[Dict], targets: List[Dict],
                          img_h: int = 736, img_w: int = 1280) -> Dict[str, float]:
    """使用 pycocotools 计算 COCO 指标"""
    try:
        if len(predictions) == 0:
            return {'mAP': 0.0, 'AP50': 0.0, 'APS': 0.0}
        
        # 定义类别
        categories = [
            {'id': 1, 'name': 'Car'},
            {'id': 2, 'name': 'Truck'},
            {'id': 3, 'name': 'Van'},
            {'id': 4, 'name': 'Bus'},
            {'id': 5, 'name': 'Pedestrian'},
            {'id': 6, 'name': 'Cyclist'},
            {'id': 7, 'name': 'Motorcyclist'},
            {'id': 8, 'name': 'Trafficcone'}
        ]
        
        # 创建 COCO 格式的 GT
        image_ids = set(t['image_id'] for t in targets)
        coco_gt = {
            'images': [{'id': img_id, 'width': img_w, 'height': img_h} for img_id in image_ids],
            'annotations': targets,
            'categories': categories,
            'info': {'description': 'DAIR-V2X Dataset', 'version': '1.0', 'year': 2024}
        }
        
        # 创建 COCO 对象
        coco_gt_obj = COCO()
        coco_gt_obj.dataset = coco_gt
        
        # 抑制 createIndex 的输出
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            coco_gt_obj.createIndex()
        finally:
            sys.stdout = old_stdout
        
        # 加载预测结果
        sys.stdout = StringIO()
        try:
            coco_dt = coco_gt_obj.loadRes(predictions)
        finally:
            sys.stdout = old_stdout
        
        # 运行 COCOeval
        coco_eval = COCOeval(coco_gt_obj, coco_dt, 'bbox')
        
        # 抑制评估过程的输出
        sys.stdout = StringIO()
        try:
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        finally:
            sys.stdout = old_stdout
        
        # 从 COCOeval.stats 中提取指标
        # stats[0] = mAP@0.5:0.95
        # stats[1] = mAP@0.5 (AP50)
        # stats[3] = mAP_s (Small objects)
        stats = coco_eval.stats
        
        metrics = {
            'mAP': float(stats[0]) / 100.0 if len(stats) > 0 else 0.0,  # 转换为 0-1 范围
            'AP50': float(stats[1]) / 100.0 if len(stats) > 1 else 0.0,
            'APS': float(stats[3]) / 100.0 if len(stats) > 3 else 0.0
        }
        
        return metrics
        
    except Exception as e:
        print(f"  ⚠ COCO 指标计算失败: {e}")
        import traceback
        traceback.print_exc()
        return {'mAP': 0.0, 'AP50': 0.0, 'APS': 0.0}


def find_latest_best_model(logs_dir: Path, model_type: str = "dset") -> Optional[Path]:
    """
    在 logs 目录下找到最新的 best_model.pth
    
    Args:
        logs_dir: logs 目录路径
        model_type: 模型类型 ("dset", "rtdetr", "deformable-detr", "yolov8s", "yolov8m", "yolov10s", "yolov10m")
    
    Returns:
        best_model_path 或 None
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
        # DSET 和 RT-DETRv2 使用 best_model.pth
        best_models = list(logs_dir.rglob('best_model.pth'))
    
    if not best_models:
        return None
    
    # 按修改时间排序，取最新的
    best_models.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return best_models[0]




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
    parser.add_argument('--input_size', type=int, nargs=2, default=[736, 1280],
                       help='输入图像尺寸 [height, width] (默认: 736 1280，对应 1280x720 缩放后尺寸，736 是 32 的倍数)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda 或 cpu)')
    parser.add_argument('--fps_iter', type=int, default=100,
                       help='FPS 测试迭代次数')
    parser.add_argument('--warmup_iter', type=int, default=20,
                       help='FPS 测试预热迭代次数（默认 20，确保模型充分预热）')
    parser.add_argument('--skip_fps', action='store_true',
                       help='跳过 FPS 测试（仅使用已有数据）')
    parser.add_argument('--model_type', type=str, default='dset',
                       choices=['dset', 'rtdetr', 'deformable-detr', 
                               'yolov8s', 'yolov8m', 'yolov10s', 'yolov10m'],
                       help='模型类型: dset, rtdetr, deformable-detr, yolov8s, yolov8m, yolov10s, yolov10m')
    parser.add_argument('--rtdetr_config', type=str, default=None,
                       help='RT-DETRv2 配置文件路径（当 model_type=rtdetr 时使用）')
    parser.add_argument('--deformable_work_dir', type=str, default=None,
                       help='Deformable-DETR work_dirs 路径（当 model_type=deformable-detr 时使用）')
    parser.add_argument('--deformable_config', type=str, default=None,
                       help='Deformable-DETR 配置文件路径（可选）')
    
    args = parser.parse_args()
    
    # 转换为绝对路径（使用显式设置的项目根目录）
    logs_dir = Path(project_root) / args.logs_dir
    config_path = Path(project_root) / args.config
    
    print("=" * 80)
    print("性能对比表生成脚本")
    print("=" * 80)
    
    # 1. 查找或使用指定的检查点
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = project_root / checkpoint_path
    else:
        print(f"\n查找最新的检查点在: {logs_dir} (模型类型: {args.model_type})")
        result = find_latest_best_model(logs_dir, args.model_type)
        if result is None:
            print(f"错误: 在 {logs_dir} 下未找到检查点")
            return
        checkpoint_path = result
    
    print(f"✓ 找到检查点: {checkpoint_path}")
    
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
    # 输入尺寸逻辑：
    # - DSET/RT-DETRv2: FLOPs 和 FPS 都使用 (1, 3, 736, 1280)
    # - YOLO: FLOPs 使用 (1, 3, 736, 1280) 公平比较，FPS 使用 (1, 3, 1280, 1280) 包含 Letterbox 开销
    input_size_flops = (1, 3, args.input_size[0], args.input_size[1])  # 用于 FLOPs 计算
    input_size_fps = input_size_flops  # 默认与 FLOPs 相同
    
    if is_yolo_model:
        # YOLO FPS 测速使用 1280x1280 以包含 Letterbox 填充开销
        input_size_fps = (1, 3, max(args.input_size), max(args.input_size))
        print(f"\n计算模型信息...")
        print(f"  FLOPs 计算尺寸: {input_size_flops[2]}x{input_size_flops[3]} (公平比较)")
        print(f"  FPS 测速尺寸: {input_size_fps[2]}x{input_size_fps[3]} (包含 Letterbox 填充开销)")
    else:
        # DSET/RT-DETRv2: FLOPs 和 FPS 使用相同尺寸
        print(f"\n计算模型信息 (输入尺寸: {input_size_flops[2]}x{input_size_flops[3]})...")
    
    # 对于 DSET 模型，确保在计算 FLOPs 之前激活 token pruning
    if args.model_type == "dset" and hasattr(model, 'encoder') and hasattr(model.encoder, 'set_epoch'):
        # 从配置中获取 warmup_epochs
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            dset_config = config.get('model', {}).get('dset', {})
            warmup_epochs = dset_config.get('token_pruning_warmup_epochs', 10)
            model.encoder.set_epoch(max(warmup_epochs, 10))
            print(f"  ✓ 已激活 DSET token pruning (epoch={max(warmup_epochs, 10)}) 以正确计算 FLOPs")
        except Exception as e:
            print(f"  ⚠ 无法读取配置以激活 token pruning: {e}")
    
    params_m, flops_g = get_model_info(model, input_size_flops, is_yolo=is_yolo_model)
    print(f"✓ 参数量: {params_m:.2f} M")
    if flops_g > 0:
        print(f"✓ FLOPs: {flops_g:.2f} G")
    else:
        print(f"⚠ FLOPs: 未计算（可能需要安装 thop: pip install thop）")
    
    # 4. 测量 FPS（使用正确的输入尺寸）
    fps = 0.0
    if not args.skip_fps:
        print(f"\n测量 FPS (迭代 {args.fps_iter} 次, 预热 {args.warmup_iter} 次)...")
        try:
            fps = measure_fps(model, input_size_fps, args.fps_iter, args.warmup_iter, args.device, is_yolo=is_yolo_model)
            print(f"✓ FPS: {fps:.1f}")
        except Exception as e:
            print(f"警告: FPS 测量失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n跳过 FPS 测试")
    
    # 5. 使用 pycocotools 在验证集上运行评估
    metrics = {'mAP': 0.0, 'AP50': 0.0, 'APS': 0.0}
    if args.model_type == "dset":
        print(f"\n运行 COCO 评估 (使用验证集)...")
        metrics = evaluate_accuracy(model, str(config_path), args.device, model_type="dset")
        print(f"✓ mAP (0.5:0.95): {metrics['mAP']:.4f}")
        print(f"✓ AP50: {metrics['AP50']:.4f}")
        print(f"✓ APS (Small): {metrics['APS']:.4f}")
    elif args.model_type == "rtdetr":
        print(f"\n运行 COCO 评估 (使用验证集)...")
        # RT-DETRv2 使用 RTDETRTrainer 进行评估
        metrics = evaluate_accuracy(model, str(rtdetr_config), args.device, model_type="rtdetr")
        print(f"✓ mAP (0.5:0.95): {metrics['mAP']:.4f}")
        print(f"✓ AP50: {metrics['AP50']:.4f}")
        print(f"✓ APS (Small): {metrics['APS']:.4f}")
    elif args.model_type == "deformable-detr":
        print("\n注意: Deformable-DETR 评估需要 MMEngine 格式，当前版本暂不支持自动评估")
        print("   请手动填入 mAP 指标或扩展评估逻辑")
    elif is_yolo_model:
        print("\n注意: YOLO 模型评估需要 ultralytics 格式，当前版本暂不支持自动评估")
        print("   请手动填入 mAP 指标或扩展评估逻辑")
    else:
        print("\n警告: 未支持的模型类型，mAP 指标将使用默认值 0.0")
    
    # 6. 输出标准化评估结果（学术论文格式）
    
    # 格式化并输出标准化评估结果（学术论文格式）
    _format_evaluation_results(
        model_type=args.model_type,
        params_m=params_m,
        flops_g=flops_g,
        fps=fps,
        metrics=metrics,
        input_resolution=(args.input_size[1], args.input_size[0]),
        is_yolo=is_yolo_model
    )


def _format_evaluation_results(model_type: str, params_m: float, flops_g: float, fps: float,
                               metrics: Dict[str, float], input_resolution: Tuple[int, int],
                               is_yolo: bool = False) -> None:
    """
    格式化并输出标准化的评估结果（学术论文格式）
    
    Args:
        model_type: 模型类型
        params_m: 参数量（百万）
        flops_g: FLOPs（十亿）
        fps: FPS
        metrics: 包含 mAP, AP50, APS 的字典
        input_resolution: 输入分辨率 (width, height)
        is_yolo: 是否为 YOLO 模型
    """
    # 模型显示名称映射
    model_name_map = {
        'dset': 'Ours (DSET-v2-R18)',
        'rtdetr': 'RT-DETRv2-R18',
        'deformable-detr': 'Deformable-DETR-R18',
        'yolov8s': 'YOLOv8-s',
        'yolov8m': 'YOLOv8-m',
        'yolov10s': 'YOLOv10-s',
        'yolov10m': 'YOLOv10-m'
    }
    model_display_name = model_name_map.get(model_type, model_type.upper())
    
    # 确定有效输入分辨率
    if is_yolo:
        effective_resolution = f"{max(input_resolution)}x{max(input_resolution)}"
    else:
        effective_resolution = f"{input_resolution[0]}x{input_resolution[1]}"
    
    # 格式化输出
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS (Academic Standard)".center(70))
    print("=" * 70)
    print(f"Model: {model_display_name}")
    print(f"Input Resolution: {effective_resolution} (Effective)")
    print("-" * 70)
    
    # 硬件指标
    params_str = f"{params_m:.2f} M" if params_m > 0 else "N/A"
    flops_str = f"{flops_g:.2f} G" if flops_g > 0 else "N/A"
    fps_str = f"{fps:.1f}" if fps > 0 else "N/A"
    
    print(f"Params: {params_str} | FLOPs: {flops_str} | FPS (RTX 3090): {fps_str}")
    
    # 精度指标
    map_str = f"{metrics['mAP']:.4f}" if metrics['mAP'] > 0 else "N/A"
    ap50_str = f"{metrics['AP50']:.4f}" if metrics['AP50'] > 0 else "N/A"
    aps_str = f"{metrics['APS']:.4f}" if metrics['APS'] > 0 else "N/A"
    
    print(f"mAP [0.5:0.95]: {map_str} | AP50: {ap50_str} | APS (Small): {aps_str}")
    print("-" * 70)
    
    # 数据来源说明
    if model_type in ["dset", "rtdetr"]:
        print("Note: Accuracy evaluated via pycocotools. FPS measured with CUDA Events.")
    else:
        print("Note: FPS measured with CUDA Events.")
    
    print("=" * 70)


if __name__ == '__main__':
    main()

