#!/usr/bin/env python3
"""
模型理论效率评估脚本 - 支持 DSET, RT-DETR, Deformable-DETR, YOLOv8, YOLOv10

功能：
1. 自动从 logs/ 目录查找最新的 best_model.pth 或使用指定的检查点
2. 使用 pycocotools 在验证集上运行 COCO 评估（仅精度指标）
3. 计算模型参数量和理论 FLOPs（考虑 token pruning 和 MoE 稀疏性）
4. 使用配置文件中的 batch_size（COCO 评估结果与 batch_size 无关，可使用更大 batch_size 加速）

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


def get_model_info(model, input_size: Tuple[int, int, int, int] = (1, 3, 736, 1280), 
                   is_yolo: bool = False, config: Dict = None, model_type: str = "dset",
                   debug: bool = False) -> Tuple[float, float, float, float]:
    """
    计算模型的参数量和理论 FLOPs
    
    Returns:
        total_params_m: 总参数量 (M)
        active_params_m: 激活参数量 (M) - 考虑 MoE 后的实际参数
        base_flops_g: 基准 FLOPs (G) - 全量运行时的计算量
        theory_flops_g: 理论 FLOPs (G) - 考虑 token pruning 和 MoE 后的理论计算量
    """
    # 计算参数量
    if is_yolo and hasattr(model, 'model'):
        pytorch_model = model.model
    else:
        pytorch_model = model
    total_params = sum(p.numel() for p in pytorch_model.parameters())
    total_params_m = total_params / 1e6
    
    # 计算激活参数量（考虑 MoE）
    active_params = total_params
    if model_type == "dset" and config is not None:
        # 从配置获取 num_experts 和 top_k
        dset_config = config.get('model', {}).get('dset', {})
        encoder_experts = dset_config.get('encoder_moe_num_experts', 1)
        encoder_top_k = dset_config.get('encoder_moe_top_k', 1)
        decoder_experts = config.get('model', {}).get('num_experts', 1)
        decoder_top_k = config.get('model', {}).get('top_k', 3)
        
        # 分别统计 Encoder 和 Decoder 的专家参数
        encoder_expert_params = 0
        decoder_expert_params = 0
        
        # 遍历所有参数，识别专家参数
        for name, param in pytorch_model.named_parameters():
            # Encoder 专家参数：参数名包含 'encoder' 且含 'expert' 或 'encoder_moe'
            if 'encoder' in name.lower() and ('expert' in name.lower() or 'encoder_moe' in name.lower()):
                encoder_expert_params += param.numel()
            # Decoder 专家参数：参数名包含 'decoder' 且含 'moe_layer'
            elif 'decoder' in name.lower() and 'moe_layer' in name.lower():
                decoder_expert_params += param.numel()
        
        # 计算激活参数
        # Encoder: Top-K 路由，激活参数 = Expert_Params × min(top_k, experts) / Num_Experts
        # Decoder: Top-K 路由，激活参数 = Expert_Params × min(top_k, experts) / Num_Experts
        encoder_active_ratio = min(encoder_top_k, encoder_experts) / max(encoder_experts, 1) if encoder_experts > 1 else 1.0
        decoder_active_ratio = min(decoder_top_k, decoder_experts) / max(decoder_experts, 1) if decoder_experts > 1 else 1.0
        encoder_active = encoder_expert_params * encoder_active_ratio
        decoder_active = decoder_expert_params * decoder_active_ratio
        
        # 总激活参数 = 总参数 - 专家总参数 + Encoder激活部分 + Decoder激活部分
        total_expert_params = encoder_expert_params + decoder_expert_params
        active_params = (total_params - total_expert_params) + encoder_active + decoder_active
    
    active_params_m = active_params / 1e6
    
    # 计算基准 FLOPs（全量运行）
    base_flops_g = 0.0
    if is_yolo:
        try:
            from copy import deepcopy
            if hasattr(model, 'model'):
                pytorch_model = model.model
                h, w = input_size[2], input_size[3]
                imgsz = [h, w] if h != w else h
                try:
                    from ultralytics.utils.torch_utils import get_flops
                    base_flops_g = get_flops(pytorch_model, imgsz=imgsz)
                except (ImportError, AttributeError):
                    if HAS_THOP:
                        pytorch_model = pytorch_model.eval()
                        device = next(pytorch_model.parameters()).device
                        dummy_input = torch.randn(input_size).to(device)
                        flops, _ = profile(deepcopy(pytorch_model), inputs=(dummy_input,), verbose=False)
                        base_flops_g = flops / 1e9
        except Exception as e:
            print(f"  ⚠ YOLO FLOPs 计算失败: {e}")
    elif HAS_THOP:
        try:
            model.eval()
            device = next(model.parameters()).device
            dummy_input = torch.randn(input_size).to(device)
            flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
            base_flops_g = flops / 1e9
            print(f"  ✓ Base FLOPs: {base_flops_g:.2f} G")
        except Exception as e:
            # mmdet 模型通常无法直接 profile(model, (tensor,))，需要构造 wrapper
            try:
                is_mmdet_model = isinstance(getattr(model, '__class__', None), type) and \
                                ('mmdet' in getattr(model.__class__, '__module__', ''))
            except Exception:
                is_mmdet_model = False
            
            if is_mmdet_model:
                try:
                    # 优先使用 mmengine 的复杂度分析工具（如果存在）
                    from mmengine.analysis import get_model_complexity_info
                    _, _, H, W = input_size
                    # get_model_complexity_info 的 input_shape 一般是 (C, H, W)
                    analysis = get_model_complexity_info(
                        model, (3, int(H), int(W)),
                        as_strings=False,
                        print_per_layer_stat=False
                    )
                    # 不同版本返回值可能是 (flops, params) 或 dict
                    if isinstance(analysis, tuple) and len(analysis) >= 1:
                        flops = analysis[0]
                        base_flops_g = float(flops) / 1e9
                        print(f"  ✓ Base FLOPs(mmengine): {base_flops_g:.2f} G")
                    elif isinstance(analysis, dict) and 'flops' in analysis:
                        base_flops_g = float(analysis['flops']) / 1e9
                        print(f"  ✓ Base FLOPs(mmengine): {base_flops_g:.2f} G")
                except Exception:
                    try:
                        # 回退：用 thop 对 mmdet detector 做 wrapper profile（构造最小 data_samples）
                        from copy import deepcopy
                        from mmdet.structures import DetDataSample
                        _, _, H, W = input_size
                        sample = DetDataSample()
                        sample.set_metainfo({
                            'ori_shape': (int(H), int(W), 3),
                            'img_shape': (int(H), int(W), 3),
                            'pad_shape': (int(H), int(W), 3),
                            'scale_factor': (1.0, 1.0),
                            'batch_input_shape': (int(H), int(W)),
                        })
                        
                        class _MMDetWrapper(nn.Module):
                            def __init__(self, det_model, data_sample):
                                super().__init__()
                                self.det_model = det_model
                                self.data_sample = data_sample
                            def forward(self, x):
                                # mode='tensor' 通常返回 head 的原始 tensor 输出（便于 profile）
                                return self.det_model(x, [self.data_sample], mode='tensor')
                        
                        wrapped = _MMDetWrapper(deepcopy(model).eval().to(device), sample)
                        flops, _ = profile(wrapped, inputs=(dummy_input,), verbose=False)
                        base_flops_g = flops / 1e9
                        print(f"  ✓ Base FLOPs(thop+mmdet wrapper): {base_flops_g:.2f} G")
                    except Exception as e2:
                        print(f"  ⚠ FLOPs 计算失败(mmdet): {e2!r}")
            else:
                print(f"  ⚠ FLOPs 计算失败: {e!r}")
    
    # 计算理论 FLOPs（考虑 token pruning 和 MoE）
    theory_flops_g = base_flops_g
    if model_type == "dset" and config is not None and HAS_THOP:
        dset_config = config.get('model', {}).get('dset', {})
        token_keep_ratio = dset_config.get('token_keep_ratio', 1.0)
        encoder_experts = dset_config.get('encoder_moe_num_experts', 1)
        encoder_top_k = dset_config.get('encoder_moe_top_k', 1)
        decoder_experts = config.get('model', {}).get('num_experts', 1)
        decoder_top_k = config.get('model', {}).get('top_k', 3)
        
        # 如果 token_keep_ratio 是字典，取平均值或主要层的值
        if isinstance(token_keep_ratio, dict):
            # 取最大 key 对应的值（通常是 P5 层，即 layer 2）
            if token_keep_ratio:
                token_keep_ratio = max(token_keep_ratio.values())
            else:
                token_keep_ratio = 1.0
        
        try:
            model.eval()
            device = next(model.parameters()).device
            dummy_img = torch.randn(input_size).to(device)
            
            from copy import deepcopy
            
            # ========== 1. Backbone: 密集计算 ==========
            if not hasattr(model, 'backbone'):
                raise AttributeError("模型缺少 backbone 属性")
            
            backbone_model = deepcopy(model.backbone).eval()
            backbone_flops, _ = profile(backbone_model, inputs=(dummy_img,), verbose=False)
            print(f"  ✓ Backbone FLOPs: {backbone_flops / 1e9:.2f} G")
            
            # ========== 2. 获取特征图作为 Encoder 的输入 ==========
            with torch.no_grad():
                backbone_feats = model.backbone(dummy_img)
                # Encoder 通常处理最后一个特征图（S5/P5）
                if isinstance(backbone_feats, (list, tuple)):
                    encoder_feat = backbone_feats[-1]  # 使用最后一个特征图
                else:
                    encoder_feat = backbone_feats
            
            # ========== 3. 解构 Encoder (针对 AIFI 模块) ==========
            if not hasattr(model, 'encoder'):
                raise AttributeError("模型缺少 encoder 属性")
            
            if encoder_feat is None:
                raise ValueError("无法获取 encoder 输入特征图")
            
            encoder_model = deepcopy(model.encoder).eval()
            
            # 准备 encoder 输入
            if isinstance(backbone_feats, (list, tuple)):
                encoder_input = backbone_feats
            else:
                encoder_input = [backbone_feats]
            
            # 先测量整个 encoder_model 的 total_enc_base
            total_enc_base, _ = profile(encoder_model, inputs=(encoder_input,), verbose=False)
            
            # 1. 明确定义隐藏维度（DSET 标准是 256）
            hidden_dim = config.get('model', {}).get('hidden_dim', 256)
            
            enc_attn_base = 0
            enc_ffn_base = 0
            
            # 调试模式：打印所有 Encoder 层名
            if debug:
                print("\n" + "=" * 80)
                print("🔍 [DEBUG] 正在扫描 Encoder 所有子模块名称")
                print("=" * 80)
                for name, module in encoder_model.named_modules():
                    if isinstance(module, nn.Linear):
                        print(f"  Linear层: {name:<60} | 类名: {module.__class__.__name__}")
                    elif isinstance(module, nn.MultiheadAttention) or "Attention" in module.__class__.__name__:
                        print(f"  Attention层: {name:<60} | 类名: {module.__class__.__name__}")
                print("=" * 80 + "\n")
            
            # 遍历 encoder 的子模块，识别 Attention 和 FFN
            for name, module in encoder_model.named_modules():
                # 针对 Attention 层
                if isinstance(module, nn.MultiheadAttention) or "Attention" in module.__class__.__name__:
                    B, _, H, W = encoder_feat.shape  # 只取 Batch, Height, Width
                    # 修复点：构造 inputs 时，最后一维必须是 hidden_dim (256)
                    dummy_seq = torch.randn(B, H * W, hidden_dim).to(device)
                    
                    attn_flops, _ = profile(module, inputs=(dummy_seq, dummy_seq, dummy_seq), verbose=False)
                    enc_attn_base += attn_flops
                
                # 针对 FFN/MoE 层
                elif isinstance(module, nn.Linear) and any(k in name.lower() for k in ['expert', 'ffn', 'moe']):
                    B, _, H, W = encoder_feat.shape
                    # 修复点：输入维度对齐 256
                    dummy_flat = torch.randn(B, H * W, hidden_dim).to(device)
                    
                    ffn_flops, _ = profile(module, inputs=(dummy_flat,), verbose=False)
                    enc_ffn_base += ffn_flops
            
            if enc_attn_base == 0 and enc_ffn_base == 0:
                raise RuntimeError("无法识别 Encoder 中的 Attention 或 FFN 层，解构失败")
            
            # 计算其他部分（Norm 层、Add 层等）：total_enc_base - enc_attn_base - enc_ffn_base
            enc_others_base = max(0, total_enc_base - enc_attn_base - enc_ffn_base)
            
            print(f"  ✓ Encoder 解构: Total={total_enc_base/1e9:.2f}G, Attn={enc_attn_base/1e9:.2f}G, FFN={enc_ffn_base/1e9:.2f}G, Others={enc_others_base/1e9:.2f}G")
            
            # Encoder 理论值：完善公式，确保 Norm 层和 Add 层也被考虑
            # Others (Norm, Add): FLOPs ∝ N，随 r 线性缩放
            # Attention: FLOPs ∝ N²，随 r² 缩放
            # FFN: FLOPs ∝ N，且 MoE 激活 Top-K experts，所以随 r × min(top_k, experts) / experts 缩放
            encoder_moe_ratio = min(encoder_top_k, encoder_experts) / max(encoder_experts, 1)
            theory_enc_flops = (enc_others_base * token_keep_ratio) + (enc_attn_base * (token_keep_ratio ** 2)) + (enc_ffn_base * token_keep_ratio * encoder_moe_ratio)
            
            # ========== 4. Decoder: 只有 FFN 部分是 MoE (Top-3) ==========
            if not hasattr(model, 'decoder'):
                raise AttributeError("模型缺少 decoder 属性")
            
            decoder_model = deepcopy(model.decoder).eval()
            
            # 准备 decoder 输入：先通过 encoder 处理 backbone 特征，得到 encoder_features
            # decoder.forward(feats, targets=None) 期望的是经过 encoder 处理后的特征
            with torch.no_grad():
                encoder_features, _ = model.encoder(backbone_feats, return_encoder_info=True)
            
            # 从配置中读取 hidden_dim（decoder 的隐藏维度，通常是 256）
            hidden_dim = config.get('model', {}).get('hidden_dim', 256)
            
            # 统计整个 decoder 的 FLOPs
            # decoder.forward(feats, targets=None) 只需要 encoder_features
            dec_base_flops, _ = profile(decoder_model, inputs=(encoder_features, None), verbose=False)
            
            dec_moe_flops = 0
            processed_moe_paths = set()  # 记录已处理的 MoE 模块路径，避免重复统计
            
            # 调试模式：打印所有 Decoder 层名
            if debug:
                print("\n" + "=" * 80)
                print("🔍 [DEBUG] 正在扫描 Decoder 所有子模块名称")
                print("=" * 80)
                for name, module in decoder_model.named_modules():
                    if isinstance(module, nn.Linear):
                        print(f"  Linear层: {name:<60} | 类名: {module.__class__.__name__}")
                    elif 'moe' in name.lower() or 'MoE' in module.__class__.__name__:
                        print(f"  ⭐ MoE相关层: {name:<60} | 类名: {module.__class__.__name__}")
                print("=" * 80 + "\n")
            
            # 遍历 decoder 的子模块，识别 MoE 层
            for name, module in decoder_model.named_modules():
                # 检查当前模块是否是已处理 MoE 模块的子模块
                # 如果当前路径是已处理路径的子路径（前缀匹配），则跳过
                is_child_of_processed = any(
                    name.startswith(processed_path + '.') for processed_path in processed_moe_paths
                )
                
                if is_child_of_processed:
                    continue  # 跳过已处理 MoE 模块的子模块
                
                # 识别 MoE 层：检查类名或参数名
                is_moe_layer = False
                if 'moe_layer' in name.lower():
                    is_moe_layer = True
                elif hasattr(module, '__class__'):
                    class_name = module.__class__.__name__
                    if 'MoE' in class_name or 'MoELayer' in class_name:
                        is_moe_layer = True
                
                if is_moe_layer:
                    # Profile MoE 层
                    # MoE 层在 decoder 的 FFN 部分，输入维度应该与 decoder hidden_dim 一致
                    # 构造 dummy 输入：MoE 层期望的输入是 [B, num_tokens, hidden_dim]
                    B = 1
                    # 从 decoder 的输出维度推断（通常是 num_queries）
                    num_queries = getattr(decoder_model, 'num_queries', 300)
                    dummy_moe_input = torch.randn(B, num_queries, hidden_dim).to(device)
                    moe_flops, _ = profile(module, inputs=(dummy_moe_input,), verbose=False)
                    dec_moe_flops += moe_flops
                    processed_moe_paths.add(name)  # 记录已处理的 MoE 模块路径
            
            if dec_moe_flops == 0:
                raise RuntimeError("无法识别 Decoder 中的 MoE 层，解构失败")
            
            dec_other_flops = max(0, dec_base_flops - dec_moe_flops)
            print(f"  ✓ Decoder 解构: Total={dec_base_flops/1e9:.2f}G, MoE={dec_moe_flops/1e9:.2f}G, Other={dec_other_flops/1e9:.2f}G")
            
            # Decoder 理论值：MoE 部分按 Top-K 路由折算，其余部分不变
            # Top-K 路由：激活 min(top_k, experts) 个专家，所以 FLOPs = 原始值 × min(top_k, experts) / experts
            # 加固边界检查：确保比例计算正确
            dec_moe_ratio = min(decoder_top_k, decoder_experts) / max(decoder_experts, 1)
            theory_dec_flops = dec_other_flops + (dec_moe_flops * dec_moe_ratio)
            
            # ========== 5. 汇总：最终 T-GFLOPs ==========
            total_theory_flops = backbone_flops + theory_enc_flops + theory_dec_flops
            theory_flops_g = total_theory_flops / 1e9
            
            print(f"  ✓ Theory FLOPs (分模块): {theory_flops_g:.2f} G")
            print(f"    - Backbone: {backbone_flops/1e9:.2f} G")
            print(f"    - Encoder (r={token_keep_ratio:.2f}, e={encoder_experts}, top-{encoder_top_k}): {theory_enc_flops/1e9:.2f} G")
            print(f"    - Decoder (e={decoder_experts}, top-{decoder_top_k}): {theory_dec_flops/1e9:.2f} G")
        except Exception as e:
            print(f"  ⚠ 理论 FLOPs 计算失败，使用基准 FLOPs: {e}")
            theory_flops_g = base_flops_g
    else:
        # 非 DSET 模型或没有 thop：理论 FLOPs = 基准 FLOPs
        theory_flops_g = base_flops_g
    
    return total_params_m, active_params_m, base_flops_g, theory_flops_g


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
    model = trainer.model
    
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
    
    return model, config


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
    
    return model, config


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
    
    # 将 Config 对象转换为 dict 以便后续处理
    config_dict = {}
    try:
        if hasattr(cfg, '_cfg_dict'):
            config_dict = cfg._cfg_dict
        elif hasattr(cfg, 'to_dict'):
            config_dict = cfg.to_dict()
        elif hasattr(cfg, '__dict__'):
            config_dict = dict(cfg.__dict__)
    except:
        pass
    
    return model, config_dict


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
    
    # 构建配置字典
    config = {
        'model_type': 'yolov8',
        'data': None
    }
    
    # 尝试从模型对象中获取数据集路径
    try:
        if hasattr(model, 'ckpt') and model.ckpt and hasattr(model.ckpt, 'data'):
            config['data'] = model.ckpt.data
    except:
        pass
    
    return model, config


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
    
    # 构建配置字典
    config = {
        'model_type': 'yolov10',
        'data': None
    }
    
    # 尝试从模型对象中获取数据集路径
    try:
        if hasattr(model, 'ckpt') and model.ckpt and hasattr(model.ckpt, 'data'):
            config['data'] = model.ckpt.data
    except:
        pass
    
    return model, config


def evaluate_yolo_accuracy(model, config_path: str, device: str = "cuda", max_samples: int = 300) -> Dict[str, float]:
    """使用 YOLO model.val() 进行评估（仅精度，无性能测试）"""
    try:
        print(f"  ✓ 使用 YOLO model.val() 进行评估")
        
        results = model.val(
            data=str(config_path),
            device=device,
            verbose=False,
            max_det=300,
            batch=1
        )
        
        metrics = {'mAP': 0.0, 'AP50': 0.0, 'APS': 0.0}
        
        if hasattr(results, 'box'):
            if hasattr(results.box, 'map'):
                metrics['mAP'] = float(results.box.map)
            if hasattr(results.box, 'map50'):
                metrics['AP50'] = float(results.box.map50)
            if hasattr(results.box, 'maps') and len(results.box.maps) > 1:
                metrics['APS'] = float(results.box.maps[1])
        
        print(f"  ✓ mAP: {metrics['mAP']:.3f}, AP50: {metrics['AP50']:.3f}, APS: {metrics['APS']:.3f}")
        return metrics
        
    except Exception as e:
        print(f"  ⚠ YOLO 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return {'mAP': 0.0, 'AP50': 0.0, 'APS': 0.0}


def evaluate_deformable_detr_accuracy(model, config_path: str, device: str = "cuda") -> Dict[str, float]:
    """评估 Deformable-DETR 模型（使用 mmdet/mmengine Runner.test()，仅精度）"""
    metrics = evaluate_deformable_detr_full(
        config_path=config_path,
        checkpoint_path=None,
        device=device,
    )
    return metrics


def _safe_get_metric(metrics: Dict, keys: List[str], default: float = 0.0) -> float:
    """Robust metric key lookup across mmdet versions."""
    for k in keys:
        if k in metrics:
            try:
                return float(metrics[k])
            except Exception:
                pass
    # 兼容：有些 evaluator 会把 key 写成 'coco/bbox_mAP' 或 'bbox_mAP'
    for k, v in metrics.items():
        if isinstance(k, str):
            for cand in keys:
                if k.endswith(cand) or cand in k:
                    try:
                        return float(v)
                    except Exception:
                        pass
    return float(default)


def _move_to_device(obj, device: str):
    """Recursively move tensors to device (for mmengine/mmdet batch dict)."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [_move_to_device(v, device) for v in obj]
        return type(obj)(moved) if not isinstance(obj, tuple) else tuple(moved)
    return obj


def evaluate_deformable_detr_full(config_path: str,
                                  checkpoint_path: Optional[str],
                                  device: str = "cuda") -> Dict[str, float]:
    """评估 Deformable-DETR 模型（仅精度，使用 mmdet Runner.test()）"""
    try:
        from mmengine.config import Config
        from mmengine.runner import Runner
        from mmdet.utils import register_all_modules
    except Exception as e:
        print(f"  ⚠ Deformable-DETR 评估依赖导入失败: {e!r}")
        return {'mAP': 0.0, 'AP50': 0.0, 'APS': 0.0}
    
    # 注册 mmdet 模块（不同版本签名略有差异）
    try:
        register_all_modules(init_default_scope=True)
    except TypeError:
        register_all_modules()
    
    cfg = Config.fromfile(config_path)
    if checkpoint_path:
        cfg.load_from = checkpoint_path
    
    # 使用配置文件中的 batch_size（不强制为 1，COCO 评估结果与 batch_size 无关）
    # 其余配置（pipeline/resize/evaluator/proposal_nums/metric_items/num_workers 等）
    # 严格沿用训练脚本（如 train_deformable_r18.py）生成的 config，保证一致性。
    
    # 避免写日志到 work_dir（Runner 需要但我们不关心）
    try:
        import tempfile
        cfg.work_dir = tempfile.mkdtemp(prefix='bench_deformable_detr_')
    except Exception:
        cfg.work_dir = cfg.get('work_dir', './work_dirs/bench_deformable_detr')
    
    runner = Runner.from_cfg(cfg)
    runner.model = runner.model.to(device)
    runner.model.eval()
    
    # 精度评估（COCO mAP）
    test_metrics = runner.test()
    
    mAP = _safe_get_metric(test_metrics, ['coco/bbox_mAP', 'bbox_mAP', 'mAP'], 0.0)
    AP50 = _safe_get_metric(test_metrics, ['coco/bbox_mAP_50', 'bbox_mAP_50', 'AP50', 'mAP_50'], 0.0)
    APS = _safe_get_metric(test_metrics, ['coco/bbox_mAP_s', 'bbox_mAP_s', 'APS', 'mAP_s'], 0.0)
    
    return {'mAP': mAP, 'AP50': AP50, 'APS': APS}


def _get_outputs_info(outputs: Dict) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """从模型输出中提取 logits 和 boxes
    
    注意：DSET 和 RT-DETR 均使用 Focal Loss，推理时应使用 Sigmoid 激活
    """
    if 'pred_logits' in outputs:
        return outputs['pred_logits'], outputs['pred_boxes'], True  # RT-DETR: sigmoid
    elif 'class_scores' in outputs:
        return outputs['class_scores'], outputs['bboxes'], True  # DSET: sigmoid (Focal Loss)
    return None, None, False


def _collect_predictions_for_coco(outputs: Dict, targets: List[Dict], batch_idx: int,
                                  all_predictions: List, all_targets: List,
                                  img_w: int, img_h: int, batch_size: int) -> None:
    """收集预测结果用于 COCO 评估"""
    pred_logits, pred_boxes, use_sigmoid = _get_outputs_info(outputs)
    if pred_logits is None:
        return
    
    batch_size_actual = pred_logits.shape[0]
    
    for i in range(batch_size_actual):
        # 尝试从 target 中直接获取原始 ID
        if i < len(targets) and 'image_id' in targets[i]:
            img_id = int(targets[i]['image_id'].item())
        else:
            # 如果没有，再退回到索引计算逻辑，但要确保 batch_size 正确
            img_id = batch_idx * batch_size + i
        
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
                        'image_id': img_id,
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
                        'image_id': img_id,
                        'category_id': int(true_labels[j].item()) + 1,
                        'bbox': [float(x), float(y), float(w), float(h)],
                        'area': float(w * h)
                    }
                    if has_iscrowd:
                        ann_dict['iscrowd'] = int(iscrowd_values[j].item())
                    all_targets.append(ann_dict)


def evaluate_accuracy(model, config_path: str, device: str = "cuda", 
                      model_type: str = "dset", max_samples: int = 300) -> Dict[str, float]:
    """使用 pycocotools 在验证集上运行 COCO 评估（仅精度，无性能测试）"""
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
        
        # 加载配置（不强制 batch_size，使用配置文件中的设置以加速评估）
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if 'misc' not in config:
            config['misc'] = {}
        config['misc']['device'] = device
        
        # 创建 DataLoader（使用配置文件中的 batch_size）
        if model_type == "dset":
            trainer = TrainerClass(config, config_file_path=str(config_path))
            _, val_loader = trainer._create_data_loaders()
        else:
            trainer = TrainerClass(config)
            trainer.model = model
            trainer.criterion = trainer.create_criterion()
            _, val_loader = trainer.create_datasets()
        
        dataset_size = len(val_loader.dataset)
        dataloader_length = len(val_loader)
        actual_batch_size = val_loader.batch_size if hasattr(val_loader, 'batch_size') else None
        
        print(f"  ✓ DataLoader: {dataloader_length} batches, {dataset_size} samples (batch_size={actual_batch_size})")
        
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
        
        print(f"  运行评估循环（仅精度评估，无性能测试）...")
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                B, C, H_tensor, W_tensor = images.shape
                images = images.to(device)
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                           for k, v in t.items()} for t in targets]
                
                # 单次前向传播获取输出
                outputs = model(images)
                
                has_predictions = isinstance(outputs, dict) and (
                    ('class_scores' in outputs and 'bboxes' in outputs) or
                    ('pred_logits' in outputs and 'pred_boxes' in outputs)
                )
                
                if has_predictions:
                    # 动态获取当前 batch 的真实图片数量 B
                    current_batch_actual_size = images.shape[0]
                    
                    _collect_predictions_for_coco(
                        outputs, targets, batch_idx, all_predictions, all_targets,
                        W_tensor, H_tensor, current_batch_actual_size  # 修复：传入真实的 B
                    )
                
                # 进度打印：每 100 个样本打印一次
                if (batch_idx + 1) % 100 == 0:
                    print(f"    进度: {batch_idx + 1}/{dataloader_length}")
        
        print(f"  ✓ 完成: {len(val_loader)} 样本, {len(all_predictions)} 预测框")
        
        # COCO 评估（仅精度指标）
        metrics = _compute_coco_metrics(all_predictions, all_targets, H_tensor, W_tensor)
        
        return metrics
        
    except Exception as e:
        print(f"  ⚠ COCO 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return {'mAP': 0.0, 'AP50': 0.0, 'APS': 0.0}


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
        
        print(f"  ✓ mAP: {metrics['mAP']:.3f}, AP50: {metrics['AP50']:.3f}, APS: {metrics['APS']:.3f}")
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


def evaluate_single_model(model_name: str, model_config: Dict, args, project_root: Path, debug: bool = False) -> Optional[Dict]:
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
    
    # 加载模型和配置
    is_yolo_model = model_type.startswith("yolov8") or model_type.startswith("yolov10")
    config = None
    try:
        if model_type == "dset":
            model, config = load_dset_model(str(config_path), str(checkpoint_path), args.device)
        elif model_type == "rtdetr":
            model, config = load_rtdetr_model(str(config_path), str(checkpoint_path), args.device)
        elif model_type == "deformable-detr":
            model = load_deformable_detr_model(str(checkpoint_path), args.device, 
                                               config_path=str(config_path) if config_path else None)
            # 加载配置
            if config_path:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
        elif model_type.startswith("yolov8"):
            model, config = load_yolov8_model(str(checkpoint_path), args.device)
        elif model_type.startswith("yolov10"):
            model, config = load_yolov10_model(str(checkpoint_path), args.device)
        else:
            print(f"  ⚠ 不支持的模型类型: {model_type}")
            return None
        print("  ✓ 模型加载成功")
    except Exception as e:
        print(f"  ⚠ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 计算参数量和理论 FLOPs
    input_size_tuple = (1, 3, input_size[0], input_size[1])
    total_params_m, active_params_m, base_flops_g, theory_flops_g = get_model_info(
        model, input_size_tuple, is_yolo=is_yolo_model, config=config, model_type=model_type, debug=debug
    )
    print(f"  ✓ Total Params: {total_params_m:.2f}M, Active Params: {active_params_m:.2f}M")
    print(f"  ✓ Base FLOPs: {base_flops_g:.2f}G, Theory FLOPs: {theory_flops_g:.2f}G")
    
    # 评估（仅精度）
    metrics = {'mAP': 0.0, 'AP50': 0.0, 'APS': 0.0}
    
    if model_type in ["dset", "rtdetr"] and config_path:
        metrics = evaluate_accuracy(model, str(config_path), args.device, 
                                    model_type=model_type, max_samples=999999)
    elif model_type == "deformable-detr" and config_path:
        metrics = evaluate_deformable_detr_full(
            config_path=str(config_path),
            checkpoint_path=str(checkpoint_path),
            device=args.device,
        )
    elif is_yolo_model:
        data_config_path = _get_yolo_data_path(model, model_config, project_root)
        if data_config_path:
            metrics = evaluate_yolo_accuracy(model, str(data_config_path), args.device, 999999)
        else:
            print(f"  ⚠ YOLO 模型需要数据集配置文件")
    
    # 清理显存
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'model_name': model_name,
        'model_type': model_type,
        'total_params_m': total_params_m,
        'active_params_m': active_params_m,
        'base_flops_g': base_flops_g,
        'theory_flops_g': theory_flops_g,
        'mAP': metrics.get('mAP', 0.0),
        'AP50': metrics.get('AP50', 0.0),
        'APS': metrics.get('APS', 0.0),
        'input_size': f"{input_size[0]}x{input_size[1]}"
    }


def print_summary_table(results: List[Dict], gpu_name: str = "GPU", save_csv: bool = True, max_samples: int = 300):
    """打印结果汇总表格并保存为 CSV（理论效率视角）"""
    if not results:
        print("\n⚠ 没有评估结果")
        return
    
    print("\n" + "=" * 140)
    print("THEORETICAL EFFICIENCY".center(140))
    print("=" * 140)
    
    header = f"{'Model':<25} {'Total':<10} {'Active':<10} {'Theory':<10} {'Resolution':<12} {'mAP':<8} {'AP50':<8} {'APS':<8}"
    print(header)
    print("-" * 140)
    print(f"{'':<25} {'Params':<10} {'Params':<10} {'GFLOPs':<10} {'':<12} {'':<8} {'':<8} {'':<8}")
    print(f"{'':<25} {'(M)':<10} {'(M)':<10} {'':<10} {'':<12} {'':<8} {'':<8} {'':<8}")
    print("-" * 140)
    
    csv_rows = [['Model', 'Type', 'Total Params(M)', 'Active Params(M)', 'Theory GFLOPs', 
                 'Resolution', 'mAP', 'AP50', 'APS', 'Input']]
    
    for r in results:
        name = r.get('model_name', 'Unknown')[:24]
        total_params = f"{r.get('total_params_m', 0):.2f}" if r.get('total_params_m', 0) > 0 else "N/A"
        active_params = f"{r.get('active_params_m', 0):.2f}" if r.get('active_params_m', 0) > 0 else "N/A"
        theory_flops = f"{r.get('theory_flops_g', 0):.2f}" if r.get('theory_flops_g', 0) > 0 else "N/A"
        resolution = r.get('input_size', 'N/A')
        
        mAP = f"{r.get('mAP', 0):.3f}" if r.get('mAP', 0) > 0 else "N/A"
        ap50 = f"{r.get('AP50', 0):.3f}" if r.get('AP50', 0) > 0 else "N/A"
        aps = f"{r.get('APS', 0):.3f}" if r.get('APS', 0) > 0 else "N/A"
        
        print(f"{name:<25} {total_params:<10} {active_params:<10} {theory_flops:<10} {resolution:<12} {mAP:<8} {ap50:<8} {aps:<8}")
        
        csv_rows.append([name, r.get('model_type', ''), total_params, active_params, theory_flops, 
                        resolution, mAP, ap50, aps, r.get('input_size', '')])
    
    print("-" * 140)
    print("Note: Theoretical FLOPs are calculated based on sparsity-aware projection (Top-K expert and token pruning ratio).")
    print("=" * 140)
    
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
    """在 logs 目录下找到最新的 best_model.pth 或 best.pt
    
    优先查找各实验目录下 weights/ 文件夹内的检查点文件，并按文件修改时间排序返回最新者。
    """
    best_models = []
    
    if model_type == "deformable-detr":
        # Deformable-DETR: 查找 best_*.pth
        # 优先查找 weights/ 目录
        best_models.extend(list(logs_dir.rglob("weights/best_*.pth")))
        best_models.extend(list(logs_dir.rglob("best_*.pth")))
    elif model_type.startswith("yolov"):
        # YOLO: 查找 best.pt 或 best_model.pth
        # 优先查找 weights/ 目录
        best_models.extend(list(logs_dir.rglob("weights/best.pt")))
        best_models.extend(list(logs_dir.rglob("weights/best_model.pth")))
        best_models.extend(list(logs_dir.rglob("best.pt")))
        best_models.extend(list(logs_dir.rglob("best_model.pth")))
    else:
        # DSET/RT-DETR: 查找 best_model.pth
        # 优先查找 weights/ 目录
        best_models.extend(list(logs_dir.rglob("weights/best_model.pth")))
        best_models.extend(list(logs_dir.rglob("best_model.pth")))
    
    # 去重并过滤存在的文件
    best_models = list(set(best_models))
    best_models = [p for p in best_models if p.exists() and p.is_file()]
    
    if not best_models:
        return None
    
    # 按文件修改时间排序，返回最新的
    best_models.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return best_models[0]


def _format_evaluation_results(model_type: str, total_params_m: float, active_params_m: float, 
                               base_flops_g: float, theory_flops_g: float,
                               metrics: Dict[str, float],
                               input_resolution: Tuple[int, int], is_yolo: bool = False,
                               gpu_name: str = "GPU") -> None:
    """格式化并输出评估结果（理论效率视角）"""
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
    print(f"Total Params: {total_params_m:.2f}M | Active Params: {active_params_m:.2f}M")
    print(f"Base FLOPs: {base_flops_g:.2f}G | Theory FLOPs: {theory_flops_g:.2f}G")
    print(f"mAP: {metrics['mAP']:.3f} | AP50: {metrics['AP50']:.3f} | APS: {metrics['APS']:.3f}")
    print("=" * 70)
    print("Note: Theoretical FLOPs are calculated based on sparsity-aware projection (Top-1 expert and token pruning ratio).")


def main():
    parser = argparse.ArgumentParser(description='生成性能对比表')
    parser.add_argument('--logs_dir', type=str, default='experiments/dset/logs')
    parser.add_argument('--config', type=str, default='experiments/dset/configs/dset4_r18_ratio0.5.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--input_size', type=int, nargs=2, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_type', type=str, default='dset',
                       choices=['dset', 'rtdetr', 'deformable-detr', 
                               'yolov8s', 'yolov8m', 'yolov10s', 'yolov10m'])
    parser.add_argument('--rtdetr_config', type=str, default=None)
    parser.add_argument('--deformable_work_dir', type=str, default=None)
    parser.add_argument('--deformable_config', type=str, default=None)
    parser.add_argument('--models_config', type=str, default=None)
    parser.add_argument('--debug', action='store_true', help='启用调试模式：打印所有模块层名以便调试 MoE 层识别')
    
    args = parser.parse_args()
    
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
        result = evaluate_single_model(model_name, model_config, args, project_root, debug=args.debug)
        if result:
            all_results.append(result)
    
    # 输出结果
    if len(all_results) > 1:
        print_summary_table(all_results, gpu_name, save_csv=True, max_samples=0)  # max_samples 不再使用，设为 0
    elif all_results:
        r = all_results[0]
        _format_evaluation_results(
            r['model_type'], 
            r.get('total_params_m', 0), r.get('active_params_m', 0),
            r.get('base_flops_g', 0), r.get('theory_flops_g', 0),
            {'mAP': r['mAP'], 'AP50': r['AP50'], 'APS': r['APS']},
            (int(r['input_size'].split('x')[1]), int(r['input_size'].split('x')[0])),
            r['model_type'].startswith("yolov"), gpu_name
        )


if __name__ == '__main__':
    main()
