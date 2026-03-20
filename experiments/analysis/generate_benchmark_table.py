#!/usr/bin/env python3
"""
模型理论效率评估脚本 - 支持 Cas_DETR, RT-DETR, Deformable-DETR, YOLOv8, YOLOv10

功能：
1. 自动从 logs/ 目录查找最新的 best_model.pth 或使用指定的检查点
2. 使用 pycocotools 在验证集上运行 COCO 评估（仅精度指标）
3. 计算模型参数量和理论 FLOPs（考虑 token pruning 和 MoE 稀疏性）
4. 使用配置文件中的 batch_size（COCO 评估结果与 batch_size 无关，可使用更大 batch_size 加速）

使用方法：
    python generate_benchmark_table.py --model_type cas_detr
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


# ==============================================================================
# Custom Ops Definitions (Based on Audit Report)
# ==============================================================================

_moe_dense_mode = False

def count_moe_layer(m, x, y):
    """MoELayer Custom Op: 计算 MoE 层的 FLOPs
    
    Dense 模式: 计算所有专家的 FLOPs (用于 Base FLOPs)
    Sparse 模式: 只计算激活专家的 FLOPs (用于 Theory FLOPs)
    """
    global _moe_dense_mode
    
    inp = x[0]
    if not torch.is_tensor(inp):
        return
    
    B, N, C = inp.shape
    num_experts = getattr(m, 'num_experts', 1)
    top_k = getattr(m, 'top_k', 1)
    dim_feedforward = getattr(m, 'dim_feedforward', 2048)
    
    router_flops = B * N * C * num_experts
    single_expert_flops = B * N * (2 * C * dim_feedforward)
    all_experts_flops = single_expert_flops * num_experts
    
    if _moe_dense_mode:
        expert_flops = all_experts_flops
    else:
        activation_ratio = top_k / max(num_experts, 1)
        expert_flops = all_experts_flops * activation_ratio
    
    total = router_flops + expert_flops
    m.total_ops += torch.DoubleTensor([int(total)])


def get_model_info(model, input_size: Tuple[int, int, int, int] = (1, 3, 736, 1280), 
                   is_yolo: bool = False, config: Dict = None, model_type: str = "cas_detr",
                   debug: bool = False) -> Tuple[float, float, float, float]:
    """计算模型的参数量和理论 FLOPs
    
    Base FLOPs: Token pruning 已应用 + MoE Dense (所有专家)
    Theory FLOPs: Token pruning 已应用 + MoE Sparse (只计算激活专家)
    """
    # ========================== 1. 参数量计算 ==========================
    if is_yolo and hasattr(model, 'model'):
        pytorch_model = model.model
    else:
        pytorch_model = model
    total_params = sum(p.numel() for p in pytorch_model.parameters())
    
    # Active Params (Simple Estimation based on config)
    active_params = total_params
    if model_type == "cas_detr" and config:
        cas_detr_cfg = config.get('model', {}).get('cas_detr', {})
        enc_k = cas_detr_cfg.get('encoder_moe_top_k', 1)
        enc_e = cas_detr_cfg.get('encoder_moe_num_experts', 1)
        dec_k = config.get('model', {}).get('top_k', 3)
        dec_e = config.get('model', {}).get('num_experts', 1)
        
        enc_r = min(enc_k, enc_e) / max(enc_e, 1) if enc_e > 0 else 1
        dec_r = min(dec_k, dec_e) / max(dec_e, 1) if dec_e > 0 else 1
        
        p_moe = 0
        for n, p in pytorch_model.named_parameters():
            if 'expert' in n.lower() or 'moe' in n.lower():
                ratio = enc_r if 'encoder' in n.lower() else dec_r
                active_params -= p.numel() * (1 - ratio)
    
    total_params_m = total_params / 1e6
    active_params_m = active_params / 1e6
    print(f"\n  📊 Params: Total={total_params_m:.2f}M, Active={active_params_m:.2f}M")

    # ========================== 2. FLOPs Calculation (Physics-Level Accurate) ==========================
    base_flops_g = 0.0
    theory_flops_g = 0.0
    
    # YOLO 使用 ultralytics 内置 get_flops，避免 thop 与 ultralytics 内部 total_ops 冲突
    if is_yolo and hasattr(model, 'model'):
        try:
            imgsz = input_size[2] if input_size[2] == input_size[3] else [input_size[2], input_size[3]]
            from ultralytics.utils.torch_utils import get_flops
            base_flops_g = get_flops(model.model, imgsz=imgsz)
            theory_flops_g = base_flops_g
            print(f"  ✓ Base FLOPs: {base_flops_g:.2f} G (ultralytics @{imgsz})")
            print(f"  ✓ Theory FLOPs: {theory_flops_g:.2f} G")
        except Exception as e:
            print(f"  ⚠ YOLO FLOPs 获取失败: {e}")
    elif HAS_THOP:
        try:
            from copy import deepcopy
            model_eval = deepcopy(model).eval()
            device = next(model_eval.parameters()).device
            dummy_img = torch.randn(input_size).to(device)
            
            # mmdet 模型（如 deformable-detr）forward 需要 batch_data_samples，用 wrapper 构造 DetDataSample
            if model_type == "deformable-detr":
                from mmdet.structures import DetDataSample

                class _MMDetFLOPsWrapper(torch.nn.Module):
                    def __init__(self, m):
                        super().__init__()
                        self._model = m

                    def forward(self, x):
                        H, W = int(x.shape[2]), int(x.shape[3])
                        ds = DetDataSample()
                        ds.set_metainfo(dict(batch_input_shape=(H, W), img_shape=(H, W)))
                        return self._model(x, [ds], mode='tensor')

                model_eval = _MMDetFLOPsWrapper(model_eval)
            
            custom_ops_map = {}
            for m in model_eval.modules():
                if "MoELayer" in m.__class__.__name__:
                    custom_ops_map[m.__class__] = count_moe_layer

            global _moe_dense_mode
            _moe_dense_mode = True
            base_macs, _ = profile(model_eval, inputs=(dummy_img,), custom_ops=custom_ops_map, verbose=False)
            base_flops_g = base_macs / 1e9
            print(f"  ✓ Base FLOPs (Dense, r=1.0): {base_flops_g:.2f} G")
            
            if model_type != "cas_detr":
                _moe_dense_mode = False
                theory_flops_g = base_flops_g
                print(f"  ✓ Theory FLOPs: {theory_flops_g:.2f} G")
            elif model_type == "cas_detr":
                if hasattr(model_eval, 'encoder') and hasattr(model_eval.encoder, 'shared_token_pruner'):
                    pruner = model_eval.encoder.shared_token_pruner
                    if pruner is not None:
                        if hasattr(pruner, 'prune_in_eval'):
                            pruner.prune_in_eval = True
                        if hasattr(pruner, 'pruning_enabled'):
                            pruner.pruning_enabled = True
                for m in model_eval.modules():
                    if hasattr(m, 'set_epoch'):
                        m.set_epoch(999)
                    if hasattr(m, 'pruning_enabled'):
                        m.pruning_enabled = True
                    if hasattr(m, 'current_epoch'):
                        m.current_epoch = 999
                cas_detr_cfg = config.get('model', {}).get('cas_detr', {})
                r = cas_detr_cfg.get('token_keep_ratio', 1.0)
                if isinstance(r, dict):
                    r = max(r.values())
                with torch.no_grad():
                    _ = model_eval(dummy_img)
                _moe_dense_mode = False
                theory_macs, _ = profile(model_eval, inputs=(dummy_img,), custom_ops=custom_ops_map, verbose=False)
                theory_flops_g = theory_macs / 1e9
                print(f"  ✓ Theory FLOPs (With Pruning, r={r:.2f}): {theory_flops_g:.2f} G")
                if r < 1.0:
                    reduction = (1 - theory_flops_g / base_flops_g) * 100 if base_flops_g > 0 else 0
                    print(f"  💡 FLOPs Reduction: {reduction:.1f}% (automatically captured by physical pruning)")

            del model_eval
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ⚠ FLOPs Calculation Failed: {e}")
            import traceback
            traceback.print_exc()
            theory_flops_g = base_flops_g

    return total_params_m, active_params_m, base_flops_g, theory_flops_g


def load_cas_detr_model(config_path: str, checkpoint_path: str, device: str = "cuda"):
    """加载 Cas_DETR 模型"""
    try:
        from experiments.cas_detr.train import Cas_DETRTrainer
    except ImportError:
        cas_detr_dir = Path(config_path).parent.parent
        if str(cas_detr_dir) not in sys.path:
            sys.path.insert(0, str(cas_detr_dir))
        from train import Cas_DETRTrainer
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if 'misc' not in config:
        config['misc'] = {}
    config['misc']['device'] = device
    
    trainer = Cas_DETRTrainer(config, config_file_path=str(config_path))
    model = trainer.model
    
    checkpoint = _load_checkpoint(checkpoint_path)
    state_dict = _extract_state_dict(checkpoint)
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        if load_result.missing_keys:
            print(f"  ⚠ missing_keys: {len(load_result.missing_keys)} 个")
            for k in load_result.missing_keys[:5]:
                print(f"      - {k}")
            if len(load_result.missing_keys) > 5:
                print(f"      ... 等 {len(load_result.missing_keys)} 个")
        if load_result.unexpected_keys:
            print(f"  ⚠ unexpected_keys: {len(load_result.unexpected_keys)} 个")
            for k in load_result.unexpected_keys[:5]:
                print(f"      - {k}")
            if len(load_result.unexpected_keys) > 5:
                print(f"      ... 等 {len(load_result.unexpected_keys)} 个")
    model.eval()
    
    # 启用 token pruning - 强制设置 epoch=100 以跨越 warmup 阶段
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'set_epoch'):
        cas_detr_config = config.get('model', {}).get('cas_detr', {})
        warmup_epochs = cas_detr_config.get('token_pruning_warmup_epochs', 10)
        target_keep_ratio = cas_detr_config.get('token_keep_ratio', 1.0)
        
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
    # 不设置 cfg.load_from：PyTorch 2.6+ 默认 weights_only=True，mmengine checkpoint 含自定义类会报错。
    # 改为手动加载（使用 weights_only=False），避免 Runner 内部 torch.load 失败。
    
    # 避免写日志到 work_dir（Runner 需要但我们不关心）
    try:
        import tempfile
        cfg.work_dir = tempfile.mkdtemp(prefix='bench_deformable_detr_')
    except Exception:
        cfg.work_dir = cfg.get('work_dir', './work_dirs/bench_deformable_detr')
    
    runner = Runner.from_cfg(cfg)
    if checkpoint_path:
        checkpoint = _load_checkpoint(checkpoint_path)
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        runner.model.load_state_dict(state_dict, strict=False)
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
    
    注意：Cas_DETR 和 RT-DETR 均使用 Focal Loss，推理时应使用 Sigmoid 激活
    """
    if 'pred_logits' in outputs:
        return outputs['pred_logits'], outputs['pred_boxes'], True  # RT-DETR: sigmoid
    elif 'class_scores' in outputs:
        return outputs['class_scores'], outputs['bboxes'], True  # Cas_DETR: sigmoid (Focal Loss)
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
                      model_type: str = "cas_detr", max_samples: int = 300) -> Dict[str, float]:
    """使用 pycocotools 在验证集上运行 COCO 评估（仅精度，无性能测试）"""
    try:
        # 导入 Trainer
        if model_type == "cas_detr":
            try:
                from experiments.cas_detr.train import Cas_DETRTrainer
            except ImportError:
                cas_detr_dir = Path(config_path).parent.parent
                if str(cas_detr_dir) not in sys.path:
                    sys.path.insert(0, str(cas_detr_dir))
                from train import Cas_DETRTrainer
            TrainerClass = Cas_DETRTrainer
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
        if model_type == "cas_detr":
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
        if model_type == "cas_detr":
            model, config = load_cas_detr_model(str(config_path), str(checkpoint_path), args.device)
        elif model_type == "rtdetr":
            model, config = load_rtdetr_model(str(config_path), str(checkpoint_path), args.device)
        elif model_type == "deformable-detr":
            model, config = load_deformable_detr_model(str(checkpoint_path), args.device, 
                                                       config_path=str(config_path) if config_path else None)
            # config 已由 load_deformable_detr_model 返回；若为空则用 Config.fromfile 补充（含 !!python/tuple 的 yaml 需 mmengine 解析）
            if not config and config_path:
                try:
                    from mmengine.config import Config
                    config = Config.fromfile(config_path)
                except ImportError:
                    config = {}
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
    
    if model_type in ["cas_detr", "rtdetr"] and config_path:
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
    print("Note: Theoretical FLOPs are calculated based on sparsity-aware projection (MoE activation ratio: top_k/expert_num, and token pruning ratio).")
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


def find_latest_best_model(logs_dir: Path, model_type: str = "cas_detr") -> Optional[Path]:
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
        # Cas_DETR/RT-DETR: 查找 best_model.pth
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
        'cas_detr': 'Cas_DETR', 'rtdetr': 'RT-DETRv2', 'deformable-detr': 'Deformable-DETR',
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
    print("Note: Theoretical FLOPs are calculated based on sparsity-aware projection (MoE activation ratio: top_k/expert_num, and token pruning ratio).")


def main():
    parser = argparse.ArgumentParser(description='生成性能对比表')
    parser.add_argument('--logs_dir', type=str, default='experiments/cas_detr/logs')
    parser.add_argument('--config', type=str, default='experiments/cas_detr/logs/cas_detr6_r18_20260126_173526/config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--input_size', type=int, nargs=2, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_type', type=str, default='cas_detr',
                       choices=['cas_detr', 'rtdetr', 'deformable-detr', 
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
    
    # 输出结果（无论单模型还是多模型，都保存 CSV）
    if all_results:
        print_summary_table(all_results, gpu_name, save_csv=True, max_samples=0)


if __name__ == '__main__':
    main()
