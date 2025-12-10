#!/usr/bin/env python3
"""
DSET Pruning Curve Benchmark Script
====================================

功能 (Functionality):
    测试不同 keep_ratio 下的模型性能，生成 Accuracy vs. Efficiency 曲线图。
    支持动态修改推理时的 keep_ratio（从 0.0 到 1.0，每 0.1 一个步长）。

使用方法 (Usage):
    测试单个模型：
        python benchmark_pruning_curve.py \
            --config experiments/dset/configs/dset6_r18_ratio0.5.yaml \
            --checkpoint experiments/dset/logs/dset6_r18_20251209_155547/best_model.pth \
            --name "DSET_r18_0.5" \
            --output pruning_tradeoff.png
    
    测试多个模型（多次调用或使用脚本循环）：
        # 方式1: 多次调用
        python benchmark_pruning_curve.py --config config1.yaml --checkpoint ckpt1.pth --name "Model1" --output results.png
        python benchmark_pruning_curve.py --config config2.yaml --checkpoint ckpt2.pth --name "Model2" --output results.png --append
    
        # 方式2: 使用配置文件（JSON格式）
        python benchmark_pruning_curve.py --models_config models.json --output results.png
    
    自定义 keep_ratio 范围：
        python benchmark_pruning_curve.py \
            --config config.yaml \
            --checkpoint ckpt.pth \
            --ratios 0.0 0.1 0.2 0.3 0.4 0.5 \
            --output results.png

运行示例 (Example):
    # 基本用法（从项目根目录运行）
    python benchmark_pruning_curve.py \
        --config experiments/dset/configs/dset6_r18_ratio0.5.yaml \
        --checkpoint experiments/dset/logs/dset6_r18_20251209_155547/best_model.pth \
        --name "DSET_r18_0.5"
    
    # 指定设备
    python benchmark_pruning_curve.py \
        --config config.yaml \
        --checkpoint ckpt.pth \
        --name "MyModel" \
        --device cuda:0

注意事项 (Notes):
    - 确保配置文件路径正确
    - 确保 checkpoint 文件存在
    - 测试范围：keep_ratio 从 0.0（完全剪枝）到 1.0（不剪枝）
    - 每个 ratio 会运行完整的验证集评估，耗时较长
"""

import sys
import os
import argparse
import yaml
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import DSETTrainer
try:
    from experiments.dset.train import DSETTrainer
except ImportError:
    # Try alternate path if running from root
    sys.path.insert(0, str(project_root / "experiments/dset"))
    from train import DSETTrainer

def setup_trainer(config_path, checkpoint_path, device='cuda'):
    """
    初始化 DSETTrainer 并加载模型权重
    Initialize DSETTrainer and load model weights
    """
    # Resolve paths if relative
    config_path_abs = os.path.abspath(config_path) if not os.path.isabs(config_path) else config_path
    checkpoint_path_abs = os.path.abspath(checkpoint_path) if not os.path.isabs(checkpoint_path) else checkpoint_path
    
    print(f"Loading config from {config_path_abs}...")
    if not os.path.exists(config_path_abs):
        raise FileNotFoundError(f"Config file not found: {config_path_abs}")
    
    with open(config_path_abs, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Force device
    if 'misc' not in config:
        config['misc'] = {}
    config['misc']['device'] = device
    
    # Initialize Trainer (this builds model and loaders)
    # We suppress training-related initializations if possible, but DSETTrainer does it all in init
    trainer = DSETTrainer(config)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path_abs}...")
    if not os.path.exists(checkpoint_path_abs):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path_abs}")
    
    try:
        checkpoint = torch.load(checkpoint_path_abs, map_location='cpu', weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions that don't support weights_only
        checkpoint = torch.load(checkpoint_path_abs, map_location='cpu')
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path_abs}: {e}")
    
    # Load weights into EMA model (which is used for validation)
    if hasattr(trainer, 'ema') and trainer.ema:
        if 'ema_state_dict' in checkpoint:
            trainer.ema.load_state_dict(checkpoint['ema_state_dict'])
        elif 'model_state_dict' in checkpoint:
            trainer.ema.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Try loading directly if it's a model state dict
            trainer.ema.module.load_state_dict(checkpoint)
        
        trainer.ema.module.eval()
        trainer.ema.module.to(device)
    else:
        # Fallback to normal model if EMA is missing
        if 'model_state_dict' in checkpoint:
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            trainer.model.load_state_dict(checkpoint)
        trainer.model.eval()
        trainer.model.to(device)
        
    return trainer

def set_inference_keep_ratio(trainer, keep_ratio):
    """
    动态修改模型的 keep_ratio
    Dynamically modify model's keep_ratio
    """
    # 1. Update Trainer's current epoch to ensure pruning is enabled (must be > warmup_epochs)
    # Warmup is usually 10, so 100 is safe
    trainer.current_epoch = 100 
    
    # 2. Update Pruners in EMA Model (Validation uses EMA)
    model = trainer.ema.module if hasattr(trainer, 'ema') and trainer.ema else trainer.model
    
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'token_pruners'):
        for pruner in model.encoder.token_pruners:
            # Set new ratio
            pruner.keep_ratio = keep_ratio
            # Force enable pruning
            pruner.set_epoch(trainer.current_epoch)
            # Ensure it's in eval mode
            pruner.prune_in_eval = True
            
    # Also update the base model just in case
    if hasattr(trainer.model, 'encoder') and hasattr(trainer.model.encoder, 'token_pruners'):
        for pruner in trainer.model.encoder.token_pruners:
            pruner.keep_ratio = keep_ratio
            pruner.set_epoch(trainer.current_epoch)

def benchmark_models(models_dict, inference_ratios, device='cuda'):
    """
    Benchmark loop
    """
    results = {}
    
    for model_name, (config_path, ckpt_path) in models_dict.items():
        print(f"\n{'='*50}")
        print(f"Benchmarking Model: {model_name}")
        print(f"{'='*50}")
        
        try:
            trainer = setup_trainer(config_path, ckpt_path, device=device)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Skipping {model_name}: {e}")
            continue
        except Exception as e:
            print(f"Error setting up {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
        mAPs = []
        
        for ratio in tqdm(inference_ratios, desc=f"Sweeping Ratios for {model_name}"):
            # Set Ratio
            set_inference_keep_ratio(trainer, ratio)
            
            # Run Validation
            # validate() returns a dict with 'mAP_0.5_0.95'
            # We suppress stdout to avoid clutter during sweep
            # sys.stdout = open(os.devnull, 'w')
            try:
                metrics = trainer.validate()
            finally:
                # sys.stdout = sys.__stdout__
                pass
            
            mAP = metrics.get('mAP_0.5_0.95', 0.0)
            mAPs.append(mAP)
            print(f"  Ratio: {ratio} -> mAP: {mAP:.4f}")
            
        results[model_name] = mAPs
        
    return results

def plot_results(inference_ratios, results, output_path="pruning_tradeoff.png"):
    """
    Plot the trade-off curves
    """
    plt.figure(figsize=(10, 6))
    
    markers = ['o', 's', '^', 'D']
    colors = ['b', 'g', 'r', 'c']
    
    for i, (model_name, mAPs) in enumerate(results.items()):
        plt.plot(inference_ratios, mAPs, 
                 marker=markers[i % len(markers)], 
                 color=colors[i % len(colors)], 
                 linewidth=2, 
                 label=model_name)
        
        # Add labels
        for x, y in zip(inference_ratios, mAPs):
            plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.xlabel('Inference Keep Ratio')
    plt.ylabel('mAP (0.5:0.95)')
    plt.title('Accuracy vs. Efficiency Trade-off (Dynamic Pruning)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="DSET Pruning Curve Benchmark - Test model performance at different keep_ratios",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model configuration (single model)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to model config YAML file (for single model testing)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint file (for single model testing)")
    parser.add_argument("--name", type=str, default="Model",
                        help="Model name for display in plot (default: 'Model')")
    
    # Multiple models configuration
    parser.add_argument("--models_config", type=str, default=None,
                        help="Path to JSON file with multiple models config. Format: "
                             '{"Model1": {"config": "path1.yaml", "checkpoint": "ckpt1.pth"}, ...}')
    
    # Testing parameters
    parser.add_argument("--ratios", type=float, nargs="+", default=None,
                        help="List of keep_ratios to test (default: 0.0 to 1.0, step 0.1)")
    parser.add_argument("--min_ratio", type=float, default=0.0,
                        help="Minimum keep_ratio (default: 0.0)")
    parser.add_argument("--max_ratio", type=float, default=1.0,
                        help="Maximum keep_ratio (default: 1.0)")
    parser.add_argument("--step", type=float, default=0.1,
                        help="Step size for keep_ratio sweep (default: 0.1)")
    
    # Output
    parser.add_argument("--output", type=str, default="pruning_tradeoff.png",
                        help="Output plot file path (default: pruning_tradeoff.png)")
    parser.add_argument("--append", action="store_true",
                        help="Append results to existing plot file (if it exists)")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (default: cuda)")
    
    args = parser.parse_args()
    
    # 1. Parse models configuration
    models = {}
    
    if args.models_config:
        # Load from JSON file
        import json
        if not os.path.exists(args.models_config):
            print(f"Error: Models config file not found: {args.models_config}")
            return
        with open(args.models_config, 'r') as f:
            models_dict = json.load(f)
        for name, cfg_dict in models_dict.items():
            if 'config' in cfg_dict and 'checkpoint' in cfg_dict:
                models[name] = (cfg_dict['config'], cfg_dict['checkpoint'])
            else:
                print(f"Warning: Invalid model entry '{name}' in config file")
    elif args.config and args.checkpoint:
        # Single model from command line
        models[args.name] = (args.config, args.checkpoint)
    else:
        print("Error: Must provide either (--config and --checkpoint) or --models_config")
        parser.print_help()
        return
    
    # 2. Parse inference ratios
    if args.ratios:
        inference_ratios = sorted(args.ratios)
    else:
        # Generate range from min to max with step
        inference_ratios = []
        current = args.min_ratio
        while current <= args.max_ratio + 1e-6:  # Add small epsilon for float comparison
            inference_ratios.append(round(current, 2))
            current += args.step
    
    print(f"Testing keep_ratios: {inference_ratios}")
    
    # 3. Check if files exist (resolve relative paths)
    valid_models = {}
    for name, (cfg, ckpt) in models.items():
        # Resolve relative paths
        cfg_abs = os.path.abspath(cfg) if not os.path.isabs(cfg) else cfg
        ckpt_abs = os.path.abspath(ckpt) if not os.path.isabs(ckpt) else ckpt
        
        if not os.path.exists(cfg_abs):
            print(f"Warning: Config not found for {name} at {cfg_abs}")
            continue
        if not os.path.exists(ckpt_abs):
            print(f"Warning: Checkpoint not found for {name} at {ckpt_abs}")
            continue
        valid_models[name] = (cfg_abs, ckpt_abs)
    
    if not valid_models:
        print("Error: No valid models found. Please check paths.")
        return
    
    # 4. Load existing results if appending
    existing_results = {}
    if args.append and os.path.exists(args.output):
        print(f"Note: --append mode is not fully implemented. Will overwrite {args.output}")
        # TODO: Implement loading existing results from plot or JSON
    
    # 5. Run Benchmark
    results = benchmark_models(valid_models, inference_ratios, device=args.device)
    
    # 7. Visualize
    if results:
        plot_results(inference_ratios, results, output_path=args.output)
    else:
        print("No results to plot.")

if __name__ == "__main__":
    main()

