#!/usr/bin/env python3
"""
DSET Pruning Curve Benchmark (CSV + Small Object Plot)
======================================================

1. Tests model at varying keep_ratios.
2. Saves ALL metrics (s/m/l) to CSV.
3. Plots Overall mAP and Small Object mAP side-by-side.
"""

import sys
import os
import argparse
import yaml
import csv
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import DSETTrainer (Adjust path as needed)
try:
    from experiments.dset.train import DSETTrainer
except ImportError:
    # Fallback path attempt
    sys.path.insert(0, str(project_root / "experiments/dset"))
    from train import DSETTrainer

def setup_trainer(config_path, checkpoint_path, device='cuda'):
    """Initialize Trainer and load checkpoint"""
    config_path = os.path.abspath(config_path)
    checkpoint_path = os.path.abspath(checkpoint_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if 'misc' not in config: config['misc'] = {}
    config['misc']['device'] = device
    
    # Init Trainer
    print(f"🔄 Initializing DSETTrainer with {config_path}...")
    trainer = DSETTrainer(config)
    
    # Load Checkpoint
    print(f"📥 Loading checkpoint from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
    # Load into EMA if available (Best practice for validation)
    target_model = trainer.ema.module if (hasattr(trainer, 'ema') and trainer.ema) else trainer.model
    state_dict = ckpt.get('ema_state_dict', ckpt.get('model_state_dict', ckpt))
    
    target_model.load_state_dict(state_dict, strict=False)
    target_model.eval()
    target_model.to(device)
    
    return trainer

def set_inference_keep_ratio(trainer, keep_ratio):
    """Dynamically update keep_ratio in all pruners"""
    trainer.current_epoch = 100 # Force prune logic enabled
    
    # Update EMA model
    model = trainer.ema.module if (hasattr(trainer, 'ema') and trainer.ema) else trainer.model
    
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'token_pruners'):
        for pruner in model.encoder.token_pruners:
            pruner.keep_ratio = keep_ratio
            pruner.set_epoch(100)
            pruner.prune_in_eval = True # Critical
    # Update base model just in case
    if hasattr(trainer.model, 'encoder') and hasattr(trainer.model.encoder, 'token_pruners'):
        for pruner in trainer.model.encoder.token_pruners:
            pruner.keep_ratio = keep_ratio
            pruner.set_epoch(100)

def save_to_csv(results, output_csv):
    """Save full metrics to CSV"""
    headers = ['Model', 'Ratio', 'mAP (All)', 'mAP_s (Small)', 'mAP_m (Mid)', 'mAP_l (Large)']
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for name, data in results.items():
            for i, ratio in enumerate(data['ratios']):
                row = [
                    name,
                    f"{ratio:.2f}",
                    f"{data['mAPs'][i]:.4f}",
                    f"{data['mAPs_s'][i]:.4f}",
                    f"{data['mAPs_m'][i]:.4f}",
                    f"{data['mAPs_l'][i]:.4f}"
                ]
                writer.writerow(row)
    print(f"\n✅ Results saved to CSV: {output_csv}")

def plot_results(results, output_plot):
    """Plot Overall vs Small Object mAP"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    markers = ['o', 's', '^', 'D']
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    
    for i, (name, data) in enumerate(results.items()):
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        
        # Plot 1: Overall
        ax1.plot(data['ratios'], data['mAPs'], marker=marker, label=name, linewidth=2, color=color)
        for x, y in zip(data['ratios'], data['mAPs']):
            if y > 0.01: ax1.annotate(f"{y:.3f}", (x,y), xytext=(0,5), textcoords='offset points', fontsize=8)
            
        # Plot 2: Small Objects (The Focus)
        ax2.plot(data['ratios'], data['mAPs_s'], marker=marker, label=name, linewidth=2, linestyle='--', color=color)
        for x, y in zip(data['ratios'], data['mAPs_s']):
            if y > 0.001: ax2.annotate(f"{y:.3f}", (x,y), xytext=(0,5), textcoords='offset points', fontsize=8)
    
    ax1.set_title('Overall mAP vs Efficiency')
    ax1.set_xlabel('Keep Ratio')
    ax1.set_ylabel('mAP')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_title('Small Object mAP (AP_s) vs Efficiency')
    ax2.set_xlabel('Keep Ratio')
    ax2.set_ylabel('mAP_small')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"✅ Plot saved to: {output_plot}")

def benchmark_models(models_dict, inference_ratios, device='cuda'):
    results = {}
    
    for name, (cfg, ckpt) in models_dict.items():
        print(f"\n{'='*40}\n🚀 Benchmarking: {name}\n{'='*40}")
        try:
            trainer = setup_trainer(cfg, ckpt, device)
        except Exception as e:
            print(f"❌ Failed to load {name}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
        # Metrics storage
        metrics_data = {'ratios': [], 'mAPs': [], 'mAPs_s': [], 'mAPs_m': [], 'mAPs_l': []}
        
        # Helper to safely extract metric
        def get_val(m, keys, default=0.0):
            for k in keys:
                if k in m: return m[k]
            return default
        
        pbar = tqdm(inference_ratios)
        for ratio in pbar:
            set_inference_keep_ratio(trainer, ratio)
            
            # Validate
            # sys.stdout = open(os.devnull, 'w') # Optional: suppress validation logs
            try:
                metrics = trainer.validate()
            except Exception as e:
                print(f"⚠️  Validation failed at ratio {ratio}: {e}")
                metrics = {}
            # sys.stdout = sys.__stdout__
            
            # Extract metrics - now trainer.validate() returns mAP_s, mAP_m, mAP_l directly
            mAP = get_val(metrics, ['mAP_0.5_0.95', 'coco/bbox_mAP'], default=0.0)
            mAP_s = get_val(metrics, ['mAP_s', 'coco/bbox_mAP_s'], default=0.0)
            mAP_m = get_val(metrics, ['mAP_m', 'coco/bbox_mAP_m'], default=0.0)
            mAP_l = get_val(metrics, ['mAP_l', 'coco/bbox_mAP_l'], default=0.0)
            
            metrics_data['ratios'].append(ratio)
            metrics_data['mAPs'].append(mAP)
            metrics_data['mAPs_s'].append(mAP_s)
            metrics_data['mAPs_m'].append(mAP_m)
            metrics_data['mAPs_l'].append(mAP_l)
            
            pbar.set_postfix({'mAP': f"{mAP:.3f}", 'mAP_s': f"{mAP_s:.3f}"})
            
        results[name] = metrics_data
        
        # Print mini table to console
        print(f"\n📊 Summary for {name}:")
        print(f"{'Ratio':<8} | {'mAP':<8} | {'mAP_s':<8} | {'mAP_m':<8} | {'mAP_l':<8}")
        print("-" * 50)
        for i in range(len(inference_ratios)):
            print(f"{metrics_data['ratios'][i]:<8.2f} | {metrics_data['mAPs'][i]:<8.4f} | {metrics_data['mAPs_s'][i]:<8.4f} | {metrics_data['mAPs_m'][i]:<8.4f} | {metrics_data['mAPs_l'][i]:<8.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="DSET Pruning Curve Benchmark - Test model performance at different keep_ratios",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Single Model Args
    parser.add_argument("--config", type=str, help="Path to model config YAML file")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint file")
    parser.add_argument("--name", type=str, default="DSET", help="Model name for display")
    # Multi Model Args
    parser.add_argument("--models_config", type=str, help="JSON path for multi-model config")
    # Sweep Args
    parser.add_argument("--ratios", type=float, nargs="+", default=None,
                        help="List of keep_ratios to test (if not provided, will use min/max/step)")
    parser.add_argument("--min_ratio", type=float, default=0.0,
                        help="Minimum keep_ratio (default: 0.0, used if --ratios not provided)")
    parser.add_argument("--max_ratio", type=float, default=1.0,
                        help="Maximum keep_ratio (default: 1.0, used if --ratios not provided)")
    parser.add_argument("--step", type=float, default=0.1,
                        help="Step size for keep_ratio sweep (default: 0.1, used if --ratios not provided)")
    # Output Args
    parser.add_argument("--output_csv", type=str, default="benchmark_results.csv",
                        help="Output CSV file path (default: benchmark_results.csv)")
    parser.add_argument("--output_plot", type=str, default="benchmark_plot.png",
                        help="Output plot file path (default: benchmark_plot.png)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # 1. Setup Models
    models = {}
    if args.models_config:
        import json
        if not os.path.exists(args.models_config):
            print(f"❌ Error: Models config file not found: {args.models_config}")
            return
        with open(args.models_config, 'r') as f:
            data = json.load(f)
            for k, v in data.items():
                if 'config' in v and 'checkpoint' in v:
                    models[k] = (v['config'], v['checkpoint'])
                else:
                    print(f"⚠️  Warning: Invalid model entry '{k}' in config file")
    elif args.config and args.checkpoint:
        models[args.name] = (args.config, args.checkpoint)
    else:
        print("❌ Error: Please provide --config/--checkpoint or --models_config")
        parser.print_help()
        return
    
    if not models:
        print("❌ Error: No valid models found")
        return
    
    # 2. Parse inference ratios
    if args.ratios and len(args.ratios) > 0:
        # User provided custom ratios
        inference_ratios = sorted(args.ratios)
    else:
        # Generate range from min to max with step
        inference_ratios = []
        current = args.min_ratio
        while current <= args.max_ratio + 1e-6:  # Add small epsilon for float comparison
            inference_ratios.append(round(current, 2))
            current += args.step
    
    print(f"📋 Testing keep_ratios: {inference_ratios}")
    
    # 3. Validate model files exist
    valid_models = {}
    for name, (cfg, ckpt) in models.items():
        cfg_abs = os.path.abspath(cfg) if not os.path.isabs(cfg) else cfg
        ckpt_abs = os.path.abspath(ckpt) if not os.path.isabs(ckpt) else ckpt
        
        if not os.path.exists(cfg_abs):
            print(f"⚠️  Warning: Config not found for {name} at {cfg_abs}")
            continue
        if not os.path.exists(ckpt_abs):
            print(f"⚠️  Warning: Checkpoint not found for {name} at {ckpt_abs}")
            continue
        valid_models[name] = (cfg_abs, ckpt_abs)
    
    if not valid_models:
        print("❌ Error: No valid models found. Please check paths.")
        return
    
    # 4. Run Benchmark
    results = benchmark_models(valid_models, inference_ratios, args.device)
    
    # 5. Save & Plot
    if results:
        # Resolve output paths
        output_csv = os.path.abspath(args.output_csv) if not os.path.isabs(args.output_csv) else args.output_csv
        output_plot = os.path.abspath(args.output_plot) if not os.path.isabs(args.output_plot) else args.output_plot
        
        save_to_csv(results, output_csv)
        plot_results(results, output_plot)
        print(f"\n✨ Benchmark complete! Results saved to:")
        print(f"   CSV: {output_csv}")
        print(f"   Plot: {output_plot}")
    else:
        print("❌ No results generated.")

if __name__ == "__main__":
    main()

