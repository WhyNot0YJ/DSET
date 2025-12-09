
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
    print(f"Loading config from {config_path}...")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Force device
    if 'misc' not in config:
        config['misc'] = {}
    config['misc']['device'] = device
    
    # Initialize Trainer (this builds model and loaders)
    # We suppress training-related initializations if possible, but DSETTrainer does it all in init
    trainer = DSETTrainer(config)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
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

def benchmark_models(models_dict, inference_ratios):
    """
    Benchmark loop
    """
    results = {}
    
    for model_name, (config_path, ckpt_path) in models_dict.items():
        print(f"\n{'='*50}")
        print(f"Benchmarking Model: {model_name}")
        print(f"{'='*50}")
        
        try:
            trainer = setup_trainer(config_path, ckpt_path)
        except FileNotFoundError:
            print(f"Skipping {model_name}: Config or Checkpoint not found.")
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
    # 1. Define Models and Checkpoints
    # USER: PLEASE UPDATE THESE PATHS
    # 格式: 'Model Name': ('Config Path', 'Checkpoint Path')
    # Default paths based on user description (assuming standard location)
    models = {
        'DSET_r18 (Train=0.3)': (
            'experiments/dset/configs/dset6_r18_ratio0.3.yaml', 
            'experiments/dset/logs/dset6_r18_ratio0.3/best_model.pth' # Placeholder
        ),
        'DSET_r18 (Train=0.5)': (
            'experiments/dset/configs/dset6_r18_ratio0.5.yaml', 
            'experiments/dset/logs/dset6_r18_ratio0.5/best_model.pth' # Placeholder
        ),
        'DSET_r18 (Train=0.9)': (
            'experiments/dset/configs/dset6_r18_ratio0.9.yaml', 
            'experiments/dset/logs/dset6_r18_ratio0.9/best_model.pth' # Placeholder
        ),
    }

    # 2. Define Inference Sweep Ratios
    inference_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # Filter based on user request [0.1, 0.2, 0.3, 0.4, 0.5]
    inference_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]

    # 3. Check if files exist (for safety)
    valid_models = {}
    for name, (cfg, ckpt) in models.items():
        if os.path.exists(cfg):
            # We allow ckpt to be missing for dry-run if needed, but warning
            if not os.path.exists(ckpt):
                print(f"Warning: Checkpoint not found for {name} at {ckpt}")
            valid_models[name] = (cfg, ckpt)
        else:
            print(f"Warning: Config not found for {name} at {cfg}")
    
    if not valid_models:
        print("No valid models found. Please check paths in the script.")
        # For demonstration, we might add a dummy entry if user wants to see logic
        # return

    # 4. Run Benchmark
    results = benchmark_models(valid_models, inference_ratios)
    
    # 5. Visualize
    if results:
        plot_results(inference_ratios, results)

if __name__ == "__main__":
    main()

