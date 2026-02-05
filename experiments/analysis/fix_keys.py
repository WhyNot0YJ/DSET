"""
fix_keys.py - The Surgeon
==========================
This script creates a new checkpoint file with renamed keys to align with
experiments/dset/train.py model naming.

Key mappings (from inspect_keys output vs train.py / rtdetrv2_decoder.py):
  - decoder: adaptive_expert_layer -> decoder_moe_layer
  - encoder: moe_layer (unchanged, already correct)

重要：必须同时 fix model_state_dict 和 ema_state_dict！
generate_benchmark_table.py 评估时加载的是 EMA，若只 fix model 不 fix EMA，
会导致 MoE 等层权重未加载，精度异常偏低。
"""

import torch
from typing import Dict, Any, Optional
import os


# ============================================================================
# CONFIGURATION
# ============================================================================
# Path to the old checkpoint file (e.g. experiments/dset/logs/S5only/dset6_r18_xxx/best_model.pth)
OLD_CKPT_PATH = "/root/autodl-tmp/DSET/experiments/dset/logs/S5only/dset6_r18_20251229_195505/best_model.pth"

# Path where the new checkpoint will be saved
NEW_CKPT_PATH = "/root/autodl-tmp/DSET/experiments/dset/logs/S5only/dset6_r18_20251229_195505/best_model_fixed.pth"

# Old substring to replace (checkpoint uses adaptive_expert_layer in decoder)
OLD_SUBSTRING = "adaptive_expert_layer"

# New substring to replace with (train.py expects decoder_moe_layer)
NEW_SUBSTRING = "decoder_moe_layer"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def rename_keys_in_state_dict(state_dict: Dict[str, Any], 
                               old_substring: str, 
                               new_substring: str) -> tuple[Dict[str, Any], int]:
    """
    Rename keys in a state_dict by replacing old_substring with new_substring.
    
    Args:
        state_dict: The state dictionary to process
        old_substring: Substring to find in keys
        new_substring: Substring to replace with
    
    Returns:
        Tuple of (new_state_dict, count_of_renamed_keys)
    """
    if not isinstance(state_dict, dict):
        print(f"  Warning: Expected dict, got {type(state_dict)}. Skipping.")
        return state_dict, 0
    
    new_state_dict = {}
    renamed_count = 0
    
    for key, value in state_dict.items():
        if old_substring in key:
            # Replace the substring in the key
            new_key = key.replace(old_substring, new_substring)
            new_state_dict[new_key] = value
            renamed_count += 1
            print(f"    Renamed: {key}")
            print(f"         -> {new_key}")
        else:
            # Keep the key as-is
            new_state_dict[key] = value
    
    return new_state_dict, renamed_count


# ============================================================================
# MAIN FUNCTION
# ============================================================================
def fix_checkpoint_keys(old_path: str, 
                       new_path: str, 
                       old_substring: str = "adaptive_expert_layer",
                       new_substring: str = "decoder_moe_layer") -> None:
    """
    Load a checkpoint, rename keys, and save the fixed version.
    
    Args:
        old_path: Path to the old checkpoint file
        new_path: Path where the new checkpoint will be saved
        old_substring: Substring to find in keys (default: "adaptive_expert_layer")
        new_substring: Substring to replace with (default: "decoder_moe_layer")
    """
    print("=" * 80)
    print("CHECKPOINT KEY FIXER")
    print("=" * 80)
    print(f"Old checkpoint: {old_path}")
    print(f"New checkpoint: {new_path}")
    print(f"Replacing '{old_substring}' with '{new_substring}'")
    print("-" * 80)
    
    try:
        # Load the old checkpoint on CPU
        print("\nLoading checkpoint...")
        try:
            checkpoint = torch.load(old_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(old_path, map_location='cpu')
        
        # Determine checkpoint structure (DSET uses model_state_dict, ema_state_dict)
        model_keys = ['model', 'state_dict', 'model_state_dict']
        has_model = isinstance(checkpoint, dict) and any(k in checkpoint for k in model_keys)
        
        total_renamed = 0
        
        if has_model:
            print("Detected wrapped checkpoint structure.")
            new_checkpoint = {}
            
            # Process model state_dict (try model_state_dict first for DSET format)
            model_sdict = None
            model_key_used = None
            for k in ['model_state_dict', 'model', 'state_dict']:
                if k in checkpoint:
                    model_sdict = checkpoint[k]
                    model_key_used = k
                    break
            
            if model_sdict is not None:
                print(f"\nProcessing '{model_key_used}'...")
                new_model_dict, count = rename_keys_in_state_dict(
                    model_sdict, old_substring, new_substring
                )
                new_checkpoint[model_key_used] = new_model_dict
                total_renamed += count
                print(f"  Renamed {count} keys in model weights.")
            
            # Process EMA weights if present (DSET uses ema_state_dict)
            # 重要：benchmark 评估时加载的是 EMA，必须同时 fix EMA 才能得到正确精度
            ema_sdict = checkpoint.get('ema_state_dict') or checkpoint.get('ema')
            if ema_sdict is not None:
                print("\nProcessing EMA weights...")
                # EMA 结构: {'module': state_dict, 'updates': N}，实际参数在 module 内
                ema_key = 'ema_state_dict' if 'ema_state_dict' in checkpoint else 'ema'
                if isinstance(ema_sdict, dict) and 'module' in ema_sdict:
                    module_dict = ema_sdict['module']
                    # 兼容 OrderedDict 等（isinstance(x, dict) 对 OrderedDict 为 True）
                    if hasattr(module_dict, 'items'):
                        ema_old_count = sum(1 for k in module_dict if old_substring in k)
                        print(f"  EMA module 中发现 {ema_old_count} 个需替换的键")
                        new_module_dict, count = rename_keys_in_state_dict(
                            module_dict, old_substring, new_substring
                        )
                        new_ema = {**ema_sdict, 'module': new_module_dict}
                        total_renamed += count
                        print(f"  Renamed {count} keys in EMA module weights.")
                    else:
                        print("  WARNING: EMA module 不是 dict 类型，跳过")
                        new_ema = ema_sdict
                else:
                    new_ema, count = rename_keys_in_state_dict(
                        ema_sdict, old_substring, new_substring
                    )
                    total_renamed += count
                    print(f"  Renamed {count} keys in EMA weights.")
                new_checkpoint[ema_key] = new_ema
            
            # Copy other keys as-is (e.g., 'optimizer_state_dict', 'epoch', etc.)
            for key in checkpoint.keys():
                if key not in model_keys + ['ema', 'ema_state_dict']:
                    new_checkpoint[key] = checkpoint[key]
        
        else:
            # Treat as flat state_dict
            print("Detected flat state_dict structure.")
            print("\nProcessing state_dict...")
            new_checkpoint, total_renamed = rename_keys_in_state_dict(
                checkpoint, old_substring, new_substring
            )
            print(f"  Renamed {total_renamed} keys.")
        
        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Renamed {total_renamed} keys in total.")
        
        if total_renamed == 0:
            print("\nWARNING: No keys were renamed!")
            print(f"Make sure '{old_substring}' exists in the checkpoint keys.")
            response = input("\nDo you still want to save the checkpoint? (yes/no): ")
            if response.lower() != 'yes':
                print("Aborted. No file saved.")
                return
        
        # Save the modified checkpoint
        print(f"\nSaving fixed checkpoint to: {new_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(new_path) if os.path.dirname(new_path) else '.', exist_ok=True)
        
        torch.save(new_checkpoint, new_path)
        print("✓ Checkpoint saved successfully!")
        
        # Verify the save（同时验证 model 和 EMA，因为 benchmark 加载的是 EMA）
        print("\nVerifying saved checkpoint...")
        try:
            verify_checkpoint = torch.load(new_path, map_location='cpu', weights_only=False)
        except TypeError:
            verify_checkpoint = torch.load(new_path, map_location='cpu')
        if isinstance(verify_checkpoint, dict):
            all_ok = True
            # 1. 验证 model_state_dict
            for k in ['model_state_dict', 'model', 'state_dict']:
                if k in verify_checkpoint:
                    verify_keys = list(verify_checkpoint[k].keys())
                    old_in_model = [x for x in verify_keys if old_substring in x]
                    if old_in_model:
                        print(f"WARNING: model 中仍有 {len(old_in_model)} 个 '{old_substring}'")
                        all_ok = False
                    else:
                        print(f"✓ model: 无 '{old_substring}'")
                    break
            # 2. 验证 EMA（benchmark 实际加载的是这个！）
            ema_verify = verify_checkpoint.get('ema_state_dict') or verify_checkpoint.get('ema')
            if ema_verify and isinstance(ema_verify, dict) and 'module' in ema_verify:
                ema_module_keys = list(ema_verify['module'].keys())
                old_in_ema = [x for x in ema_module_keys if old_substring in x]
                new_in_ema = [x for x in ema_module_keys if new_substring in x]
                if old_in_ema:
                    print(f"WARNING: EMA module 中仍有 {len(old_in_ema)} 个 '{old_substring}'（benchmark 加载 EMA，会导致精度异常！）")
                    for k in old_in_ema[:3]:
                        print(f"    - {k}")
                    all_ok = False
                else:
                    print(f"✓ EMA module: 无 '{old_substring}'，有 {len(new_in_ema)} 个 '{new_substring}'")
            elif ema_verify:
                print("  (EMA 结构无 module，跳过 EMA 验证)")
            if all_ok:
                print("✓ 验证通过: model 和 EMA 均已正确替换")
        
    except FileNotFoundError:
        print(f"ERROR: Checkpoint file not found at: {old_path}")
        print("Please update OLD_CKPT_PATH at the top of this script.")
    except Exception as e:
        print(f"ERROR: Failed to process checkpoint: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    fix_checkpoint_keys(OLD_CKPT_PATH, NEW_CKPT_PATH, OLD_SUBSTRING, NEW_SUBSTRING)
