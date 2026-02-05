"""
inspect_keys.py - The Scout
============================
This script helps inspect checkpoint keys to confirm exact key names.
It intelligently handles checkpoint structures (raw state_dict or wrapped dictionaries)
and can print all parameter keys (optionally as a tree) or filter by keyword.
"""

import os
import torch
from typing import Dict, Any, Optional, List


# ============================================================================
# CONFIGURATION
# ============================================================================
# Path to the checkpoint file you want to inspect.
CHECKPOINT_PATH = "/root/autodl-tmp/DSET/experiments/dset/logs/S5only/dset6_r18_20251229_195505/best_model.pth"  # put your .pth here or use path to logs/.../best_model.pth

# Print mode: "all" = 打印所有层级的参数; "filter" = 仅打印包含关键词的键
PRINT_MODE = "all"  # "all" | "filter"

# Keyword to search for when PRINT_MODE == "filter"
SEARCH_KEYWORD = "moe"

# True = 按层级树形展示; False = 仅打印键名列表（更简洁）
SHOW_AS_TREE = False


# ============================================================================
# HELPERS
# ============================================================================
def _state_dict_keys(state_dict: Dict[str, Any]) -> List[str]:
    return sorted(state_dict.keys()) if isinstance(state_dict, dict) else []


def _build_key_tree(keys: List[str]) -> Dict[str, Any]:
    """把 'a.b.c.weight' 形式的键列表转成嵌套 dict，便于树形打印。"""
    root: Dict[str, Any] = {}
    for k in keys:
        parts = k.split(".")
        d = root
        for i, p in enumerate(parts):
            is_leaf = i == len(parts) - 1
            if p not in d:
                d[p] = None if is_leaf else {}
            elif not is_leaf and d[p] is None:
                d[p] = {}
            if not is_leaf:
                d = d[p]
    return root


def _tree_sort_key(name: str) -> tuple:
    """排序：数字按数值排，其余按字符串，保证 layer1 < layer2 且 0 < 1。"""
    if name.isdigit():
        return (0, int(name))
    return (1, name)


def _print_tree(node: Dict[str, Any], indent: int = 0) -> None:
    """递归打印树，每层缩进 2 格。"""
    if node is None:
        return
    for name in sorted(node.keys(), key=_tree_sort_key):
        val = node[name]
        line = "  " * indent + name
        if val is None:
            print(line)
        else:
            print(line)
            _print_tree(val, indent + 1)


def _print_keys(keys: List[str], as_tree: bool) -> None:
    if as_tree:
        tree = _build_key_tree(keys)
        _print_tree(tree)
    else:
        for k in keys:
            print(f"  {k}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================
def inspect_checkpoint_keys(
    checkpoint_path: str,
    *,
    keyword: str = "moe",
    print_mode: str = "all",
    show_as_tree: bool = True,
) -> None:
    """
    Load a checkpoint and print parameter keys (all or filtered by keyword).
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        keyword: Keyword to filter keys when print_mode == "filter"
        print_mode: "all" = 打印所有参数键; "filter" = 仅打印包含 keyword 的键
        show_as_tree: True = 按层级树形展示; False = 平铺逐行打印完整键名
    """
    mode_desc = "所有层级的参数" if print_mode == "all" else f"包含 '{keyword}' 的键"
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Print mode: {print_mode} | Display: {'tree' if show_as_tree else 'flat'}")
    print(f"Scope: {mode_desc}")
    print("-" * 80)
    
    try:
        # PyTorch 2.6+ 默认 weights_only=True，checkpoint 含 numpy 等对象需设为 False
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                print("Found wrapped checkpoint structure with 'model' key.")
                model_state_dict = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                print("Found wrapped checkpoint structure with 'model_state_dict' key.")
                model_state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                print("Found wrapped checkpoint structure with 'state_dict' key.")
                model_state_dict = checkpoint['state_dict']
            else:
                print("Treating checkpoint as flat state_dict.")
                model_state_dict = checkpoint
            
            if 'ema' in checkpoint:
                print("Found EMA weights ('ema' key).")
                ema_state_dict = checkpoint['ema']
            elif 'ema_state_dict' in checkpoint:
                print("Found EMA weights ('ema_state_dict' key).")
                ema_state_dict = checkpoint['ema_state_dict']
            else:
                ema_state_dict = None
        else:
            print("Treating checkpoint as raw state_dict.")
            model_state_dict = checkpoint
            ema_state_dict = None
        
        model_keys = _state_dict_keys(model_state_dict)
        if print_mode == "filter":
            kw = keyword.lower()
            model_keys = [k for k in model_keys if kw in k.lower()]
        
        print("\n=== MODEL WEIGHTS ===")
        if not model_keys:
            print("  (none)" if print_mode == "filter" else "  (empty)")
        else:
            _print_keys(model_keys, show_as_tree)
            print(f"  [共 {len(model_keys)} 个键]")
        
        if ema_state_dict is not None:
            # EMA 结构: {'module': state_dict, 'updates': N}，实际参数在 module 内
            ema_module = None
            if isinstance(ema_state_dict, dict) and 'module' in ema_state_dict:
                ema_module = ema_state_dict['module']
            ema_keys = _state_dict_keys(ema_module) if ema_module is not None else _state_dict_keys(ema_state_dict)
            if print_mode == "filter":
                kw = keyword.lower()
                ema_keys = [k for k in ema_keys if kw in k.lower()]
            print("\n=== EMA WEIGHTS ===")
            if ema_module is not None:
                print("  (来自 ema_state_dict['module'])")
            if not ema_keys:
                print("  (none)" if print_mode == "filter" else "  (empty)")
            else:
                _print_keys(ema_keys, show_as_tree)
                print(f"  [共 {len(ema_keys)} 个键]")
        
        # Top-level checkpoint keys (brief)
        if isinstance(checkpoint, dict):
            print("\n--- Checkpoint 顶层键 ---")
            for key in checkpoint.keys():
                print(f"  - {key}")
        
    except FileNotFoundError:
        print(f"ERROR: Checkpoint file not found at: {checkpoint_path}")
        print("Please update CHECKPOINT_PATH at the top of this script.")
        print("Tip: use the path to your trained checkpoint, e.g. logs/dset6_r18_xxx/best_model.pth")
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    path = CHECKPOINT_PATH
    if not os.path.isabs(path) and not os.path.isfile(path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt = os.path.join(script_dir, path)
        if os.path.isfile(alt):
            path = alt
    inspect_checkpoint_keys(
        path,
        keyword=SEARCH_KEYWORD,
        print_mode=PRINT_MODE,
        show_as_tree=SHOW_AS_TREE,
    )
