#!/usr/bin/env python3
"""
读取 benchmark 配置 JSON 的脚本。

默认读取 experiments/analysis/generate_benchmark_table_dset.json，
可通过 --json 指定其他文件。

用法:
    python read_benchmark_json.py
    python read_benchmark_json.py --json path/to/other.json
    python read_benchmark_json.py --run   # 读取后执行 generate_benchmark_table
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path


# 默认 JSON 路径（与脚本同目录）
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_JSON_PATH = SCRIPT_DIR / "generate_benchmark_table_dset.json"


def load_benchmark_json(json_path: str | Path) -> dict:
    """加载 benchmark 配置 JSON"""
    path = Path(json_path)
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="读取 benchmark 配置 JSON")
    parser.add_argument(
        "--json", "-j",
        type=str,
        default=str(DEFAULT_JSON_PATH),
        help=f"JSON 配置文件路径（默认: {DEFAULT_JSON_PATH.name}）",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="读取后执行 generate_benchmark_table 进行评估",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="格式化打印 JSON 内容",
    )
    args = parser.parse_args()

    try:
        config = load_benchmark_json(args.json)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return 1

    if args.run:
        # 调用 generate_benchmark_table，传入 JSON 路径
        json_abs = str(Path(args.json).resolve())
        script = SCRIPT_DIR / "generate_benchmark_table.py"
        return subprocess.call([sys.executable, str(script), "--models_config", json_abs])

    # 仅打印
    print(f"已加载: {args.json}")
    print(f"共 {len(config)} 个模型配置:\n")
    for name, cfg in config.items():
        if isinstance(cfg, dict):
            mtype = cfg.get("type", "?")
            ckpt = cfg.get("checkpoint", "?")
            cfg_path = cfg.get("config", "?")
            print(f"  {name}")
            print(f"    type: {mtype}")
            print(f"    config: {cfg_path}")
            print(f"    checkpoint: {ckpt}")
            print()
        else:
            print(f"  {name}: {cfg}")

    if args.pretty:
        print("\n--- 原始 JSON ---")
        print(json.dumps(config, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    exit(main())
