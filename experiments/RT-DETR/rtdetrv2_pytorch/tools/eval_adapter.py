#!/usr/bin/env python3
"""Standalone evaluation script for RT-DETR experiment directories."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.misc import dist_utils
from src.core import YAMLConfig
from src.solver import TASKS


def _resolve_checkpoint(log_dir: Path) -> Path:
    for name in ("best.pth", "last.pth", "checkpoint.pth"):
        path = log_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"checkpoint not found in {log_dir}")


def _resolve_config(log_dir: Path, config_arg: str | None) -> Path:
    if config_arg:
        return Path(config_arg).resolve()
    for name in ("config.yml", "config.yaml"):
        path = log_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"config.yml or config.yaml not found in {log_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RT-DETR checkpoint reevaluation")
    parser.add_argument("--log-dir", type=str, required=True, help="experiment output directory")
    parser.add_argument("--config", type=str, default=None, help="explicit config path")
    parser.add_argument("--checkpoint", type=str, default=None, help="explicit checkpoint path")
    parser.add_argument("--device", type=str, default=None, help="override device")
    parser.add_argument("--data-root", type=str, default=None, help="override dataset root")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="evaluation split")
    parser.add_argument("--seed", type=int, default=None, help="evaluation seed")
    args = parser.parse_args()

    log_dir = Path(args.log_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else _resolve_checkpoint(log_dir)
    config_path = _resolve_config(log_dir, args.config)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
        force=True,
    )
    logger = logging.getLogger("eval_adapter")
    logger.info("log dir: %s", log_dir)
    logger.info("config: %s", config_path)
    logger.info("checkpoint: %s", checkpoint_path)
    logger.info("split: %s", args.split)

    kwargs = {
        "resume": str(checkpoint_path),
        "output_dir": str(log_dir),
    }
    if args.device is not None:
        kwargs["device"] = args.device
    if args.data_root is not None:
        kwargs.setdefault("val_dataloader", {}).setdefault("dataset", {})["data_root"] = args.data_root
    if args.split != "val":
        kwargs.setdefault("val_dataloader", {}).setdefault("dataset", {})["split"] = args.split

    dist_utils.setup_distributed(print_rank=0, print_method='builtin', seed=args.seed)
    try:
        cfg = YAMLConfig(str(config_path), **kwargs)
        solver = TASKS[cfg.yaml_cfg['task']](cfg)
        solver.val()
    finally:
        dist_utils.cleanup()


if __name__ == "__main__":
    main()
