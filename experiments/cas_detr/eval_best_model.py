#!/usr/bin/env python3
"""独立评估：对已有 DETR 实验目录的 best_model.pth 做 val/test 重评。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

_cas_detr_dir = Path(__file__).resolve().parent
_experiments_root = _cas_detr_dir.parent
if str(_cas_detr_dir) not in sys.path:
    sys.path.insert(0, str(_cas_detr_dir))
if str(_experiments_root) not in sys.path:
    sys.path.insert(0, str(_experiments_root))


def _resolve_checkpoint(log_dir: Path) -> Path:
    best_path = log_dir / "best_model.pth"
    latest_path = log_dir / "latest_checkpoint.pth"
    if best_path.exists():
        return best_path
    if latest_path.exists():
        return latest_path
    raise FileNotFoundError(
        f"未找到 best_model.pth 或 latest_checkpoint.pth: {log_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DETR best_model 重新评估（val/test）")
    parser.add_argument("--log_dir", type=str, required=True, help="实验目录")
    parser.add_argument("--device", type=str, default=None, help="覆盖配置中的 device")
    args = parser.parse_args()

    log_dir = Path(args.log_dir).resolve()
    config_yaml = log_dir / "config.yaml"
    if not config_yaml.exists():
        raise FileNotFoundError(f"config.yaml 不存在: {config_yaml}")

    checkpoint_path = _resolve_checkpoint(log_dir)

    with config_yaml.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    if args.device is not None:
        config.setdefault("misc", {})["device"] = args.device

    # 强制绑定到该实验目录，避免重新创建新的 log_dir。
    config["resume_from_checkpoint"] = str(checkpoint_path)
    config.setdefault("checkpoint", {})["resume_from_checkpoint"] = str(checkpoint_path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
        force=True,
    )
    logger = logging.getLogger("eval_best_detr")
    logger.info("实验目录: %s", log_dir)
    logger.info("评估检查点: %s", checkpoint_path)
    logger.info("推理设备: %s", config.get("misc", {}).get("device", "cpu"))

    from train import CaS_DETRTrainer

    trainer = CaS_DETRTrainer(config=config, config_file_path=None)
    trainer.logger = logger
    trainer._evaluate_best_model_and_print_all_ap()


if __name__ == "__main__":
    main()
