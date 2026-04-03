#!/usr/bin/env python3
"""Adapter launcher for RT-DETR training without changing original configs.

``-t`` / ``tuning`` 加载整网权重时，``load_tuning_state`` 只加载 **形状一致** 的参数。
若 checkpoint 为 COCO（num_classes=80）而当前任务为 DAIR/UA（8/4 类），
decoder 侧分类头（``pred_logits``、``enc_score_head``、``denoising_class_embed`` 等）
最后一维与 checkpoint 不一致，会进入日志里的 **unmatched** 而不会加载，相当于随机初始化；
骨干等同形状层仍会继承预训练。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import yaml

_RTDETR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, _RTDETR_ROOT)
# ``experiments`` 根目录（含 ``common/``），供 CaS 风格评估
_EXP_ROOT = str(Path(_RTDETR_ROOT).resolve().parent.parent)
if _EXP_ROOT not in sys.path:
    sys.path.insert(0, _EXP_ROOT)

from src.misc import dist_utils
from src.core import YAMLConfig, yaml_utils
from src.solver import TASKS


DATASET_CONFIGS = {
    "uadetrac": "configs/dataset/uadetrac_detection.yml",
    "dairv2x": "configs/dataset/dairv2x_detection.yml",
}


def _resolve_dataset_config(args) -> str | None:
    if args.dataset_config:
        return str(Path(args.dataset_config).resolve())
    if args.dataset:
        return str((Path(__file__).resolve().parent.parent / DATASET_CONFIGS[args.dataset]).resolve())
    return None


def _make_runtime_config(base_config: str, dataset_config: str | None) -> Path:
    includes = [str(Path(base_config).resolve())]
    if dataset_config:
        includes.append(str(Path(dataset_config).resolve()))

    temp = tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False, encoding="utf-8")
    yaml.safe_dump({"__include__": includes}, temp, sort_keys=False, allow_unicode=True)
    temp.close()
    return Path(temp.name)


def _build_override_dict(args) -> dict:
    update_dict = yaml_utils.parse_cli(args.update)
    if getattr(args, "tuning", None) is None and os.environ.get("RTDETR_TUNING_CKPT"):
        update_dict.setdefault("tuning", os.environ["RTDETR_TUNING_CKPT"].strip())
    cli_updates = {k: v for k, v in args.__dict__.items() if k not in ['update', 'dataset', 'dataset_config'] and v is not None}
    update_dict.update(cli_updates)

    if args.data_root:
        update_dict.setdefault("train_dataloader", {}).setdefault("dataset", {})["data_root"] = args.data_root
        update_dict.setdefault("val_dataloader", {}).setdefault("dataset", {})["data_root"] = args.data_root

    return update_dict


def _maybe_warn_tuning_num_classes(cfg: YAMLConfig) -> None:
    """COCO 等预训练与自定义 ``num_classes`` 并存时，提醒分类头可能无法加载。"""
    tuning = getattr(cfg, "tuning", None)
    if not tuning:
        return
    if not dist_utils.is_main_process():
        return
    nc = cfg.yaml_cfg.get("num_classes", "?")
    print(
        "[train_adapter] tuning 已启用: 若 checkpoint 的类别数与当前 num_classes 不一致 "
        f"(当前 num_classes={nc})，decoder 分类头相关权重会因形状不匹配被跳过（仍为随机初始化），"
        "仅同形状的层（多为 backbone/encoder）会加载。请以训练日志中 "
        "`Load model.state_dict` 的 `unmatched` 为准。"
    )


def _persist_runtime_config(cfg: YAMLConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.yml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.yaml_cfg, f, sort_keys=False, allow_unicode=True)


def main(args) -> None:
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    dataset_config = _resolve_dataset_config(args)
    runtime_config = _make_runtime_config(args.config, dataset_config)

    try:
        update_dict = _build_override_dict(args)
        cfg = YAMLConfig(str(runtime_config), **update_dict)
        print('cfg: ', cfg.__dict__)

        _persist_runtime_config(cfg)
        _maybe_warn_tuning_num_classes(cfg)

        solver = TASKS[cfg.yaml_cfg['task']](cfg)
        if args.test_only:
            solver.val()
        else:
            solver.fit()
            if getattr(args, "cas_eval", False):
                logging.basicConfig(
                    level=logging.INFO,
                    format="%(message)s",
                    force=True,
                )
                from common.rtdetr_cas_eval import run_rtdetr_cas_style_eval_after_fit

                run_rtdetr_cas_style_eval_after_fit(
                    solver,
                    cfg,
                    Path(args.config).resolve(),
                    experiment_name=args.experiment_name,
                )
    finally:
        runtime_config.unlink(missing_ok=True)
        dist_utils.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str, required=True, help='base RT-DETR config')
    parser.add_argument('--dataset', type=str, choices=sorted(DATASET_CONFIGS.keys()), help='dataset preset')
    parser.add_argument('--dataset-config', type=str, help='custom dataset overlay yaml')
    parser.add_argument('--data-root', type=str, help='override dataset root for train/val')

    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    parser.add_argument(
        '-t',
        '--tuning',
        type=str,
        help=(
            "从整网 .pth 微调；未传时可设 RTDETR_TUNING_CKPT。"
            "若与当前 num_classes 不一致，分类头会跳过加载（见 unmatched）。"
        ),
    )
    parser.add_argument('-d', '--device', type=str, help='device')
    parser.add_argument('--seed', type=int, help='exp reproducibility')
    parser.add_argument('--use-amp', action='store_true', help='auto mixed precision training')
    parser.add_argument('--output-dir', type=str, help='output directory')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summary')
    parser.add_argument('--test-only', action='store_true', default=False)
    parser.add_argument(
        '--cas-eval',
        action='store_true',
        default=False,
        help='训练结束后运行与 CaS_DETR 一致的 val/test 指标（mAP、E/M/H、S/M/L）并写入 eval_metrics.csv',
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='rtdetr',
        help='写入 CSV / benchmark 显示用的实验名',
    )

    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')
    parser.add_argument('--local-rank', type=int, help='local rank id')
    args = parser.parse_args()

    main(args)
