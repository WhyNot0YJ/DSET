#!/usr/bin/env python3
"""YOLOX 训练与与 Ultralytics 一致的 KITTI/scale 评估。"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch

_experiments_root = Path(__file__).resolve().parent.parent
if str(_experiments_root) not in sys.path:
    sys.path.insert(0, str(_experiments_root))

_yolo_dir = Path(__file__).resolve().parent
_yolox_root = _yolo_dir / "external" / "YOLOX"

from base_yolo_trainer import BaseYOLOTrainer
from yolox_predict import YOLOXEvalPredictor, load_yolox_for_eval


class YOLOXTrainer(BaseYOLOTrainer):
    VERSION = "yolox"

    def create_model(self):
        """训练由 YOLOX ``tools.train`` 完成，此处不创建 Ultralytics 模型。"""
        return None

    def _resolve_yolox_exp_file(self) -> Path:
        rel = self.model_config.get("yolox_exp_file")
        if not rel:
            raise ValueError("model.yolox_exp_file 未配置")
        p = Path(rel)
        if not p.is_absolute():
            p = Path(self.config_path).parent / p if self.config_path else _yolo_dir / p
        if not p.exists():
            p = _yolo_dir / self.model_config.get("yolox_exp_file", "")
        if not p.exists():
            raise FileNotFoundError(f"找不到 YOLOX exp 文件: {rel}")
        return p.resolve()

    def _coco_data_root(self) -> str:
        root = self.data_config.get("coco_data_root")
        if not root:
            raise ValueError("data.coco_data_root 未设置（训练 YOLOX 需 COCO 根目录）")
        return str(Path(root).expanduser())

    def _load_yolox_exp(self):
        from yolox.exp import get_exp

        exp = get_exp(exp_file=str(self._resolve_yolox_exp_file()))
        exp.data_dir = self._coco_data_root()
        imgsz = int(self.training_config.get("imgsz", 640))
        exp.input_size = (imgsz, imgsz)
        exp.test_size = (imgsz, imgsz)
        if self.class_names:
            exp.num_classes = len(self.class_names)
        exp.data_num_workers = int(self.misc_config.get("num_workers", 4))
        exp.max_epoch = int(self.training_config.get("epochs", 300))
        ne = exp.max_epoch
        exp.eval_interval = min(10, max(1, ne // 5))
        exp.print_interval = min(50, max(10, exp.max_epoch))
        exp.save_history_ckpt = False
        exp.output_dir = str(self.log_dir.parent)
        exp.test_conf = 0.01
        if hasattr(exp, "warmup_epochs"):
            exp.warmup_epochs = min(exp.warmup_epochs, max(1, exp.max_epoch // 10))
        return exp

    def _build_train_cli_args(self, exp, epochs_override: Optional[int]) -> argparse.Namespace:
        pretrained = self._resolve_model_path()
        ckpt = None
        if pretrained and str(pretrained).endswith((".pth", ".pt")) and Path(pretrained).exists():
            ckpt = str(pretrained)

        max_ep = int(self.training_config.get("epochs", 100))
        if epochs_override is not None:
            max_ep = int(epochs_override)
        exp.max_epoch = max_ep

        resume_ckpt = None
        p = getattr(self, "_resume_checkpoint_path", None)
        if p and Path(p).exists():
            resume_ckpt = str(p)
        resume = resume_ckpt is not None

        devices = min(torch.cuda.device_count(), 1) if torch.cuda.is_available() else 1
        if devices < 1:
            devices = 1

        args = argparse.Namespace(
            experiment_name=self.log_dir.name,
            name=None,
            dist_backend="nccl",
            dist_url=None,
            batch_size=int(self.training_config.get("batch_size", 16)),
            devices=devices,
            exp_file=str(self._resolve_yolox_exp_file()),
            resume=resume,
            ckpt=resume_ckpt if resume else ckpt,
            start_epoch=None,
            num_machines=1,
            machine_rank=0,
            fp16=bool(self.training_config.get("fp16", True)),
            cache=None,
            occupy=False,
            logger="tensorboard",
            opts=[],
        )
        return args

    def start_training(
        self, resume_checkpoint: Optional[str] = None, epochs_override: Optional[int] = None
    ):
        try:
            from yolox.tools.train import main as yolox_train_main
        except ImportError as exc:
            raise ImportError(
                "无法导入 yolox.tools.train。请在 experiments/yolo/external/YOLOX 执行 "
                "`python3 -m pip install -e .`（见 external/INSTALL_YOLOX.txt）。"
            ) from exc
        from yolox.exp import check_exp_value
        from yolox.utils import configure_module

        self._resume_checkpoint_path = resume_checkpoint
        self.setup_logging()

        configure_module()
        exp = self._load_yolox_exp()
        if epochs_override is not None:
            exp.max_epoch = int(epochs_override)

        args = self._build_train_cli_args(exp, epochs_override)
        check_exp_value(exp)

        self.logger.info("=" * 80)
        self.logger.info("🚀 YOLOX 训练（Megvii）")
        self.logger.info(f"  exp_file: {args.exp_file}")
        self.logger.info(f"  data_dir: {exp.data_dir}")
        self.logger.info(f"  output_dir: {exp.output_dir} / {args.experiment_name}")
        self.logger.info(f"  epochs: {exp.max_epoch}, batch: {args.batch_size}, fp16: {args.fp16}")
        self.logger.info("=" * 80)

        if not _yolox_root.is_dir():
            raise FileNotFoundError(
                f"未找到 YOLOX 源码: {_yolox_root}，请 clone Megvii-BaseDetection/YOLOX 到此路径。"
            )

        yolox_train_main(exp, args)

        self._sync_yolox_weights_to_weights_dir()
        self._post_training_processing(None)

    def _sync_yolox_weights_to_weights_dir(self):
        wd = self.log_dir / "weights"
        wd.mkdir(parents=True, exist_ok=True)
        best = self.log_dir / "best_ckpt.pth"
        if best.exists():
            shutil.copy2(best, wd / "best_ckpt.pth")
        latest = self.log_dir / "latest_ckpt.pth"
        if latest.exists():
            shutil.copy2(latest, wd / "latest_ckpt.pth")

    def _optional_post_train_benchmark(self, model) -> Optional[dict]:
        pred, _ = self._get_kitti_eval_predictor(None)
        if pred is None:
            return None
        return self._benchmark_eval_predictor(pred)

    def _can_run_kitti_eval_without_ultralytics_model(self) -> bool:
        return (
            (self.log_dir / "weights" / "best_ckpt.pth").exists()
            or (self.log_dir / "best_ckpt.pth").exists()
        )

    def _get_kitti_eval_predictor(self, model):
        ckpt = self.log_dir / "weights" / "best_ckpt.pth"
        if not ckpt.exists():
            ckpt = self.log_dir / "best_ckpt.pth"
        if not ckpt.exists():
            self.logger.warning("未找到 YOLOX best_ckpt.pth，跳过评估")
            return None, max(len(self.class_names), 1)

        exp = self._load_yolox_exp()
        device = self.misc_config.get("device", "cuda")
        pred = load_yolox_for_eval(exp, ckpt, device=device)
        nc = len(self.class_names) if self.class_names else int(exp.num_classes)
        return pred, nc

    def _predict_batch_kitti_eval(self, predictor, batch_paths, imgsz, device):
        if isinstance(predictor, YOLOXEvalPredictor):
            return predictor.predict_paths(list(batch_paths), conf=0.01)
        return super()._predict_batch_kitti_eval(predictor, batch_paths, imgsz, device)

    def _benchmark_eval_predictor(self, eval_predictor) -> Optional[dict]:
        if isinstance(eval_predictor, YOLOXEvalPredictor):
            from yolo_benchmark import benchmark_yolox
            from common.model_benchmark import benchmark_to_dict, log_benchmark

            model_name = self.model_config.get("model_name", "yolox_s")
            if model_name.endswith((".pt", ".pth")):
                model_name = model_name.rsplit(".", 1)[0]
            imgsz = int(self.training_config.get("imgsz", 640))
            try:
                res = benchmark_yolox(
                    eval_predictor.model,
                    eval_predictor.exp,
                    imgsz=imgsz,
                    device=next(eval_predictor.model.parameters()).device,
                    model_name=model_name,
                )
                log_benchmark(self.logger.info, res, header=model_name)
                return benchmark_to_dict(res)
            except Exception as exc:
                self.logger.warning(f"YOLOX benchmark 失败: {exc}")
                return None
        return super()._benchmark_eval_predictor(eval_predictor)
