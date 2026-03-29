#!/usr/bin/env python3
"""
Faster R-CNN (torchvision) 训练器 — 继承 BaseYOLOTrainer，
复用 KITTI / multi-scale 评估管线与 eval_metrics.csv 输出。
"""

import sys
import time
import math
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

_experiments_root = Path(__file__).resolve().parent.parent
if str(_experiments_root) not in sys.path:
    sys.path.insert(0, str(_experiments_root))

_yolo_dir = Path(__file__).resolve().parent
if str(_yolo_dir) not in sys.path:
    sys.path.insert(0, str(_yolo_dir))

from base_yolo_trainer import BaseYOLOTrainer
from fasterrcnn_dataset import (
    YOLOFormatDetectionDataset,
    RandomHorizontalFlipDetection,
    detection_collate_fn,
    resolve_split_dirs,
)
from common.model_benchmark import benchmark_model, benchmark_to_dict, log_benchmark

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result wrapper — mimic Ultralytics result for base-class eval
# ---------------------------------------------------------------------------

class _FasterRCNNResult:
    """Thin wrapper that presents torchvision output in Ultralytics-like API."""

    def __init__(self, orig_shape: Tuple[int, int], output: Dict[str, torch.Tensor]):
        self.orig_shape = orig_shape  # (H, W)
        boxes = output["boxes"]
        scores = output["scores"]
        labels = output["labels"] - 1  # torchvision 1-indexed → YOLO 0-indexed
        self.boxes = SimpleNamespace(
            xyxy=boxes.cpu(),
            conf=scores.cpu(),
            cls=labels.float().cpu(),
        )


# ---------------------------------------------------------------------------
# Benchmark wrapper — convert (1,3,H,W) batch tensor to list for torchvision
# ---------------------------------------------------------------------------

class _BenchmarkInputAdapter(nn.Module):
    """Wraps a torchvision detection model so ``benchmark_model`` can feed
    it a standard ``(1, 3, H, W)`` tensor (converted to ``[tensor(3,H,W)]``)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        return self.model([x[0]])


# ---------------------------------------------------------------------------
# FasterRCNNTrainer
# ---------------------------------------------------------------------------

class FasterRCNNTrainer(BaseYOLOTrainer):
    VERSION = "fasterrcnn"

    def __init__(
        self,
        config: Dict,
        config_path: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        resume_checkpoint: Optional[str] = None,
    ):
        super().__init__(config, config_path, class_names, resume_checkpoint=resume_checkpoint)

    # ── model creation ────────────────────────────────────────────────

    def create_model(self) -> nn.Module:
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        self.logger.info("✓ 创建 Faster R-CNN (ResNet-50 FPN, COCO V1 预训练)")
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        nc_with_bg = self.num_classes + 1
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, nc_with_bg)
        self.logger.info(
            f"  替换分类头: in_features={in_features}, "
            f"num_classes={nc_with_bg} (含背景)"
        )
        return model

    def _create_fresh_model(self) -> nn.Module:
        """Build an un-trained model skeleton for weight loading."""
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, self.num_classes + 1
        )
        return model

    # ── training loop ─────────────────────────────────────────────────

    def start_training(
        self,
        resume_checkpoint: Optional[str] = None,
        epochs_override: Optional[int] = None,
    ):
        self._resume_checkpoint_path = resume_checkpoint
        self.setup_logging()

        device = torch.device(self.misc_config.get("device", "cuda"))
        model = self.create_model()
        model.to(device)

        epochs = epochs_override or self.training_config.get("epochs", 100)
        batch_size = self.training_config.get("batch_size", 4)
        num_workers = self.misc_config.get("num_workers", 2)
        data_yaml = self._resolve_data_yaml()

        # datasets
        train_img_dir, train_lbl_dir = resolve_split_dirs(data_yaml, "train")
        val_img_dir, val_lbl_dir = resolve_split_dirs(data_yaml, "val")

        train_ds = YOLOFormatDetectionDataset(
            str(train_img_dir), str(train_lbl_dir),
            transform=RandomHorizontalFlipDetection(0.5),
        )
        val_ds = YOLOFormatDetectionDataset(str(val_img_dir), str(val_lbl_dir))

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=detection_collate_fn,
            pin_memory=True, drop_last=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=detection_collate_fn,
            pin_memory=True,
        )

        # resume（checkpoint 内 epoch 为「已成功结束的上一个 epoch 的 1-based 编号」，
        # 与保存时 epoch_loop+1 一致；下次训练从该值作为 0-based 下标开始）
        start_epoch = 0
        resume_ckpt: Optional[Dict[str, Any]] = None
        if resume_checkpoint and Path(resume_checkpoint).exists():
            resume_ckpt = torch.load(resume_checkpoint, map_location=device)
            if isinstance(resume_ckpt, dict) and "model_state_dict" in resume_ckpt:
                model.load_state_dict(resume_ckpt["model_state_dict"])
                start_epoch = int(resume_ckpt.get("epoch", 0))
                self.logger.info(
                    f"📦 从检查点恢复: 下一轮将从 Epoch {start_epoch + 1}/{epochs} 继续 "
                    f"(checkpoint['epoch']={start_epoch})"
                )
            else:
                model.load_state_dict(resume_ckpt)
                self.logger.info(f"📦 从检查点恢复权重: {resume_checkpoint}")

        # optimizer & scheduler
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, lr=0.005, momentum=0.9, weight_decay=0.0005,
        )
        warmup_epochs = min(3, epochs)
        main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(epochs * 0.6), int(epochs * 0.8)],
            gamma=0.1,
        )

        best_val_loss = float("inf")
        results_rows: List[Dict[str, Any]] = []
        csv_path = self.log_dir / "results.csv"
        if (
            resume_ckpt is not None
            and csv_path.exists()
        ):
            try:
                import pandas as pd
                prev = pd.read_csv(csv_path)
                results_rows = prev.to_dict("records")
                if results_rows:
                    best_val_loss = float(
                        min(float(r["val/total_loss"]) for r in results_rows)
                    )
                self.logger.info(
                    f"📈 已载入历史 results.csv 共 {len(results_rows)} 行, "
                    f"历史 best val_loss≈{best_val_loss:.4f}"
                )
            except Exception as exc:
                self.logger.warning(f"读取已有 results.csv 失败（将从头记曲线）: {exc}")

        if resume_ckpt is not None and isinstance(resume_ckpt, dict):
            if "optimizer_state_dict" in resume_ckpt:
                try:
                    optimizer.load_state_dict(
                        resume_ckpt["optimizer_state_dict"]
                    )
                except Exception as exc:
                    self.logger.warning(
                        "优化器状态加载失败（将用全新优化器）: %s", exc
                    )
            if "scheduler_state_dict" in resume_ckpt:
                try:
                    main_scheduler.load_state_dict(
                        resume_ckpt["scheduler_state_dict"]
                    )
                except Exception as exc:
                    self.logger.warning(
                        "调度器状态加载失败，尝试按 epoch 快进: %s", exc
                    )
                    n_ff = max(0, start_epoch - warmup_epochs)
                    for _ in range(n_ff):
                        main_scheduler.step()
            elif start_epoch > 0:
                self.logger.warning(
                    "检查点无 scheduler 状态：仅按 epoch 快进 MultiStepLR（旧版 .pt）"
                )
                n_ff = max(0, start_epoch - warmup_epochs)
                for _ in range(n_ff):
                    main_scheduler.step()
            if "best_val_loss" in resume_ckpt:
                try:
                    best_val_loss = float(resume_ckpt["best_val_loss"])
                except (TypeError, ValueError):
                    pass

        weights_dir = self.log_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        if start_epoch > 0:
            self._log_fasterrcnn_resume(epochs, batch_size, data_yaml, device, train_ds, val_ds, start_epoch)
        else:
            self._log_fasterrcnn_config(
                epochs, batch_size, data_yaml, device, train_ds, val_ds,
            )

        total_batches = math.ceil(len(train_ds) / batch_size)

        for epoch in range(start_epoch, epochs):
            # ── warmup LR (linear) ──
            if epoch < warmup_epochs:
                warmup_factor = min(1.0, (epoch + 1) / warmup_epochs)
                for pg in optimizer.param_groups:
                    pg["lr"] = 0.005 * warmup_factor

            # ── train ──
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            t0 = time.time()

            pbar = tqdm(
                train_loader, total=total_batches,
                desc=f"Epoch {epoch + 1}/{epochs}",
                ncols=120, leave=True,
            )
            for images, targets in pbar:
                images = [img.to(device) for img in images]
                targets = [
                    {k: v.to(device) for k, v in t.items()} for t in targets
                ]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
                optimizer.step()

                batch_loss = losses.item()
                epoch_loss += batch_loss
                n_batches += 1
                pbar.set_postfix(loss=f"{batch_loss:.3f}", lr=f"{optimizer.param_groups[0]['lr']:.5f}")

            if epoch >= warmup_epochs:
                main_scheduler.step()

            train_loss = epoch_loss / max(n_batches, 1)
            epoch_time = time.time() - t0

            # ── validation loss ──
            self.logger.info(f"  计算验证集 loss ...")
            val_loss = self._validate_loss(model, val_loader, device)

            lr_now = optimizer.param_groups[0]["lr"]
            self.logger.info(
                f"Epoch {epoch + 1}/{epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"lr={lr_now:.6f}  time={epoch_time:.1f}s"
            )

            results_rows.append({
                "epoch": epoch + 1,
                "train/total_loss": train_loss,
                "val/total_loss": val_loss,
                "lr/pg0": lr_now,
            })

            # ── checkpointing ──
            ckpt_payload = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": main_scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            }
            torch.save(ckpt_payload, weights_dir / "last.pt")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(ckpt_payload, weights_dir / "best.pt")
                self.logger.info(
                    f"  ✓ Best model saved (val_loss={val_loss:.4f})"
                )

        # ── write results.csv ──
        self._write_results_csv(results_rows)

        # ── post-training (benchmark + KITTI/scale eval) ──
        best_pt = weights_dir / "best.pt"
        if best_pt.exists():
            ckpt = torch.load(best_pt, map_location=device)
            state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state)
        model.eval()
        self._post_training_processing(model)

    # ── helpers for training loop ─────────────────────────────────────

    @torch.no_grad()
    def _validate_loss(self, model, val_loader, device) -> float:
        model.train()  # torchvision returns losses only in train mode
        total_loss = 0.0
        n = 0
        for images, targets in tqdm(val_loader, desc="  Val", ncols=100, leave=False):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            total_loss += sum(l.item() for l in loss_dict.values())
            n += 1
        return total_loss / max(n, 1)

    def _write_results_csv(self, rows: List[Dict[str, Any]]):
        import pandas as pd
        if not rows:
            return
        df = pd.DataFrame(rows)
        csv_path = self.log_dir / "results.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"✓ 训练曲线数据: {csv_path}")

    def _log_fasterrcnn_resume(
        self,
        epochs,
        batch_size,
        data_yaml,
        device,
        train_ds,
        val_ds,
        start_epoch: int,
    ):
        self.logger.info("=" * 80)
        self.logger.info("▶ 恢复 Faster R-CNN (ResNet-50 FPN) 训练")
        self.logger.info("=" * 80)
        self.logger.info(f"  数据集路径: {data_yaml}")
        self.logger.info(f"  训练集: {len(train_ds)} 张")
        self.logger.info(f"  验证集: {len(val_ds)} 张")
        self.logger.info(
            f"  进度: 从 Epoch {start_epoch + 1}/{epochs} 继续 → 共 {epochs} epoch 配置"
        )
        self.logger.info(f"  批次大小: {batch_size}")
        self.logger.info(f"  设备: {device}")
        self.logger.info(f"  输出目录: {self.log_dir}")
        self.logger.info("=" * 80)

    def _log_fasterrcnn_config(self, epochs, batch_size, data_yaml, device, train_ds, val_ds):
        self.logger.info("=" * 80)
        self.logger.info("🚀 开始 Faster R-CNN (ResNet-50 FPN) 训练")
        self.logger.info("=" * 80)
        self.logger.info(f"  数据集路径: {data_yaml}")
        self.logger.info(f"  训练集: {len(train_ds)} 张")
        self.logger.info(f"  验证集: {len(val_ds)} 张")
        self.logger.info(f"  训练轮数: {epochs}")
        self.logger.info(f"  批次大小: {batch_size}")
        self.logger.info(f"  优化器: SGD (lr=0.005, momentum=0.9, wd=0.0005)")
        self.logger.info(f"  设备: {device}")
        self.logger.info(f"  类别数: {self.num_classes}  类别: {self.class_names}")
        self.logger.info(f"  输出目录: {self.log_dir}")
        self.logger.info("=" * 80)

    # ── plot override (reads our results.csv) ─────────────────────────

    def _plot_training_curves(self):
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        csv_path = self.log_dir / "results.csv"
        if not csv_path.exists():
            return
        try:
            df = pd.read_csv(csv_path)
            epochs = df["epoch"].values

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle("Faster R-CNN Training Curves", fontsize=16, fontweight="bold")

            if "train/total_loss" in df.columns:
                axes[0].plot(epochs, df["train/total_loss"], "b-o", label="Train Loss", linewidth=2, markersize=3)
            if "val/total_loss" in df.columns:
                axes[0].plot(epochs, df["val/total_loss"], "r-s", label="Val Loss", linewidth=2, markersize=3)
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Loss Curves")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            if "lr/pg0" in df.columns:
                axes[1].plot(epochs, df["lr/pg0"], color="orange", linewidth=2)
                axes[1].set_yscale("log")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Learning Rate")
            axes[1].set_title("Learning Rate Schedule")
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = self.log_dir / "training_curves.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            self.logger.info(f"✓ 训练曲线已保存: {save_path}")
        except Exception as exc:
            self.logger.warning(f"绘制训练曲线失败: {exc}")

    # ── KITTI / scale eval overrides ──────────────────────────────────

    def _get_kitti_eval_predictor(self, model):
        device = torch.device(self.misc_config.get("device", "cuda"))
        best_pt = self.log_dir / "weights" / "best.pt"

        if best_pt.exists():
            eval_model = self._create_fresh_model()
            ckpt = torch.load(best_pt, map_location=device)
            state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            eval_model.load_state_dict(state)
            eval_model.to(device).eval()
        elif model is not None:
            eval_model = model
            eval_model.eval()
        else:
            return None, max(len(self.class_names), 1)

        nc = max(len(self.class_names), 1)
        return eval_model, nc

    def _predict_batch_kitti_eval(self, predictor, batch_paths, imgsz, device):
        device = torch.device(device) if isinstance(device, str) else device
        predictor.eval()

        images = []
        orig_shapes = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            w, h = img.size
            orig_shapes.append((h, w))
            images.append(TF.to_tensor(img).to(device))

        with torch.no_grad():
            outputs = predictor(images)

        results = []
        for orig_shape, out in zip(orig_shapes, outputs):
            results.append(_FasterRCNNResult(orig_shape, out))
        return results

    def _can_run_kitti_eval_without_ultralytics_model(self) -> bool:
        return True

    # ── benchmark overrides ───────────────────────────────────────────

    def _run_model_benchmark(self, model_or_predictor) -> Optional[dict]:
        try:
            if isinstance(model_or_predictor, nn.Module):
                raw_model = model_or_predictor
            else:
                return None

            raw_model.eval()
            wrapper = _BenchmarkInputAdapter(raw_model)
            model_name = self.model_config.get("model_name", "fasterrcnn_resnet50_fpn")
            result = benchmark_model(
                wrapper,
                imgsz=self.training_config.get("imgsz", 640),
                device=self.misc_config.get("device", "cuda"),
                model_name=model_name,
                includes_nms=True,
            )
            log_benchmark(self.logger.info, result, header=model_name)
            return benchmark_to_dict(result)
        except Exception as exc:
            self.logger.warning(f"Faster R-CNN benchmark 失败: {exc}")
            return None

    def _benchmark_eval_predictor(self, eval_predictor) -> Optional[dict]:
        return self._run_model_benchmark(eval_predictor)

    def _optional_post_train_benchmark(self, model) -> Optional[dict]:
        if model is None:
            return None
        return self._run_model_benchmark(model)

    # ── file-naming override (weights are state_dict, not Ultralytics) ─

    def _align_file_naming(self):
        """Copy weights to standard names if they exist."""
        import shutil
        weights_dir = self.log_dir / "weights"
        if not weights_dir.exists():
            return
        copies = [
            (weights_dir / "best.pt", self.log_dir / "best_model.pth"),
            (weights_dir / "last.pt", self.log_dir / "latest_checkpoint.pth"),
        ]
        for src, dst in copies:
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                self.logger.info(f"✓ 已创建: {dst.name}")
