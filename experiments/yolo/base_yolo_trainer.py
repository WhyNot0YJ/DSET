#!/usr/bin/env python3
"""
YOLO 训练器基类：减少重复，支持 YOLO v8、v10、v11、v12
"""

import sys
import os
import json
import yaml
import torch
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from abc import ABC, abstractmethod
import shutil

_experiments_root = Path(__file__).resolve().parent.parent
if str(_experiments_root) not in sys.path:
    sys.path.insert(0, str(_experiments_root))

# 本地 ``external/ultralytics`` 与 ``external/YOLOX``（非 site-packages）
_yolo_dir = Path(__file__).resolve().parent
_external = _yolo_dir / "external"
if _external.is_dir() and str(_external) not in sys.path:
    sys.path.insert(0, str(_external))
_yolox_repo = _external / "YOLOX"
if _yolox_repo.is_dir() and str(_yolox_repo) not in sys.path:
    sys.path.insert(0, str(_yolox_repo))

from common.vram_batch import (
    compute_vram_batch_adjustment,
    format_vram_batch_log,
    resolve_cuda_device_index,
)
from common.model_benchmark import (
    BENCHMARK_EVAL_METRIC_KEYS,
    benchmark_to_dict,
    format_benchmark_eval_line,
    format_eval_csv_cell,
    log_benchmark,
    merge_benchmark_dict_into_metrics,
)

from yolo_validator_utils import MetricsLogger, MultiScaleMetricsCalculator


# ---------------------------------------------------------------------------
# Module-level AP helpers (used by BaseYOLOTrainer._evaluate_kitti_scale_*)
# ---------------------------------------------------------------------------

def _iou_xyxy(a, b) -> float:
    """IoU between two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ba - inter
    return inter / union if union > 0 else 0.0


def _compute_ap_for_class(preds, pos_gt_by_img, ign_gt_by_img, iou_thr=0.5) -> float:
    """
    VOC-style AP (AUC with monotone precision envelope) for one class.

    preds         : [(img_idx, score, xyxy_list), ...]
    pos_gt_by_img : {img_idx: [xyxy, ...]}  positive GT boxes
    ign_gt_by_img : {img_idx: [xyxy, ...]}  ignore GT boxes
                    (predictions matching these are skipped – neither TP nor FP)
    """
    n_pos = sum(len(v) for v in pos_gt_by_img.values())
    if n_pos == 0 or not preds:
        return 0.0

    preds_sorted = sorted(preds, key=lambda x: -x[1])
    matched = {img_idx: [False] * len(boxes) for img_idx, boxes in pos_gt_by_img.items()}
    tps: List[int] = []
    fps: List[int] = []

    for img_idx, _score, pred_box in preds_sorted:
        # 1) 先尝试匹配正样本 GT（pos 优先于 ign）
        pos_boxes = pos_gt_by_img.get(img_idx, [])
        best_iou, best_j = 0.0, -1
        for j, gt_box in enumerate(pos_boxes):
            iou = _iou_xyxy(pred_box, gt_box)
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_iou >= iou_thr and best_j >= 0 and not matched[img_idx][best_j]:
            tps.append(1)
            fps.append(0)
            matched[img_idx][best_j] = True
            continue

        # 2) 未匹配到 pos → 查 ign：若与 ign GT 重叠则跳过（不算 FP）
        ign_boxes = ign_gt_by_img.get(img_idx, [])
        if ign_boxes and max(_iou_xyxy(pred_box, gb) for gb in ign_boxes) >= iou_thr:
            continue

        # 3) 既不匹配 pos 也不在 ign 区域 → FP
        tps.append(0)
        fps.append(1)

    if not tps:
        return 0.0

    tp_c = np.cumsum(tps, dtype=float)
    fp_c = np.cumsum(fps, dtype=float)
    recalls = tp_c / n_pos
    precisions = tp_c / (tp_c + fp_c)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


# 与 COCO 检测 AP 一致：0.50, 0.55, …, 0.95 共 10 档，用于「面积 S/M/L 的 mAP@0.5:0.95」
IOU_THRESHOLDS_COCO = tuple(float(x) for x in np.linspace(0.5, 0.95, 10))


def _compute_ap_mean_over_ious(
    preds, pos_gt_by_img, ign_gt_by_img, iou_thresholds
) -> float:
    """多 IoU 阈值下各类 AP 的算术平均（与 COCO 的 AP@0.5:0.95 口径一致）。"""
    if not iou_thresholds:
        return 0.0
    vals = [
        _compute_ap_for_class(preds, pos_gt_by_img, ign_gt_by_img, iou_thr=float(t))
        for t in iou_thresholds
    ]
    return float(np.mean(vals))


class BaseYOLOTrainer(ABC):
    """所有YOLO训练器的基类"""
    
    # 子类需要实现的版本名称
    VERSION = "base"
    
    def __init__(self, config: Dict, config_path: Optional[str] = None, class_names: Optional[List[str]] = None):
        """
        初始化基础训练器
        
        Args:
            config: 配置字典
            config_path: 配置文件路径
            class_names: 类别名称列表
        """
        self.config = config
        self.config_path = config_path
        self.class_names = class_names or []
        self.num_classes = len(self.class_names)
        
        # 提取配置段
        self.model_config = config.get('model', {})
        self.training_config = config.get('training', {})
        self.data_config = config.get('data', {})
        self.checkpoint_config = config.get('checkpoint', {})
        self.misc_config = config.get('misc', {})
        
        # 日志和指标记录
        self.logger = None
        self.log_dir = None
        self.metrics_logger = None
        
        # 设置日志
        self.setup_logging()
        
        self._apply_vram_batch_size_rule()
        
        # 验证配置
        self._validate_config()
        
        self._log_initialization_info()
    
    def setup_logging(self):
        """设置日志系统"""
        resume_checkpoint = getattr(self, '_resume_checkpoint_path', None)
        
        if resume_checkpoint and Path(resume_checkpoint).exists():
            self.log_dir = Path(resume_checkpoint).parent
            self.experiment_name = self.log_dir.name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.model_config.get('model_name', f'yolo{self.VERSION}n')
            if model_name.endswith('.pt'):
                model_name = model_name[:-3]
            
            self.experiment_name = f"yolo_{model_name.replace(f'yolo{self.VERSION}', f'v{self.VERSION}')}"
            log_base = self.checkpoint_config.get('log_dir', 'logs')
            data_yaml = self.data_config.get('data_yaml', '')
            ds_stem = Path(data_yaml).stem if data_yaml else 'unknown'
            if 'dairv2x' in ds_stem.lower() or 'dair' in ds_stem.lower():
                ds_dir = 'dairv2x'
            elif 'uadetrac' in ds_stem.lower() or 'ua' in ds_stem.lower() or ds_stem == 'data':
                ds_dir = 'uadetrac'
            else:
                ds_dir = ds_stem
            self.log_dir = Path(f"{log_base}/{ds_dir}/{self.experiment_name}_{timestamp}")
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置日志处理器
        handlers = [
            logging.FileHandler(self.log_dir / 'training.log', mode='a'),
            logging.StreamHandler()
        ]
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers,
            force=True
        )
        
        self.logger = logging.getLogger(__name__)
        
        # 保存配置文件（仅新训练时）
        if not resume_checkpoint:
            config_save_path = self.log_dir / 'config.yaml'
            with open(config_save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            self.logger.info(f"✓ 配置已保存到: {config_save_path}")
        
        # 初始化指标记录器
        self.metrics_logger = MetricsLogger(self.log_dir)
    
    def _validate_config(self):
        """验证配置文件"""
        required_keys = {
            'model': ['model_name'],
            'training': ['epochs', 'batch_size'],
            'data': ['data_yaml']
        }
        
        missing_keys = []
        for section, keys in required_keys.items():
            if section not in self.config:
                missing_keys.append(f"缺少配置节: {section}")
                continue
            for key in keys:
                if key not in self.config[section]:
                    missing_keys.append(f"{section}.{key}")
        
        if missing_keys:
            error_msg = f"配置文件缺少必需的配置项:\n"
            error_msg += "\n".join(f"  - {key}" for key in missing_keys)
            raise ValueError(error_msg)
    
    def _log_initialization_info(self):
        """记录初始化信息"""
        if self.logger:
            self.logger.info(f"✓ 初始化YOLO{self.VERSION}训练器")
            self.logger.info(f"  类别数量: {self.num_classes}")
            if self.class_names:
                self.logger.info(f"  类别: {', '.join(self.class_names)}")
    
    @abstractmethod
    def create_model(self):
        """创建YOLO模型（由子类实现）"""
        pass
    
    def _resolve_model_path(self, model_name: Optional[str] = None) -> str:
        """
        解析预训练权重路径
        
        Args:
            model_name: 模型名称或路径
            
        Returns:
            解析后的模型路径
        """
        model_name = model_name or self.model_config.get('model_name', f'yolov8n.pt')
        pretrained_weights = self.model_config.get('pretrained_weights', None)
        
        if not pretrained_weights:
            return model_name
        
        pretrained_path = Path(pretrained_weights)
        if not pretrained_path.is_absolute():
            candidates: List[Path] = []
            if self.config_path:
                candidates.append(
                    Path(self.config_path).resolve().parent / pretrained_weights
                )
            candidates.append(Path(__file__).resolve().parent / pretrained_weights)
            found: Optional[Path] = None
            for c in candidates:
                if c.is_file():
                    found = c
                    break
            pretrained_path = found if found is not None else candidates[-1]
        
        if pretrained_path.exists():
            self.logger.info(f"✓ 加载预训练权重: {pretrained_path}")
            return str(pretrained_path)
        else:
            self.logger.warning(f"⚠️  预训练权重文件不存在: {pretrained_path}")
            self.logger.info(f"   将使用模型名称自动加载: {model_name}")
            return model_name
    
    def _resolve_data_yaml(self) -> str:
        """解析数据 YAML 路径（与 DETR 共用 ``common.detr_data_root.resolve_autodl_fs_path``）。"""
        from common.detr_data_root import resolve_autodl_fs_path

        data_yaml = self.data_config.get('data_yaml')
        return resolve_autodl_fs_path(data_yaml)
    
    def _apply_vram_batch_size_rule(self):
        """使用配置中的 batch / workers；CUDA 下记录显存信息（与 cas_detr 共用 common.vram_batch）。"""
        device_str = self.misc_config.get('device', 'cuda')
        if 'cuda' not in str(device_str) or not torch.cuda.is_available():
            return

        idx = resolve_cuda_device_index(str(device_str))
        orig_bs = int(self.training_config.get('batch_size', 16))
        orig_nw = int(self.misc_config.get('num_workers', 2))
        orig_pf = int(self.misc_config.get('prefetch_factor', 1))

        r = compute_vram_batch_adjustment(
            orig_bs, orig_nw, orig_pf, device_index=idx
        )
        if r is None:
            return

        self.training_config['batch_size'] = r.batch_size
        self.misc_config['num_workers'] = r.num_workers
        self.misc_config['prefetch_factor'] = r.prefetch_factor

        if self.logger:
            self.logger.info(format_vram_batch_log(r))

    def _build_train_kwargs(self) -> Dict:
        """
        构建YOLO训练参数
        
        Returns:
            训练参数字典
        """
        train_kwargs = {
            'data': self._resolve_data_yaml(),
            'epochs': self.training_config.get('epochs', 100),
            'batch': self.training_config.get('batch_size', 16),
            'imgsz': self.training_config.get('imgsz', 640),
            'device': self.misc_config.get('device', 'cuda'),
            'workers': self.misc_config.get('num_workers', 8),
            'project': str(self.log_dir.parent),
            'name': self.log_dir.name,
            'exist_ok': True,
            'plots': True,
            'save': True,
            'save_period': self.training_config.get('save_period', 10),
            'val': True,
            'verbose': True,
        }
        
        # 优化器和学习率参数
        optim_params = ['optimizer', 'lr0', 'lrf', 'momentum', 'weight_decay', 
                       'warmup_epochs', 'warmup_momentum', 'warmup_bias_lr', 'cos_lr']
        for param in optim_params:
            if param in self.training_config:
                train_kwargs[param] = self.training_config[param]
        
        # 其他训练参数
        other_params = ['seed', 'deterministic', 'patience', 'max_det']
        for param in other_params:
            if param in self.training_config:
                train_kwargs[param] = self.training_config[param]
        
        return train_kwargs
    
    def _log_training_config(self, train_kwargs: Dict):
        """记录训练配置信息"""
        self.logger.info("=" * 80)
        self.logger.info(f"🚀 开始YOLO{self.VERSION}训练")
        self.logger.info("=" * 80)
        self.logger.info("📝 训练配置:")
        self.logger.info(f"  数据集路径: {train_kwargs['data']}")
        self.logger.info(f"  训练轮数: {train_kwargs['epochs']}")
        self.logger.info(f"  批次大小: {train_kwargs['batch']}")
        self.logger.info(f"  优化器: {self.training_config.get('optimizer', 'auto')}")
        self.logger.info(f"  初始学习率: {self.training_config.get('lr0', 0.01)}")
        self.logger.info(f"  Weight decay: {self.training_config.get('weight_decay', 0.0001)}")
        self.logger.info(f"  输出目录: {self.log_dir}")
        if self.model_config.get('pretrained_weights'):
            self.logger.info(f"  预训练权重: {self.model_config['pretrained_weights']}")
        self.logger.info("=" * 80)
    
    def start_training(self, resume_checkpoint: Optional[str] = None, 
                      epochs_override: Optional[int] = None):
        """
        开始训练
        
        Args:
            resume_checkpoint: 恢复训练的检查点路径
            epochs_override: 覆盖epochs（用于测试）
        """
        self._resume_checkpoint_path = resume_checkpoint
        self.setup_logging()  # 重新初始化日志
        
        # 创建模型
        model = self.create_model()
        
        # 构建训练参数
        train_kwargs = self._build_train_kwargs()
        
        # 覆盖epochs如果提供了
        if epochs_override is not None:
            config_epochs = train_kwargs['epochs']
            train_kwargs['epochs'] = epochs_override
            self.logger.info(f"⚠️  测试模式：使用命令行参数覆盖epochs ({config_epochs} → {epochs_override})")
        
        # 恢复训练
        if resume_checkpoint and Path(resume_checkpoint).exists():
            self.logger.info(f"📦 从检查点恢复训练: {resume_checkpoint}")
            if Path(resume_checkpoint).is_file():
                train_kwargs['resume'] = str(resume_checkpoint)
            else:
                train_kwargs['resume'] = True
        
        # 记录配置
        self._log_training_config(train_kwargs)
        
        # 执行训练
        try:
            results = model.train(**train_kwargs)
            self._post_training_processing(model)
            return results
        except Exception as e:
            self.logger.error(f"训练失败: {e}")
            raise
    
    # ------------------------------------------------------------------
    # Truncation normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_truncation(tr: float, *, dair_categorical: bool = False) -> float:
        """
        Map dataset-specific truncation fields to a KITTI-style ratio in [0, 1]
        for ``categorize_by_kitti_difficulty`` (thresholds 0.15 / 0.30 / 0.50).

        **DAIR-V2X（官方）**：``truncated_state ∈ {0,1,2}`` 表示
        不截断 / **横向截断** / **纵向截断**，不是「被截断的百分比」。
        这里将 1、2 映射为同一代理比例（均视为「有截断」、且 ≤ hard 上限），
        横向与纵向在 KITTI 难度里不区分轴向，仅区分是否超过比例阈值。

        **UA-DETRAC**：``truncated_state`` 为连续截断比例，直接裁剪到 [0, 1]。
        """
        tr = float(tr)
        if dair_categorical:
            k = int(round(tr))
            if k == 0:
                return 0.0
            if k == 1:
                return 0.20   # 横向截断 → 通过 moderate(≤0.30) 和 hard(≤0.50)
            if k == 2:
                return 0.40   # 纵向截断 → 只通过 hard(≤0.50)
            return max(0.0, min(1.0, tr))
        return max(0.0, min(1.0, tr))

    # ------------------------------------------------------------------
    # Post-training KITTI / multi-scale evaluation
    # ------------------------------------------------------------------

    def _resolve_kitti_eval_splits(
        self, data_cfg: dict, root: Path
    ) -> List[Tuple[str, Path, Path]]:
        """
        返回 [(split, images_dir, labels_meta_dir), ...]，顺序 **val → test**。
        对应目录须存在且 labels_meta 下有 JSON。
        """
        eval_test = self.data_config.get('eval_test_after_training', True)
        out: List[Tuple[str, Path, Path]] = []

        val_rel = str(data_cfg.get('val', 'images/val')).strip()
        val_img_dir = (
            Path(val_rel) if Path(val_rel).is_absolute() else root / val_rel
        )
        lm_val = root / 'labels_meta' / 'val'
        if lm_val.is_dir() and any(lm_val.glob('*.json')):
            out.append(('val', val_img_dir, lm_val))

        test_rel = str(data_cfg.get('test', '')).strip()
        if eval_test and test_rel:
            test_dir = Path(test_rel) if Path(test_rel).is_absolute() else root / test_rel
            lm_test = root / 'labels_meta' / 'test'
            if (
                test_dir.is_dir()
                and lm_test.is_dir()
                and any(lm_test.glob('*.json'))
            ):
                out.append(('test', test_dir, lm_test))

        return out

    def _get_kitti_eval_predictor(self, model):
        """
        Return (predictor, num_classes) for KITTI/scale eval.
        Default: Ultralytics ``YOLO`` loaded from ``weights/best.pt``.
        """
        best_pt = self.log_dir / 'weights' / 'best.pt'
        from ultralytics import YOLO as _YOLO
        eval_model = _YOLO(str(best_pt)) if best_pt.exists() else model
        nc = (
            len(eval_model.names)
            if eval_model is not None
            and hasattr(eval_model, 'names')
            and eval_model.names
            else max(len(self.class_names), 1)
        )
        return eval_model, nc

    def _predict_batch_kitti_eval(self, predictor, batch_paths, imgsz, device):
        """Run batch inference for KITTI eval (Ultralytics API)."""
        return predictor.predict(
            source=[str(p) for p in batch_paths],
            conf=0.01,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )

    def _benchmark_eval_predictor(self, eval_predictor) -> Optional[dict]:
        """GFLOPs/FPS on the same weights used for KITTI eval (e.g. ``best.pt``)."""
        return self._run_model_benchmark(eval_predictor)

    def _optional_post_train_benchmark(self, model) -> Optional[dict]:
        """After training: GFLOPs/FPS. Override for non-Ultralytics backends."""
        if model is None:
            return None
        return self._run_model_benchmark(model)

    def _can_run_kitti_eval_without_ultralytics_model(self) -> bool:
        """If True, run KITTI/scale eval even when ``model`` is None (e.g. YOLOX)."""
        return False

    def _evaluate_kitti_scale_after_training(self, model, bench_dict=None) -> dict:
        """
        训练结束后的 KITTI / multi-scale mAP：

        - **val**：写 ``eval_metrics.csv`` 一行；
        - **test**：当 ``data.eval_test_after_training`` 为真且存在 ``test`` 与 ``labels_meta/test`` 时再评一行。
        - Benchmark（GFLOPs/FPS 等）只算一次，两行共用。

        返回值：若跑了 test 则返回 test 的 metrics，否则返回 val。
        """
        # ── 1. Resolve dataset root ───────────────────────────────────────
        try:
            data_yaml_path = Path(self._resolve_data_yaml())
        except FileNotFoundError as exc:
            self.logger.warning(f"无法定位 data.yaml，跳过 KITTI/scale 评估: {exc}")
            return {}

        with data_yaml_path.open(encoding='utf-8') as fh:
            data_cfg = yaml.safe_load(fh) or {}

        # DAIR-V2X: truncated_state is axis category {0,1,2}; UA-DETRAC: continuous ratio
        dair_categorical = 'DAIR-V2X' in str(data_yaml_path) or 'dairv2x' in str(
            data_yaml_path
        ).lower()

        # Resolve dataset root from the optional 'path' field in data.yaml
        root = data_yaml_path.parent.resolve()
        path_field = str(data_cfg.get('path', '')).strip()
        if path_field:
            pc = Path(path_field)
            if pc.is_absolute():
                if pc.is_dir():
                    root = pc
            else:
                proj = Path(__file__).resolve().parent.parent.parent
                for cand in [
                    (data_yaml_path.parent / pc).resolve(),
                    (proj / path_field).resolve(),
                    (proj.parent / path_field).resolve(),
                    (proj.parent / 'datasets' / path_field).resolve(),
                ]:
                    if cand.is_dir():
                        root = cand
                        break

        splits = self._resolve_kitti_eval_splits(data_cfg, root)
        if not splits:
            self.logger.warning(
                '未找到可用的 KITTI/scale 评估划分（需 labels_meta/val 或 labels_meta/test 且含 JSON）'
            )
            return {}

        # ── 2. Load best weights & benchmark（各 split 共用）──────────────
        eval_predictor, nc = self._get_kitti_eval_predictor(model)
        if eval_predictor is None:
            self.logger.warning("无可用评估权重/预测器，跳过 KITTI/scale 评估")
            return {}

        device = self.misc_config.get('device', 'cuda')
        imgsz = self.training_config.get('imgsz', 640)

        if bench_dict is None and eval_predictor is not None:
            bench_dict = self._benchmark_eval_predictor(eval_predictor)

        model_name = self.model_config.get('model_name', f'yolov{self.VERSION}n')
        if model_name.endswith('.pt'):
            model_name = model_name[:-3]
        dataset_name = Path(self.data_config.get('data_yaml', '')).stem or 'unknown'
        if 'dairv2x' in dataset_name.lower() or 'dair' in dataset_name.lower():
            dataset_name = 'DAIR-V2X'
        elif 'uadetrac' in dataset_name.lower() or 'ua-detrac' in dataset_name.lower():
            dataset_name = 'UA-DETRAC'

        import csv
        fieldnames = [
            'model', 'dataset', 'eval_split',
            'mAP_50_all', 'mAP_5095_all',
            'mAP_easy', 'mAP_moderate', 'mAP_hard',
            'mAP_small', 'mAP_medium', 'mAP_large',
            'mAP_small_5095', 'mAP_medium_5095', 'mAP_large_5095',
            *BENCHMARK_EVAL_METRIC_KEYS,
        ]
        class_names = self.class_names if self.class_names else [f'cls_{i}' for i in range(nc)]
        for name in class_names:
            fieldnames.append(f'AP50_{name}')
            fieldnames.append(f'AP5095_{name}')

        summary_csv = self.log_dir.parent / 'eval_metrics.csv'
        write_header = not summary_csv.exists() or summary_csv.stat().st_size == 0
        last_metrics: Dict[str, Any] = {}

        with summary_csv.open('a', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
            if write_header:
                writer.writeheader()

            for eval_split, eval_img_dir, labels_meta_dir in splits:
                # ── 3. Per-split: images + meta ───────────────────────────
                meta_by_stem = {p.stem: p for p in labels_meta_dir.glob('*.json')}
                if not meta_by_stem:
                    self.logger.warning(
                        f"labels_meta/{eval_split} 为空，跳过: {labels_meta_dir}"
                    )
                    continue

                eval_images = sorted(
                    p
                    for ext in ('.jpg', '.jpeg', '.png')
                    for p in eval_img_dir.glob(f'*{ext}')
                    if p.stem in meta_by_stem
                )
                if not eval_images:
                    self.logger.warning(
                        f"{eval_split} 无与 meta 匹配的图像: {eval_img_dir}"
                    )
                    continue

                self.logger.info(f"📊 评估 [{eval_split}, {len(eval_images)} 张]")

                # ── 4. Collect GT and raw predictions ───────────────────────
                gt_info: Dict[int, list] = {c: [] for c in range(nc)}
                preds_by_cls: Dict[int, list] = {c: [] for c in range(nc)}

                BATCH = 32
                for batch_start in range(0, len(eval_images), BATCH):
                    batch_paths = eval_images[batch_start: batch_start + BATCH]
                    batch_results = self._predict_batch_kitti_eval(
                        eval_predictor, batch_paths, imgsz, device
                    )
                    for i_in_batch, (result, img_path) in enumerate(
                        zip(batch_results, batch_paths)
                    ):
                        img_idx = batch_start + i_in_batch
                        img_h, img_w = result.orig_shape
                        raw = json.loads(
                            meta_by_stem[img_path.stem].read_text(encoding='utf-8')
                        )
                        if isinstance(raw, list):
                            entries = raw
                        elif isinstance(raw, dict) and 'objects' in raw:
                            entries = raw['objects']
                        else:
                            entries = []

                        for entry in entries:
                            cls = int(entry['class_id'])
                            if not (0 <= cls < nc):
                                continue
                            if 'bbox_yolo' in entry:
                                byo = entry['bbox_yolo']
                                cx, cy, bw, bh = byo['cx'], byo['cy'], byo['w'], byo['h']
                                px1 = (cx - bw / 2) * img_w
                                py1 = (cy - bh / 2) * img_h
                                px2 = (cx + bw / 2) * img_w
                                py2 = (cy + bh / 2) * img_h
                            elif 'bbox_xyxy' in entry:
                                px1, py1, px2, py2 = map(float, entry['bbox_xyxy'][:4])
                            else:
                                continue
                            h_px = py2 - py1
                            area_px = (px2 - px1) * h_px

                            occ = float(entry.get('occluded_state', 0))
                            tr = self._normalize_truncation(
                                float(entry.get('truncated_state', 0)),
                                dair_categorical=dair_categorical,
                            )
                            diff = MultiScaleMetricsCalculator.categorize_by_kitti_difficulty(
                                h_px, occ, tr
                            )
                            scale = MultiScaleMetricsCalculator.categorize_by_scale(area_px)
                            gt_info[cls].append(
                                (img_idx, (px1, py1, px2, py2), diff, scale, h_px)
                            )

                        if result.boxes is not None:
                            for box, conf, cls_t in zip(
                                result.boxes.xyxy.cpu().numpy(),
                                result.boxes.conf.cpu().numpy(),
                                result.boxes.cls.cpu().numpy().astype(int),
                            ):
                                c = int(cls_t)
                                if 0 <= c < nc:
                                    preds_by_cls[c].append(
                                        (img_idx, float(conf), box.tolist())
                                    )

                # ── 5. AP：难度 + scale + per-class ─────────────────────────
                DIFF_INCLUDE = {
                    'easy':     {'easy'},
                    'moderate': {'moderate'},
                    'hard':     {'hard'},
                }
                SCALE_LEVELS = ('small', 'medium', 'large')
                metrics: Dict[str, Any] = {}

                _MIN_H = MultiScaleMetricsCalculator.MIN_HEIGHT
                for diff_key, include in DIFF_INCLUDE.items():
                    aps = []
                    for cls in range(nc):
                        pos: Dict[int, list] = {}
                        ign: Dict[int, list] = {}
                        for img_idx, xyxy, d, _s, _h in gt_info[cls]:
                            if d in include:
                                pos.setdefault(img_idx, []).append(xyxy)
                            else:
                                ign.setdefault(img_idx, []).append(xyxy)
                        aps.append(_compute_ap_for_class(preds_by_cls[cls], pos, ign))
                    metrics[f'mAP_{diff_key}'] = float(np.mean(aps))

                for scale in SCALE_LEVELS:
                    aps50: List[float] = []
                    aps5095: List[float] = []
                    for cls in range(nc):
                        pos: Dict[int, list] = {}
                        ign: Dict[int, list] = {}
                        for img_idx, xyxy, _d, s, h in gt_info[cls]:
                            if s == scale and h >= _MIN_H:
                                pos.setdefault(img_idx, []).append(xyxy)
                            else:
                                ign.setdefault(img_idx, []).append(xyxy)
                        aps50.append(
                            _compute_ap_for_class(preds_by_cls[cls], pos, ign, iou_thr=0.5)
                        )
                        aps5095.append(
                            _compute_ap_mean_over_ious(
                                preds_by_cls[cls], pos, ign, IOU_THRESHOLDS_COCO
                            )
                        )
                    metrics[f'mAP_{scale}'] = float(np.mean(aps50))
                    metrics[f'mAP_{scale}_5095'] = float(np.mean(aps5095))

                per_cls_50: List[float] = []
                per_cls_5095: List[float] = []
                for cls in range(nc):
                    pos_all: Dict[int, list] = {}
                    ign_all: Dict[int, list] = {}
                    for img_idx, xyxy, _d, _s, h in gt_info[cls]:
                        if h >= _MIN_H:
                            pos_all.setdefault(img_idx, []).append(xyxy)
                        else:
                            ign_all.setdefault(img_idx, []).append(xyxy)
                    ap50 = _compute_ap_for_class(
                        preds_by_cls[cls], pos_all, ign_all, iou_thr=0.5
                    )
                    ap5095 = _compute_ap_mean_over_ious(
                        preds_by_cls[cls], pos_all, ign_all, IOU_THRESHOLDS_COCO
                    )
                    per_cls_50.append(ap50)
                    per_cls_5095.append(ap5095)
                    metrics[f'AP50_{class_names[cls]}'] = ap50
                    metrics[f'AP5095_{class_names[cls]}'] = ap5095

                metrics['mAP_50_all'] = float(np.mean(per_cls_50))
                metrics['mAP_5095_all'] = float(np.mean(per_cls_5095))
                metrics['eval_split'] = eval_split
                merge_benchmark_dict_into_metrics(metrics, bench_dict)

                self.logger.info(
                    f"🎯 [{eval_split}] KITTI@0.5  E/M/H = "
                    f"{metrics['mAP_easy']:.4f} / {metrics['mAP_moderate']:.4f} / "
                    f"{metrics['mAP_hard']:.4f}"
                )
                self.logger.info(
                    f"📐 [{eval_split}] S/M/L  "
                    f"@0.5: {metrics['mAP_small']:.4f} / {metrics['mAP_medium']:.4f} / "
                    f"{metrics['mAP_large']:.4f}  |  "
                    f"@0.5:0.95: {metrics['mAP_small_5095']:.4f} / "
                    f"{metrics['mAP_medium_5095']:.4f} / {metrics['mAP_large_5095']:.4f}"
                )
                cls_50_str = ' | '.join(
                    f'{class_names[i]}={per_cls_50[i]:.4f}' for i in range(nc)
                )
                cls_5095_str = ' | '.join(
                    f'{class_names[i]}={per_cls_5095[i]:.4f}' for i in range(nc)
                )
                self.logger.info(f"📋 [{eval_split}] Per-class AP@0.5:  {cls_50_str}")
                self.logger.info(f"📋 [{eval_split}] Per-class AP@0.5:0.95:  {cls_5095_str}")
                if (bm_line := format_benchmark_eval_line(metrics)):
                    self.logger.info(bm_line)

                row = {}
                for k in fieldnames:
                    if k in ('model', 'dataset'):
                        continue
                    row[k] = (
                        format_eval_csv_cell(k, metrics[k]) if k in metrics else ''
                    )
                row['model'] = model_name
                row['dataset'] = dataset_name
                writer.writerow(row)
                last_metrics = metrics

        self.logger.info(f"✓ 指标已追加: {summary_csv}")
        self.logger.info(f"{'='*80}")

        return last_metrics

    def _run_model_benchmark(self, model):
        """运行 GFLOPs / FPS benchmark 并记录日志（仅一次）。"""
        from yolo_benchmark import benchmark_yolo
        bench_dict = None
        try:
            model_name = self.model_config.get('model_name', f'yolov{self.VERSION}n')
            if model_name.endswith('.pt'):
                model_name = model_name[:-3]
            bench_result = benchmark_yolo(
                model, imgsz=self.training_config.get('imgsz', 640),
                device=self.misc_config.get('device', 'cuda'),
                model_name=model_name,
            )
            log_benchmark(self.logger.info, bench_result, header=model_name)
            bench_dict = benchmark_to_dict(bench_result)
        except Exception as exc:
            self.logger.warning(f"Model benchmark 失败（不影响评估结果）: {exc}")
        return bench_dict

    def _post_training_processing(self, model=None):
        """训练后处理"""
        self.logger.info("=" * 80)
        self.logger.info("✅ 训练完成！")
        self.logger.info("=" * 80)

        self._plot_training_curves()
        self._align_file_naming()

        best_model_path = self.log_dir / "best_model.pth"
        if best_model_path.exists():
            self.logger.info(f"✓ 最佳模型: {best_model_path}")

        bench_dict = self._optional_post_train_benchmark(model)
        run_eval = model is not None or self._can_run_kitti_eval_without_ultralytics_model()
        if run_eval:
            try:
                self._evaluate_kitti_scale_after_training(model, bench_dict=bench_dict)
            except Exception as exc:
                self.logger.warning(f"KITTI/scale 评估出错（训练结果不受影响）: {exc}")

        self.logger.info(f"✓ 所有输出已保存到: {self.log_dir}")
        self.logger.info("=" * 80)
    
    def _parse_and_print_training_results(self):
        """解析results.csv并输出"""
        try:
            results_csv = self.log_dir / "results.csv"
            if not results_csv.exists():
                self.logger.warning(f"未找到results.csv文件: {results_csv}")
                return
            
            df = pd.read_csv(results_csv)
            self.logger.info("=" * 80)
            self.logger.info("训练过程摘要:")
            self.logger.info("=" * 80)
            
            # 提取关键指标列
            train_loss_cols = [c for c in df.columns if 'train/box_loss' in c.lower() or 
                             'train/cls_loss' in c.lower() or 'train/dfl_loss' in c.lower()]
            val_loss_cols = [c for c in df.columns if 'val/box_loss' in c.lower() or 
                            'val/cls_loss' in c.lower() or 'val/dfl_loss' in c.lower()]
            
            train_loss = df[train_loss_cols].sum(axis=1) if train_loss_cols else pd.Series(0, index=df.index)
            val_loss = df[val_loss_cols].sum(axis=1) if val_loss_cols else pd.Series(0, index=df.index)
            
            # 查找mAP列
            map50_col = next((c for c in df.columns if 'map50(b)' in c.lower()), None)
            map50_95_col = next((c for c in df.columns if 'map50-95(b)' in c.lower()), None)
            
            # 打印最后5个epoch的结果
            display_rows = min(5, len(df))
            for idx in range(len(df) - display_rows, len(df)):
                row = df.iloc[idx]
                epoch = int(row.get('epoch', idx + 1))
                line = f"Epoch {epoch}: Loss={train_loss.iloc[idx]:.2f}|{val_loss.iloc[idx]:.2f}"
                if map50_col and not pd.isna(row.get(map50_col)):
                    line += f" | mAP@0.5={row[map50_col]:.4f}"
                if map50_95_col and not pd.isna(row.get(map50_95_col)):
                    line += f" | mAP@0.5:0.95={row[map50_95_col]:.4f}"
                self.logger.info(line)
            
            self.logger.info("=" * 80)
        
        except Exception as e:
            self.logger.warning(f"解析训练结果失败: {e}")
    
    def _plot_training_curves(self):
        """生成训练曲线"""
        try:
            results_csv = self.log_dir / "results.csv"
            if not results_csv.exists():
                return
            
            df = pd.read_csv(results_csv)
            epochs = df.get('epoch', range(1, len(df) + 1)).values
            
            # 提取损失
            train_loss_cols = [c for c in df.columns if 'train/box_loss' in c.lower() or 
                             'train/cls_loss' in c.lower() or 'train/dfl_loss' in c.lower()]
            val_loss_cols = [c for c in df.columns if 'val/box_loss' in c.lower() or 
                            'val/cls_loss' in c.lower() or 'val/dfl_loss' in c.lower()]
            
            train_loss = df[train_loss_cols].sum(axis=1).values if train_loss_cols else None
            val_loss = df[val_loss_cols].sum(axis=1).values if val_loss_cols else None
            
            # 提取mAP
            map50_col = next((c for c in df.columns if 'map50(b)' in c.lower()), None)
            map50_95_col = next((c for c in df.columns if 'map50-95(b)' in c.lower()), None)
            
            map50 = df[map50_col].values if map50_col else None
            map50_95 = df[map50_95_col].values if map50_95_col else None
            
            # 提取学习率
            lr_col = next((c for c in df.columns if 'lr' in c.lower()), None)
            lr = df[lr_col].values if lr_col else None
            
            # 绘制
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f'YOLO{self.VERSION} Training Curves', fontsize=16, fontweight='bold')
            
            # 损失曲线
            if train_loss is not None:
                axes[0].plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2, markersize=4)
            if val_loss is not None:
                axes[0].plot(epochs, val_loss, 'r-s', label='Val Loss', linewidth=2, markersize=4)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Loss Curves')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # mAP曲线
            if map50 is not None:
                axes[1].plot(epochs, map50, 'g-^', label='mAP@0.5', linewidth=2, markersize=4)
            if map50_95 is not None:
                axes[1].plot(epochs, map50_95, 'm-d', label='mAP@[0.5:0.95]', linewidth=2, markersize=4)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('mAP')
            axes[1].set_title('mAP Metrics')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # 学习率曲线
            if lr is not None:
                axes[2].plot(epochs, lr, 'orange', linewidth=2)
                axes[2].set_yscale('log')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Learning Rate')
            axes[2].set_title('Learning Rate Schedule')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = self.log_dir / 'training_curves.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✓ 训练曲线已保存到: {save_path}")
        
        except Exception as e:
            self.logger.warning(f"绘制训练曲线失败: {e}")
    
    def _align_file_naming(self):
        """统一文件命名"""
        try:
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
        
        except Exception as e:
            self.logger.warning(f"统一文件命名失败: {e}")
