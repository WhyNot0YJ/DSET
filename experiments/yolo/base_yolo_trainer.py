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
from collections import Counter
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
from common.det_eval_metrics import (
    PYCOCOTOOLS_AVAILABLE,
    coco_ap_at_iou50_all,
    coco_area_ap_at_iou50,
    coco_area_bucket_counts_from_xywh_annotations,
    extract_per_category_ap_from_coco_eval,
    format_area_bucket_counts,
    run_coco_bbox_eval,
)

from yolo_validator_utils import MetricsLogger

# Ultralytics ``cfg/default.yaml``：未指定 batch 时为 16
DEFAULT_TRAIN_BATCH = 16


class BaseYOLOTrainer(ABC):
    """所有YOLO训练器的基类"""
    
    # 子类需要实现的版本名称
    VERSION = "base"
    
    def __init__(
        self,
        config: Dict,
        config_path: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        resume_checkpoint: Optional[str] = None,
    ):
        """
        初始化基础训练器
        
        Args:
            config: 配置字典
            config_path: 配置文件路径
            class_names: 类别名称列表
            resume_checkpoint: 若从 ``weights/*.pt`` 续训，传入路径可在首次 ``setup_logging`` 时锚定实验根目录
        """
        self.config = config
        self.config_path = config_path
        self.class_names = class_names or []
        self.num_classes = len(self.class_names)
        self._resume_checkpoint_path = resume_checkpoint

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

        def _experiment_root_from_ckpt(ckpt: Path) -> Path:
            ckpt = ckpt.resolve()
            parent = ckpt.parent
            if parent.name == "weights":
                exp = parent.parent
                if (exp / "config.yaml").is_file():
                    return exp
            return parent
        
        if resume_checkpoint and Path(resume_checkpoint).exists():
            self.log_dir = _experiment_root_from_ckpt(Path(resume_checkpoint))
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
            # 锚定到本文件所在目录（experiments/yolo），避免 cwd 与 YOLOX/Ultralytics 不一致时路径错位
            self.log_dir = (_yolo_dir / log_base / ds_dir / f"{self.experiment_name}_{timestamp}").resolve()
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
            'training': ['epochs'],
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
        """解析 Ultralytics 用的 data.yaml 路径（与 DETR 共用路径解析）。"""
        from common.detr_data_root import resolve_autodl_fs_path
        from common.dataset_registry import (
            load_dataset_registry,
            find_dataset_profile_by_coco_root,
        )

        data_yaml = self.data_config.get("data_yaml")
        if data_yaml and str(data_yaml).strip():
            return resolve_autodl_fs_path(data_yaml)

        # CaS_DETR 等仅配置 data_root（COCO 根），无 data_yaml：从 datasets.yaml 按 coco_data_root 反查
        root_raw = self.data_config.get("data_root") or self.data_config.get("coco_data_root")
        if root_raw and str(root_raw).strip():
            resolved_root = resolve_autodl_fs_path(str(root_raw).strip())
            registry_path = Path(__file__).resolve().parent / "configs" / "datasets.yaml"
            if registry_path.is_file():
                try:
                    datasets = load_dataset_registry(registry_path)
                    profile = find_dataset_profile_by_coco_root(datasets, resolved_root)
                    dy = (profile or {}).get("data_yaml")
                    if dy and str(dy).strip():
                        return resolve_autodl_fs_path(str(dy).strip())
                except Exception:
                    pass
            raise FileNotFoundError(
                f"配置中无 data_yaml，且无法根据 data_root={resolved_root!r} "
                f"在 {registry_path} 中匹配到 coco_data_root；请显式设置 data.data_yaml"
            )

        raise ValueError("路径为空：data.data_yaml 未设置且 data.data_root 为空")
    
    def _apply_vram_batch_size_rule(self):
        """使用配置中的 batch / workers；CUDA 下记录显存信息（与 cas_detr 共用 common.vram_batch）。"""
        device_str = self.misc_config.get('device', 'cuda')
        if 'cuda' not in str(device_str) or not torch.cuda.is_available():
            return

        idx = resolve_cuda_device_index(str(device_str))
        orig_bs = int(self.training_config.get('batch_size', DEFAULT_TRAIN_BATCH))
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
        构建传给 ``model.train()`` 的参数：仅包含 YAML 中已写的项，以及输出目录所需的
        ``data`` / ``project`` / ``name``；其余由 Ultralytics 默认配置补齐。
        """
        train_kwargs: Dict = {
            'data': self._resolve_data_yaml(),
            'project': str(self.log_dir.parent),
            'name': self.log_dir.name,
        }
        for k, v in self.training_config.items():
            if k == 'batch_size':
                train_kwargs['batch'] = v
            else:
                train_kwargs[k] = v
        if 'device' in self.misc_config:
            train_kwargs['device'] = self.misc_config['device']
        if 'num_workers' in self.misc_config:
            train_kwargs['workers'] = self.misc_config['num_workers']
        if 'exist_ok' in self.checkpoint_config:
            train_kwargs['exist_ok'] = self.checkpoint_config['exist_ok']
        return train_kwargs

    def _log_training_config(self, train_kwargs: Dict):
        """记录训练配置信息"""
        self.logger.info("=" * 80)
        self.logger.info(f"🚀 开始YOLO{self.VERSION}训练")
        self.logger.info("=" * 80)
        self.logger.info("📝 训练配置:")
        self.logger.info(f"  数据集路径: {train_kwargs['data']}")
        if 'epochs' in train_kwargs:
            self.logger.info(f"  训练轮数: {train_kwargs['epochs']}")
        if 'batch' in train_kwargs:
            self.logger.info(f"  批次大小: {train_kwargs['batch']}")
        for k in ('optimizer', 'lr0', 'weight_decay', 'imgsz'):
            if k in self.training_config:
                self.logger.info(f"  {k}: {self.training_config[k]}")
        self.logger.info(f"  输出目录: {self.log_dir}")
        if self.model_config.get('pretrained_weights'):
            self.logger.info(f"  预训练权重: {self.model_config['pretrained_weights']}")
        if 'seed' in train_kwargs:
            self.logger.info(f"  随机种子 seed: {train_kwargs['seed']}")
        if 'deterministic' in train_kwargs:
            self.logger.info(f"  deterministic: {train_kwargs['deterministic']}")
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
    # Post-training KITTI / multi-scale evaluation
    # ------------------------------------------------------------------

    def _labels_meta_split_dir(self, root: Path, split: str) -> Path:
        """``root/labels_meta/{split}``；若为空且配置了 ``data.coco_data_root``，则尝试该根下的同名路径。"""
        primary = root / 'labels_meta' / split
        if primary.is_dir() and any(primary.glob('*.json')):
            return primary
        cr = self.data_config.get('coco_data_root')
        if not cr:
            return primary
        alt = Path(str(cr)).expanduser().resolve() / 'labels_meta' / split
        if alt.resolve() == primary.resolve():
            return primary
        if alt.is_dir() and any(alt.glob('*.json')):
            self.logger.info(
                "KITTI/scale 使用 data.coco_data_root 下的 labels_meta/%s: %s",
                split,
                alt,
            )
            return alt
        return primary

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
        lm_val = self._labels_meta_split_dir(root, 'val')
        if lm_val.is_dir() and any(lm_val.glob('*.json')):
            out.append(('val', val_img_dir, lm_val))

        test_rel = str(data_cfg.get('test', '')).strip()
        if eval_test and test_rel:
            test_dir = Path(test_rel) if Path(test_rel).is_absolute() else root / test_rel
            lm_test = self._labels_meta_split_dir(root, 'test')
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
            cr = self.data_config.get('coco_data_root')
            cr_res = Path(str(cr)).expanduser().resolve() if cr else None
            self.logger.warning(
                '未找到可用的 KITTI/scale 评估划分：在 YAML path 对应 root=%s 与 data.coco_data_root=%s '
                '下均未找到含 JSON 的 labels_meta/val 或 labels_meta/test',
                root,
                cr_res,
            )
            return {}

        # ── 2. Load best weights & benchmark（各 split 共用）──────────────
        eval_predictor, nc = self._get_kitti_eval_predictor(model)
        if eval_predictor is None:
            best_pt = (self.log_dir / "weights" / "best.pt").resolve()
            last_pt = (self.log_dir / "weights" / "last.pt").resolve()
            self.logger.warning(
                "无可用评估权重/预测器，跳过 KITTI/scale 评估：未找到 %s "
                "（eval_best_model 未传入内存中的 model，必须依赖该文件）",
                best_pt,
            )
            if last_pt.is_file():
                self.logger.warning(
                    "  发现 last.pt，可执行: cp %s %s 后再评估",
                    last_pt,
                    best_pt,
                )
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
                debug_gt_annotations: List[Dict[str, Any]] = []
                debug_pred_annotations: List[Dict[str, Any]] = []
                debug_image_ids: set[int] = set()
                # COCOeval：全类 mAP / S/M/L / 每类 AP
                img_sizes: Dict[int, Tuple[int, int]] = {}
                coco_annotations: List[Dict[str, Any]] = []
                coco_predictions: List[Dict[str, Any]] = []
                ann_id = 0

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
                        img_sizes[img_idx] = (int(img_w), int(img_h))
                        debug_image_ids.add(img_idx)
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
                            w_px = (px2 - px1)
                            area_px = w_px * h_px
                            if w_px <= 0 or h_px <= 0:
                                continue

                            ann_id += 1
                            coco_annotations.append(
                                {
                                    "id": ann_id,
                                    "image_id": img_idx,
                                    "category_id": cls + 1,
                                    "bbox": [float(px1), float(py1), float(w_px), float(h_px)],
                                    "area": float(w_px * h_px),
                                    "iscrowd": 0,
                                }
                            )
                            debug_gt_annotations.append(
                                {
                                    "image_id": img_idx,
                                    "category_id": cls + 1,
                                    "bbox": [float(px1), float(py1), float(w_px), float(h_px)],
                                }
                            )

                        if result.boxes is not None:
                            for box, conf, cls_t in zip(
                                result.boxes.xyxy.cpu().numpy(),
                                result.boxes.conf.cpu().numpy(),
                                result.boxes.cls.cpu().numpy().astype(int),
                            ):
                                c = int(cls_t)
                                if 0 <= c < nc:
                                    box_list = box.tolist()
                                    x1, y1, x2, y2 = map(float, box_list)
                                    w_px = x2 - x1
                                    h_px = y2 - y1
                                    if w_px > 0 and h_px > 0:
                                        coco_predictions.append(
                                            {
                                                "image_id": img_idx,
                                                "category_id": c + 1,
                                                "bbox": [x1, y1, w_px, h_px],
                                                "score": float(conf),
                                            }
                                        )
                                        debug_pred_annotations.append(
                                            {
                                                "image_id": img_idx,
                                                "category_id": c + 1,
                                                "bbox": [x1, y1, w_px, h_px],
                                            }
                                        )

                if os.getenv("CAS_DEBUG_AREA_METRICS", "0") == "1":
                    gt_counts = coco_area_bucket_counts_from_xywh_annotations(debug_gt_annotations)
                    pred_counts = coco_area_bucket_counts_from_xywh_annotations(debug_pred_annotations)
                    self.logger.info(
                        "[DEBUG][YOLO][AREA][%s] images=%d  %s  %s",
                        eval_split,
                        len(debug_image_ids),
                        format_area_bucket_counts("gt", gt_counts),
                        format_area_bucket_counts("pred", pred_counts),
                    )

                self.logger.info(
                    "[%s] COCO 评估输入: pycocotools=%s, GT=%d 条, pred=%d 条",
                    eval_split,
                    PYCOCOTOOLS_AVAILABLE,
                    len(coco_annotations),
                    len(coco_predictions),
                )

                # ── 5. AP：全类 mAP / S/M/L / 每类 AP（一次 COCOeval，全 GT iscrowd=0）
                metrics: Dict[str, Any] = {}

                categories_coco = [
                    {'id': i + 1, 'name': class_names[i]} for i in range(nc)
                ]
                coco_gt = {
                    'images': [
                        {
                            'id': i,
                            'width': img_sizes[i][0],
                            'height': img_sizes[i][1],
                        }
                        for i in range(len(eval_images))
                    ],
                    'categories': categories_coco,
                    'annotations': coco_annotations,
                }

                coco_eval = run_coco_bbox_eval(coco_gt, coco_predictions)
                per_cls_50: List[float] = []
                per_cls_5095: List[float] = []
                if coco_eval is None:
                    if not PYCOCOTOOLS_AVAILABLE:
                        self.logger.warning(
                            f"[{eval_split}] COCOeval 跳过：未安装 pycocotools，"
                            "请执行: pip install pycocotools  （COCO 口径指标已置 0）"
                        )
                    elif not coco_annotations:
                        self.logger.warning(
                            f"[{eval_split}] COCOeval 跳过：未解析到任何 GT（labels_meta 与图像是否匹配、"
                            "bbox_yolo/bbox_xyxy 是否存在），COCO 口径指标置 0"
                        )
                    else:
                        self.logger.warning(
                            f"[{eval_split}] COCOeval 失败（pycocotools 运行异常），"
                            "COCO 口径指标置 0"
                        )
                    metrics['mAP_50_all'] = 0.0
                    metrics['mAP_5095_all'] = 0.0
                    metrics['mAP_small'] = 0.0
                    metrics['mAP_medium'] = 0.0
                    metrics['mAP_large'] = 0.0
                    metrics['mAP_small_5095'] = 0.0
                    metrics['mAP_medium_5095'] = 0.0
                    metrics['mAP_large_5095'] = 0.0
                    per_cls_50 = [0.0] * nc
                    per_cls_5095 = [0.0] * nc
                    for i in range(nc):
                        nm = class_names[i]
                        metrics[f'AP50_{nm}'] = 0.0
                        metrics[f'AP5095_{nm}'] = 0.0
                else:
                    metrics['mAP_50_all'] = coco_ap_at_iou50_all(coco_eval)
                    metrics['mAP_5095_all'] = (
                        max(0.0, float(coco_eval.stats[0]))
                        if len(coco_eval.stats) > 0
                        else 0.0
                    )
                    s50, m50, l50 = coco_area_ap_at_iou50(coco_eval)
                    metrics['mAP_small'] = s50
                    metrics['mAP_medium'] = m50
                    metrics['mAP_large'] = l50
                    if len(coco_eval.stats) >= 6:
                        metrics['mAP_small_5095'] = max(0.0, float(coco_eval.stats[3]))
                        metrics['mAP_medium_5095'] = max(0.0, float(coco_eval.stats[4]))
                        metrics['mAP_large_5095'] = max(0.0, float(coco_eval.stats[5]))
                    else:
                        metrics['mAP_small_5095'] = 0.0
                        metrics['mAP_medium_5095'] = 0.0
                        metrics['mAP_large_5095'] = 0.0

                    per_cat_50, per_cat_5095 = extract_per_category_ap_from_coco_eval(
                        coco_eval, categories_coco
                    )
                    per_cls_50 = [
                        per_cat_50.get(class_names[i], 0.0) for i in range(nc)
                    ]
                    per_cls_5095 = [
                        per_cat_5095.get(class_names[i], 0.0) for i in range(nc)
                    ]
                    for i in range(nc):
                        nm = class_names[i]
                        metrics[f'AP50_{nm}'] = per_cat_50.get(nm, 0.0)
                        metrics[f'AP5095_{nm}'] = per_cat_5095.get(nm, 0.0)

                metrics['eval_split'] = eval_split
                merge_benchmark_dict_into_metrics(metrics, bench_dict)

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

        bench_dict = None
        try:
            self._plot_training_curves()
            self._align_file_naming()

            best_model_path = self.log_dir / "best_model.pth"
            if best_model_path.exists():
                self.logger.info(f"✓ 最佳模型: {best_model_path}")

            try:
                bench_dict = self._optional_post_train_benchmark(model)
            except Exception as exc:
                self.logger.exception(
                    "训练后 benchmark/预测器准备失败（不影响已保存权重；eval 阶段会重试加载）: %s",
                    exc,
                )

            run_eval = model is not None or self._can_run_kitti_eval_without_ultralytics_model()
            if not run_eval:
                self.logger.warning(
                    "跳过 KITTI/scale 与 eval_metrics：无 Ultralytics 模型且未在 log_dir 检测到可评估权重。"
                    " log_dir=%s （请确认 best_ckpt.pth / weights/best.pt 是否在此目录下）",
                    self.log_dir.resolve(),
                )
            else:
                try:
                    self._evaluate_kitti_scale_after_training(model, bench_dict=bench_dict)
                except Exception as exc:
                    self.logger.warning(f"KITTI/scale 评估出错（训练结果不受影响）: {exc}")
        finally:
            self.logger.info(f"✓ 所有输出已保存到: {self.log_dir.resolve()}")
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
