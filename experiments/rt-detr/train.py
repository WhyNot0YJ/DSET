import sys
import os
import argparse
from pathlib import Path

_experiments_root = Path(__file__).resolve().parent.parent
if str(_experiments_root) not in sys.path:
    sys.path.insert(0, str(_experiments_root))
from common.dataset_registry import (
    load_dataset_registry,
    resolve_dataset_profile,
    find_dataset_profile_by_data_yaml,
    apply_detr_dataset_profile,
    default_detr_registry_path,
)
from common.vram_batch import (
    compute_vram_batch_adjustment,
    format_vram_batch_log,
)
from common.det_eval_metrics import (
    kitti_difficulty_from_coco_ann,
    coco_ap_at_iou50_all,
    coco_area_ap_at_iou50,
    dataset_dir_name,
)
from common.detr_eval_utils import (
    evaluate_best_model_after_training,
    log_detr_eval_summary,
    write_detr_eval_csv,
)
from common.detr_data_root import resolve_detr_data_root
from common.eval_schedule import (
    should_run_validation,
    describe_eval_schedule,
)
import yaml
import torch
import numpy as np
import re
import logging
from typing import Optional, Dict, Union, List, Tuple
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

project_root = Path(__file__).parent.resolve()
if str(os.getcwd()) not in sys.path:
    sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))

from seed_utils import set_seed, seed_worker
from src.misc.training_visualizer import TrainingVisualizer
from src.misc.early_stopping import EarlyStopping
from src.data import DataLoader
from src.optim.ema import ModelEMA
from src.optim.warmup import WarmupLR
from src.data.dataset.dairv2x_detection import DAIRV2XDetection
from src.data.dataset.coco_folder_detection import CocoFolderDetection
from src.nn.postprocessor.detr_postprocessor import DetDETRPostProcessor
from src.nn.postprocessor.box_revert import box_revert, BoxProcessFormat
import cv2
import torchvision.transforms as T

try:
    from batch_inference import postprocess_outputs, draw_boxes, inference_from_preprocessed_image
    USE_BATCH_INFERENCE_LOGIC = True
except ImportError:
    USE_BATCH_INFERENCE_LOGIC = False


def create_backbone(backbone_type: str, **kwargs):
    """创建backbone的工厂函数"""
    from src.nn.backbone.presnet import PResNet
    from src.nn.backbone.hgnetv2 import HGNetv2
    from src.nn.backbone.csp_resnet import CSPResNet
    from src.nn.backbone.csp_darknet import CSPDarkNet
    
    if backbone_type.startswith('presnet'):
        depth_match = re.search(r'(\d+)', backbone_type)
        if depth_match:
            depth = int(depth_match.group(1))
        else:
            raise ValueError(f"无法从backbone类型 {backbone_type} 解析depth")
        
        default_params = {
            'depth': depth,
            'variant': 'd',
            'return_idx': [1, 2, 3],
            'freeze_at': 0,
            'freeze_norm': True,
            'pretrained': False
        }
        default_params.update(kwargs)
        return PResNet(**default_params)
    
    elif backbone_type.startswith('hgnetv2'):
        name_map = {'hgnetv2_l': 'L', 'hgnetv2_x': 'X', 'hgnetv2_h': 'H'}
        if backbone_type not in name_map:
            raise ValueError(f"不支持的HGNetv2类型: {backbone_type}")
        
        default_params = {
            'name': name_map[backbone_type],
            'return_idx': [1, 2, 3],
            'freeze_at': 0,
            'freeze_norm': True,
            'pretrained': False
        }
        default_params.update(kwargs)
        return HGNetv2(**default_params)
    
    elif backbone_type.startswith('cspresnet'):
        name_map = {'cspresnet_s': 's', 'cspresnet_m': 'm', 'cspresnet_l': 'l', 'cspresnet_x': 'x'}
        if backbone_type not in name_map:
            raise ValueError(f"不支持的CSPResNet类型: {backbone_type}")
        
        default_params = {
            'name': name_map[backbone_type],
            'return_idx': [1, 2, 3],
            'pretrained': False
        }
        default_params.update(kwargs)
        return CSPResNet(**default_params)
    
    elif backbone_type == 'cspdarknet':
        default_params = {'return_idx': [2, 3, -1]}
        default_params.update(kwargs)
        return CSPDarkNet(**default_params)
    
    else:
        raise ValueError(f"不支持的backbone类型: {backbone_type}")


class RTDETRTrainer:
    
    def __init__(self, config: Union[str, dict], pretrained_weights: Optional[str] = None, 
                 data_root: Optional[str] = None, epochs: Optional[int] = None,
                 batch_size: Optional[int] = None, warmup_epochs: Optional[int] = None):
        """初始化训练器
        
        Args:
            config: 配置文件路径或配置字典
            pretrained_weights: 预训练权重路径（可选，会覆盖配置文件）
            data_root: 数据集根目录（可选，会覆盖配置文件）
            epochs: 训练轮数（可选，会覆盖配置文件）
            batch_size: 批次大小（可选，会覆盖配置文件）
            warmup_epochs: 学习率预热轮数（可选，会覆盖配置文件）
        """
        self.pretrained_weights = pretrained_weights
        
        # 加载配置
        using_config_file = isinstance(config, str)
        if using_config_file:
            # 从文件加载配置
            self.config_path = config
            with open(config, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            # 直接使用配置字典
            self.config_path = None
            self.config = config
        
        # 如果使用配置文件，验证必需的配置项是否存在
        if using_config_file:
            self._validate_config_file()
        
        if 'training' in self.config:
            if 'pretrained_lr' in self.config['training']:
                self.config['training']['pretrained_lr'] = float(self.config['training']['pretrained_lr'])
            if 'new_lr' in self.config['training']:
                self.config['training']['new_lr'] = float(self.config['training']['new_lr'])
            if 'eta_min' in self.config['training']:
                self.config['training']['eta_min'] = float(self.config['training']['eta_min'])
            if 'weight_decay' in self.config['training']:
                self.config['training']['weight_decay'] = float(self.config['training']['weight_decay'])
        
        if data_root is not None:
            self.config['data']['data_root'] = data_root
        
        if epochs is not None:
            self.config['training']['epochs'] = epochs
        
        if batch_size is not None:
            self.config['training']['batch_size'] = batch_size
        
        if warmup_epochs is not None:
            self.config['training']['warmup_epochs'] = warmup_epochs

        if using_config_file:
            if 'misc' not in self.config or 'device' not in self.config['misc']:
                raise ValueError(f"配置文件 {self.config_path} 缺少必需的配置项: misc.device")
            device_str = self.config['misc']['device']
        else:
            device_str = self.config.get('misc', {}).get('device', 'cuda')
        self.device = torch.device(device_str)
        self.log_dir = None
        self.logger = None
        self.experiment_name = None
        self._create_directories()
        
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.warmup_scheduler = None
        self.ema = None
        self.scaler = None
        self.visualizer = None
        self.postprocessor = None
        
        self.class_names = [
            "Car", "Truck", "Van", "Bus", "Pedestrian", 
            "Cyclist", "Motorcyclist", "Trafficcone"
        ]
        self.colors = [
            (255, 0, 0),      # Car - 红色
            (0, 255, 0),      # Truck - 绿色
            (255, 128, 0),    # Van - 橙色
            (0, 0, 255),      # Bus - 蓝色
            (255, 255, 0),    # Pedestrian - 黄色
            (255, 0, 255),    # Cyclist - 品红
            (0, 255, 255),    # Motorcyclist - 青色
            (128, 128, 128),  # Trafficcone - 灰色
        ]
        self.num_classes = 8
        self._init_dataset_derived_fields()

    def _init_dataset_derived_fields(self):
        """从 config['data'] 同步 num_classes / class_names / colors。"""
        dc = self.config.setdefault("data", {})
        self.num_classes = int(dc.get("num_classes", self.num_classes))
        cn = dc.get("class_names")
        if isinstance(cn, list) and cn:
            self.class_names = [str(x) for x in cn]
        while len(self.colors) < len(self.class_names):
            self.colors.append((128, 128, 128))
        self.colors = self.colors[: len(self.class_names)]
        self._dair_truncation_categorical = (
            dc.get("dataset_class") == "DAIRV2XDetection"
            or "DAIR-V2X" in str(dc.get("data_root", ""))
        )

    def _resolve_dataset_class(self):
        name = self.config.get("data", {}).get("dataset_class", "DAIRV2XDetection")
        if name == "DAIRV2XDetection":
            return DAIRV2XDetection
        if name == "CocoFolderDetection":
            return CocoFolderDetection
        raise ValueError(f"未知 dataset_class: {name}")

    def _apply_vram_batch_size_rule(self):
        """按显存动态设置 batch（逻辑见 common.vram_batch）。"""
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            return

        base_bs = int(self.config.get('training', {}).get('batch_size', 16))
        orig_nw = int(self.config.get('misc', {}).get('num_workers', 4))
        orig_pf = int(self.config.get('misc', {}).get('prefetch_factor', 1))
        device_index = (
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device()
        )

        r = compute_vram_batch_adjustment(
            base_bs, orig_nw, orig_pf, device_index=device_index
        )
        if r is None:
            return

        self.config['training']['batch_size'] = r.batch_size
        self.config.setdefault('misc', {})['num_workers'] = r.num_workers
        self.config.setdefault('misc', {})['prefetch_factor'] = r.prefetch_factor

        if getattr(self, "logger", None):
            self.logger.info(format_vram_batch_log(r))
    
    def _validate_config_file(self):
        """验证配置文件是否包含所有必需的配置项"""
        required_keys = {
            'model': ['backbone', 'num_decoder_layers', 'hidden_dim', 'num_queries'],
            'training': ['epochs', 'batch_size', 'pretrained_lr', 'new_lr'],
            'data': ['data_root'],
            'misc': ['device', 'num_workers']
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
            error_msg = f"配置文件 {self.config_path} 缺少必需的配置项:\n"
            error_msg += "\n".join(f"  - {key}" for key in missing_keys)
            raise ValueError(error_msg)
    
    def setup_logging(self):
        """设置日志系统"""
        # 检查是否从检查点恢复
        resume_checkpoint = getattr(self, '_resume_checkpoint_path', None)
        
        if resume_checkpoint and Path(resume_checkpoint).exists():
            # 恢复训练：使用检查点所在目录（不创建新目录）
            self.log_dir = Path(resume_checkpoint).parent
            # 从目录名中提取实验名称（去掉时间戳部分）
            dir_name = self.log_dir.name
            # 假设格式为 rtdetr_r50_20240101_120000，提取 rtdetr_r50
            parts = dir_name.rsplit('_', 2)  # 分割最后两部分（日期和时间）
            if len(parts) >= 2:
                self.experiment_name = '_'.join(parts[:-2]) if len(parts) > 2 else parts[0]
            else:
                self.experiment_name = dir_name
        else:
            # 新训练：创建带时间戳的目录
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 从配置中获取backbone类型，加入到目录名中
            backbone = self.config['model']['backbone']
            # 移除presnet前缀，只保留数字部分（如presnet18 -> r18, presnet34 -> r34）
            backbone_short = backbone.replace('presnet', 'r').replace('pres', 'r') if 'presnet' in backbone or 'pres' in backbone else backbone
            # 生成实验名称（不带时间戳）
            self.experiment_name = f"rtdetr_{backbone_short}"
            ds_dir = dataset_dir_name(self.config)
            self.log_dir = Path(f"logs/{ds_dir}/{self.experiment_name}_{timestamp}")
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
            force=True  # 强制重新配置
        )
        
        self.logger = logging.getLogger(__name__)
        
        # 如果是恢复训练，记录日志
        if resume_checkpoint and Path(resume_checkpoint).exists():
            self.logger.info(f"📦 恢复训练，使用现有日志目录: {self.log_dir}")
        
        # 保存配置文件（仅新训练时）
        if not resume_checkpoint:
            config_save_path = self.log_dir / 'config.yaml'
            with open(config_save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            self.logger.info(f"✓ 配置已保存到: {config_save_path}")
    
    def _create_directories(self):
        """创建必要的目录（已在setup_logging中创建）"""
        # log_dir 已在 setup_logging 中创建
        # 所有输出都保存在 log_dir 中，无需额外创建目录
        pass
    
    def create_model(self):
        """创建模型"""
        # 从配置文件读取backbone类型
        backbone_type = self.config['model']['backbone']
        
        # 动态创建backbone
        backbone = create_backbone(backbone_type)
        
        # 从配置文件读取encoder配置
        encoder_config = self.config['model']['encoder']
        in_channels = encoder_config['in_channels']
        expansion = encoder_config['expansion']
        
        self.logger.info(f"✓ Backbone: {backbone_type}")
        self.logger.info(f"✓ HybridEncoder: in_channels={in_channels}, expansion={expansion}")
        
        # 创建encoder
        from src.zoo.rtdetr.hybrid_encoder import HybridEncoder
        encoder = HybridEncoder(
            in_channels=in_channels,
            feat_strides=[8, 16, 32],
            hidden_dim=256,
            use_encoder_idx=[2],
            num_encoder_layers=1,
            expansion=expansion,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.0,
            enc_act='gelu',
            act='silu'
        )
        
        # 从配置文件读取模型参数
        num_decoder_layers = self.config['model']['num_decoder_layers']
        hidden_dim = self.config['model']['hidden_dim']
        num_queries = self.config['model']['num_queries']
        
        # 创建decoder（添加denoising训练）
        from src.zoo.rtdetr.rtdetrv2_decoder import RTDETRTransformerv2
        decoder = RTDETRTransformerv2(
            num_classes=self.num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_layers=num_decoder_layers, 
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation='relu',
            feat_channels=[256, 256, 256],
            feat_strides=[8, 16, 32],
            num_levels=3,
            # 添加denoising训练参数
            num_denoising=100,
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            num_points=[4, 4, 4]
        )
        
        self.logger.info(f"✓ Decoder配置: {num_decoder_layers}层, hidden_dim={hidden_dim}, queries={num_queries}")
        
        # 创建RT-DETR模型
        from src.zoo.rtdetr.rtdetr import RTDETR
        model = RTDETR(backbone=backbone, encoder=encoder, decoder=decoder)
        
        self.logger.info("✓ 模型创建完成（已启用backbone预训练）")
        
        return model
    
    def load_pretrained_weights(self, model, pretrained_path: str):
        """加载预训练权重
        
        支持多种checkpoint格式：
        - EMA格式: {'ema': {'module': {...}}}
        - 标准格式: {'model': {...}} 或 {'model_state_dict': {...}}
        - 直接权重: state_dict
        
        Args:
            model: RT-DETR模型
            pretrained_path: 预训练权重路径
        """
        try:
            pretrained_file = Path(pretrained_path)
            if not pretrained_file.exists():
                self.logger.warning(f"⚠ 预训练权重文件不存在: {pretrained_path}")
                self.logger.info("将使用随机初始化权重")
                return
            
            self.logger.info(f"正在从本地文件加载预训练权重: {pretrained_path}")
            checkpoint = torch.load(pretrained_file, map_location='cpu', weights_only=False)
            
            # 处理不同的checkpoint格式
            if isinstance(checkpoint, dict):
                if 'ema' in checkpoint and 'module' in checkpoint['ema']:
                    # EMA格式: {'ema': {'module': {...}}}
                    state_dict = checkpoint['ema']['module']
                    self.logger.info("✓ 检测到EMA checkpoint格式")
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
        
            filtered_state_dict = {}
            skipped_class_params = 0
            
            for k, v in state_dict.items():
                # 跳过类别相关的参数（这些参数的形状会不匹配）
                if any(keyword in k for keyword in ['class_embed', 'score_head', 'denoising_class_embed']):
                    skipped_class_params += 1
                    continue
                filtered_state_dict[k] = v
            
            # 加载过滤后的参数
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
            
            # 统计加载结果
            # 注意：missing_keys 可能包含预训练模型中不存在的参数（如不同模型结构的差异）
            # 只统计预训练模型中实际存在的 missing_keys
            actual_missing_keys = [k for k in missing_keys if k in filtered_state_dict]
            total_params = len(filtered_state_dict)
            loaded_params = total_params - len(actual_missing_keys)
            
            self.logger.info(f"✓ 成功加载预训练权重: {loaded_params}/{total_params} 个参数")
            
            # 报告跳过的类别参数
            if skipped_class_params > 0:
                self.logger.info(f"  - 跳过类别相关参数: {skipped_class_params} 个（COCO 80类 → DAIR-V2X 8类）")
            
            # 统计各部分的参数（只统计预训练模型中实际存在的参数）
            backbone_loaded = sum(1 for k in filtered_state_dict.keys() if k not in actual_missing_keys and 'backbone' in k)
            encoder_loaded = sum(1 for k in filtered_state_dict.keys() if k not in actual_missing_keys and 'encoder' in k)
            decoder_loaded = sum(1 for k in filtered_state_dict.keys() if k not in actual_missing_keys and 'decoder' in k)
            
            self.logger.info(f"  - Backbone: {backbone_loaded} 个参数")
            self.logger.info(f"  - Encoder: {encoder_loaded} 个参数")
            self.logger.info(f"  - Decoder: {decoder_loaded} 个参数")
            
            if len(actual_missing_keys) > 0:
                # actual_missing_keys是filtered_state_dict中有但当前模型没有的参数
                self.logger.info(f"  - 预训练模型缺少参数: {len(actual_missing_keys)} 个（当前模型新增）")
                # 显示前3个示例
                if len(actual_missing_keys) <= 5:
                    self.logger.info(f"    示例: {list(actual_missing_keys)}")
                else:
                    self.logger.info(f"    示例: {list(actual_missing_keys)[:3]} ...")
            
            # 如果 missing_keys 中有预训练模型中不存在的参数，说明是模型结构差异
            model_only_missing = [k for k in missing_keys if k not in filtered_state_dict]
            if len(model_only_missing) > 0:
                self.logger.debug(f"  - 模型结构差异导致的 missing_keys: {len(model_only_missing)} 个（预训练模型中不存在，不影响加载统计）")
            
            if len(unexpected_keys) > 0:
                self.logger.info(f"  - 模型新增参数: {len(unexpected_keys)} 个（将随机初始化）")
            
        except Exception as e:
            self.logger.error(f"✗ 加载预训练权重失败: {e}")
            self.logger.info("将使用随机初始化权重")
    
    def create_criterion(self):
        """创建损失函数"""
        from src.zoo.rtdetr.matcher import HungarianMatcher
        from src.zoo.rtdetr.rtdetrv2_criterion import RTDETRCriterionv2
        
        # 创建matcher
        matcher = HungarianMatcher(
            weight_dict={'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
            use_focal_loss=False,
            alpha=0.25,
            gamma=2.0
        )
        
        # 主损失权重
        main_weight_dict = {
            'loss_vfl': 1.0,
            'loss_bbox': 5.0,
            'loss_giou': 2.0
        }
        
        # 辅助损失权重（decoder的前N-1层）
        num_decoder_layers = self.config['model']['num_decoder_layers']
        aux_weight_dict = {}
        for i in range(num_decoder_layers - 1):  # 前N-1层
            aux_weight_dict[f'loss_vfl_aux_{i}'] = 1.0
            aux_weight_dict[f'loss_bbox_aux_{i}'] = 5.0
            aux_weight_dict[f'loss_giou_aux_{i}'] = 2.0
        
        # Encoder辅助损失
        aux_weight_dict['loss_vfl_enc_0'] = 1.0
        aux_weight_dict['loss_bbox_enc_0'] = 5.0
        aux_weight_dict['loss_giou_enc_0'] = 2.0
        
        # Denoising辅助损失
        num_denoising_layers = num_decoder_layers  # 和decoder层数一致
        for i in range(num_denoising_layers):
            aux_weight_dict[f'loss_vfl_dn_{i}'] = 1.0
            aux_weight_dict[f'loss_bbox_dn_{i}'] = 5.0
            aux_weight_dict[f'loss_giou_dn_{i}'] = 2.0
        
        # 合并所有权重
        weight_dict = {**main_weight_dict, **aux_weight_dict}
        
        # 创建criterion（与MoE RT-DETR保持一致）
        criterion = RTDETRCriterionv2(
            matcher=matcher,
            weight_dict=weight_dict,
            losses=['vfl', 'boxes'],
            alpha=0.75,
            gamma=2.0,
            num_classes=self.num_classes,
            boxes_weight_format=None,
            share_matched_indices=False
        )
        
        return criterion
    
    def create_datasets(self):
        """创建数据集"""
        from src.data.dataloader import BatchImageCollateFuncion
        
        raw_root = self.config['data']['data_root']
        data_root = resolve_detr_data_root(raw_root)
        if data_root != raw_root:
            self.logger.info(f"✓ 数据集根目录: {raw_root} → {data_root}")
        self.config['data']['data_root'] = data_root
        ds_cls = self._resolve_dataset_class()
        augmentation_config = self.config.get('augmentation', {})

        train_dataset = ds_cls(
            data_root=data_root,
            split='train',
            augmentation_config=augmentation_config
        )
        
        val_dataset = ds_cls(
            data_root=data_root,
            split='val',
            augmentation_config=augmentation_config
        )

        # 多尺度训练配置 (从config中读取或使用默认值)
        scales = self.config.get('augmentation', {}).get('scales', [576, 608, 640, 640, 640, 672, 704])
        stop_epoch = self.config.get('augmentation', {}).get('stop_epoch', 71)
        train_collate_fn = BatchImageCollateFuncion(scales=scales, stop_epoch=stop_epoch)
        # Keep validation deterministic at the dataset's fixed resize target (640 by config).
        val_collate_fn = BatchImageCollateFuncion(scales=None, stop_epoch=stop_epoch)
        
        # num_workers在misc配置中
        num_workers = self.config.get('misc', {}).get('num_workers', 16)
        pin_memory = self.config.get('misc', {}).get('pin_memory', True)
        prefetch_factor = self.config.get('misc', {}).get('prefetch_factor', 2)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            collate_fn=train_collate_fn,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            collate_fn=val_collate_fn,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )
        
        return train_loader, val_loader
    
    def create_optimizer(self):
        """创建优化器（使用分组学习率）"""
        # 获取配置中的学习率，确保是浮点数类型（直接使用配置文件字段名）
        new_lr = float(self.config['training']['new_lr'])
        pretrained_lr = float(self.config['training']['pretrained_lr'])
        weight_decay = float(self.config['training'].get('weight_decay', 0.0001))
        
        # 分组参数
        param_groups = []
        
        # 定义新增结构的关键词（rt-detr没有MoE/CaS_DETR结构，所以为空）
        new_structure_keywords = []
        
        # 1. 预训练参数组（backbone、encoder、decoder的标准层，排除norm层和新增结构）
        pretrained_params = []
        pretrained_names = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # 判断是否为预训练部分（backbone、encoder、decoder）
                is_pretrained = any(part in name for part in ['backbone', 'encoder', 'decoder'])
                # 排除norm层
                is_norm = any(norm in name for norm in ['norm', 'bn', 'gn', 'ln'])
                # 排除新增结构（即使它们在encoder/decoder中）
                is_new_structure = any(keyword in name.lower() for keyword in new_structure_keywords)
                
                if is_pretrained and not is_norm and not is_new_structure:
                    pretrained_params.append(param)
                    pretrained_names.append(name)
        
        if pretrained_params:
            param_groups.append({
                'params': pretrained_params,
                'lr': pretrained_lr,
                'weight_decay': weight_decay
            })
            self.logger.info(f"✓ 预训练参数组: {len(pretrained_params)} 个参数, lr={pretrained_lr}")
        
        # 2. Norm层参数（无weight decay）
        norm_params = []
        norm_names = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and any(norm in name for norm in ['norm', 'bn', 'gn', 'ln']):
                norm_params.append(param)
                norm_names.append(name)
        
        if norm_params:
            param_groups.append({
                'params': norm_params,
                'lr': new_lr,
                'weight_decay': 0.0  # Norm层不使用weight decay
            })
            self.logger.info(f"✓ Norm层参数组: {len(norm_params)} 个参数, lr={new_lr}, wd=0")
        
        # 3. 新参数组（MoE层、CaS_DETR层等新增结构，即使它们在encoder/decoder中）
        new_params = []
        new_names = []
        processed_params = set(id(p) for p in pretrained_params + norm_params)
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and id(param) not in processed_params:
                new_params.append(param)
                new_names.append(name)
        
        if new_params:
            param_groups.append({
                'params': new_params,
                'lr': new_lr,
                'weight_decay': weight_decay
            })
            self.logger.info(f"✓ 新参数组: {len(new_params)} 个参数, lr={new_lr}")
        
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999)
        )
        
        return optimizer
    
    def create_scheduler(self):
        """创建学习率调度器"""
        scheduler_type = self.config['training'].get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            eta_min = float(self.config['training'].get('eta_min', 1e-7))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=eta_min
            )
            self.logger.info(f"✓ 使用CosineAnnealingLR调度器 (eta_min={eta_min})")
        else:
            # MultiStepLR
            milestones = self.config['training'].get('milestones', [60, 80])
            gamma = float(self.config['training'].get('gamma', 0.1))
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=gamma
            )
            self.logger.info(f"✓ 使用MultiStepLR调度器 (milestones={milestones})")
        
        return scheduler
    
    def create_warmup_scheduler(self):
        """创建学习率预热调度器（与MoE RT-DETR保持一致）"""
        warmup_epochs = self.config['training'].get('warmup_epochs', 3)
        
        # 确保warmup_end_lr是浮点数
        warmup_end_lr = float(self.config['training']['new_lr'])
        warmup_scheduler = WarmupLR(
            optimizer=self.optimizer,
            warmup_epochs=warmup_epochs,
            warmup_start_lr=1e-7,
            warmup_end_lr=warmup_end_lr
        )
        
        self.logger.info(f"✓ 学习率预热: {warmup_epochs} 轮")
        return warmup_scheduler
    
    def _save_latest_checkpoint(self):
        """保存最新检查点用于断点续训（与moe-rtdetr一致）"""
        try:
            checkpoint = {
                'epoch': self.last_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'config': self.config,
                'best_loss': self.best_loss,
                'best_loss_epoch': getattr(self, 'best_loss_epoch', -1),
                'best_map': self.best_map,
                'best_map_epoch': getattr(self, 'best_map_epoch', -1),
                'best_mAP_50': getattr(self, 'best_mAP_50', 0.0),
                'best_mAP_50_epoch': getattr(self, 'best_mAP_50_epoch', -1),
                'best_mAP_075': getattr(self, 'best_mAP_075', 0.0),
                'best_mAP_075_epoch': getattr(self, 'best_mAP_075_epoch', -1),
                'global_step': self.global_step
            }
            
            # 添加可选组件状态
            if hasattr(self, 'warmup_scheduler') and self.warmup_scheduler:
                checkpoint['warmup_scheduler_state_dict'] = self.warmup_scheduler.state_dict()
            
            if hasattr(self, 'ema') and self.ema:
                checkpoint['ema_state_dict'] = self.ema.state_dict()
            
            if hasattr(self, 'scaler') and self.scaler:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
            if hasattr(self, 'visualizer') and self.visualizer:
                checkpoint['visualizer_state_dict'] = self.visualizer.state_dict()
            
            if hasattr(self, 'early_stopping') and self.early_stopping:
                checkpoint['early_stopping_state'] = self.early_stopping.state_dict()
            
            # 保存到 log_dir
            latest_path = self.log_dir / 'latest_checkpoint.pth'
            torch.save(checkpoint, latest_path)
            self.logger.info(f"💾 保存最新检查点: {latest_path}")
            
        except Exception as e:
            self.logger.warning(f"保存最新检查点失败: {e}")
    
    def _save_best_checkpoint(self, epoch):
        """保存最佳模型检查点（基于mAP）"""
        try:
            # 保存当前EMA模型的state_dict（用于推理时确保使用best_model的参数）
            best_ema_state = None
            if hasattr(self, 'ema') and self.ema:
                best_ema_state = self.ema.state_dict()
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'config': self.config,
                'best_loss': self.best_loss,
                'best_loss_epoch': getattr(self, 'best_loss_epoch', -1),
                'best_map': self.best_map,
                'best_map_epoch': getattr(self, 'best_map_epoch', epoch),
                'best_mAP_50': getattr(self, 'best_mAP_50', 0.0),
                'best_mAP_50_epoch': getattr(self, 'best_mAP_50_epoch', -1),
                'best_mAP_075': getattr(self, 'best_mAP_075', 0.0),
                'best_mAP_075_epoch': getattr(self, 'best_mAP_075_epoch', -1),
                'global_step': self.global_step
            }
            
            # 添加可选组件状态
            if hasattr(self, 'warmup_scheduler') and self.warmup_scheduler:
                checkpoint['warmup_scheduler_state_dict'] = self.warmup_scheduler.state_dict()
            
            if hasattr(self, 'ema') and self.ema:
                checkpoint['ema_state_dict'] = best_ema_state
            
            if hasattr(self, 'scaler') and self.scaler:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
            if hasattr(self, 'visualizer') and self.visualizer:
                checkpoint['visualizer_state_dict'] = self.visualizer.state_dict()
            
            if hasattr(self, 'early_stopping') and self.early_stopping:
                checkpoint['early_stopping_state'] = self.early_stopping.state_dict()
            
            # 保存到 log_dir
            best_path = self.log_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"💾 保存最佳模型: {best_path}")
            
        except Exception as e:
            self.logger.warning(f"保存最新检查点失败: {e}")
    
    def start_training(self, resume_checkpoint=None):
        """开始训练"""
        # 保存恢复检查点路径（用于日志设置）
        self._resume_checkpoint_path = resume_checkpoint
        
        # 重新设置日志（现在可以正确处理恢复训练的情况）
        self.setup_logging()

        # 按显存动态调整 batch_size（YAML 为基准；16G×3、32G×6）
        self._apply_vram_batch_size_rule()
        
        self.logger.info("=" * 80)
        self.logger.info("🚀 开始RT-DETR训练")
        self.logger.info("=" * 80)
        
        # 显示关键配置信息
        self.logger.info("📝 训练配置:")
        self.logger.info(f"  数据集路径: {self.config['data']['data_root']}")
        self.logger.info(f"  训练轮数: {self.config['training']['epochs']}")
        self.logger.info(f"  批次大小: {self.config['training']['batch_size']}")
        self.logger.info(f"  新组件学习率: {self.config['training']['new_lr']}")
        self.logger.info(f"  预训练组件学习率: {self.config['training']['pretrained_lr']}")
        self.logger.info(f"  输出目录: {self.log_dir}")
        pretrained_weights_display = self.pretrained_weights or self.config.get('model', {}).get('pretrained_weights', None)
        if pretrained_weights_display:
            self.logger.info(f"  预训练权重: {pretrained_weights_display}")
        if resume_checkpoint:
            self.logger.info(f"  恢复检查点: {resume_checkpoint}")
        self.logger.info("=" * 80)
        
        # 1. 创建模型
        self.model = self.create_model()
        
        # 2. 加载预训练权重（如果提供）
        pretrained_weights = self.pretrained_weights or self.config.get('model', {}).get('pretrained_weights', None)
        if pretrained_weights and not resume_checkpoint:
            self.logger.info(f"🔗 加载预训练权重: {pretrained_weights}")
            self.load_pretrained_weights(self.model, pretrained_weights)
        else:
            self.logger.info("ℹ️  使用随机初始化权重")
        
        # 将模型移到设备
        self.model = self.model.to(self.device)
        
        # 启用GPU优化设置
        if torch.cuda.is_available():
            # 启用cudnn benchmark以加速卷积操作（输入尺寸固定时）
            torch.backends.cudnn.benchmark = True
            # 启用TensorFloat-32（RTX 5090支持，可加速某些操作）
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.logger.info("✓ 已启用GPU优化: cudnn.benchmark=True, TF32=True")
        
        # 3. 创建其他组件
        self.criterion = self.create_criterion()
        self.train_dataloader, self.val_dataloader = self.create_datasets()
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        self.warmup_scheduler = self.create_warmup_scheduler()
        
        # 4. 创建EMA和梯度缩放器
        ema_decay = self.config['training'].get('ema_decay', 0.9999)
        self.ema = ModelEMA(self.model, decay=ema_decay)
        self.scaler = torch.amp.GradScaler('cuda')
        self.logger.info(f"✓ EMA decay={ema_decay}, 混合精度训练已启用")
        
        # 5. 创建可视化器（使用log_dir）
        self.visualizer = TrainingVisualizer(
            log_dir=self.log_dir,
            model_type='standard',
            experiment_name=self.experiment_name
        )
        
        # 5.5 创建推理后处理器
        self.postprocessor = DetDETRPostProcessor(
            num_classes=self.num_classes,
            use_focal_loss=True,
            num_top_queries=300,
            box_process_format=BoxProcessFormat.RESIZE
        )
        
        # 创建推理输出目录
        self.inference_output_dir = self.log_dir / 'inference_samples'
        self.inference_output_dir.mkdir(exist_ok=True)
        self.logger.info(f"✓ 推理样本输出目录: {self.inference_output_dir}")
        
        # 6. 设置训练属性
        self.last_epoch = -1
        self.best_loss = float('inf')
        self.best_loss_epoch = -1
        self.best_map = 0.0  # 记录最佳mAP
        self.best_map_epoch = -1
        self.best_mAP_50 = 0.0
        self.best_mAP_50_epoch = -1
        self.best_mAP_075 = 0.0
        self.best_mAP_075_epoch = -1
        self.global_step = 0  # 全局步数（与moe-rtdetr/cas_detr保持一致）
        
        # 6.5 初始化Early Stopping
        self.early_stopping = self._create_early_stopping()
        
        # 7. 设置梯度裁剪参数
        self.clip_max_norm = self.config['training'].get('clip_max_norm', 10.0)
        self.logger.info(f"✓ 梯度裁剪: max_norm={self.clip_max_norm}")
        
        # 8. 恢复训练（如果提供checkpoint）
        if resume_checkpoint:
            self.logger.info(f"📦 从检查点恢复训练: {resume_checkpoint}")
            checkpoint = torch.load(resume_checkpoint, map_location=self.device, weights_only=False)
            
            # 恢复模型和优化器状态
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # 恢复warmup调度器
            if 'warmup_scheduler_state_dict' in checkpoint and self.warmup_scheduler:
                self.warmup_scheduler.load_state_dict(checkpoint['warmup_scheduler_state_dict'])
            
            # 恢复EMA
            if 'ema_state_dict' in checkpoint and self.ema:
                self.ema.load_state_dict(checkpoint['ema_state_dict'])
            
            # 恢复可视化器历史记录
            if 'visualizer_state_dict' in checkpoint and self.visualizer:
                self.visualizer.load_state_dict(checkpoint['visualizer_state_dict'])
                self.logger.info(f"✓ 已恢复训练历史记录")
            
            # 恢复early stopping状态
            if 'early_stopping_state' in checkpoint and self.early_stopping:
                self.early_stopping.load_state_dict(checkpoint['early_stopping_state'])
                self.logger.info(f"✓ 已恢复Early Stopping状态")
            
            # 恢复epoch计数和最佳指标
            self.last_epoch = checkpoint.get('epoch', -1)
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            self.best_loss_epoch = checkpoint.get('best_loss_epoch', -1)
            self.best_map = checkpoint.get('best_map', 0.0)
            self.best_map_epoch = checkpoint.get('best_map_epoch', -1)
            self.best_mAP_50 = checkpoint.get('best_mAP_50', 0.0)
            self.best_mAP_50_epoch = checkpoint.get('best_mAP_50_epoch', -1)
            self.best_mAP_075 = checkpoint.get('best_mAP_075', 0.0)
            self.best_mAP_075_epoch = checkpoint.get('best_mAP_075_epoch', -1)
            self.global_step = checkpoint.get('global_step', 0)
            self.logger.info(f'✓ 从epoch {self.last_epoch + 1}恢复训练')
            
            # 显示恢复的训练信息
            if 'best_metric' in checkpoint:
                self.logger.info(f"✓ 历史最佳指标: {checkpoint['best_metric']}")
            self.logger.info(f"✓ 最佳loss: {self.best_loss:.4f}, 最佳mAP: {self.best_map:.4f}")
            if 'train_loss' in checkpoint:
                self.logger.info(f"✓ 上次训练损失: {checkpoint['train_loss']:.4f}")
            if 'val_loss' in checkpoint:
                self.logger.info(f"✓ 上次验证损失: {checkpoint['val_loss']:.4f}")
        
        # 9. 打印训练配置摘要
        self.logger.info("=" * 80)
        self.logger.info("训练配置摘要:")
        self.logger.info(f"  - 训练轮数: {self.config['training']['epochs']}")
        self.logger.info(f"  - 批次大小: {self.config['training']['batch_size']}")
        self.logger.info(f"  - 新组件学习率: {self.config['training']['new_lr']}")
        self.logger.info(f"  - 预训练组件学习率: {self.config['training']['pretrained_lr']}")
        self.logger.info(f"  - Weight decay: {self.config['training']['weight_decay']}")
        self.logger.info(f"  - Warmup轮数: {self.config['training'].get('warmup_epochs', 3)}")
        self.logger.info(f"  - 梯度裁剪: {self.clip_max_norm}")
        self.logger.info(f"  - 设备: {self.device}")
        self.logger.info("=" * 80)
        
        self._custom_training_loop()
        
        # 保存最终的 latest_checkpoint（用于断点续训）
        self._save_latest_checkpoint()
    
    def _create_early_stopping(self) -> Optional[EarlyStopping]:
        """创建Early Stopping。"""
        training_config = self.config.get('training', {})
        patience = training_config.get('early_stopping_patience', None)
        
        if patience is None or patience <= 0:
            self.logger.info("⏱️  Early Stopping: 未启用")
            return None
        
        metric_name = training_config.get('early_stopping_metric', 'mAP_0.5_0.95')
        mode = 'max' if 'mAP' in metric_name or 'AP' in metric_name else 'min'
        
        self.logger.info(f"⏱️  Early Stopping: 启用 (patience={patience}, metric={metric_name}, mode={mode})")
        
        return EarlyStopping(patience=patience, metric_name=metric_name, logger=self.logger)

    def _early_stopping_metric_improved(self, val_metrics: Dict[str, float]) -> bool:
        """与 is_best_* 同一套比较规则，在更新 best_* 之前调用。"""
        name = self.config['training'].get('early_stopping_metric', 'mAP_0.5_0.95')
        if 'loss' in name.lower():
            return val_metrics.get('total_loss', float('inf')) < self.best_loss
        if 'mAP_0.75' in name:
            return val_metrics.get('mAP_0.75', 0.0) > self.best_mAP_075
        if 'mAP_0.5_0.95' in name or 'mAP_0.5:0.95' in name:
            return val_metrics.get('mAP_0.5_0.95', 0.0) > self.best_map
        if 'mAP_0.5' in name:
            return val_metrics.get('mAP_0.5', 0.0) > self.best_mAP_50
        return val_metrics.get('mAP_0.5_0.95', 0.0) > self.best_map

    def _early_stopping_best_snapshot(self) -> Tuple[float, int]:
        """更新 best_* 之后，用于早停日志的「当前监控指标最佳值 / epoch」。"""
        name = self.config['training'].get('early_stopping_metric', 'mAP_0.5_0.95')
        if 'loss' in name.lower():
            return self.best_loss, self.best_loss_epoch
        if 'mAP_0.75' in name:
            return self.best_mAP_075, self.best_mAP_075_epoch
        if 'mAP_0.5_0.95' in name or 'mAP_0.5:0.95' in name:
            return self.best_map, self.best_map_epoch
        if 'mAP_0.5' in name:
            return self.best_mAP_50, self.best_mAP_50_epoch
        return self.best_map, self.best_map_epoch

    def _custom_training_loop(self):
        """自定义训练循环"""
        epochs = self.config['training']['epochs']
        self.logger.info(f"开始训练 {epochs} epochs")
        eval_sched = self.config.get("training", {}).get("eval_schedule")
        self.logger.info(f"✓ 验证策略: {describe_eval_schedule(eval_sched)}")

        for epoch in range(self.last_epoch + 1, epochs):
            self.last_epoch = epoch
            
            # 更新训练集 epoch
            self.train_dataloader.set_epoch(epoch)
            if hasattr(self.train_dataloader.collate_fn, 'set_epoch'):
                self.train_dataloader.collate_fn.set_epoch(epoch)

            # 训练一个epoch
            train_metrics = self._train_epoch()
            
            should_validate = should_run_validation(epoch, eval_sched)

            if should_validate:
                val_metrics = self._validate_epoch()
            else:
                val_metrics = {}
            
            # 学习率调度（与moe-rtdetr/cas_detr保持一致）
            if self.last_epoch < self.warmup_scheduler.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                self.scheduler.step()
            
            # 输出日志（不输出mAP，只在best_model时输出）
            self.logger.info(f"Epoch {epoch}:")
            if should_validate:
                current_map = val_metrics.get('mAP_0.5_0.95', 0.0)
                current_map_50 = val_metrics.get('mAP_0.5', 0.0)
                self.logger.info(f"  训练损失: {train_metrics.get('total_loss', 0.0):.2f} | 验证损失: {val_metrics.get('total_loss', 0.0):.2f}")
                self.logger.info(f"  📊 当前mAP: {current_map:.4f} (mAP@50: {current_map_50:.4f})")
            else:
                self.logger.info(f"  训练损失: {train_metrics.get('total_loss', 0.0):.2f} | 验证损失: Skipped")
            
            # 记录到可视化器
            current_lr = self.optimizer.param_groups[0]['lr']
            self.visualizer.record(
                epoch=epoch,
                train_loss=train_metrics.get('total_loss', 0.0),
                val_loss=val_metrics.get('total_loss', 0.0),
                mAP_0_5=val_metrics.get('mAP_0.5', 0.0),
                mAP_0_75=val_metrics.get('mAP_0.75', 0.0),
                mAP_0_5_0_95=val_metrics.get('mAP_0.5_0.95', 0.0),
                learning_rate=current_lr,
                ap_easy=0.0,
                ap_moderate=0.0,
                ap_hard=0.0
            )
            
            # 验证时才更新 best_*；早停只计数，「是否改善」与上面比较规则一致
            if should_validate:
                es_improved = self._early_stopping_metric_improved(val_metrics)
                vl = val_metrics.get('total_loss', float('inf'))
                v5095 = val_metrics.get('mAP_0.5_0.95', 0.0)
                v50 = val_metrics.get('mAP_0.5', 0.0)
                v75 = val_metrics.get('mAP_0.75', 0.0)
                is_best_loss = vl < self.best_loss
                is_best_map = v5095 > self.best_map
                is_best_map50 = v50 > self.best_mAP_50
                is_best_map75 = v75 > self.best_mAP_075

                if is_best_loss:
                    self.best_loss = vl
                    self.best_loss_epoch = epoch
                    self.logger.info(f"  🎉 新的最佳验证损失: {self.best_loss:.2f}")

                if is_best_map:
                    self.best_map = v5095
                    self.best_map_epoch = epoch
                    self.logger.info(f"  🎉 新的最佳mAP: {self.best_map:.4f}")
                    self._save_best_checkpoint(epoch)

                if is_best_map50:
                    self.best_mAP_50 = v50
                    self.best_mAP_50_epoch = epoch
                if is_best_map75:
                    self.best_mAP_075 = v75
                    self.best_mAP_075_epoch = epoch

                if self.early_stopping:
                    bv, be = self._early_stopping_best_snapshot()
                    if self.early_stopping.step(es_improved, best_value=bv, best_epoch=be):
                        self.logger.info(f"Early Stopping在epoch {epoch}触发，停止训练")
                        break
            
            if epoch % 5 == 0 or epoch == self.config['training']['epochs'] - 1:
                self._save_latest_checkpoint()
                try:
                    self.visualizer.plot()
                except Exception as e:
                    self.logger.warning(f"绘制训练曲线失败: {e}")
        
        self.logger.info("✓ 训练完成！")
        try:
            self.visualizer.plot()
            self.visualizer.export_to_csv()
        except Exception as e:
            self.logger.warning(f"绘制训练曲线失败: {e}")

        self._evaluate_best_model_and_print_all_ap()
        try:
            best_model_path = self.log_dir / 'best_model.pth'
            latest_checkpoint_path = self.log_dir / 'latest_checkpoint.pth'
            checkpoint_path = best_model_path if best_model_path.exists() else latest_checkpoint_path
            if checkpoint_path.exists():
                if checkpoint_path == latest_checkpoint_path:
                    self.logger.warning("未找到best_model.pth，改用latest_checkpoint.pth进行推理")
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                best_ema_state = checkpoint.get('ema_state_dict', None)
                best_epoch = checkpoint.get('epoch', None)
                self._run_inference_on_best_model(best_ema_state, best_epoch=best_epoch)
            else:
                self.logger.warning("未找到best_model.pth和latest_checkpoint.pth，跳过推理")
        except Exception as e:
            self.logger.warning(f"训练结束时推理失败（不影响训练结果）: {e}")

    def _build_test_dataloader_optional(self):
        """若存在 test 数据则构建 DataLoader，否则 None（与 YOLO 训练后 test 评估一致）。"""
        if not self.config.get("data", {}).get("eval_test_after_training", True):
            return None
        from src.data.dataloader import BatchImageCollateFuncion

        data_root = self.config["data"]["data_root"]
        ds_cls = self._resolve_dataset_class()
        augmentation_config = self.config.get("augmentation", {})
        try:
            test_dataset = ds_cls(
                data_root=data_root,
                split="test",
                augmentation_config=augmentation_config,
            )
        except FileNotFoundError:
            return None
        if len(test_dataset) == 0:
            return None

        scales = self.config.get("augmentation", {}).get(
            "scales", [576, 608, 640, 640, 640, 672, 704]
        )
        stop_epoch = self.config.get("augmentation", {}).get("stop_epoch", 71)
        # Keep test-time evaluation deterministic at the dataset's fixed resize target.
        collate_fn = BatchImageCollateFuncion(scales=None, stop_epoch=stop_epoch)
        num_workers = self.config.get("misc", {}).get("num_workers", 16)
        pin_memory = self.config.get("misc", {}).get("pin_memory", True)
        prefetch_factor = self.config.get("misc", {}).get("prefetch_factor", 2)
        return DataLoader(
            test_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )

    def _run_ema_eval_on_dataloader(self, dataloader, split_label: str,
                                     best_epoch=None, bench_dict=None):
        """使用当前 EMA（已加载 best 权重）在给定 loader 上算 mAP。"""
        all_predictions = []
        all_targets = []
        current_h, current_w = 640, 640

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                _, _, current_h, current_w = images.shape
                images = images.to(self.device, non_blocking=True)
                targets = [
                    {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                    for t in targets
                ]

                outputs = self.ema.module(images, targets)
                has_predictions = ("pred_logits" in outputs and "pred_boxes" in outputs) or (
                    "class_scores" in outputs and "bboxes" in outputs
                )
                if has_predictions:
                    self._collect_predictions(
                        outputs, targets, batch_idx, all_predictions, all_targets, current_w, current_h
                    )

        if len(all_predictions) == 0 or len(all_targets) == 0:
            self.logger.warning(f"best_model 在 {split_label} 上无有效预测或标注，跳过 AP")
            return

        metrics = self._compute_map_metrics(
            all_predictions,
            all_targets,
            img_h=current_h,
            img_w=current_w,
            print_per_category=True,
            compute_difficulty=True,
        )

        log_detr_eval_summary(self.logger, split_label, metrics, bench_dict)
        csv_path = write_detr_eval_csv(
            self.log_dir, self.config, self.experiment_name,
            split_label, metrics, self.class_names, bench_dict,
        )
        self.logger.info(f"✓ best_model [{split_label}] 评估完成 (epoch={best_epoch})  → {csv_path}")

    def _evaluate_best_model_and_print_all_ap(self):
        """训练结束后在 val 上评估；若存在 test 划分与标注则再评一次。"""
        evaluate_best_model_after_training(
            log_dir=self.log_dir,
            device=self.device,
            config=self.config,
            experiment_name=self.experiment_name,
            logger=self.logger,
            ema=self.ema,
            val_loader=self.val_dataloader,
            build_test_loader_fn=self._build_test_dataloader_optional,
            run_eval_fn=self._run_ema_eval_on_dataloader,
        )

    def _run_inference_on_best_model(self, best_ema_state=None, best_epoch=None):
        """使用best_model运行推理，输出5张验证图像的推理结果
        
        Args:
            best_ema_state: best_model的EMA模型state_dict，如果提供则使用它进行推理
            best_epoch: best_model保存时的epoch，用于文件名
        """
        try:
            # 保存当前EMA模型状态（推理后恢复）
            original_ema_state = None
            if best_ema_state is not None and hasattr(self, 'ema') and self.ema:
                original_ema_state = self.ema.state_dict()
                # 加载best_model的EMA参数
                self.ema.load_state_dict(best_ema_state)
            
            # 从验证数据加载器中获取一个batch用于推理
            inference_images, inference_targets = next(iter(self.val_dataloader))
            inference_images = inference_images.to(self.device)
            inference_targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                 for k, v in t.items()} for t in inference_targets]
            
            # 打印前5张推理结果
            batch_size = len(inference_targets)
            num_inference_images = min(5, batch_size)
            # 使用best_epoch作为文件名，如果没有提供则使用last_epoch（向后兼容）
            epoch_for_filename = best_epoch if best_epoch is not None else self.last_epoch
            self.logger.info(f"  生成best_model推理结果（前{num_inference_images}张，epoch={epoch_for_filename}）...")
            
            for img_idx in range(num_inference_images):
                self._inference_single_image_from_batch(
                    inference_images, inference_targets, 0, image_idx=img_idx, 
                    suffix=f"best_model_epoch_{epoch_for_filename}"
                )
            
            self.logger.info(f"  ✓ 推理结果已保存到: {self.inference_output_dir}")
            
            # 恢复原始EMA模型状态
            if original_ema_state is not None and hasattr(self, 'ema') and self.ema:
                self.ema.load_state_dict(original_ema_state)
                
        except Exception as e:
            # 如果推理失败，不影响训练，但尝试恢复EMA状态
            if hasattr(self, 'logger'):
                self.logger.warning(f"best_model推理失败（不影响训练）: {e}")
            if original_ema_state is not None and hasattr(self, 'ema') and self.ema:
                try:
                    self.ema.load_state_dict(original_ema_state)
                except:
                    pass
    
    def _inference_single_image_from_batch(self, images, targets, batch_idx, image_idx=0, suffix=None):
        """从batch中选择一张图片进行推理并保存结果（直接复用batch_inference.py的逻辑）
        
        Args:
            images: 图像tensor
            targets: 目标列表
            batch_idx: batch索引
            image_idx: 图像在batch中的索引
            suffix: 文件名后缀（默认使用epoch，如"epoch_0"或"best_model"）
        """
        try:
            # 使用EMA模型进行推理
            self.ema.module.eval()
            
            # 选择batch中的第一张图片（或指定索引）
            single_image = images[image_idx:image_idx+1]  # [1, 3, H, W]
            single_target = targets[image_idx] if image_idx < len(targets) else None
            
            if single_target is None:
                return
            
            # 获取image_id用于命名和查找原始图像
            image_id = single_target['image_id'].item() if 'image_id' in single_target else batch_idx
            
            # 获取原始图像路径
            ds = self.val_dataloader.dataset
            orig_image_path = None
            if hasattr(ds, "get_image_path"):
                orig_image_path = ds.get_image_path(int(image_id))
            if orig_image_path is None or not orig_image_path.exists():
                data_root = Path(self.config["data"]["data_root"])
                orig_image_path = data_root / "image" / f"{int(image_id):06d}.jpg"
            if not orig_image_path.exists():
                return
            
            # 使用batch_inference.py中的函数进行推理
            if USE_BATCH_INFERENCE_LOGIC:
                result_image = inference_from_preprocessed_image(
                    single_image,
                    self.ema.module,
                    self.postprocessor,
                    orig_image_path,
                    conf_threshold=0.3,
                    target_size=640,
                    device=str(self.device),
                    class_names=self.class_names,
                    colors=self.colors,
                    verbose=False
                )
                
                if result_image is None:
                    self.ema.module.train()
                    return
                
                # 保存结果：图片名_suffix.jpg
                image_name = orig_image_path.stem
                if suffix is None:
                    suffix = f"epoch_{self.last_epoch}"
                output_filename = f"{image_name}_{suffix}.jpg"
                output_path = self.inference_output_dir / output_filename
                cv2.imwrite(str(output_path), result_image)
            else:
                # 备用逻辑
                with torch.no_grad():
                    outputs = self.ema.module(single_image)
                
                # 根据 box_revert 文档，orig_sizes 应该是 (w, h)
                _, _, h, w = single_image.shape
                eval_sizes = torch.tensor([[w, h]], device=self.device)
                
                results = self.postprocessor(outputs, orig_sizes=eval_sizes)
                
                if len(results) > 0:
                    result = results[0]
                    labels = result['labels'].cpu().numpy()
                    boxes = result['boxes'].cpu().numpy()
                    scores = result['scores'].cpu().numpy()
                    
                    mask = scores >= 0.3
                    labels = labels[mask]
                    boxes = boxes[mask]
                    scores = scores[mask]
                    
                    if len(labels) > 0:
                        orig_image = cv2.imread(str(orig_image_path))
                        if orig_image is not None:
                            result_image = draw_boxes(
                                orig_image.copy(), labels, boxes, scores,
                                class_names=self.class_names,
                                colors=self.colors
                            )
                            image_name = orig_image_path.stem
                            if suffix is None:
                                suffix = f"epoch_{self.last_epoch}"
                            output_filename = f"{image_name}_{suffix}.jpg"
                            output_path = self.inference_output_dir / output_filename
                            cv2.imwrite(str(output_path), result_image)
            
            # 恢复训练模式
            self.ema.module.train()
            
        except Exception as e:
            # 如果推理失败，不影响训练
            if hasattr(self, 'logger'):
                self.logger.debug(f"推理失败（不影响训练）: {e}")
            if hasattr(self, 'ema') and hasattr(self.ema, 'module'):
                self.ema.module.train()
    
    def _train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = torch.tensor(0.0, device=self.device)
        detection_loss = torch.tensor(0.0, device=self.device)
        
        for batch_idx, (images, targets) in enumerate(self.train_dataloader):
            images = images.to(self.device, non_blocking=True)
            targets = [{k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            self.optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = self.model(images, targets)
                # 使用criterion计算损失
                loss_dict = self.criterion(outputs, targets)
                loss = sum(loss_dict.values())
            
            # 反向传播
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_max_norm)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.ema.update(self.model)
            
            # detach 后累加，避免每步 GPU→CPU 同步
            total_loss += loss.detach()
            det_components = sum(
                loss_dict[k].detach() for k in ('loss_vfl', 'loss_bbox', 'loss_giou') if k in loss_dict
            )
            detection_loss += det_components
            
            if batch_idx % 100 == 0:
                self.logger.info(
                    f'Epoch {self.last_epoch} | Batch {batch_idx} | '
                    f'Loss: {loss.item():.2f} (Det: {det_components.item():.2f})'
                )
            
            self.global_step += 1
        
        num_batches = len(self.train_dataloader)
        return {
            'total_loss': (total_loss / num_batches).item(),
            'detection_loss': (detection_loss / num_batches).item(),
        }
    
    def _validate_epoch(self):
        """验证模型并计算mAP"""
        self.ema.module.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        # 初始化默认尺寸 (防止 val_loader 为空)
        current_h, current_w = 640, 640
        
        # 验证逻辑
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(self.val_dataloader):
                B, C, H_tensor, W_tensor = images.shape
                current_h, current_w = H_tensor, W_tensor

                images = images.to(self.device, non_blocking=True)
                targets = [{k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                           for k, v in t.items()} for t in targets]
                
                outputs = self.ema.module(images, targets)
                
                # 计算损失（兼容两种方式：模型内部计算或外部计算）
                if isinstance(outputs, dict) and 'total_loss' in outputs:
                    # 模型内部已计算损失（与moe-rtdetr/cas_detr保持一致）
                    loss = outputs['total_loss']
                    total_loss += loss.item()
                else:
                    # 使用criterion计算损失（rt-detr标准方式）
                    loss_dict = self.criterion(outputs, targets)
                    loss = sum(loss_dict.values())
                    total_loss += loss.item()
                
                # 收集预测结果（只在需要计算mAP时收集，前30个epoch跳过）
                # 兼容两种输出格式：pred_logits/pred_boxes 或 class_scores/bboxes
                has_predictions = (
                    ('pred_logits' in outputs and 'pred_boxes' in outputs) or
                    ('class_scores' in outputs and 'bboxes' in outputs)
                )
                if has_predictions:
                    self._collect_predictions(outputs, targets, batch_idx, all_predictions, all_targets, W_tensor, H_tensor)
        
        # 保存预测结果用于后续打印每个类别mAP（避免重复计算）
        self._last_val_predictions = all_predictions
        self._last_val_targets = all_targets
        
        avg_loss = total_loss / len(self.val_dataloader)
        
        # 计算mAP（不计算每个类别的mAP，只在best_model时计算）
        mAP_metrics = self._compute_map_metrics(all_predictions, all_targets, 
                                              img_h=current_h, img_w=current_w,
                                              print_per_category=False)
        
        return {
            'total_loss': avg_loss,
            'mAP_0.5': mAP_metrics.get('mAP_0.5', 0.0),
            'mAP_0.75': mAP_metrics.get('mAP_0.75', 0.0),
            'mAP_0.5_0.95': mAP_metrics.get('mAP_0.5_0.95', 0.0)
        }
    
    @staticmethod
    def _cxcywh_to_xywh_orig(boxes: torch.Tensor, img_w: int, img_h: int,
                              orig_w: float, orig_h: float) -> np.ndarray:
        """cxcywh (归一化或像素) → COCO xywh in original image coords. Returns numpy."""
        scale = min(img_w / orig_w, img_h / orig_h)
        normalized = boxes.max() <= 1.01
        if normalized:
            cx = boxes[:, 0] * img_w
            cy = boxes[:, 1] * img_h
            bw = boxes[:, 2] * img_w
            bh = boxes[:, 3] * img_h
        else:
            cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        out = torch.stack([
            ((cx - bw / 2) / scale).clamp(0, orig_w),
            ((cy - bh / 2) / scale).clamp(0, orig_h),
            (bw / scale).clamp(1, orig_w),
            (bh / scale).clamp(1, orig_h),
        ], dim=1)
        return out.cpu().numpy()

    def _collect_predictions(self, outputs: Dict, targets: List[Dict], batch_idx: int,
                            all_predictions: List, all_targets: List, img_w: int, img_h: int) -> None:
        """向量化收集预测/GT，单次 .cpu().numpy() 替代逐元素 .item()。"""
        if 'pred_logits' in outputs:
            pred_logits = outputs['pred_logits']
            pred_boxes = outputs['pred_boxes']
        elif 'class_scores' in outputs:
            pred_logits = outputs['class_scores']
            pred_boxes = outputs['bboxes']
        else:
            return

        bs = self.config['training']['batch_size']
        batch_size = pred_logits.shape[0]

        for i in range(batch_size):
            image_id = batch_idx * bs + i
            orig_h, orig_w = targets[i]['orig_size'].tolist()

            # ── predictions ──
            scores_sig = torch.sigmoid(pred_logits[i])
            max_scores, pred_cls = scores_sig.max(dim=-1)
            valid = ~torch.all(pred_boxes[i] == 1.0, dim=1)
            if valid.any():
                fb = pred_boxes[i][valid]
                fc = pred_cls[valid]
                fs = max_scores[valid]
                boxes_np = self._cxcywh_to_xywh_orig(fb, img_w, img_h, orig_w, orig_h)
                cls_np = (fc.cpu().numpy() + 1).tolist()
                scores_np = fs.cpu().numpy().tolist()
                all_predictions.extend(
                    {'image_id': image_id, 'category_id': c, 'bbox': boxes_np[j].tolist(), 'score': s}
                    for j, (c, s) in enumerate(zip(cls_np, scores_np))
                )

            # ── ground truth ──
            if i >= len(targets) or 'labels' not in targets[i] or 'boxes' not in targets[i]:
                continue
            true_labels = targets[i]['labels']
            true_boxes = targets[i]['boxes']
            if len(true_labels) == 0:
                continue

            gt_boxes_np = self._cxcywh_to_xywh_orig(true_boxes, img_w, img_h, orig_w, orig_h)
            gt_cls_np = (true_labels.cpu().numpy() + 1)
            gt_area = gt_boxes_np[:, 2] * gt_boxes_np[:, 3]
            gt_h = gt_boxes_np[:, 3]

            has_iscrowd = 'iscrowd' in targets[i]
            has_occ = 'occluded_state' in targets[i]
            has_trunc = 'truncated_state' in targets[i]
            iscrowd_np = targets[i]['iscrowd'].cpu().numpy() if has_iscrowd else None
            occ_np = targets[i]['occluded_state'].cpu().numpy() if has_occ else None
            trunc_np = targets[i]['truncated_state'].cpu().numpy() if has_trunc else None

            for j in range(len(gt_cls_np)):
                ann = {
                    'image_id': image_id,
                    'category_id': int(gt_cls_np[j]),
                    'bbox': gt_boxes_np[j].tolist(),
                    'area': float(gt_area[j]),
                    'bbox_height': float(gt_h[j]),
                }
                if iscrowd_np is not None:
                    ann['iscrowd'] = int(iscrowd_np[j])
                if occ_np is not None:
                    ann['occluded_state'] = float(occ_np[j])
                if trunc_np is not None:
                    ann['truncated_state'] = float(trunc_np[j])
                all_targets.append(ann)

    def _get_kitti_difficulty(self, target: Dict) -> str:
        """与 YOLO 侧 ``categorize_by_kitti_difficulty`` + DAIR 截断映射一致。"""
        return kitti_difficulty_from_coco_ann(
            target,
            dair_categorical_trunc=self._dair_truncation_categorical,
        )

    def _run_coco_eval(self, predictions: List[Dict], targets: List[Dict], categories: List[Dict],
                       img_h: int, img_w: int, print_summary: bool = False):
        """执行一次 COCOeval，返回 coco_eval 对象；无目标时返回 None。"""
        if len(targets) == 0:
            return None

        coco_gt = {
            'images': [],
            'annotations': [],
            'categories': categories,
            'info': {
                'description': 'DAIR-V2X Dataset',
                'version': '1.0',
                'year': 2024
            }
        }

        image_ids = set(target['image_id'] for target in targets)
        for img_id in image_ids:
            coco_gt['images'].append({'id': img_id, 'width': img_w, 'height': img_h})

        for i, target in enumerate(targets):
            ann = target.copy()
            ann['id'] = i + 1
            coco_gt['annotations'].append(ann)

        from io import StringIO
        import sys

        coco_gt_obj = COCO()
        coco_gt_obj.dataset = coco_gt

        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            coco_gt_obj.createIndex()
            # 过滤 predictions，只保留当前 targets 中存在的 image_id，减少 loadRes 匹配压力
            target_img_ids = set(coco_gt_obj.imgs.keys())
            filtered_predictions = [p for p in predictions if p['image_id'] in target_img_ids]
            
            if not filtered_predictions:
                return None
                
            coco_dt = coco_gt_obj.loadRes(filtered_predictions)
            coco_eval = COCOeval(coco_gt_obj, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
        except Exception as e:
            return None
        finally:
            sys.stdout = old_stdout

        if print_summary:
            coco_eval.summarize()
        else:
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                coco_eval.summarize()
            finally:
                sys.stdout = old_stdout

        return coco_eval

    def _compute_difficulty_aps(self, predictions: List[Dict], targets: List[Dict], categories: List[Dict],
                                img_h: int, img_w: int) -> Dict[str, float]:
        """计算 KITTI 风格 AP_easy / AP_moderate / AP_hard。"""
        import copy
        easy_targets = []
        moderate_targets = []
        hard_targets = []

        for target in targets:
            level = self._get_kitti_difficulty(target)
            
            # 独立模式：每档只评该档 GT，其余难度作 iscrowd=1（不计 FP）
            t_easy = copy.deepcopy(target)
            if level != 'easy':
                t_easy['iscrowd'] = 1
            easy_targets.append(t_easy)

            t_mod = copy.deepcopy(target)
            if level != 'moderate':
                t_mod['iscrowd'] = 1
            moderate_targets.append(t_mod)

            t_hard = copy.deepcopy(target)
            if level != 'hard':
                t_hard['iscrowd'] = 1
            hard_targets.append(t_hard)

        easy_eval = self._run_coco_eval(predictions, easy_targets, categories, img_h, img_w, print_summary=False)
        moderate_eval = self._run_coco_eval(predictions, moderate_targets, categories, img_h, img_w, print_summary=False)
        hard_eval = self._run_coco_eval(predictions, hard_targets, categories, img_h, img_w, print_summary=False)

        # AP@IoU=0.50（stats[1]），与 YOLO 训练后 KITTI 难度 mAP@0.5 对齐；勿用 stats[0]（0.5:0.95）
        return {
            "AP_easy": coco_ap_at_iou50_all(easy_eval),
            "AP_moderate": coco_ap_at_iou50_all(moderate_eval),
            "AP_hard": coco_ap_at_iou50_all(hard_eval),
        }
    
    def _extract_per_category_ap_from_eval(
        self, coco_eval, categories: List[Dict]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """从一次 COCOeval 的 precision[T,R,K,A,M] 提取各类别 AP@50 与 AP@0.5:0.95（T=0 为 IoU=0.5）。"""
        per_cat_50 = {cat["name"]: 0.0 for cat in categories}
        per_cat_5095 = {cat["name"]: 0.0 for cat in categories}
        if coco_eval is None or not hasattr(coco_eval, "eval") or "precision" not in coco_eval.eval:
            return per_cat_50, per_cat_5095

        try:
            precision = coco_eval.eval["precision"]
            area_index = 0
            max_det_index = len(coco_eval.params.maxDets) - 1
            cat_id_to_index = {cat_id: idx for idx, cat_id in enumerate(coco_eval.params.catIds)}

            for cat in categories:
                cat_id = cat["id"]
                cat_name = cat["name"]
                if cat_id not in cat_id_to_index:
                    continue

                cat_index = cat_id_to_index[cat_id]
                # AP@0.5: 仅 IoU 第 0 档（0.50），对 recall 维取均值
                p50 = precision[0, :, cat_index, area_index, max_det_index]
                v50 = p50[p50 > -1]
                per_cat_50[cat_name] = float(np.mean(v50)) if v50.size > 0 else 0.0
                # AP@0.5:0.95: 对所有 IoU 档与 recall 维取均值（与 COCO 各类 AP 一致）
                p5095 = precision[:, :, cat_index, area_index, max_det_index]
                v5095 = p5095[p5095 > -1]
                per_cat_5095[cat_name] = float(np.mean(v5095)) if v5095.size > 0 else 0.0
        except Exception as e:
            self.logger.debug(f"从COCOeval提取每类AP失败: {e}")

        return per_cat_50, per_cat_5095
    
    def _compute_map_metrics(self, predictions: List[Dict], targets: List[Dict], 
                             img_h: int = 640, img_w: int = 640,
                             print_per_category: bool = False,
                             compute_difficulty: bool = False) -> Dict[str, float]:
        """计算mAP指标。
        
        Args:
            predictions: 预测结果列表
            targets: 真实标签列表
            img_h: 图像高度 (Tensor Shape)
            img_w: 图像宽度 (Tensor Shape)
            print_per_category: 是否打印每个类别的详细mAP（默认False，只在best_model时打印）
        """
        try:
            if len(predictions) == 0:
                return {
                    'mAP_0.5': 0.0,
                    'mAP_0.75': 0.0,
                    'mAP_0.5_0.95': 0.0,
                    'AP_small': 0.0,
                    'AP_medium': 0.0,
                    'AP_large': 0.0,
                    'AP_small_50': 0.0,
                    'AP_medium_50': 0.0,
                    'AP_large_50': 0.0,
                    'AP_easy': 0.0,
                    'AP_moderate': 0.0,
                    'AP_hard': 0.0
                }
            
            # 获取类别信息
            if hasattr(self, 'val_dataloader') and hasattr(self.val_dataloader.dataset, 'get_categories'):
                categories = self.val_dataloader.dataset.get_categories()
            else:
                categories = [
                    {'id': 1, 'name': 'Car'},
                    {'id': 2, 'name': 'Truck'},
                    {'id': 3, 'name': 'Van'},
                    {'id': 4, 'name': 'Bus'},
                    {'id': 5, 'name': 'Pedestrian'},
                    {'id': 6, 'name': 'Cyclist'},
                    {'id': 7, 'name': 'Motorcyclist'},
                    {'id': 8, 'name': 'Trafficcone'}
                ]
            
            coco_eval = self._run_coco_eval(
                predictions, targets, categories, img_h, img_w,
                print_summary=print_per_category
            )
            if coco_eval is None:
                return {
                    'mAP_0.5': 0.0,
                    'mAP_0.75': 0.0,
                    'mAP_0.5_0.95': 0.0,
                    'AP_small': 0.0,
                    'AP_medium': 0.0,
                    'AP_large': 0.0,
                    'AP_small_50': 0.0,
                    'AP_medium_50': 0.0,
                    'AP_large_50': 0.0,
                    'AP_easy': 0.0,
                    'AP_moderate': 0.0,
                    'AP_hard': 0.0
                }
            
            # 每类 AP@50 / AP@0.5:0.95 从同一次 COCOeval 的 precision 张量提取
            per_cat_50: Dict[str, float] = {}
            per_cat_5095: Dict[str, float] = {}
            if print_per_category:
                per_cat_50, per_cat_5095 = self._extract_per_category_ap_from_eval(
                    coco_eval, categories
                )
            
            difficulty_metrics = {
                'AP_easy': 0.0,
                'AP_moderate': 0.0,
                'AP_hard': 0.0,
            }
            if compute_difficulty:
                difficulty_metrics = self._compute_difficulty_aps(predictions, targets, categories, img_h, img_w)
            
            s50, m50, l50 = coco_area_ap_at_iou50(coco_eval)
            result = {
                'mAP_0.5': coco_eval.stats[1] if len(coco_eval.stats) > 1 else 0.0,
                'mAP_0.75': coco_eval.stats[2] if len(coco_eval.stats) > 2 else 0.0,
                'mAP_0.5_0.95': coco_eval.stats[0] if len(coco_eval.stats) > 0 else 0.0,
                'AP_small': coco_eval.stats[3] if len(coco_eval.stats) > 3 else 0.0,
                'AP_medium': coco_eval.stats[4] if len(coco_eval.stats) > 4 else 0.0,
                'AP_large': coco_eval.stats[5] if len(coco_eval.stats) > 5 else 0.0,
                'AP_small_50': s50,
                'AP_medium_50': m50,
                'AP_large_50': l50,
                'AP_easy': difficulty_metrics['AP_easy'],
                'AP_moderate': difficulty_metrics['AP_moderate'],
                'AP_hard': difficulty_metrics['AP_hard'],
            }
            
            for cat_name in per_cat_5095.keys():
                result[f"AP50_{cat_name}"] = per_cat_50.get(cat_name, 0.0)
                result[f"AP5095_{cat_name}"] = per_cat_5095[cat_name]
            
            return result
            
        except Exception as e:
            self.logger.warning(f"mAP计算失败: {e}")
            return {
                'mAP_0.5': 0.0,
                'mAP_0.75': 0.0,
                'mAP_0.5_0.95': 0.0,
                'AP_small': 0.0,
                'AP_medium': 0.0,
                'AP_large': 0.0,
                'AP_small_50': 0.0,
                'AP_medium_50': 0.0,
                'AP_large_50': 0.0,
                'AP_easy': 0.0,
                'AP_moderate': 0.0,
                'AP_hard': 0.0
            }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RT-DETR训练脚本')
    parser.add_argument('--backbone', type=str, default='presnet50', 
                       choices=['presnet18', 'presnet34', 'presnet50', 'presnet101',
                               'hgnetv2_l', 'hgnetv2_x', 'hgnetv2_h',
                               'cspresnet_s', 'cspresnet_m', 'cspresnet_l', 'cspresnet_x',
                               'cspdarknet', 'mresnet'],
                       help='Backbone类型')
    parser.add_argument('--data_root', type=str, default='/root/autodl-fs/datasets/DAIR-V2X', 
                       help='DAIR-V2X数据集路径')
                       
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--pretrained_lr', type=float, default=1e-5, help='预训练组件学习率')
    parser.add_argument('--new_lr', type=float, default=1e-4, help='新组件学习率')
    parser.add_argument('--warmup_epochs', type=int, default=3, 
                       help='学习率预热轮数')
    parser.add_argument('--pretrained_weights', type=str, default=None,
                       help='预训练权重路径（RT-DETR COCO预训练模型）')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='从检查点恢复训练（检查点文件路径）')
    parser.add_argument('--config', type=str, default=None,
                       help='YAML配置文件路径')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子，用于确保实验可重复性（默认：42）')
    parser.add_argument('--deterministic', action='store_true',
                       help='使用确定性算法（会降低速度但保证完全可重复）')
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='数据集键名或别名（与 yolo/configs/datasets.yaml 中一致，可设置 coco_data_root / detr_dataset_class）',
    )
    parser.add_argument(
        '--dataset_registry',
        type=str,
        default=str(default_detr_registry_path()),
        help='数据集注册表 YAML 路径（默认与 YOLO 共用）',
    )
    
    args = parser.parse_args()
    
    # 设置随机种子（必须在所有操作之前）
    print("\n" + "="*60)
    print("🔧 初始化训练环境")
    print("="*60)
    set_seed(args.seed, deterministic=args.deterministic)
    
    # 加载配置
    if args.config and args.config.endswith('.yaml'):
        # 从YAML文件加载配置
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"📄 从配置文件加载: {args.config}")
        
        # 确保学习率相关值是浮点数（YAML中的科学计数法可能被解析为字符串）
        # 直接使用配置文件中的字段名：pretrained_lr, new_lr
        if 'training' in config:
            # 类型转换确保是浮点数
            if 'pretrained_lr' in config['training']:
                config['training']['pretrained_lr'] = float(config['training']['pretrained_lr'])
            if 'new_lr' in config['training']:
                config['training']['new_lr'] = float(config['training']['new_lr'])
            if 'eta_min' in config['training']:
                config['training']['eta_min'] = float(config['training']['eta_min'])
            if 'weight_decay' in config['training']:
                config['training']['weight_decay'] = float(config['training']['weight_decay'])
        
        # 允许命令行参数覆盖配置文件
        if args.backbone != 'presnet50':
            config['model']['backbone'] = args.backbone
        if args.epochs != 100:
            config['training']['epochs'] = args.epochs
        if args.batch_size != 16:
            config['training']['batch_size'] = args.batch_size
        if args.pretrained_lr != 1e-5:
            config['training']['pretrained_lr'] = args.pretrained_lr
        if args.new_lr != 1e-4:
            config['training']['new_lr'] = args.new_lr
        if args.warmup_epochs != 3:
            config['training']['warmup_epochs'] = args.warmup_epochs
        if args.pretrained_weights:
            config['model']['pretrained_weights'] = args.pretrained_weights
    else:
        # 创建默认配置
        config = {
        'model': {
            'hidden_dim': 256,
            'num_queries': 100,
            'backbone': args.backbone
        },
        'data': {
            'data_root': args.data_root
        },
        'train_dataloader': {
            'dataset': 'DAIRV2XDetection',
            'batch_size': args.batch_size,
            'shuffle': True,
            'num_workers': 16,
            'collate_fn': None
        },
        'val_dataloader': {
            'dataset': 'DAIRV2XDetection',
            'batch_size': args.batch_size,
            'shuffle': False,
            'num_workers': 16,
            'collate_fn': None
        },
        'training': {
            'device': 'cuda',
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'new_lr': args.new_lr,
            'pretrained_lr': args.pretrained_lr,
            'weight_decay': 0.0001,
            'num_workers': 16,
            'save_interval': 10,
            'print_freq': 50,
            'log_dir': 'logs',
            'save_dir': 'checkpoints',
            'output_dir': 'checkpoints',
            'ema_decay': 0.9999,
            'scheduler': 'cosine',
            'eta_min': 0.0000001,
            'warmup_epochs': args.warmup_epochs,
            'clip_max_norm': 10.0
        },
        'validation': {
            'interval': 5,
            'metrics': ['mAP', 'mAP_50', 'mAP_75']
        },
        'augmentation': {
            'mixup': {'enabled': False, 'alpha': 0.2},
            'cutmix': {'enabled': False, 'alpha': 1.0},
            'mosaic': {'enabled': False, 'prob': 0.0}  # 禁用Mosaic，不适合路测探头场景（会破坏空间关系）
        },
        'data_augmentation': {
            'brightness': 0.4,
            'contrast': 0.4,
            'saturation': 0.7,
            'hue': 0.015,
            'flip_prob': 0.5,
            'color_jitter_prob': 0.0
        },
        'misc': {
            'device': 'cuda',
            'num_workers': 16  # 数据加载器worker数量
        }
    }

    registry_path = Path(args.dataset_registry)
    if not registry_path.is_absolute():
        registry_path = Path(__file__).resolve().parent / registry_path
    datasets_map = load_dataset_registry(registry_path)
    if args.dataset:
        profile = resolve_dataset_profile(datasets_map, args.dataset)
        config = apply_detr_dataset_profile(config, profile)
        print(f"🗂️  DETR 数据集: {args.dataset} -> data_root={config.get('data', {}).get('data_root')}")
    elif args.config and args.config.endswith('.yaml') and args.data_root != '/root/autodl-fs/datasets/DAIR-V2X':
        config.setdefault('data', {})['data_root'] = args.data_root
    elif not (args.config and args.config.endswith('.yaml')):
        if args.data_root != '/root/autodl-fs/datasets/DAIR-V2X':
            config.setdefault('data', {})['data_root'] = args.data_root
    
    # 创建训练器
    # 如果使用配置文件，只传递显式传递的参数（不等于默认值的），其他传递None让配置文件的值生效
    if args.config and args.config.endswith('.yaml'):
        # 使用配置文件：只传递显式传递的参数，默认值参数传递None
        data_root_arg = None if args.data_root == '/root/autodl-fs/datasets/DAIR-V2X' else args.data_root
        epochs_arg = None if args.epochs == 100 else args.epochs
        batch_size_arg = None if args.batch_size == 16 else args.batch_size
        warmup_epochs_arg = None if args.warmup_epochs == 3 else args.warmup_epochs
    else:
        # 不使用配置文件：传递所有参数（包括默认值）
        data_root_arg = args.data_root
        epochs_arg = args.epochs
        batch_size_arg = args.batch_size
        warmup_epochs_arg = args.warmup_epochs
    
    trainer = RTDETRTrainer(
        config=config,
        pretrained_weights=args.pretrained_weights,
        data_root=data_root_arg,
        epochs=epochs_arg,
        batch_size=batch_size_arg,
        warmup_epochs=warmup_epochs_arg
    )
    
    # 开始训练
    trainer.start_training(resume_checkpoint=args.resume_from_checkpoint)


if __name__ == '__main__':
    main()
