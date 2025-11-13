import sys
import os
import argparse
import yaml
import torch
import numpy as np
import re
from pathlib import Path
import logging
from typing import Optional, Dict, Union, List
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

# 添加项目路径
project_root = Path(__file__).parent.resolve()
# 确保当前工作目录在路径中（重要：当从不同目录运行时）
if str(os.getcwd()) not in sys.path:
    sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))  # 添加experiments目录

# 导入随机种子工具
from seed_utils import set_seed, seed_worker

# 导入现有工具
from src.misc.training_visualizer import TrainingVisualizer
from src.misc.early_stopping import EarlyStopping
from src.data import DataLoader
from src.optim.ema import ModelEMA
from src.optim.amp import GradScaler
from src.optim.warmup import WarmupLR
from src.data.dataset.dairv2x_detection import DAIRV2XDetection


def create_backbone(backbone_type: str, **kwargs):
    """创建backbone的工厂函数。
    
    Args:
        backbone_type: backbone类型（presnet18/34/50/101, hgnetv2_l等）
        **kwargs: backbone特定参数（会覆盖默认配置）
    
    Returns:
        nn.Module: backbone模型实例
        
    Raises:
        ValueError: 不支持的backbone类型
    """
    from src.nn.backbone.presnet import PResNet
    from src.nn.backbone.hgnetv2 import HGNetv2
    from src.nn.backbone.csp_resnet import CSPResNet
    from src.nn.backbone.csp_darknet import CSPDarkNet
    
    # PResNet配置（通过正则表达式解析depth）
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
            'freeze_at': 0,  # 冻结第一个stage
            'freeze_norm': True,  # 冻结BN层
            'pretrained': False
        }
        default_params.update(kwargs)
        return PResNet(**default_params)
    
    # HGNetv2配置
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
    
    # CSPResNet配置
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
    
    # CSPDarkNet配置
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
        
        # 确保学习率相关值是浮点数（YAML中的科学计数法可能被解析为字符串）
        # 直接使用配置文件中的字段名：pretrained_lr, new_lr
        if 'training' in self.config:
            # 类型转换确保是浮点数
            if 'pretrained_lr' in self.config['training']:
                self.config['training']['pretrained_lr'] = float(self.config['training']['pretrained_lr'])
            if 'new_lr' in self.config['training']:
                self.config['training']['new_lr'] = float(self.config['training']['new_lr'])
            if 'eta_min' in self.config['training']:
                self.config['training']['eta_min'] = float(self.config['training']['eta_min'])
            if 'weight_decay' in self.config['training']:
                self.config['training']['weight_decay'] = float(self.config['training']['weight_decay'])
        
        # 命令行参数覆盖配置文件（只有在显式传递时才覆盖）
        if data_root is not None:
            self.config['data']['data_root'] = data_root
        
        if epochs is not None:
            self.config['training']['epochs'] = epochs
        
        if batch_size is not None:
            self.config['training']['batch_size'] = batch_size
        
        if warmup_epochs is not None:
            self.config['training']['warmup_epochs'] = warmup_epochs
        
        # 设置基本属性（device在misc配置中）
        if using_config_file:
            # 如果使用配置文件，device必须存在，否则报错
            if 'misc' not in self.config or 'device' not in self.config['misc']:
                raise ValueError(f"配置文件 {self.config_path} 缺少必需的配置项: misc.device")
            device_str = self.config['misc']['device']
        else:
            device_str = self.config.get('misc', {}).get('device', 'cuda')
        self.device = torch.device(device_str)
        self.setup_logging()
        self._create_directories()
        
        # 初始化组件
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.warmup_scheduler = None
        self.ema = None
        self.scaler = None
        self.visualizer = None
    
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
            # 恢复训练：使用检查点所在目录
            self.log_dir = Path(resume_checkpoint).parent
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"📦 恢复训练，使用现有日志目录: {self.log_dir}")
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
            self.log_dir = Path(f"logs/{self.experiment_name}_{timestamp}")
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
            act='silu',
            eval_spatial_size=[640, 640]
        )
        
        # 从配置文件读取模型参数
        num_decoder_layers = self.config['model']['num_decoder_layers']
        hidden_dim = self.config['model']['hidden_dim']
        num_queries = self.config['model']['num_queries']
        
        # 创建decoder（添加denoising训练）
        from src.zoo.rtdetr.rtdetrv2_decoder import RTDETRTransformerv2
        decoder = RTDETRTransformerv2(
            num_classes=6,
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
            checkpoint = torch.load(pretrained_file, map_location='cpu')
            
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
            
            # 统计加载结果（使用filtered_state_dict）
            total_params = len(filtered_state_dict)
            loaded_params = total_params - len(missing_keys)
            
            self.logger.info(f"✓ 成功加载预训练权重: {loaded_params}/{total_params} 个参数")
            
            # 报告跳过的类别参数
            if skipped_class_params > 0:
                self.logger.info(f"  - 跳过类别相关参数: {skipped_class_params} 个（COCO 80类 → DAIR-V2X 6类）")
            
            # 统计各部分的参数
            backbone_loaded = sum(1 for k in filtered_state_dict.keys() if k not in missing_keys and 'backbone' in k)
            encoder_loaded = sum(1 for k in filtered_state_dict.keys() if k not in missing_keys and 'encoder' in k)
            decoder_loaded = sum(1 for k in filtered_state_dict.keys() if k not in missing_keys and 'decoder' in k)
            
            self.logger.info(f"  - Backbone: {backbone_loaded} 个参数")
            self.logger.info(f"  - Encoder: {encoder_loaded} 个参数")
            self.logger.info(f"  - Decoder: {decoder_loaded} 个参数")
            
            if len(missing_keys) > 0:
                # missing_keys是filtered_state_dict中有但当前模型没有的参数
                self.logger.info(f"  - 预训练模型缺少参数: {len(missing_keys)} 个（当前模型新增）")
                # 显示前3个示例
                if len(missing_keys) <= 5:
                    self.logger.info(f"    示例: {list(missing_keys)}")
                else:
                    self.logger.info(f"    示例: {list(missing_keys)[:3]} ...")
            
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
            num_classes=6,
            boxes_weight_format=None,
            share_matched_indices=False
        )
        
        return criterion
    
    def create_datasets(self):
        """创建数据集"""
        from src.data.dataloader import BaseCollateFunction
        
        # 创建collate_fn类
        class CustomCollateFunction(BaseCollateFunction):
            def __call__(self, batch):
                images, targets = zip(*batch)
                
                if isinstance(images[0], np.ndarray):
                    images = torch.stack([
                        torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 
                        for img in images
                    ], dim=0)
                else:
                    images = torch.stack(images, 0)
                
                return images, list(targets)
        
        # 直接使用DAIRV2XDetection类（直接使用配置文件中的data.data_root）
        data_root = self.config['data']['data_root']
        use_mosaic = self.config.get('training', {}).get('use_mosaic', True)
        target_size = 640
        
        train_dataset = DAIRV2XDetection(
            data_root=data_root,
            split='train',
            transforms=None,
            use_mosaic=use_mosaic,
            target_size=target_size
        )
        
        val_dataset = DAIRV2XDetection(
            data_root=data_root,
            split='val',
            transforms=None,
            use_mosaic=False,  # 验证时不使用Mosaic
            target_size=target_size
        )
        
        collate_fn = CustomCollateFunction()
        
        # num_workers在misc配置中
        num_workers = self.config.get('misc', {}).get('num_workers', 4)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
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
        
        # 1. Backbone参数（使用较小学习率，排除norm层）
        backbone_params = []
        backbone_names = []
        for name, param in self.model.named_parameters():
            if 'backbone' in name and param.requires_grad:
                # 排除norm层
                if not any(norm in name for norm in ['norm', 'bn', 'gn', 'ln']):
                    backbone_params.append(param)
                    backbone_names.append(name)
        
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': pretrained_lr,
                'weight_decay': weight_decay
            })
            self.logger.info(f"✓ Backbone参数组: {len(backbone_params)} 个参数, lr={pretrained_lr}")
        
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
        
        # 3. 其他参数（encoder、decoder等）
        other_params = []
        other_names = []
        processed_params = set(id(p) for p in backbone_params + norm_params)
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and id(param) not in processed_params:
                other_params.append(param)
                other_names.append(name)
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': new_lr,
                'weight_decay': weight_decay
            })
            self.logger.info(f"✓ 其他参数组: {len(other_params)} 个参数, lr={new_lr}")
        
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
                'best_map': self.best_map,
                'best_metric': getattr(self, 'best_metric', 0.0),
                'global_step': getattr(self, 'global_step', 0)
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
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'config': self.config,
                'best_loss': self.best_loss,
                'best_map': self.best_map,
                'global_step': getattr(self, 'global_step', 0)
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
        
        # 3. 创建其他组件
        self.criterion = self.create_criterion()
        self.train_dataloader, self.val_dataloader = self.create_datasets()
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        self.warmup_scheduler = self.create_warmup_scheduler()
        
        # 4. 创建EMA和梯度缩放器
        ema_decay = self.config['training'].get('ema_decay', 0.9999)
        self.ema = ModelEMA(self.model, decay=ema_decay)
        self.scaler = GradScaler()
        self.logger.info(f"✓ EMA decay={ema_decay}, 混合精度训练已启用")
        
        # 5. 创建可视化器（使用log_dir）
        self.visualizer = TrainingVisualizer(
            log_dir=self.log_dir,
            model_type='standard',
            experiment_name=self.experiment_name
        )
        
        # 6. 设置训练属性
        self.last_epoch = -1
        self.best_loss = float('inf')
        self.best_map = 0.0  # 记录最佳mAP
        
        # 6.5 初始化Early Stopping
        self.early_stopping = self._create_early_stopping()
        
        # 7. 设置梯度裁剪参数
        self.clip_max_norm = self.config['training'].get('clip_max_norm', 10.0)
        self.logger.info(f"✓ 梯度裁剪: max_norm={self.clip_max_norm}")
        
        # 8. 恢复训练（如果提供checkpoint）
        if resume_checkpoint:
            self.logger.info(f"📦 从检查点恢复训练: {resume_checkpoint}")
            checkpoint = torch.load(resume_checkpoint, map_location=self.device)
            
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
            self.best_map = checkpoint.get('best_map', 0.0)
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
        
        return EarlyStopping(
            patience=patience,
            mode=mode,
            min_delta=0.0001,
            metric_name=metric_name,
            logger=self.logger
        )
    
    def _custom_training_loop(self):
        """自定义训练循环"""
        epochs = self.config['training']['epochs']
        self.logger.info(f"开始训练 {epochs} epochs")
        
        for epoch in range(self.last_epoch + 1, epochs):
            self.last_epoch = epoch
            
            # 训练一个epoch
            train_metrics = self._train_epoch()
            
            # 验证
            val_metrics = self._validate_epoch()
            
            # 学习率调度
            if hasattr(self, 'warmup_scheduler') and self.warmup_scheduler and not self.warmup_scheduler.finished():
                self.warmup_scheduler.step()
            else:
                self.scheduler.step()
            
            # 输出日志
            self.logger.info(f"Epoch {epoch}:")
            self.logger.info(f"  训练损失: {train_metrics.get('total_loss', 0.0):.2f} | 验证损失: {val_metrics.get('total_loss', 0.0):.2f}")
            self.logger.info(f"  mAP@0.5: {val_metrics.get('mAP_0.5', 0.0):.4f} | mAP@0.75: {val_metrics.get('mAP_0.75', 0.0):.4f} | "
                           f"mAP@[0.5:0.95]: {val_metrics.get('mAP_0.5_0.95', 0.0):.4f}")
            self.logger.info(f"  预测/目标: {val_metrics['num_predictions']}/{val_metrics['num_targets']}")
            
            # 记录到可视化器
            current_lr = self.optimizer.param_groups[0]['lr']
            self.visualizer.record(
                epoch=epoch,
                train_loss=train_metrics.get('total_loss', 0.0),
                val_loss=val_metrics.get('total_loss', 0.0),
                mAP_0_5=val_metrics.get('mAP_0.5', 0.0),
                mAP_0_75=val_metrics.get('mAP_0.75', 0.0),
                mAP_0_5_0_95=val_metrics.get('mAP_0.5_0.95', 0.0),
                learning_rate=current_lr
            )
            
            # 保存检查点 - 同时考虑loss和mAP
            is_best_loss = val_metrics.get('total_loss', float('inf')) < self.best_loss
            is_best_map = val_metrics.get('mAP_0.5_0.95', 0.0) > self.best_map
            
            if is_best_loss:
                self.best_loss = val_metrics.get('total_loss', float('inf'))
                self.logger.info(f"  🎉 新的最佳验证损失: {self.best_loss:.2f}")
            
            if is_best_map:
                self.best_map = val_metrics.get('mAP_0.5_0.95', 0.0)
                self.logger.info(f"  🎉 新的最佳mAP: {self.best_map:.4f}")
                # 保存最佳模型（基于mAP）
                self._save_best_checkpoint(epoch)
            
            # Early Stopping检查
            if self.early_stopping:
                # 获取要监控的指标值
                metric_name = self.early_stopping.metric_name
                if 'mAP_0.5_0.95' in metric_name or 'mAP_0.5:0.95' in metric_name:
                    metric_value = val_metrics.get('mAP_0.5_0.95', 0.0)
                elif 'mAP_0.5' in metric_name:
                    metric_value = val_metrics.get('mAP_0.5', 0.0)
                elif 'mAP_0.75' in metric_name:
                    metric_value = val_metrics.get('mAP_0.75', 0.0)
                elif 'loss' in metric_name.lower():
                    metric_value = val_metrics.get('total_loss', float('inf'))
                else:
                    metric_value = val_metrics.get('mAP_0.5_0.95', 0.0)  # 默认
                
                if self.early_stopping(metric_value, epoch):
                    self.logger.info(f"Early Stopping在epoch {epoch}触发，停止训练")
                    break
            
            # 每个epoch都保存latest用于断点续训
            self._save_latest_checkpoint()
            
            # 绘制训练曲线（每个epoch都更新）
            try:
                self.visualizer.plot()
            except Exception as e:
                self.logger.warning(f"绘制训练曲线失败: {e}")
        
        # 训练完成后，绘制最终的训练曲线并导出CSV
        self.logger.info("✓ 训练完成！")
        try:
            self.visualizer.plot()
            self.visualizer.export_to_csv()
            self.logger.info(f"✓ 训练曲线已保存到: {self.log_dir}/training_curves.png")
            self.logger.info(f"✓ 训练历史已导出到: {self.log_dir}/training_history.csv")
            self.logger.info(f"✓ 所有输出已保存到: {self.log_dir}")
        except Exception as e:
            self.logger.warning(f"绘制最终训练曲线失败: {e}")
    
    def _train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        detection_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(self.train_dataloader):
            images = images.to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            # 前向传播
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
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
            
            # 统计损失
            total_loss += loss.item()
            
            # 计算检测损失（主要损失项）
            det_loss_val = 0.0
            if 'loss_vfl' in loss_dict:
                det_loss_val += loss_dict['loss_vfl'].item()
            if 'loss_bbox' in loss_dict:
                det_loss_val += loss_dict['loss_bbox'].item()
            if 'loss_giou' in loss_dict:
                det_loss_val += loss_dict['loss_giou'].item()
            
            detection_loss += det_loss_val
            
            # 每50个batch打印一次（参照moe-rtdetr格式）
            if batch_idx % 50 == 0:
                self.logger.info(f'Epoch {self.last_epoch} | Batch {batch_idx} | '
                               f'Loss: {loss.item():.2f} (Det: {det_loss_val:.2f})')
        
        # 计算平均值
        num_batches = len(self.train_dataloader)
        avg_loss = total_loss / num_batches
        avg_detection_loss = detection_loss / num_batches
        
        return {
            'total_loss': avg_loss,
            'detection_loss': avg_detection_loss
        }
    
    def _validate_epoch(self):
        """验证模型并计算mAP"""
        self.ema.module.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        total_raw_predictions = 0  # 原始query总数
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(self.val_dataloader):
                images = images.to(self.device)
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in t.items()} for t in targets]
                
                outputs = self.ema.module(images, targets)
                
                # 使用criterion计算损失
                loss_dict = self.criterion(outputs, targets)
                loss = sum(loss_dict.values())
                total_loss += loss.item()
                
                # 统计原始预测数（所有queries）
                if 'pred_logits' in outputs:
                    total_raw_predictions += outputs['pred_logits'].shape[0] * outputs['pred_logits'].shape[1]
                
                # 收集预测结果
                if 'pred_logits' in outputs and 'pred_boxes' in outputs:
                    self._collect_predictions(outputs, targets, batch_idx, all_predictions, all_targets)
        
        # 计算mAP
        mAP_metrics = self._compute_map_metrics(all_predictions, all_targets)
        
        avg_loss = total_loss / len(self.val_dataloader)
        
        return {
            'total_loss': avg_loss,
            'mAP_0.5': mAP_metrics.get('mAP_0.5', 0.0),
            'mAP_0.75': mAP_metrics.get('mAP_0.75', 0.0),
            'mAP_0.5_0.95': mAP_metrics.get('mAP_0.5_0.95', 0.0),
            'num_predictions': len(all_predictions),
            'num_raw_predictions': len(all_predictions),  # 修复：使用实际预测数
            'num_targets': len(all_targets)
        }
    
    def _collect_predictions(self, outputs: Dict, targets: List[Dict], batch_idx: int,
                            all_predictions: List, all_targets: List) -> None:
        """收集预测结果用于mAP计算。保留所有有效预测框，不做top-k限制。"""
        pred_logits = outputs['pred_logits']  # [B, Q, C]
        pred_boxes = outputs['pred_boxes']    # [B, Q, 4]
        
        batch_size = pred_logits.shape[0]
        
        for i in range(batch_size):
            pred_scores = torch.softmax(pred_logits[i], dim=-1)  # [Q, C]
            max_scores, pred_classes = torch.max(pred_scores, dim=-1)  # [Q]
            
            # 过滤无效框（padding框），保留所有有效预测框
            valid_boxes_mask = ~torch.all(pred_boxes[i] == 1.0, dim=1)
            valid_indices = torch.where(valid_boxes_mask)[0]
            if len(valid_indices) > 0:
                filtered_boxes = pred_boxes[i][valid_indices]
                filtered_classes = pred_classes[valid_indices]
                filtered_scores = max_scores[valid_indices]
                
                # 转换为COCO格式
                if filtered_boxes.shape[0] > 0:
                    boxes_coco = torch.zeros_like(filtered_boxes)
                    if filtered_boxes.max() <= 1.0:
                        # 归一化坐标 -> 像素坐标
                        boxes_coco[:, 0] = (filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2) * 640
                        boxes_coco[:, 1] = (filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2) * 640
                        boxes_coco[:, 2] = filtered_boxes[:, 2] * 640
                        boxes_coco[:, 3] = filtered_boxes[:, 3] * 640
                    else:
                        boxes_coco = filtered_boxes.clone()
                    
                    # Clamp坐标
                    boxes_coco[:, 0] = torch.clamp(boxes_coco[:, 0], 0, 640)
                    boxes_coco[:, 1] = torch.clamp(boxes_coco[:, 1], 0, 640)
                    boxes_coco[:, 2] = torch.clamp(boxes_coco[:, 2], 1, 640)
                    boxes_coco[:, 3] = torch.clamp(boxes_coco[:, 3], 1, 640)
                    
                    for j in range(filtered_boxes.shape[0]):
                        all_predictions.append({
                            'image_id': batch_idx * self.config['training']['batch_size'] + i,
                            'category_id': int(filtered_classes[j].item()) + 1,
                            'bbox': boxes_coco[j].cpu().numpy().tolist(),
                            'score': float(filtered_scores[j].item())
                        })
            
            # 处理真实标签
            if i < len(targets) and 'labels' in targets[i] and 'boxes' in targets[i]:
                true_labels = targets[i]['labels']
                true_boxes = targets[i]['boxes']
                
                if len(true_labels) > 0:
                    img_size = 640
                    max_val = float(true_boxes.max().item()) if true_boxes.numel() > 0 else 0.0
                    scale = img_size if max_val <= 1.0 + 1e-6 else 1.0
                    
                    true_boxes_coco = torch.zeros_like(true_boxes)
                    true_boxes_coco[:, 0] = (true_boxes[:, 0] - true_boxes[:, 2] / 2) * scale
                    true_boxes_coco[:, 1] = (true_boxes[:, 1] - true_boxes[:, 3] / 2) * scale
                    true_boxes_coco[:, 2] = true_boxes[:, 2] * scale
                    true_boxes_coco[:, 3] = true_boxes[:, 3] * scale
                    
                    true_boxes_coco[:, 0] = torch.clamp(true_boxes_coco[:, 0], 0, img_size)
                    true_boxes_coco[:, 1] = torch.clamp(true_boxes_coco[:, 1], 0, img_size)
                    true_boxes_coco[:, 2] = torch.clamp(true_boxes_coco[:, 2], 1, img_size)
                    true_boxes_coco[:, 3] = torch.clamp(true_boxes_coco[:, 3], 1, img_size)
                    
                    for j in range(len(true_labels)):
                        all_targets.append({
                            'image_id': batch_idx * self.config['training']['batch_size'] + i,
                            'category_id': int(true_labels[j].item()) + 1,
                            'bbox': true_boxes_coco[j].cpu().numpy().tolist(),
                            'area': float((true_boxes_coco[j, 2] * true_boxes_coco[j, 3]).item()),
                            'iscrowd': 0
                        })
    
    def _compute_map_metrics(self, predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
        """计算mAP指标。"""
        try:
            if len(predictions) == 0:
                return {
                    'mAP_0.5': 0.0,
                    'mAP_0.75': 0.0,
                    'mAP_0.5_0.95': 0.0
                }
            
            # 获取类别信息
            if hasattr(self, 'val_dataloader') and hasattr(self.val_dataloader.dataset, 'get_categories'):
                categories = self.val_dataloader.dataset.get_categories()
            else:
                categories = [
                    {'id': 1, 'name': 'car'},
                    {'id': 2, 'name': 'truck'},
                    {'id': 3, 'name': 'bus'},
                    {'id': 4, 'name': 'person'},
                    {'id': 5, 'name': 'bicycle'},
                    {'id': 6, 'name': 'motorcycle'}
                ]
            
            # 创建COCO格式数据
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
            
            # 添加图像信息
            image_ids = set(target['image_id'] for target in targets)
            for img_id in image_ids:
                coco_gt['images'].append({
                    'id': img_id, 
                    'width': 640, 
                    'height': 640
                })
            
            # 添加标注
            for i, target in enumerate(targets):
                target['id'] = i + 1
                coco_gt['annotations'].append(target)
            
            # 使用pycocotools评估
            coco_gt_obj = COCO()
            coco_gt_obj.dataset = coco_gt
            coco_gt_obj.createIndex()
            
            coco_dt = coco_gt_obj.loadRes(predictions)
            
            coco_eval = COCOeval(coco_gt_obj, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            return {
                'mAP_0.5': coco_eval.stats[1],
                'mAP_0.75': coco_eval.stats[2],
                'mAP_0.5_0.95': coco_eval.stats[0]
            }
            
        except Exception as e:
            self.logger.warning(f"mAP计算失败: {e}")
            return {
                'mAP_0.5': 0.0,
                'mAP_0.75': 0.0,
                'mAP_0.5_0.95': 0.0
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
    parser.add_argument('--data_root', type=str, default='datasets/DAIR-V2X', 
                       help='DAIR-V2X数据集路径')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
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
        if args.batch_size != 32:
            config['training']['batch_size'] = args.batch_size
        if args.pretrained_lr != 1e-5:
            config['training']['pretrained_lr'] = args.pretrained_lr
        if args.new_lr != 1e-4:
            config['training']['new_lr'] = args.new_lr
        if args.warmup_epochs != 3:
            config['training']['warmup_epochs'] = args.warmup_epochs
        if args.data_root != 'datasets/DAIR-V2X':
            config['data']['data_root'] = args.data_root
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
            'num_workers': 4,
            'collate_fn': None
        },
        'val_dataloader': {
            'dataset': 'DAIRV2XDetection',
            'batch_size': args.batch_size,
            'shuffle': False,
            'num_workers': 4,
            'collate_fn': None
        },
        'training': {
            'device': 'cuda',
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'new_lr': args.new_lr,
            'pretrained_lr': args.pretrained_lr,
            'weight_decay': 0.0001,
            'num_workers': 4,
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
            'mosaic': {'enabled': True, 'prob': 0.5}
        }
    }
    
    # 创建训练器
    # 如果使用配置文件，只传递显式传递的参数（不等于默认值的），其他传递None让配置文件的值生效
    if args.config and args.config.endswith('.yaml'):
        # 使用配置文件：只传递显式传递的参数，默认值参数传递None
        data_root_arg = None if args.data_root == 'datasets/DAIR-V2X' else args.data_root
        epochs_arg = None if args.epochs == 100 else args.epochs
        batch_size_arg = None if args.batch_size == 32 else args.batch_size
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
