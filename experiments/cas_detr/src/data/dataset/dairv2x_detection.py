"""
DAIR-V2X数据集检测器 - 基于COCO格式
专门为DAIR-V2X车路协同数据集设计，支持COCO格式评估
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import cv2
import torchvision.transforms.v2 as T

from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor, BoundingBoxes
from ...core import register
from ..transforms import (
    RandomPhotometricDistort, RandomHorizontalFlip,
    ConvertBoxes, Normalize, SanitizeBoundingBoxes,
    RandomZoomOut, RandomIoUCrop, ConvertPILImage, build_square_input_transform,
)

__all__ = ['DAIRV2XDetection']


def _normalize_state(value) -> int:
    """将状态值标准化到 [0, 1, 2]。"""
    try:
        state = int(float(value))
    except (TypeError, ValueError):
        return 0
    if state < 0:
        return 0
    if state > 2:
        return 2
    return state


def _parse_state(value) -> int:
    """解析原始状态值（不做范围裁剪），无效值回退为0。"""
    try:
        state = int(float(value))
    except (TypeError, ValueError):
        return 0
    return state

@register()
class DAIRV2XDetection(DetDataset):
    """DAIR-V2X数据集检测器 - 支持COCO格式评估"""
    
    # 默认的增强配置
    DEFAULT_AUGMENTATION_CONFIG = {
        'target_size': 640,
        'stop_epoch': 71,
        'normalize_mean': [0.485, 0.456, 0.406],
        'normalize_std': [0.229, 0.224, 0.225],
        'photometric_distort_p': 0.5,
        'zoom_out_enabled': True,
        'iou_crop_p': 0.8,
        'horizontal_flip_p': 0.5,
        # letterbox 填充：与 configs 中 augmentation 统一为 114（COCO/YOLO 常用灰边）
        'letterbox_fill': 114,
    }
    
    def __init__(self, data_root: str, split: str = "train", transforms=None, 
                 target_size: int = 640,
                 stop_epoch: int = 71,
                 augmentation_config: Dict = None):
        """
        初始化DAIR-V2X数据集
        
        Args:
            data_root: 数据集根目录
            split: 数据集分割 ('train' 或 'val')
            transforms: 数据变换 (如果为None，将使用默认的增强策略)
            target_size: 目标图像尺寸 (保留参数以兼容)
            stop_epoch: Training epoch at which to stop heavy augmentation (Crop, ZoomOut)
            augmentation_config: 增强配置字典，覆盖默认配置
        """
        super().__init__()
        
        self.data_root = Path(data_root)
        self.split = split
        
        # 合并增强配置
        self.aug_config = self.DEFAULT_AUGMENTATION_CONFIG.copy()
        if augmentation_config is not None:
            self.aug_config.update(augmentation_config)
        
        # 向后兼容：如果传入target_size和stop_epoch，覆盖config中的值
        if target_size != 640:
            self.aug_config['target_size'] = target_size
        if stop_epoch != 71:
            self.aug_config['stop_epoch'] = stop_epoch
        
        self.target_size = self.aug_config['target_size']
        self.stop_epoch = self.aug_config['stop_epoch']
        

        # DAIR-V2X类别定义（8类：前7类是交通参与者，Trafficcone是道路设施）
        self.class_names = [
            "Car", "Truck", "Van", "Bus", "Pedestrian", 
            "Cyclist", "Motorcyclist", "Trafficcone"
        ]
        self.class_to_id = {name: i for i, name in enumerate(self.class_names)}
        
        # 类别合并映射：Barrowlist -> Cyclist (ID=5)
        self.class_merge_map = {
            "Barrowlist": 5,
        }
        
        # Ignore类别列表（训练时应过滤，不参与AP计算）
        self.ignore_classes = [
            "PedestrianIgnore", "CarIgnore", "OtherIgnore", 
            "Unknown_movable", "Unknown_unmovable"
        ]

        self.image_idx_to_coco_id = {}
        self.annotations_by_coco_image_id = {}
        self._load_instances_annotations()
        
        self.data_info = self._load_data_info()
        self.split_indices = self._load_split_indices()
        
        self.set_epoch(0)
        self._init_transforms()

    def _resolve_instances_json_path(self, split_name: str) -> Path:
        primary = self.data_root / "annotations" / f"instances_{split_name}.json"
        if primary.exists():
            return primary
        if split_name == "test":
            alt = self.data_root.parent / "DAIR-V2X_YOLO" / "instances_test.json"
            if alt.exists():
                return alt
        raise FileNotFoundError(f"instances标注文件不存在: {primary}")

    def _init_transforms(self):
        """初始化或更新变换策略"""
        target_size = self.aug_config['target_size']
        normalize_mean = self.aug_config['normalize_mean']
        normalize_std = self.aug_config['normalize_std']
        photometric_distort_p = self.aug_config['photometric_distort_p']
        zoom_out_enabled = self.aug_config['zoom_out_enabled']
        iou_crop_p = self.aug_config['iou_crop_p']
        horizontal_flip_p = self.aug_config['horizontal_flip_p']
        to_square = build_square_input_transform(self.aug_config)
        
        # 逻辑: 根据当前 epoch 切换增强策略
        if self.split == 'train':
            # 判断是否进入最后阶段
            if self.epoch >= self.stop_epoch:
                # 阶段 2: 仅保留基础 Resize 和 Normalize
                self.transforms = T.Compose([
                    to_square,
                    ConvertPILImage(),
                    T.ToDtype(torch.float32, scale=True),
                    Normalize(mean=normalize_mean, std=normalize_std),
                    ConvertBoxes(fmt='cxcywh', normalize=True)
                ])
            else:
                # 阶段 1: 官方标准顺序增强
                transforms_list = [
                    # 1. 颜色增强
                    RandomPhotometricDistort(p=photometric_distort_p),
                ]
                
                # 2. 空间扩展 (可配置)
                if zoom_out_enabled:
                    transforms_list.append(RandomZoomOut(fill=0))
                
                # 3. IoU 约束裁剪
                transforms_list.extend([
                    RandomIoUCrop(p=iou_crop_p),
                    # 4. bbox 清洗
                    SanitizeBoundingBoxes(),
                    # 5. 随机翻转
                    RandomHorizontalFlip(p=horizontal_flip_p),
                ])
                
                # 6-10. Resize 和格式转换
                transforms_list.extend([
                    to_square,
                    ConvertPILImage(),
                    T.ToDtype(torch.float32, scale=True),
                    Normalize(mean=normalize_mean, std=normalize_std),
                    ConvertBoxes(fmt='cxcywh', normalize=True)
                ])
                
                self.transforms = T.Compose(transforms_list)
        else:
            # 验证/推理配置
            self.transforms = T.Compose([
                to_square,
                ConvertPILImage(),
                T.ToDtype(torch.float32, scale=True),
                Normalize(mean=normalize_mean, std=normalize_std),
                ConvertBoxes(fmt='cxcywh', normalize=True)
            ])

    def set_epoch(self, epoch):
        """更新当前 epoch 并根据需要重新初始化 transforms"""
        super().set_epoch(epoch)
        # 重新初始化以应用分阶段策略
        self._init_transforms()

    def _load_data_info(self):
        """加载数据信息"""
        data_info_path = self.data_root / "metadata" / "data_info.json"
        if data_info_path.exists():
            with open(data_info_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"数据信息文件不存在: {data_info_path}")
    
    def _load_split_indices(self):
        """加载训练/验证/测试分割（与 metadata/split_data.json 一致）。"""
        split_path = self.data_root / "metadata" / "split_data.json"
        if split_path.exists():
            with open(split_path, 'r', encoding='utf-8') as f:
                split_data = json.load(f)
                indices = split_data.get(self.split, [])
                valid_indices = []
                for idx in indices:
                    idx_int = int(idx)
                    image_path = self.data_root / "image" / f"{idx_int:06d}.jpg"
                    if image_path.exists():
                        valid_indices.append(idx_int)
                return valid_indices
        if self.split == "test":
            return []
        return self._create_random_split()
    
    def _create_random_split(self):
        """创建随机分割（按路口分层随机分割）"""
        import random
        from collections import defaultdict
        
        intersection_groups = defaultdict(list)
        for idx, item in enumerate(self.data_info):
            loc = item.get('intersection_loc', 'Unknown')
            intersection_groups[loc].append(idx)
        
        random.seed(42)
        
        train_indices = []
        val_indices = []
        
        for loc, indices in intersection_groups.items():
            shuffled = indices.copy()
            random.shuffle(shuffled)
            split_point = int(len(shuffled) * 0.8)
            train_indices.extend(shuffled[:split_point])
            val_indices.extend(shuffled[split_point:])
        
        random.shuffle(train_indices)
        random.shuffle(val_indices)
        
        if self.split == "train":
            return train_indices
        else:
            return val_indices

    def _load_instances_annotations(self):
        """加载COCO instances标注并建立图像索引。"""
        if self.split == "train":
            split_name = "train"
        elif self.split == "test":
            split_name = "test"
        else:
            split_name = "val"
        ann_path = self._resolve_instances_json_path(split_name)

        with open(ann_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        self.image_idx_to_coco_id = {}
        for image_item in coco_data.get('images', []):
            file_name = image_item.get('file_name', '')
            stem = Path(file_name).stem
            try:
                image_idx = int(stem)
            except (TypeError, ValueError):
                continue
            self.image_idx_to_coco_id[image_idx] = int(image_item['id'])

        self.annotations_by_coco_image_id = {}
        for ann in coco_data.get('annotations', []):
            image_id = int(ann.get('image_id', -1))
            if image_id < 0:
                continue
            self.annotations_by_coco_image_id.setdefault(image_id, []).append(ann)
    
    def __len__(self):
        return len(self.split_indices)
    
    def load_item(self, idx):
        """Load a single item without transforms (for Mosaic/Mixup)"""
        if idx >= len(self.split_indices):
            idx = idx % len(self.split_indices)
            
        actual_idx = self.split_indices[idx]
        
        # 加载图像 (PIL)
        image_path = self.data_root / "image" / f"{actual_idx:06d}.jpg"
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        
        # 加载标注
        coco_image_id = self.image_idx_to_coco_id.get(actual_idx)
        annotations = self._load_annotations(coco_image_id)
        
        # 准备Target
        if len(annotations) > 0:
            boxes = torch.tensor([ann['bbox'] for ann in annotations], dtype=torch.float32) # XYXY
            labels = torch.tensor([ann['class_id'] for ann in annotations], dtype=torch.int64)
            areas = torch.tensor([ann['area'] for ann in annotations], dtype=torch.float32)
            iscrowd = torch.tensor([ann.get('iscrowd', 0) for ann in annotations], dtype=torch.int64)
            occluded_states = torch.tensor([ann.get('occluded_state', 0) for ann in annotations], dtype=torch.int64)
            truncated_states = torch.tensor([ann.get('truncated_state', 0) for ann in annotations], dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
            occluded_states = torch.zeros((0,), dtype=torch.int64)
            truncated_states = torch.zeros((0,), dtype=torch.int64)
        
        # 包装成 BoundingBoxes
        boxes = BoundingBoxes(boxes, format='xyxy', canvas_size=(h, w))
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'area': areas,
            'image_id': torch.tensor([actual_idx if coco_image_id is None else coco_image_id]),
            'iscrowd': iscrowd,
            'occluded_state': occluded_states,
            'truncated_state': truncated_states,
            'orig_size': torch.tensor([h, w]), # H, W
        }
        
        return image, target

    def __getitem__(self, idx):
        """获取数据项 - 返回COCO格式的数据"""
        # 1. Load Base Item
        image, target = self.load_item(idx)
        
        # 2. Apply Mosaic / Mixup (if enabled and in training)
        # Note: Mosaic/Mixup are now handled through hardcoded transform pipeline
        
        # 3. 应用标准变换
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        # 更新size（变换后的尺寸）
        if isinstance(image, torch.Tensor):
            target['size'] = torch.tensor(image.shape[-2:]) # H, W
        
        return image, target
    
    def _load_annotations(self, coco_image_id: Optional[int]) -> List[Dict]:
        """从COCO instances加载标注，Ignore框标记为iscrowd=1。"""
        if coco_image_id is None:
            return []

        annotations = self.annotations_by_coco_image_id.get(int(coco_image_id), [])
        processed_annotations = []
        for ann in annotations:
            category_id = int(ann.get('category_id', 0))
            class_id = category_id - 1
            if class_id < 0 or class_id >= len(self.class_names):
                continue

            # COCO bbox: [x, y, w, h]
            bbox_xywh = ann.get('bbox', [0, 0, 0, 0])
            if len(bbox_xywh) != 4:
                continue
            x1 = float(bbox_xywh[0])
            y1 = float(bbox_xywh[1])
            width = float(bbox_xywh[2])
            height = float(bbox_xywh[3])
            x2 = x1 + width
            y2 = y1 + height
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # XYXY format
            bbox = [x1, y1, x2, y2]
            area = float(ann.get('area', width * height))
            raw_occluded_state = _parse_state(ann.get("occluded_state", ann.get("occulated_state", 0)))
            raw_truncated_state = _parse_state(ann.get("truncated_state", ann.get("turncated_state", 0)))
            occluded_state = _normalize_state(raw_occluded_state)
            truncated_state = _normalize_state(raw_truncated_state)

            is_beyond_hard = (
                raw_occluded_state > 2 or raw_occluded_state < 0 or
                raw_truncated_state > 2 or raw_truncated_state < 0
            )

            is_ignore = bool(ann.get('iscrowd', 0)) or is_beyond_hard
            processed_annotations.append({
                'class_id': class_id,
                'bbox': bbox,
                'area': area,
                'iscrowd': 1 if is_ignore else 0,
                'occluded_state': occluded_state,
                'truncated_state': truncated_state
            })
        
        # 如果是训练集，且为了避免ignore框干扰训练（如DETR matching），可以过滤掉iscrowd=1的框
        # 但Transform可能会crop掉它们。
        # 这里我们保留，但在__getitem__中可以根据需要过滤
        # 按照之前的逻辑：训练时过滤ignore框
        if self.split == 'train':
            processed_annotations = [ann for ann in processed_annotations if ann['iscrowd'] == 0]
            
        return processed_annotations

    def get_image_path(self, coco_image_id: int) -> Optional[Path]:
        """供训练结束可视化：由 COCO image_id 解析原始图像路径。"""
        for idx, cid in self.image_idx_to_coco_id.items():
            if int(cid) == int(coco_image_id):
                return self.data_root / "image" / f"{int(idx):06d}.jpg"
        return None
    
    def get_categories(self):
        """获取类别信息 - COCO格式"""
        categories = []
        for i, class_name in enumerate(self.class_names):
            categories.append({
                'id': i + 1,
                'name': class_name,
                'supercategory': 'object'
            })
        return categories
    
    def get_image_info(self, image_id: int):
        """获取图像信息 - COCO格式"""
        if image_id < len(self.data_info):
            data_item = self.data_info[image_id]
            # 注意：这里的size可能是原始尺寸，或者默认target_size
            # 实际上应该读取图片获取真实尺寸，或者使用metadata中的信息
            # 简单起见，这里暂时不读取图片
            return {
                'id': image_id,
                'file_name': data_item["image_path"],
                # 'width': ..., 'height': ... 
            }
        return None
