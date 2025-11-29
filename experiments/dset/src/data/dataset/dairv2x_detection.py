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
    RandomPhotometricDistort, RandomIoUCrop, RandomHorizontalFlip, 
    RandomResize, ConvertBoxes, Normalize, SanitizeBoundingBoxes
)

__all__ = ['DAIRV2XDetection']

@register()
class DAIRV2XDetection(DetDataset):
    """DAIR-V2X数据集检测器 - 支持COCO格式评估"""
    
    def __init__(self, data_root: str, split: str = "train", transforms=None, 
                 target_size: int = 640,
                 aug_brightness: float = 0.0,
                 aug_contrast: float = 0.0,
                 aug_saturation: float = 0.0,
                 aug_hue: float = 0.0,
                 aug_color_jitter_prob: float = 0.0):
        """
        初始化DAIR-V2X数据集
        
        Args:
            data_root: 数据集根目录
            split: 数据集分割 ('train' 或 'val')
            transforms: 数据变换 (如果为None，将使用默认的Unified Task-Adapted Augmentation)
            target_size: 目标图像尺寸 (保留参数以兼容，但会被新的增强策略覆盖)
            aug_*: 保留参数以兼容，但会被新的增强策略覆盖
        """
        super().__init__()
        
        self.data_root = Path(data_root)
        self.split = split
        self.target_size = target_size
        
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
        
        # 加载数据信息
        self.data_info = self._load_data_info()
        self.split_indices = self._load_split_indices()
        
        # 初始化变换策略 (Unified Task-Adapted Augmentation)
        if transforms is None:
            scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
            if split == 'train':
                self.transforms = T.Compose([
                    RandomPhotometricDistort(
                        brightness=aug_brightness, 
                        contrast=aug_contrast, 
                        saturation=aug_saturation, 
                        hue=aug_hue
                    ),
                    RandomIoUCrop(min_scale=aug_crop_min, max_scale=aug_crop_max, p=1.0),
                    RandomHorizontalFlip(p=aug_flip_prob),
                    RandomResize(scales=scales, max_size=1333),
                    SanitizeBoundingBoxes(),
                    T.ToImage(),
                    T.ToDtype(torch.float32, scale=True),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ConvertBoxes(fmt='cxcywh', normalize=True)
                ])
            else:
                # 验证/推理配置：矩形推理 (Rectangular Inference)
                # Resize到1280 (保持长宽比)，短边自适应
                self.transforms = T.Compose([
                    T.Resize(size=1280, max_size=1333, antialias=True),
                    T.ToImage(),
                    T.ToDtype(torch.float32, scale=True),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ConvertBoxes(fmt='cxcywh', normalize=True)
                ])
        else:
            self.transforms = transforms
    
    def _load_data_info(self):
        """加载数据信息"""
        data_info_path = self.data_root / "metadata" / "data_info.json"
        if data_info_path.exists():
            with open(data_info_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"数据信息文件不存在: {data_info_path}")
    
    def _load_split_indices(self):
        """加载训练/验证分割"""
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
        else:
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
    
    def __len__(self):
        return len(self.split_indices)
    
    def __getitem__(self, idx):
        """获取数据项 - 返回COCO格式的数据"""
        actual_idx = self.split_indices[idx]
        
        # 加载图像 (PIL)
        image_path = self.data_root / "image" / f"{actual_idx:06d}.jpg"
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        
        # 加载标注
        annotation_path = self.data_root / "annotations" / "camera" / f"{actual_idx:06d}.json"
        annotations = self._load_annotations(annotation_path)
        
        # 准备Target
        if len(annotations) > 0:
            boxes = torch.tensor([ann['bbox'] for ann in annotations], dtype=torch.float32) # XYXY
            labels = torch.tensor([ann['class_id'] for ann in annotations], dtype=torch.int64)
            areas = torch.tensor([ann['area'] for ann in annotations], dtype=torch.float32)
            iscrowd = torch.tensor([ann.get('iscrowd', 0) for ann in annotations], dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        
        # 包装成 BoundingBoxes
        boxes = BoundingBoxes(boxes, format='xyxy', canvas_size=(h, w))
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'area': areas,
            'image_id': torch.tensor([actual_idx]),
            'iscrowd': iscrowd,
            'orig_size': torch.tensor([h, w]), # H, W
        }
        
        # 应用变换
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        # 更新size（变换后的尺寸）
        if isinstance(image, torch.Tensor):
            target['size'] = torch.tensor(image.shape[-2:]) # H, W
        
        return image, target
    
    def _load_annotations(self, annotation_path: Path) -> List[Dict]:
        """加载标注文件，Ignore框标记为iscrowd=1"""
        if not annotation_path.exists():
            return []
        
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        processed_annotations = []
        for ann in annotations:
            class_name = ann["type"]
            
            # 获取2D边界框
            bbox_2d = ann["2d_box"]
            x1 = float(bbox_2d["xmin"])
            y1 = float(bbox_2d["ymin"])
            x2 = float(bbox_2d["xmax"])
            y2 = float(bbox_2d["ymax"])
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # XYXY format
            bbox = [x1, y1, x2, y2]
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            if class_name in self.ignore_classes:
                # 训练时：如果split是train，可以过滤掉ignore框
                # 但为了保持代码一致性，这里保留并标记iscrowd=1
                # SanitizeBoundingBoxes 可能不会过滤 iscrowd=1
                processed_annotations.append({
                    'class_id': 0,
                    'bbox': bbox,
                    'area': area,
                    'iscrowd': 1
                })
            elif class_name in self.class_merge_map:
                class_id = self.class_merge_map[class_name]
                processed_annotations.append({
                    'class_id': class_id,
                    'bbox': bbox,
                    'area': area,
                    'iscrowd': 0
                })
            elif class_name in self.class_to_id:
                class_id = self.class_to_id[class_name]
                processed_annotations.append({
                    'class_id': class_id,
                    'bbox': bbox,
                    'area': area,
                    'iscrowd': 0
                })
        
        # 如果是训练集，且为了避免ignore框干扰训练（如DETR matching），可以过滤掉iscrowd=1的框
        # 但Transform可能会crop掉它们。
        # 这里我们保留，但在__getitem__中可以根据需要过滤
        # 按照之前的逻辑：训练时过滤ignore框
        if self.split == 'train':
            processed_annotations = [ann for ann in processed_annotations if ann['iscrowd'] == 0]
            
        return processed_annotations
    
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
