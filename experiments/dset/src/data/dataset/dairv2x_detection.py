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
    """DAIR-V2X Dataset Detector - Supports COCO format evaluation"""
    
    def __init__(self, data_root: str, split: str = "train", transforms=None, 
                 aug_brightness: float = 0.0,
                 aug_contrast: float = 0.0,
                 aug_saturation: float = 0.0,
                 aug_hue: float = 0.0,
                 aug_color_jitter_prob: float = 0.0,
                 aug_crop_min: float = 0.3,
                 aug_crop_max: float = 1.0,
                 aug_flip_prob: float = 0.5,
                 train_scales_min: int = 480,
                 train_scales_max: int = 800,
                 train_scales_step: int = 32,
                 train_max_size: int = 1333,
                 aug_mosaic_prob: float = 0.0,
                 aug_mixup_prob: float = 0.0):
        """
        Initialize DAIR-V2X Dataset
        
        Args:
            data_root: Dataset root directory
            split: Dataset split ('train' or 'val')
            transforms: Data transforms (if None, default Unified Task-Adapted Augmentation will be used)
            aug_*: Retained for compatibility, but overridden by new augmentation strategy
            train_scales_min: Minimum short edge size for multi-scale training
            train_scales_max: Maximum short edge size for multi-scale training
            train_scales_step: Step size for generating scale options
            train_max_size: Maximum long edge size for multi-scale training
            aug_mosaic_prob: Probability of applying Mosaic augmentation
            aug_mixup_prob: Probability of applying Mixup augmentation
        """
        super().__init__()
        
        self.data_root = Path(data_root)
        self.split = split
        
        # Store augmentation probabilities for use in __getitem__ or custom wrapper
        self.aug_mosaic_prob = aug_mosaic_prob
        self.aug_mixup_prob = aug_mixup_prob
        
        # DAIR-V2X Class Definitions (8 classes: first 7 are traffic participants, Trafficcone is road facility)
        self.class_names = [
            "Car", "Truck", "Van", "Bus", "Pedestrian", 
            "Cyclist", "Motorcyclist", "Trafficcone"
        ]
        self.class_to_id = {name: i for i, name in enumerate(self.class_names)}
        
        # Class Merge Mapping: Barrowlist -> Cyclist (ID=5)
        self.class_merge_map = {
            "Barrowlist": 5,
        }
        
        # Ignore Class List (should be filtered during training, not involved in AP calculation)
        self.ignore_classes = [
            "PedestrianIgnore", "CarIgnore", "OtherIgnore", 
            "Unknown_movable", "Unknown_unmovable"
        ]
        
        # Load Data Info
        self.data_info = self._load_data_info()
        self.split_indices = self._load_split_indices()
        
        # Initialize Transform Strategy (Unified Task-Adapted Augmentation)
        if transforms is None:
            scales = list(range(train_scales_min, train_scales_max + 1, train_scales_step))
            if split == 'train':
                # Note: Mosaic and Mixup are handled in __getitem__ or separate wrapper because they need multiple images
                # Here we only define the per-image transform pipeline
                
                self.transforms = T.Compose([
                    RandomPhotometricDistort(
                        brightness=(max(0, 1 - aug_brightness), 1 + aug_brightness), 
                        contrast=(max(0, 1 - aug_contrast), 1 + aug_contrast), 
                        saturation=(max(0, 1 - aug_saturation), 1 + aug_saturation), 
                        hue=(-aug_hue, aug_hue)
                    ),
                    RandomIoUCrop(min_scale=aug_crop_min, max_scale=aug_crop_max, p=1.0),
                    RandomHorizontalFlip(p=aug_flip_prob),
                    RandomResize(scales=scales, max_size=train_max_size),
                    SanitizeBoundingBoxes(),
                    T.ToImage(),
                    T.ToDtype(torch.float32, scale=True),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ConvertBoxes(fmt='cxcywh', normalize=False)
                ])
                
                # Initialize Mosaic/Mixup helpers if needed
                if self.aug_mosaic_prob > 0:
                    from ..transforms.mosaic import Mosaic
                    # Initialize Mosaic helper (not as a transform pipeline but as a helper object)
                    self.mosaic_helper = Mosaic(size=train_scales_max) 
                    
            else:
                # Val/Inference Config: Rectangular Inference
                # Resize to 720 (short edge), max 1280 (long edge, keep aspect ratio)
                # 1920x1080 -> 1280x720 (16:9)
                self.transforms = T.Compose([
                    T.Resize(size=720, max_size=1280, antialias=True),
                    T.ToImage(),
                    T.ToDtype(torch.float32, scale=True),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ConvertBoxes(fmt='cxcywh', normalize=False)
                ])
        else:
            self.transforms = transforms
    
    def _load_data_info(self):
        """Load data info"""
        data_info_path = self.data_root / "metadata" / "data_info.json"
        if data_info_path.exists():
            with open(data_info_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Data info file not found: {data_info_path}")
    
    def _load_split_indices(self):
        """Load train/val split"""
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
        """Create random split (stratified by intersection location)"""
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
    
    def load_item(self, idx):
        """Load a single item without transforms (for Mosaic/Mixup)"""
        if idx >= len(self.split_indices):
            idx = idx % len(self.split_indices)
            
        actual_idx = self.split_indices[idx]
        
        # Load Image (PIL)
        image_path = self.data_root / "image" / f"{actual_idx:06d}.jpg"
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        
        # Load Annotations
        annotation_path = self.data_root / "annotations" / "camera" / f"{actual_idx:06d}.json"
        annotations = self._load_annotations(annotation_path)
        
        # Prepare Target
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
        
        # Wrap as BoundingBoxes
        boxes = BoundingBoxes(boxes, format='xyxy', canvas_size=(h, w))
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'area': areas,
            'image_id': torch.tensor([actual_idx]),
            'iscrowd': iscrowd,
            'orig_size': torch.tensor([h, w]), # H, W
        }
        
        return image, target

    def __getitem__(self, idx):
        """Get data item - returns COCO format data"""
        # 1. Load Base Item
        image, target = self.load_item(idx)
        
        # 2. Apply Mosaic / Mixup (if enabled and in training)
        if self.split == 'train' and self.transforms is not None:
            # Mosaic Augmentation
            if hasattr(self, 'aug_mosaic_prob') and self.aug_mosaic_prob > 0:
                import random
                if random.random() < self.aug_mosaic_prob:
                    # Pass self (dataset) to mosaic helper to load other images
                    # Note: self.mosaic_helper was initialized in __init__
                    if hasattr(self, 'mosaic_helper'):
                         image, target, _ = self.mosaic_helper(image, target, self)

            # Mixup Augmentation (Future Work)
            # if hasattr(self, 'aug_mixup_prob') and self.aug_mixup_prob > 0:
            #     pass

        # 3. Apply Standard Transforms
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        # Update size (transformed size)
        if isinstance(image, torch.Tensor):
            target['size'] = torch.tensor(image.shape[-2:]) # H, W
        
        return image, target
    
    def _load_annotations(self, annotation_path: Path) -> List[Dict]:
        """Load annotation file, mark ignore boxes with iscrowd=1"""
        if not annotation_path.exists():
            return []
        
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        processed_annotations = []
        for ann in annotations:
            class_name = ann["type"]
            
            # Get 2D BBox
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
                # Training: If split is train, can filter ignore boxes
                # But for consistency, we keep and mark iscrowd=1
                # SanitizeBoundingBoxes might not filter iscrowd=1
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
        
        # If training set, filter out iscrowd=1 boxes to avoid interference
        # But Transform might crop them out.
        # We keep them here, but filter in __getitem__ if needed
        # Logic: Filter ignore boxes during training
        if self.split == 'train':
            processed_annotations = [ann for ann in processed_annotations if ann['iscrowd'] == 0]
            
        return processed_annotations
    
    def get_categories(self):
        """Get category info - COCO format"""
        categories = []
        for i, class_name in enumerate(self.class_names):
            categories.append({
                'id': i + 1,
                'name': class_name,
                'supercategory': 'object'
            })
        return categories
    
    def get_image_info(self, image_id: int):
        """Get image info - COCO format"""
        if image_id < len(self.data_info):
            data_item = self.data_info[image_id]
            # Should read image for real size or use metadata
            # For simplicity, not reading image here
            return {
                'id': image_id,
                'file_name': data_item["image_path"],
                # 'width': ..., 'height': ... 
            }
        return None
