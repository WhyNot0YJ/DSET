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

from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor
from ...core import register

__all__ = ['DAIRV2XDetection']

@register()
class DAIRV2XDetection(DetDataset):
    """DAIR-V2X数据集检测器 - 支持COCO格式评估"""
    
    def __init__(self, data_root: str, split: str = "train", transforms=None, 
                 use_mosaic: bool = False, target_size: int = 640):
        """
        初始化DAIR-V2X数据集
        
        Args:
            data_root: 数据集根目录
            split: 数据集分割 ('train' 或 'val')
            transforms: 数据变换
            use_mosaic: 是否使用Mosaic数据增强
            target_size: 目标图像尺寸
        """
        super().__init__()
        
        self.data_root = Path(data_root)
        self.split = split
        self.transforms = transforms
        self.use_mosaic = use_mosaic and split == "train"
        self.target_size = target_size
        
        # DAIR-V2X类别定义（8类：前7类是交通参与者，Trafficcone是道路设施）
        # 注意：Tricyclist 数据集中不存在，Barrowlist 样本过少已合并到 Cyclist
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
        
        # 初始化Mosaic增强
        if self.use_mosaic:
            from ..transforms.mosaic import Mosaic
            self.mosaic_transform = Mosaic(size=target_size, max_size=target_size)
            print(f"启用Mosaic数据增强 for {split} split")
        else:
            self.mosaic_transform = None
    
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
                # 将字符串索引转换为整数，并检查对应的图片文件是否存在
                valid_indices = []
                for idx in indices:
                    idx_int = int(idx)
                    # 检查对应的图片文件是否存在
                    image_path = self.data_root / "image" / f"{idx_int:06d}.jpg"
                    if image_path.exists():
                        valid_indices.append(idx_int)
                return valid_indices
        else:
            # 如果没有分割文件，使用随机分割（按路口分层随机分割）
            return self._create_random_split()
    
    def _create_random_split(self):
        """创建随机分割（按路口分层随机分割）"""
        import random
        from collections import defaultdict
        
        # 按路口分组
        intersection_groups = defaultdict(list)
        for idx, item in enumerate(self.data_info):
            loc = item.get('intersection_loc', 'Unknown')
            intersection_groups[loc].append(idx)
        
        # 设置随机种子（使用固定的种子42，确保可复现）
        random.seed(42)
        
        # 对每个路口的数据进行随机分割
        train_indices = []
        val_indices = []
        
        for loc, indices in intersection_groups.items():
            # 打乱当前路口的数据
            shuffled = indices.copy()
            random.shuffle(shuffled)
            
            # 按比例分割（80%训练，20%验证）
            split_point = int(len(shuffled) * 0.8)
            train_indices.extend(shuffled[:split_point])
            val_indices.extend(shuffled[split_point:])
        
        # 打乱最终结果（可选，保持随机性）
        random.shuffle(train_indices)
        random.shuffle(val_indices)
        
        # 根据 split 返回对应的索引
        if self.split == "train":
            return train_indices
        else:
            return val_indices
    
    def __len__(self):
        return len(self.split_indices)
    
    def __getitem__(self, idx):
        """获取数据项 - 返回COCO格式的数据"""
        # 获取实际的图片索引
        actual_idx = self.split_indices[idx]
        
        # 直接构建图像路径（不依赖data_info）
        image_path = self.data_root / "image" / f"{actual_idx:06d}.jpg"
        image = self._load_image(image_path)
        
        # 加载标注
        annotation_path = self.data_root / "annotations" / "camera" / f"{actual_idx:06d}.json"
        annotations = self._load_annotations(annotation_path)
        
        # 预处理图像和标注
        processed_image, (scale_h, scale_w), pad_h, pad_w, new_h, new_w = self._preprocess_image(image)
        processed_annotations = self._adjust_annotations(annotations, scale_h, scale_w, pad_h, pad_w, new_h, new_w)
        
        # 方案B：训练时过滤ignore框，评估时保留（标记为iscrowd=1）
        if self.split == 'train':
            # 训练时：完全丢弃ignore框，不传给模型
            valid_annotations = [ann for ann in processed_annotations if ann.get('iscrowd', 0) == 0]
        else:
            # 评估时：保留所有框（包括ignore），让COCOeval自动处理
            valid_annotations = processed_annotations
        
        if len(valid_annotations) > 0:
            boxes = torch.stack([torch.tensor(ann['bbox']) for ann in valid_annotations])
            labels = torch.stack([torch.tensor(ann['class_id']) for ann in valid_annotations])
            areas = torch.stack([torch.tensor(ann['area']) for ann in valid_annotations])
            iscrowd = torch.stack([torch.tensor(ann.get('iscrowd', 0)) for ann in valid_annotations])
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        
        target = {
            'image_id': torch.tensor([actual_idx]),
            'boxes': boxes,
            'labels': labels,
            'area': areas,
            'orig_size': torch.tensor([image.shape[0], image.shape[1]]),
            'size': torch.tensor([self.target_size, self.target_size])
        }
        
        # 评估时保留iscrowd字段，让COCOeval自动处理
        if self.split != 'train':
            target['iscrowd'] = iscrowd
        
        # 返回预处理后的图像（processed_image），而不是原始图
        output_image = processed_image
        # 应用数据变换（若有）
        if self.transforms is not None:
            output_image, target, _ = self.transforms(output_image, target, self)
        
        return output_image, target
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """加载图像"""
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        # 使用OpenCV加载图像（BGR格式）
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        # 转换为RGB格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def _load_annotations(self, annotation_path: Path) -> List[Dict]:
        """加载标注文件，Ignore框标记为iscrowd=1"""
        if not annotation_path.exists():
            return []
        
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        processed_annotations = []
        for ann in annotations:
            # 获取类别（直接使用数据集中的类别名称，不转小写）
            class_name = ann["type"]  # 保持原始大小写：Car, Pedestrian, etc.
            
            # 获取2D边界框
            bbox_2d = ann["2d_box"]
            x1 = float(bbox_2d["xmin"])
            y1 = float(bbox_2d["ymin"])
            x2 = float(bbox_2d["xmax"])
            y2 = float(bbox_2d["ymax"])
            
            # 检查边界框是否有效
            if x2 <= x1 or y2 <= y1:
                continue
            
            # 转换为COCO格式 [x, y, w, h]
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            # 处理Ignore类别：标记为iscrowd=1
            if class_name in self.ignore_classes:
                processed_annotations.append({
                    'id': len(processed_annotations) + 1,
                    'class_id': 0,
                    'bbox': [x1, y1, width, height],
                    'area': area,
                    'iscrowd': 1
                })
            # 处理类别合并：Barrowlist 合并到 Cyclist
            elif class_name in self.class_merge_map:
                class_id = self.class_merge_map[class_name]
                processed_annotations.append({
                    'id': len(processed_annotations) + 1,
                    'class_id': class_id,  # 映射到 Cyclist (ID=5)
                    'bbox': [x1, y1, width, height],  # COCO格式
                    'area': area,
                    'iscrowd': 0  # 正常检测目标
                })
            # 处理正式检测类别
            elif class_name in self.class_to_id:
                class_id = self.class_to_id[class_name]
                processed_annotations.append({
                    'id': len(processed_annotations) + 1,
                    'class_id': class_id,  # 直接存储 class_id (0-7)，训练时使用
                    'bbox': [x1, y1, width, height],  # COCO格式
                    'area': area,
                    'iscrowd': 0  # 正常检测目标
                })
            # 跳过未知类别
            else:
                continue
        
        return processed_annotations
    
    def _preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[float, float], int, int, int, int]:
        """预处理图像：保持宽高比缩放到目标尺寸，确保尺寸是32的倍数"""
        H, W, C = image.shape
        
        # 计算缩放比例（保持宽高比）
        scale = min(self.target_size / H, self.target_size / W)
        new_h, new_w = int(H * scale), int(W * scale)
        
        # 确保缩放后的尺寸是32的倍数（backbone需要）
        new_h = ((new_h + 31) // 32) * 32
        new_w = ((new_w + 31) // 32) * 32
        
        # 确保不超过目标尺寸
        new_h = min(new_h, self.target_size)
        new_w = min(new_w, self.target_size)
        
        # 验证尺寸是32的倍数
        assert new_h % 32 == 0, f"new_h ({new_h}) 不是32的倍数"
        assert new_w % 32 == 0, f"new_w ({new_w}) 不是32的倍数"
        assert self.target_size % 32 == 0, f"target_size ({self.target_size}) 不是32的倍数"
        
        # 转换为tensor进行缩放 - 使用更高效的插值
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)  # HWC -> CHW
        resized_image = torch.nn.functional.interpolate(
            image_tensor.unsqueeze(0), 
            size=(new_h, new_w), 
            mode='bilinear', 
            align_corners=False,
            antialias=False  # 关闭抗锯齿以加速
        ).squeeze(0)
        
        # 创建填充后的图像，并按偏移量将缩放图像居中放置
        padded_image = torch.zeros(C, self.target_size, self.target_size, dtype=resized_image.dtype)
        # 计算填充偏移
        pad_h = (self.target_size - new_h) // 2
        pad_w = (self.target_size - new_w) // 2
        padded_image[:, pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized_image
        
        # ⚠️ 关键修复：返回实际的缩放比例（考虑32倍数调整）
        # 使用调整后的 new_h 和 new_w 计算实际缩放比例，确保坐标转换一致
        actual_scale_h = new_h / H
        actual_scale_w = new_w / W
        
        return padded_image, (actual_scale_h, actual_scale_w), pad_h, pad_w, new_h, new_w
    
    def _adjust_annotations(self, annotations: List[Dict], scale_h: float, scale_w: float, 
                           pad_h: int, pad_w: int, new_h: int, new_w: int) -> List[Dict]:
        """调整标注坐标以匹配预处理后的图像
        
        转换为RT-DETR期望的格式：
        - 格式：[cx, cy, w, h] (中心点坐标)
        - 归一化：相对于target_size归一化到[0, 1]
        
        Args:
            scale_h: 高度的实际缩放比例 (new_h / orig_h)
            scale_w: 宽度的实际缩放比例 (new_w / orig_w)
            pad_h: 高度方向的填充偏移
            pad_w: 宽度方向的填充偏移
            new_h: resize后的实际高度
            new_w: resize后的实际宽度
        """
        adjusted_annotations = []
        
        for ann in annotations:
            bbox = ann['bbox']  # [x, y, w, h] COCO格式（左上角）
            x, y, w, h = bbox
            
            # ⚠️ 关键修复：使用实际的缩放比例（考虑32倍数调整）
            # 分别使用 scale_w 和 scale_h，确保坐标转换准确
            x *= scale_w
            y *= scale_h
            w *= scale_w
            h *= scale_h
            
            # 添加填充偏移
            x += pad_w
            y += pad_h
            
            # 确保坐标在有效范围内
            x = max(0, min(self.target_size - 1, x))
            y = max(0, min(self.target_size - 1, y))
            w = max(1, min(self.target_size - x, w))
            h = max(1, min(self.target_size - y, h))
            
            # 转换为RT-DETR格式：[cx, cy, w, h]并归一化
            cx = (x + w / 2) / self.target_size  # 中心x，归一化到[0,1]
            cy = (y + h / 2) / self.target_size  # 中心y，归一化到[0,1]
            w_norm = w / self.target_size        # 宽度，归一化到[0,1]
            h_norm = h / self.target_size        # 高度，归一化到[0,1]
            
            # 更新标注（RT-DETR格式）
            adjusted_ann = ann.copy()
            adjusted_ann['bbox'] = [cx, cy, w_norm, h_norm]  # 归一化的中心点坐标
            adjusted_ann['area'] = w * h  # area保持像素值（用于评估）
            
            adjusted_annotations.append(adjusted_ann)
        
        return adjusted_annotations
    
    def get_categories(self):
        """获取类别信息 - COCO格式"""
        categories = []
        for i, class_name in enumerate(self.class_names):
            categories.append({
                'id': i + 1,  # COCO类别从1开始
                'name': class_name,
                'supercategory': 'object'
            })
        return categories
    
    def get_image_info(self, image_id: int):
        """获取图像信息 - COCO格式"""
        if image_id < len(self.data_info):
            data_item = self.data_info[image_id]
            return {
                'id': image_id,
                'file_name': data_item["image_path"],
                'width': self.target_size,
                'height': self.target_size
            }
        return None
