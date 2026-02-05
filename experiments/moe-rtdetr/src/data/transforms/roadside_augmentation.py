"""
适合路测/监控数据集的数据增强
专注于颜色、光照等不影响空间关系的增强
"""

import torch
import torchvision.transforms.v2 as T
from typing import Optional, Dict, Any
import numpy as np


class RoadsideAugmentation:
    """
    路测数据集数据增强
    包含颜色、光照等增强，不破坏空间关系
    """
    
    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
        color_jitter_prob: float = 0.0,
    ):
        """
        初始化路测数据增强
        
        Args:
            brightness: 亮度调整范围 (0.0 = 不增强, 建议 0.1-0.2)
            contrast: 对比度调整范围 (0.0 = 不增强, 建议 0.1-0.2)
            saturation: 饱和度调整范围 (0.0 = 不增强, 建议 0.1-0.2)
            hue: 色相调整范围 (0.0 = 不增强, 建议 0.05-0.1)
            color_jitter_prob: 应用颜色抖动的概率 (0.0 = 不应用, 建议 0.5-0.8)
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.color_jitter_prob = color_jitter_prob
        
        # 构建增强管道
        transforms = []
        
        # 颜色抖动（如果启用）
        if color_jitter_prob > 0 and (brightness > 0 or contrast > 0 or saturation > 0 or hue > 0):
            color_jitter = T.ColorJitter(
                brightness=(max(0, 1 - brightness), 1 + brightness) if brightness > 0 else None,
                contrast=(max(0, 1 - contrast), 1 + contrast) if contrast > 0 else None,
                saturation=(max(0, 1 - saturation), 1 + saturation) if saturation > 0 else None,
                hue=(-hue, hue) if hue > 0 else None,
            )
            transforms.append(T.RandomApply([color_jitter], p=color_jitter_prob))
        
        self.transforms = T.Compose(transforms) if transforms else None
    
    def __call__(self, image: torch.Tensor, target: Dict[str, Any]) -> tuple:
        """
        应用数据增强
        
        Args:
            image: 输入图像 (torch.Tensor, CHW 格式)
            target: 目标字典（包含 boxes, labels 等）
        
        Returns:
            (增强后的图像, 目标字典)
        """
        if self.transforms is not None:
            # 确保图像是 torch.Tensor 且格式正确
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).float()
            
            # 如果是 CHW 格式，需要转换为 PIL Image 或使用 torchvision transforms
            # torchvision.transforms.v2 可以直接处理 CHW 格式的 tensor
            image, target = self.transforms(image, target)
        
        return image, target
    
    def __repr__(self):
        return (f"RoadsideAugmentation("
                f"brightness={self.brightness}, "
                f"contrast={self.contrast}, "
                f"saturation={self.saturation}, "
                f"hue={self.hue}, "
                f"color_jitter_prob={self.color_jitter_prob})")

