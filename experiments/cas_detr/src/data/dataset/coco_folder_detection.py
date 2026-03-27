"""
COCO 检测数据集：按 split 子目录存放图像（如 UA-DETRAC_COCO/train、val、test）。
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchvision.transforms.v2 as T
from PIL import Image

from ._dataset import DetDataset
from .._misc import BoundingBoxes
from ...core import register
from ..transforms import (
    RandomPhotometricDistort,
    RandomHorizontalFlip,
    ConvertBoxes,
    Normalize,
    SanitizeBoundingBoxes,
    RandomZoomOut,
    RandomIoUCrop,
    ConvertPILImage,
    build_square_input_transform,
)

__all__ = ["CocoFolderDetection"]


@register()
class CocoFolderDetection(DetDataset):
    """COCO 格式；图像位于 data_root/<split>/<file_name>。"""

    DEFAULT_AUGMENTATION_CONFIG = {
        "target_size": 640,
        "stop_epoch": 21,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "photometric_distort_p": 0.5,
        "zoom_out_enabled": True,
        "iou_crop_p": 0.8,
        "horizontal_flip_p": 0.5,
        "letterbox_fill": 114,
    }

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transforms=None,
        target_size: int = 640,
        stop_epoch: int = 21,
        augmentation_config: Dict = None,
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.split = split

        self.aug_config = self.DEFAULT_AUGMENTATION_CONFIG.copy()
        if augmentation_config is not None:
            self.aug_config.update(augmentation_config)
        if target_size != 640:
            self.aug_config["target_size"] = target_size
        if stop_epoch != 21:
            self.aug_config["stop_epoch"] = stop_epoch

        self.target_size = self.aug_config["target_size"]
        self.stop_epoch = self.aug_config["stop_epoch"]

        ann_path = self.data_root / "annotations" / f"instances_{split}.json"
        if not ann_path.exists():
            raise FileNotFoundError(f"instances 标注不存在: {ann_path}")

        with open(ann_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        cats = sorted(coco_data.get("categories", []), key=lambda c: int(c["id"]))
        self.class_names = [str(c["name"]) for c in cats]
        self.cat_id_to_class = {int(c["id"]): i for i, c in enumerate(cats)}

        self._images: List[Dict] = list(coco_data.get("images", []))
        self._coco_id_to_path: Dict[int, Path] = {}
        for im in self._images:
            cid = int(im["id"])
            fn = im.get("file_name", "")
            self._coco_id_to_path[cid] = self.data_root / split / fn

        self.annotations_by_coco_image_id: Dict[int, List] = {}
        for ann in coco_data.get("annotations", []):
            iid = int(ann.get("image_id", -1))
            if iid < 0:
                continue
            self.annotations_by_coco_image_id.setdefault(iid, []).append(ann)

        self.set_epoch(0)
        self._init_transforms()

    def _init_transforms(self):
        target_size = self.aug_config["target_size"]
        normalize_mean = self.aug_config["normalize_mean"]
        normalize_std = self.aug_config["normalize_std"]
        photometric_distort_p = self.aug_config["photometric_distort_p"]
        zoom_out_enabled = self.aug_config["zoom_out_enabled"]
        iou_crop_p = self.aug_config["iou_crop_p"]
        horizontal_flip_p = self.aug_config["horizontal_flip_p"]
        to_square = build_square_input_transform(self.aug_config)

        if self.split == "train":
            if self.epoch >= self.stop_epoch:
                self.transforms = T.Compose(
                    [
                        to_square,
                        ConvertPILImage(),
                        T.ToDtype(torch.float32, scale=True),
                        Normalize(mean=normalize_mean, std=normalize_std),
                        ConvertBoxes(fmt="cxcywh", normalize=True),
                    ]
                )
            else:
                transforms_list = [RandomPhotometricDistort(p=photometric_distort_p)]
                if zoom_out_enabled:
                    transforms_list.append(RandomZoomOut(fill=0))
                transforms_list.extend(
                    [
                        RandomIoUCrop(p=iou_crop_p),
                        SanitizeBoundingBoxes(),
                        RandomHorizontalFlip(p=horizontal_flip_p),
                    ]
                )
                transforms_list.extend(
                    [
                        to_square,
                        ConvertPILImage(),
                        T.ToDtype(torch.float32, scale=True),
                        Normalize(mean=normalize_mean, std=normalize_std),
                        ConvertBoxes(fmt="cxcywh", normalize=True),
                    ]
                )
                self.transforms = T.Compose(transforms_list)
        else:
            self.transforms = T.Compose(
                [
                    to_square,
                    ConvertPILImage(),
                    T.ToDtype(torch.float32, scale=True),
                    Normalize(mean=normalize_mean, std=normalize_std),
                    ConvertBoxes(fmt="cxcywh", normalize=True),
                ]
            )

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self._init_transforms()

    def __len__(self):
        return len(self._images)

    def get_image_path(self, coco_image_id: int) -> Optional[Path]:
        p = self._coco_id_to_path.get(int(coco_image_id))
        if p is None:
            return None
        p = Path(p)
        return p if p.exists() else None

    def load_item(self, idx):
        if idx >= len(self._images):
            idx = idx % len(self._images)
        img_info = self._images[idx]
        coco_id = int(img_info["id"])
        image_path = self._coco_id_to_path[coco_id]
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        annotations = self._load_annotations(coco_id)
        if len(annotations) > 0:
            boxes = torch.tensor([ann["bbox"] for ann in annotations], dtype=torch.float32)
            labels = torch.tensor([ann["class_id"] for ann in annotations], dtype=torch.int64)
            areas = torch.tensor([ann["area"] for ann in annotations], dtype=torch.float32)
            iscrowd = torch.tensor([ann.get("iscrowd", 0) for ann in annotations], dtype=torch.int64)
            # float32：UA-DETRAC 为连续比例，DAIR-V2X 为整数类别，均可精确表示
            occluded_states = torch.tensor(
                [ann.get("occluded_state", 0.0) for ann in annotations], dtype=torch.float32
            )
            truncated_states = torch.tensor(
                [ann.get("truncated_state", 0.0) for ann in annotations], dtype=torch.float32
            )
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
            occluded_states = torch.zeros((0,), dtype=torch.float32)
            truncated_states = torch.zeros((0,), dtype=torch.float32)

        boxes = BoundingBoxes(boxes, format="xyxy", canvas_size=(h, w))
        target = {
            "boxes": boxes,
            "labels": labels,
            "area": areas,
            "image_id": torch.tensor([coco_id]),
            "iscrowd": iscrowd,
            "occluded_state": occluded_states,
            "truncated_state": truncated_states,
            "orig_size": torch.tensor([h, w]),
        }
        return image, target

    def __getitem__(self, idx):
        image, target = self.load_item(idx)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        if isinstance(image, torch.Tensor):
            target["size"] = torch.tensor(image.shape[-2:])
        return image, target

    def _load_annotations(self, coco_image_id: int) -> List[Dict]:
        raw = self.annotations_by_coco_image_id.get(int(coco_image_id), [])
        out: List[Dict] = []
        for ann in raw:
            cat_id = int(ann.get("category_id", 0))
            if cat_id not in self.cat_id_to_class:
                continue
            class_id = self.cat_id_to_class[cat_id]
            bbox_xywh = ann.get("bbox", [0, 0, 0, 0])
            if len(bbox_xywh) != 4:
                continue
            x1, y1 = float(bbox_xywh[0]), float(bbox_xywh[1])
            w, h = float(bbox_xywh[2]), float(bbox_xywh[3])
            x2, y2 = x1 + w, y1 + h
            if x2 <= x1 or y2 <= y1:
                continue
            area = float(ann.get("area", w * h))
            iscrowd = 1 if int(ann.get("iscrowd", 0)) else 0

            # ── occlusion ──
            # DAIR-V2X: "occluded_state"  (离散类别 0/1/2)
            # UA-DETRAC: "occlusion_status" (离散 0/1)
            # 统一存为 float；下游 _normalize_occlusion_level 自动识别：
            #   值 >= 1.0 → 按类别等级处理，值 < 1.0 → 按比例区间处理
            if "occluded_state" in ann:
                occ_val = float(ann["occluded_state"])
            elif "occlusion_status" in ann:
                occ_val = float(ann["occlusion_status"])
            else:
                occ_val = 0.0

            # ── truncation ──
            # DAIR-V2X: "truncated_state"  (离散类别 0/1/2; dair_categorical=True 路径)
            # UA-DETRAC: "truncation_ratio" (连续比例 0.0~1.0; dair_categorical=False 路径)
            # 下游 normalize_truncation 根据 _dair_truncation_categorical 标志选择规则
            if "truncated_state" in ann:
                trunc_val = float(ann["truncated_state"])
            elif "truncation_ratio" in ann:
                trunc_val = float(ann["truncation_ratio"])
            else:
                trunc_val = 0.0

            out.append(
                {
                    "class_id": class_id,
                    "bbox": [x1, y1, x2, y2],
                    "area": area,
                    "iscrowd": iscrowd,
                    "occluded_state": occ_val,
                    "truncated_state": trunc_val,
                }
            )
        if self.split == "train":
            out = [a for a in out if a["iscrowd"] == 0]
        return out

    def get_categories(self):
        categories = []
        for i, class_name in enumerate(self.class_names):
            categories.append({
                "id": i + 1,
                "name": class_name,
                "supercategory": "object",
            })
        return categories
