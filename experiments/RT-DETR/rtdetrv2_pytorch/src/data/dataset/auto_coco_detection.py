"""COCO dataset adapter with auto-resolved image/annotation paths."""

from pathlib import Path

from ...core import register
from .coco_dataset import CocoDetection

__all__ = ["AutoCocoDetection"]


@register()
class AutoCocoDetection(CocoDetection):
    __inject__ = ['transforms']
    __share__ = ['remap_mscoco_category']

    def __init__(
        self,
        data_root,
        split,
        transforms,
        return_masks=False,
        remap_mscoco_category=False,
        img_folder=None,
        ann_file=None,
    ):
        root = Path(data_root)

        if ann_file is not None:
            resolved_ann = Path(ann_file)
        else:
            resolved_ann = root / "annotations" / f"instances_{split}.json"

        if img_folder is not None:
            resolved_img = Path(img_folder)
        elif (root / split).is_dir():
            resolved_img = root / split
        elif (root / "image").is_dir():
            resolved_img = root / "image"
        else:
            raise FileNotFoundError(
                f"Cannot infer image folder from data_root={root}; expected {root / split} or {root / 'image'}"
            )

        super().__init__(
            img_folder=str(resolved_img),
            ann_file=str(resolved_ann),
            transforms=transforms,
            return_masks=return_masks,
            remap_mscoco_category=remap_mscoco_category,
        )
        self.data_root = str(root)
        self.split = split
