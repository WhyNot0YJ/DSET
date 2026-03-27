#!/usr/bin/env python3
"""Shared YOLOX Exp base for CaS_DETR COCO-style datasets (train/val/test image folders)."""

import os

import cv2

from yolox.data import COCODataset, TrainTransform, ValTransform
from yolox.exp import Exp as MyExp


class DairCocoDataset(COCODataset):
    """
    DAIR-V2X COCO images are stored directly under ``data_dir/image/*.jpg``.
    Their ``file_name`` already includes the ``image/`` prefix, so we should not
    prepend ``train/val/test`` again.
    """

    def load_image(self, index):
        file_name = self.annotations[index][3]
        img_file = os.path.join(self.data_dir, file_name)

        img = cv2.imread(img_file)
        assert img is not None, f"file named {img_file} not found"
        return img


class CasYoloxExp(MyExp):
    """
    Overrides COCO image folder names: ``train``, ``val``, ``test`` (not COCO2017 defaults).
    Subclasses set ``num_classes`` and optionally ``train_ann`` / ``val_ann`` / ``test_ann``.
    """

    def __init__(self):
        super().__init__()
        self.depth = 0.33
        self.width = 0.50
        self.train_ann = "instances_train.json"
        self.val_ann = "instances_val.json"
        self.test_ann = "instances_test.json"
        self.image_layout = "split_subdir"

    def _dataset_cls(self):
        return DairCocoDataset if self.image_layout == "flat_image_dir" else COCODataset

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        return self._dataset_cls()(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="train",
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", False)
        json_file = self.test_ann if testdev else self.val_ann
        name = "test" if testdev else "val"
        return self._dataset_cls()(
            data_dir=self.data_dir,
            json_file=json_file,
            name=name,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )
