"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.utils.data as data


class DetDataset(data.Dataset):
    def __getitem__(self, index):
        img, target = self.load_item(index)
        if self.transforms is not None:
            img, target, _ = self.transforms(img, target, self)
        return img, target

    def load_item(self, index):
        raise NotImplementedError("Please implement this function to return item before `transforms`.")

    def get_image_path(self, coco_image_id: int) -> Optional[Path]:
        """由 COCO ``image_id`` 解析磁盘上的原图路径（训练 Token 热力图、推理落盘等）。

        不同数据集目录布局不同（如 ``data_root/<split>/file_name`` 与 ``data_root/image/000xxx.jpg``），
        仅在此处实现映射，调用方统一走本接口即可。
        """
        return None

    def set_epoch(self, epoch) -> None:
        self._epoch = epoch 

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1
