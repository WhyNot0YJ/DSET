"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import default_collate

import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as VT
from torchvision.transforms.v2 import functional as VF, InterpolationMode

import random
from functools import partial
from typing import List, Dict, Any

from ..core import register


__all__ = [
    'DataLoader',
    'BaseCollateFunction', 
    'BatchImageCollateFuncion',
    'batch_image_collate_fn'
]


@register()
class DataLoader(data.DataLoader):
    __inject__ = ['dataset', 'collate_fn']

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ['dataset', 'batch_size', 'num_workers', 'drop_last', 'collate_fn']:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string

    def set_epoch(self, epoch):
        self._epoch = epoch 
        self.dataset.set_epoch(epoch)
        self.collate_fn.set_epoch(epoch)
    
    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        assert isinstance(shuffle, bool), 'shuffle must be a boolean'
        self._shuffle = shuffle


@register()
def batch_image_collate_fn(items):
    """only batch image
    """
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]


class BaseCollateFunction(object):
    def set_epoch(self, epoch):
        self._epoch = epoch 

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    def __call__(self, items):
        raise NotImplementedError('')


@register()
class BatchImageCollateFuncion(BaseCollateFunction):
    def __init__(
        self, 
        scales=None, 
        stop_epoch=None, 
    ) -> None:
        super().__init__()
        self.scales = scales
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
        # self.interpolation = interpolation

    @staticmethod
    def _adjust_targets_after_resize(
        targets: List[Dict[str, Any]],
        old_h: int,
        old_w: int,
        new_h: int,
        new_w: int,
    ) -> None:
        """与 ``F.interpolate`` 后的张量对齐 targets。

        - 归一化 ``cxcywh``（相对当前输入张量）在均匀宽高缩放下方阵上不变，不修改 ``boxes``。
        - ``letterbox_pad``、``letterbox_scale`` 为像素/几何量，需按缩放比例更新，否则
          训练里用 ``orig_size`` + letterbox 反算或验证后处理会错位。
        """
        if old_h == new_h and old_w == new_w:
            return
        sx = new_w / float(old_w)
        sy = new_h / float(old_h)
        for tg in targets:
            if not tg:
                continue
            pad = tg.get("letterbox_pad")
            if pad is not None and isinstance(pad, torch.Tensor) and pad.numel() >= 2:
                p = pad.detach().float().clone()
                p[0] *= sx
                p[1] *= sy
                tg["letterbox_pad"] = p.to(device=pad.device, dtype=pad.dtype)
            sc = tg.get("letterbox_scale")
            if sc is not None and isinstance(sc, torch.Tensor):
                # 配置里多尺度为方形 sz×sz 时 sx==sy；非方形时用几何平均近似各向同性 letterbox 尺度
                factor = (sx * sy) ** 0.5
                tg["letterbox_scale"] = (sc.float() * factor).to(
                    device=sc.device, dtype=sc.dtype
                )
            sz_t = tg.get("size")
            if sz_t is not None and isinstance(sz_t, torch.Tensor):
                tg["size"] = torch.tensor(
                    [new_h, new_w], dtype=sz_t.dtype, device=sz_t.device
                )

    def __call__(self, items):
        images = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]

        if self.scales is not None and self.epoch < self.stop_epoch:
            sz = random.choice(self.scales)
            # Ensure sz is (h, w)
            if isinstance(sz, int):
                sz = (sz, sz)
            old_h, old_w = int(images.shape[2]), int(images.shape[3])
            images = F.interpolate(images, size=sz, mode="bilinear", align_corners=False)
            new_h, new_w = int(images.shape[2]), int(images.shape[3])
            self._adjust_targets_after_resize(targets, old_h, old_w, new_h, new_w)

        return images, targets

