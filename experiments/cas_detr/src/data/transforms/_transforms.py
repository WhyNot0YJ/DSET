""""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 

import torchvision
torchvision.disable_beta_transforms_warning()

import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

import PIL
import PIL.Image

from typing import Any, Dict, List, Optional, Union

from .._misc import convert_to_tv_tensor, _boxes_keys
from .._misc import Image, Video, Mask, BoundingBoxes
from .._misc import SanitizeBoundingBoxes
from ...core import register

from .letterbox_geom import compute_letterbox_layout


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
SanitizeBoundingBoxes = register(name='SanitizeBoundingBoxes')(SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


@register()
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )

    @staticmethod
    def _spatial_size(inpt: Any):
        get_spatial_size = getattr(F, 'get_spatial_size', None)
        if callable(get_spatial_size):
            return get_spatial_size(inpt)

        get_size = getattr(F, 'get_size', None)
        if callable(get_size):
            size = get_size(inpt)
            return size[0], size[1]

        if isinstance(inpt, PIL.Image.Image):
            w, h = inpt.size
            return h, w

        if hasattr(inpt, 'shape') and len(inpt.shape) >= 2:
            return int(inpt.shape[-2]), int(inpt.shape[-1])

        raise TypeError(f'Unsupported input type for spatial size: {type(inpt)}')

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = self._spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return self._get_params(flat_inputs)

    def __init__(self, size, fill=0, padding_mode='constant') -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _resolve_fill(self, inpt: Any):
        if isinstance(self._fill, dict):
            for cls in type(inpt).__mro__:
                if cls in self._fill:
                    return self._fill[cls]
        return self.fill

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:        
        fill = self._resolve_fill(inpt)
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p 

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (
        BoundingBoxes,
    )
    def __init__(self, fmt='', normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:  
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(inpt, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)
            
        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (
        PIL.Image.Image,
    )
    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:  
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == 'float32':
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.

        inpt = Image(inpt)

        return inpt

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)

@register()
class RandomResize(T.Transform):
    def __init__(self, scales, max_size=None, antialias=True):
        super().__init__()
        self.scales = scales
        self.max_size = max_size
        self.antialias = antialias

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        import random
        target_size = random.choice(self.scales)
        return T.Resize(target_size, max_size=self.max_size, antialias=self.antialias)(inputs)


@register()
class LetterboxResize(torch.nn.Module):
    """等比缩放后居中 pad 到固定方形画布（letterbox），避免 16:9 等比例被拉成正方形产生畸变。

    在 target 中写入 ``letterbox_scale``、``letterbox_pad``（左上 padding，单位像素），供验证 / 推理阶段
    将预测框映射回原图分辨率。
    """

    def __init__(self, size: int, fill: Union[int, float] = 0, antialias: bool = True) -> None:
        super().__init__()
        self.size = int(size)
        self.fill = int(fill)
        self.antialias = antialias

    def forward(self, *inputs: Any) -> Any:
        image, target = inputs
        tw = th = self.size
        w, h = image.size

        L = compute_letterbox_layout(w, h, self.size)
        scale = L['scale']
        new_w, new_h = int(L['new_w']), int(L['new_h'])
        pad_left = int(L['pad_left'])
        pad_right = int(L['pad_right'])
        pad_top = int(L['pad_top'])
        pad_bottom = int(L['pad_bottom'])

        image = F.resize(image, [new_h, new_w], antialias=self.antialias)

        if 'boxes' in target and isinstance(target['boxes'], BoundingBoxes):
            boxes_t = target['boxes'].clone()
            if boxes_t.numel() > 0:
                boxes_t = boxes_t * scale
            target['boxes'] = convert_to_tv_tensor(
                boxes_t, key='boxes', box_format='XYXY', spatial_size=(new_h, new_w)
            )

        image = F.pad(image, [pad_left, pad_top, pad_right, pad_bottom], fill=self.fill, padding_mode='constant')

        if 'boxes' in target and isinstance(target['boxes'], BoundingBoxes):
            boxes_t = target['boxes'].clone()
            if boxes_t.numel() > 0:
                off = boxes_t.new_tensor([pad_left, pad_top, pad_left, pad_top])
                boxes_t = boxes_t + off
            target['boxes'] = convert_to_tv_tensor(
                boxes_t, key='boxes', box_format='XYXY', spatial_size=(th, tw)
            )

        target['letterbox_scale'] = torch.tensor(scale, dtype=torch.float32)
        target['letterbox_pad'] = torch.tensor(
            [float(pad_left), float(pad_top)], dtype=torch.float32
        )
        return image, target
