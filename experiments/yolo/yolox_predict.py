"""YOLOX inference for KITTI/scale eval (Ultralytics-compatible result objects)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import List, Union

import cv2
import numpy as np
import torch
import torch.nn as nn

from yolox.data.data_augment import ValTransform
from yolox.utils import fuse_model, postprocess


def _torch_load_yolox_ckpt(path: Union[str, Path], map_location="cpu"):
    """Load Megvii YOLOX ``.pth`` dicts. PyTorch 2.6+ defaults ``weights_only=True``, which rejects numpy scalars in pickles."""
    p = str(path)
    try:
        return torch.load(p, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(p, map_location=map_location)


class _EvalBoxes:
    __slots__ = ("xyxy", "conf", "cls")

    def __init__(self, xyxy: torch.Tensor, conf: torch.Tensor, cls: torch.Tensor):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls


class YOLOXEvalPredictor:
    """
    Batch image paths -> list of objects with ``orig_shape`` and ``boxes`` (like Ultralytics).
    """

    def __init__(
        self,
        model: nn.Module,
        exp,
        device: Union[str, torch.device] = "cuda",
        fp16: bool = False,
    ):
        self.model = model
        self.exp = exp
        self.num_classes = exp.num_classes
        self.test_size = exp.test_size
        self.confthre = getattr(exp, "test_conf", 0.01)
        self.nmsthre = getattr(exp, "nmsthre", 0.65)
        self.device = torch.device(
            "cuda:0" if str(device).startswith("cuda") and torch.cuda.is_available() else device
        )
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=False)
        self.model.eval()
        self.model.to(self.device)
        if self.fp16 and self.device.type == "cuda":
            self.model.half()

    @torch.no_grad()
    def predict_paths(self, paths: List[Path], conf: float = 0.01) -> List[SimpleNamespace]:
        """Returns one SimpleNamespace per image with ``orig_shape`` (h,w) and ``boxes`` or None."""
        self.confthre = conf
        out: List[SimpleNamespace] = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                h, w = 1, 1
                res = SimpleNamespace(
                    orig_shape=(h, w),
                    boxes=None,
                )
                out.append(res)
                continue
            h, w = img.shape[:2]
            img_proc, _ = self.preproc(img, None, self.test_size)
            img_t = torch.from_numpy(img_proc).unsqueeze(0).float()
            if self.device.type == "cuda":
                img_t = img_t.cuda()
            if self.fp16:
                img_t = img_t.half()

            raw = self.model(img_t)
            raw = postprocess(
                raw,
                self.num_classes,
                self.confthre,
                self.nmsthre,
                class_agnostic=False,
            )
            det = raw[0] if raw else None
            if det is None or det.numel() == 0:
                res = SimpleNamespace(orig_shape=(h, w), boxes=None)
                out.append(res)
                continue

            det = det.cpu()
            ratio = min(self.test_size[0] / float(h), self.test_size[1] / float(w))
            bboxes = det[:, :4] / ratio
            scores = det[:, 4] * det[:, 5]
            clss = det[:, 6].long()

            res = SimpleNamespace(
                orig_shape=(h, w),
                boxes=_EvalBoxes(
                    xyxy=bboxes.float(),
                    conf=scores.float(),
                    cls=clss.float(),
                ),
            )
            out.append(res)
        return out


def load_yolox_for_eval(
    exp,
    ckpt_path: Union[str, Path],
    device: str = "cuda",
    fuse_bn: bool = True,
) -> YOLOXEvalPredictor:
    from yolox.utils import load_ckpt

    model = exp.get_model()
    ckpt = _torch_load_yolox_ckpt(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model = load_ckpt(model, state)
    if fuse_bn:
        model = fuse_model(model)
    fp16 = False
    return YOLOXEvalPredictor(model, exp, device=device, fp16=fp16)


def benchmark_yolox_forward_nms(model: nn.Module, exp):
    """Returns postprocess_fn for ``benchmark_model`` (forward + YOLOX NMS)."""
    num_classes = exp.num_classes
    confthre = getattr(exp, "test_conf", 0.25)
    nmsthre = getattr(exp, "nmsthre", 0.65)

    def _pp(raw_output):
        if isinstance(raw_output, (tuple, list)):
            preds = raw_output[0] if len(raw_output) > 0 else raw_output
        else:
            preds = raw_output
        return postprocess(preds, num_classes, confthre, nmsthre, class_agnostic=False)

    return _pp
