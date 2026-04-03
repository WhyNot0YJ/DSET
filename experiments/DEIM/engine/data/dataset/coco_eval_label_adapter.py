"""
Train-index labels to COCO category_id for validation.

Predictions use indices 0..N-1. Annotations use category id; order follows
coco_gt.dataset['categories'], matching CocoDetection.category2label.
Train and val JSON should list categories in the same order.
"""
from __future__ import annotations

from typing import Any, Dict

import torch

from ...core import register
from .coco_eval import CocoEvaluator


class CocoEvaluatorTrainLabelAdapter:
    """Delegates to an inner evaluator after remapping prediction labels."""

    def __init__(self, inner):
        self._inner = inner
        cats = inner.coco_gt.dataset.get("categories", [])
        self._label_to_category_id = {i: c["id"] for i, c in enumerate(cats)}

    def _map_labels(self, labels: torch.Tensor) -> torch.Tensor:
        flat = labels.flatten().tolist()
        mapped = [self._label_to_category_id.get(int(x), int(x)) for x in flat]
        return torch.tensor(mapped, device=labels.device, dtype=labels.dtype).reshape(labels.shape)

    def _remap_predictions(self, predictions: Dict[Any, Dict]) -> Dict[Any, Dict]:
        out: Dict[Any, Dict] = {}
        for image_id, pred in predictions.items():
            pred = dict(pred)
            if "labels" in pred:
                pred["labels"] = self._map_labels(pred["labels"])
            out[image_id] = pred
        return out

    @property
    def coco_gt(self):
        return self._inner.coco_gt

    @property
    def coco_eval(self):
        return self._inner.coco_eval

    @property
    def iou_types(self):
        return self._inner.iou_types

    def cleanup(self):
        return self._inner.cleanup()

    def update(self, predictions):
        return self._inner.update(self._remap_predictions(predictions))

    def synchronize_between_processes(self):
        return self._inner.synchronize_between_processes()

    def accumulate(self):
        return self._inner.accumulate()

    def summarize(self):
        return self._inner.summarize()


@register()
class CocoEvaluatorTrainLabelMapping:
    """Like CocoEvaluator, but maps prediction label indices to category ids in update."""

    def __init__(self, coco_gt, iou_types):
        self._impl = CocoEvaluatorTrainLabelAdapter(CocoEvaluator(coco_gt, iou_types))

    def __getattr__(self, name):
        return getattr(self._impl, name)
