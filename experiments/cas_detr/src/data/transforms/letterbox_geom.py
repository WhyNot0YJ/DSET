"""Shared letterbox geometry: training (LetterboxResize), inference, and map alignment."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import torch


def compute_letterbox_layout(width: int, height: int, target_size: int) -> Dict[str, Union[int, float]]:
    """PIL-style width/height → scale, resized size, symmetric padding into target_size square.

    Matches ``LetterboxResize`` / ``preprocess_image`` (center pad).
    """
    tw = th = int(target_size)
    scale = min(tw / float(width), th / float(height))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    dw, dh = tw - new_w, th - new_h
    pad_left = dw // 2
    pad_right = dw - pad_left
    pad_top = dh // 2
    pad_bottom = dh - pad_top
    return {
        "scale": float(scale),
        "new_w": new_w,
        "new_h": new_h,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "pad_right": pad_right,
        "pad_bottom": pad_bottom,
        "padded_w": tw,
        "padded_h": th,
    }


def _tensor_scalar(x: Union[torch.Tensor, float, int]) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.reshape(-1)[0].item())
    return float(x)


def build_letterbox_meta_for_postprocess(
    target: Dict[str, Any],
    input_h: int,
    input_w: int,
) -> Dict[str, Any]:
    """Build ``meta`` dict for ``batch_inference.postprocess_outputs`` from a collated target."""
    orig = target["orig_size"]
    if isinstance(orig, torch.Tensor):
        orig_h, orig_w = float(orig[0].item()), float(orig[1].item())
    else:
        orig_h, orig_w = float(orig[0]), float(orig[1])

    meta: Dict[str, Any] = {
        "orig_size": torch.tensor([[int(orig_h), int(orig_w)]], dtype=torch.float32),
        "padded_h": int(input_h),
        "padded_w": int(input_w),
        "letterbox_uniform": True,
    }

    lb_pad = target.get("letterbox_pad")
    lb_scale = target.get("letterbox_scale")
    if lb_pad is not None and lb_scale is not None:
        meta["pad_left"] = _tensor_scalar(lb_pad[0])
        meta["pad_top"] = _tensor_scalar(lb_pad[1])
        meta["scale"] = _tensor_scalar(lb_scale)
        meta["new_w"] = max(1, int(round(orig_w * meta["scale"])))
        meta["new_h"] = max(1, int(round(orig_h * meta["scale"])))
        return meta

    # Legacy: input assumed to fill the tensor (non-letterbox stretch / no meta)
    meta["letterbox_uniform"] = False
    meta["pad_left"] = 0.0
    meta["pad_top"] = 0.0
    meta["scale_h"] = float(input_h) / orig_h if orig_h > 0 else 1.0
    meta["scale_w"] = float(input_w) / orig_w if orig_w > 0 else 1.0
    return meta


def align_feature_map_to_original_np(
    map_2d: np.ndarray,
    h_feat: int,
    w_feat: int,
    H_tensor: int,
    W_tensor: int,
    orig_h: int,
    orig_w: int,
    pad_left: Optional[float] = None,
    pad_top: Optional[float] = None,
    new_h: Optional[int] = None,
    new_w: Optional[int] = None,
    *,
    normalize_before_resize: bool = False,
    interp_tensor: int = cv2.INTER_LINEAR,
    interp_orig: int = cv2.INTER_CUBIC,
) -> np.ndarray:
    """Upsample a feature/grid map to input tensor space, crop letterbox content, resize to original image.

    If pad/inner sizes are omitted, treats the full ``H_tensor``×``W_tensor`` as valid content (legacy).
    """
    m = np.asarray(map_2d, dtype=np.float32)
    if m.shape[0] != h_feat or m.shape[1] != w_feat:
        m = cv2.resize(m, (w_feat, h_feat), interpolation=cv2.INTER_NEAREST)

    m = cv2.resize(m, (W_tensor, H_tensor), interpolation=interp_tensor)

    if pad_left is None or pad_top is None or new_h is None or new_w is None:
        pl = pt = 0
        nh, nw = H_tensor, W_tensor
    else:
        pl = int(round(pad_left))
        pt = int(round(pad_top))
        nh, nw = int(new_h), int(new_w)

    y1, y2 = max(0, pt), min(H_tensor, pt + nh)
    x1, x2 = max(0, pl), min(W_tensor, pl + nw)
    if y2 <= y1 or x2 <= x1:
        m_crop = m
    else:
        m_crop = m[y1:y2, x1:x2]

    if normalize_before_resize:
        mn, mx = float(m_crop.min()), float(m_crop.max())
        if mx - mn > 1e-8:
            m_crop = (m_crop - mn) / (mx - mn)
        else:
            m_crop = np.zeros_like(m_crop)

    return cv2.resize(m_crop, (orig_w, orig_h), interpolation=interp_orig)
