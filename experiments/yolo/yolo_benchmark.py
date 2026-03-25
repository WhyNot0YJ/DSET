"""
YOLO 系列模型 GFLOPs / FPS 测量封装。

调用 ``common.model_benchmark`` 统一核心，保证与 DETR 系列公平对比。

**关键公平性设计**：YOLO 的 FPS / Latency **包含 NMS 后处理**。
DETR 系列是端到端模型（forward 直出检测结果），而 YOLO 的 forward 只
输出原始预测，还需要 NMS 才得到最终检测框。因此 YOLO 侧的 FPS 测量
计时范围 = forward + NMS，与 DETR 的 forward（已含全部计算）对齐。

支持三种输入方式：
1. Ultralytics YOLO 对象（会自动提取内部 nn.Module）
2. 权重文件路径（.pt）
3. 裸 nn.Module

用法示例::

    from yolo_benchmark import benchmark_yolo
    from common.model_benchmark import format_benchmark_report
    result = benchmark_yolo("best.pt", imgsz=640, device="cuda")
    print(format_benchmark_report(result))
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Optional, Union, List, Tuple

import torch
import torch.nn as nn

_experiments_root = Path(__file__).resolve().parent.parent
if str(_experiments_root) not in sys.path:
    sys.path.insert(0, str(_experiments_root))

from common.model_benchmark import (
    BenchmarkResult,
    benchmark_model,
    format_benchmark_report,
    benchmark_to_dict,
)

logger = logging.getLogger(__name__)


# ── NMS 后处理包装 ─────────────────────────────────────────────────────

def _build_nms_postprocess(conf_thres: float = 0.25, iou_thres: float = 0.7, max_det: int = 300):
    """
    构建 YOLO NMS 后处理函数，用于 FPS 测量。

    返回的函数接收模型原始输出，执行 NMS，保证 FPS 计时包含后处理。
    """
    try:
        from ultralytics.utils.nms import non_max_suppression
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression

    def _nms_fn(raw_output):
        if isinstance(raw_output, (tuple, list)):
            preds = raw_output[0] if len(raw_output) > 0 else raw_output
        else:
            preds = raw_output
        return non_max_suppression(
            preds,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
        )
    return _nms_fn


# ── 模型提取 ───────────────────────────────────────────────────────────

def _extract_yolo_nn_module(model_or_path) -> Tuple[nn.Module, str]:
    """
    从 Ultralytics YOLO 对象或 .pt 路径中提取裸 nn.Module。

    Returns:
        (nn_module, model_name_str)
    """
    if isinstance(model_or_path, (str, Path)):
        pt_path = Path(model_or_path)
        if not pt_path.exists():
            raise FileNotFoundError(f"权重文件不存在: {pt_path}")
        from ultralytics import YOLO
        yolo_obj = YOLO(str(pt_path))
        name = pt_path.stem
        return _unwrap_ultralytics(yolo_obj), name

    if hasattr(model_or_path, "model") and isinstance(model_or_path.model, nn.Module):
        name = getattr(model_or_path, "model_name", "yolo")
        if hasattr(model_or_path, "ckpt_path"):
            name = Path(str(model_or_path.ckpt_path)).stem
        return _unwrap_ultralytics(model_or_path), str(name)

    if isinstance(model_or_path, nn.Module):
        return model_or_path, "yolo"

    raise TypeError(
        f"不支持的模型类型: {type(model_or_path)}。"
        "请传入 Ultralytics YOLO 对象、.pt 路径或 nn.Module。"
    )


def _unwrap_ultralytics(yolo_obj) -> nn.Module:
    """从 ultralytics.YOLO 对象提取可前向传播的 nn.Module。"""
    inner = getattr(yolo_obj, "model", yolo_obj)
    if hasattr(inner, "module"):
        inner = inner.module
    if hasattr(inner, "fuse"):
        try:
            inner.fuse()
        except Exception:
            pass
    return inner


# ── Benchmark 入口 ─────────────────────────────────────────────────────

def benchmark_yolo(
    model_or_path,
    imgsz: Union[int, List[int], Tuple[int, int]] = 640,
    device: Optional[Union[str, torch.device]] = None,
    model_name: Optional[str] = None,
    warmup_iters: int = 50,
    measure_iters: int = 200,
    use_fp16: bool = False,
    conf_thres: float = 0.25,
    iou_thres: float = 0.7,
    max_det: int = 300,
) -> BenchmarkResult:
    """
    YOLO 系列 benchmark 入口（FPS 包含 NMS 后处理）。

    与 ``common.model_benchmark.benchmark_detr_model`` 使用同一测量核心，
    通过 ``postprocess_fn`` 机制将 NMS 纳入计时，保证与 DETR 端到端推理公平。

    Args:
        model_or_path: Ultralytics YOLO 对象、.pt 权重路径或裸 nn.Module
        imgsz: 输入图像尺寸（int 或 [H, W]）
        device: 设备（默认跟随模型）
        model_name: 模型显示名称
        warmup_iters: GPU warmup 次数
        measure_iters: 测量迭代次数
        use_fp16: 是否使用 FP16 推理
        conf_thres: NMS 置信度阈值
        iou_thres: NMS IoU 阈值
        max_det: NMS 最大检测数

    Returns:
        BenchmarkResult（includes_nms=True）
    """
    nn_module, auto_name = _extract_yolo_nn_module(model_or_path)
    name = model_name or auto_name

    if device is None:
        try:
            device = next(nn_module.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nms_fn = _build_nms_postprocess(
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=max_det,
    )

    return benchmark_model(
        nn_module,
        imgsz=imgsz,
        device=device,
        model_name=name,
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
        use_fp16=use_fp16,
        postprocess_fn=nms_fn,
        includes_nms=True,
    )
