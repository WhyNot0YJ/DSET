"""
统一模型 GFLOPs / FPS / 参数量 基准测量工具。

YOLO 和 DETR 系列共用本模块，保证对比公平：
- 同一 GFLOPs 计算方法（统一使用 torch.profiler）
- 同一 FPS 测量协议（固定 warmup / 迭代数 / CUDA sync / batch=1）
- 同一输入尺寸（默认 640×640）

用法示例::

    from common.model_benchmark import benchmark_model, format_benchmark_report
    result = benchmark_model(model, imgsz=640, device="cuda")
    print(format_benchmark_report(result))
"""

from __future__ import annotations

import time
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── 数据结构 ────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    """基准测量结果。"""
    model_name: str = ""
    gflops: float = 0.0
    params_total: int = 0
    params_active: int = 0
    params_trainable: int = 0
    fps: float = 0.0
    latency_ms: float = 0.0
    imgsz: Tuple[int, int] = (640, 640)
    device: str = "cuda"
    warmup_iters: int = 0
    measure_iters: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    includes_nms: bool = False


# ── GFLOPs ──────────────────────────────────────────────────────────────

def compute_gflops(
    model: nn.Module,
    imgsz: Union[int, List[int], Tuple[int, int]] = 640,
    device: Optional[Union[str, torch.device]] = None,
) -> float:
    """
    计算模型 GFLOPs（十亿次浮点运算）。

    统一使用 torch.profiler 进行前向算子 FLOPs 统计，避免不同模型
    在 thop 支持度不一致时产生口径差异。对 YOLO（带 stride 属性）
    和 DETR（无 stride）均兼容。
    """
    model = _unwrap(model)
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    if not isinstance(imgsz, (list, tuple)):
        imgsz = [int(imgsz), int(imgsz)]
    else:
        imgsz = [int(imgsz[0]), int(imgsz[1])]

    # 确定输入通道数
    in_channels = _guess_in_channels(model)

    return _gflops_via_torch_profiler(model, imgsz, in_channels, device)


def _gflops_via_torch_profiler(
    model: nn.Module, imgsz: List[int], in_channels: int, device: torch.device
) -> float:
    if not hasattr(torch, "profiler") or not hasattr(torch.profiler, "profile"):
        return 0.0
    model_copy = deepcopy(model).to(device).eval()
    try:
        im = torch.empty((1, in_channels, *imgsz), device=device)
        with torch.profiler.profile(with_flops=True) as prof:
            model_copy(im)
        return sum(x.flops for x in prof.key_averages()) / 1e9
    except Exception:
        return 0.0
    finally:
        del model_copy


# ── 参数量 ──────────────────────────────────────────────────────────────

def compute_params(model: nn.Module) -> Tuple[int, int]:
    """返回 (总参数量, 可训练参数量)。"""
    model = _unwrap(model)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_active_params(model: nn.Module) -> int:
    """
    估算“实际激活参数量”。

    默认等于总参数量；对于 MoE 层，仅按 top_k / num_experts 折算 expert 参数，
    router 与其它共享参数按全量计入。
    """
    model = _unwrap(model)
    total = sum(p.numel() for p in model.parameters())
    active = float(total)

    for module in model.modules():
        is_moe = (
            module.__class__.__name__ == "MoELayer"
            and hasattr(module, "num_experts")
            and hasattr(module, "top_k")
        )
        if not is_moe:
            continue

        num_experts = int(getattr(module, "num_experts", 1))
        top_k = min(int(getattr(module, "top_k", 1)), max(num_experts, 1))
        if num_experts <= 0:
            continue

        active_ratio = top_k / num_experts
        expert_param_count = 0
        for name, param in module.named_parameters(recurse=True):
            lname = name.lower()
            if "experts." in lname or lname.startswith("expert_"):
                expert_param_count += param.numel()

        active -= expert_param_count * (1.0 - active_ratio)

    return max(int(round(active)), 0)


# ── FPS / Latency ───────────────────────────────────────────────────────

def measure_fps(
    model: nn.Module,
    imgsz: Union[int, List[int], Tuple[int, int]] = 640,
    device: Optional[Union[str, torch.device]] = None,
    warmup_iters: int = 50,
    measure_iters: int = 200,
    use_fp16: bool = False,
    postprocess_fn: Optional[Callable] = None,
) -> Tuple[float, float, List[float]]:
    """
    测量模型 FPS 和推理延迟（batch_size=1，单帧）。

    协议：
    1. model.eval() + torch.no_grad()
    2. warmup_iters 次前向 → 不计时
    3. measure_iters 次前向 → 每次 cuda.synchronize → 记录时间
    4. 取中位数延迟 → 计算 FPS

    公平性：
    - DETR 端到端无需后处理 → ``postprocess_fn=None``
    - YOLO 需要 NMS 后处理 → 传入 NMS 包装函数，计时包含 NMS

    Args:
        postprocess_fn: 可选后处理函数，接收模型输出并执行后处理（如 NMS）。
                        传入时计时范围 = forward + postprocess，保证公平对比。

    Returns:
        (fps, latency_ms_median, latencies_ms_list)
    """
    model = _unwrap(model)
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model = model.to(device).eval()

    if not isinstance(imgsz, (list, tuple)):
        imgsz = [int(imgsz), int(imgsz)]
    else:
        imgsz = [int(imgsz[0]), int(imgsz[1])]

    in_channels = _guess_in_channels(model)
    dummy = torch.randn(1, in_channels, *imgsz, device=device)

    if use_fp16 and device.type == "cuda":
        model = model.half()
        dummy = dummy.half()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            out = model(dummy)
            if postprocess_fn is not None:
                postprocess_fn(out)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    # Measure
    latencies: List[float] = []
    with torch.no_grad():
        for _ in range(measure_iters):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            out = model(dummy)
            if postprocess_fn is not None:
                postprocess_fn(out)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    latencies_arr = np.array(latencies)
    median_ms = float(np.median(latencies_arr))
    fps = 1000.0 / median_ms if median_ms > 0 else 0.0
    return fps, median_ms, latencies


# ── 综合 Benchmark ──────────────────────────────────────────────────────

def benchmark_model(
    model: nn.Module,
    imgsz: Union[int, List[int], Tuple[int, int]] = 640,
    device: Optional[Union[str, torch.device]] = None,
    model_name: str = "",
    warmup_iters: int = 50,
    measure_iters: int = 200,
    use_fp16: bool = False,
    postprocess_fn: Optional[Callable] = None,
    includes_nms: bool = False,
) -> BenchmarkResult:
    """
    一站式 benchmark：GFLOPs + 参数量 + FPS/Latency。

    公平性保证：同一函数、同一协议，YOLO 和 DETR 都调用此接口。

    - DETR（端到端）: ``postprocess_fn=None`` — forward 即最终输出
    - YOLO（需要 NMS）: 传入 NMS 包装函数 → FPS/Latency 包含后处理
    """
    model = _unwrap(model)
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    if not isinstance(imgsz, (list, tuple)):
        imgsz_tuple = (int(imgsz), int(imgsz))
    else:
        imgsz_tuple = (int(imgsz[0]), int(imgsz[1]))

    gflops = compute_gflops(model, list(imgsz_tuple), device)
    total_p, train_p = compute_params(model)
    active_p = compute_active_params(model)

    fps, lat_ms, lats = 0.0, 0.0, []
    if device.type == "cuda" and torch.cuda.is_available():
        fps, lat_ms, lats = measure_fps(
            model, list(imgsz_tuple), device,
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
            use_fp16=use_fp16,
            postprocess_fn=postprocess_fn,
        )
    else:
        fps, lat_ms, lats = measure_fps(
            model, list(imgsz_tuple), device,
            warmup_iters=max(warmup_iters // 5, 5),
            measure_iters=max(measure_iters // 5, 20),
            use_fp16=False,
            postprocess_fn=postprocess_fn,
        )

    return BenchmarkResult(
        model_name=model_name,
        gflops=gflops,
        params_total=total_p,
        params_active=active_p,
        params_trainable=train_p,
        fps=fps,
        latency_ms=lat_ms,
        imgsz=imgsz_tuple,
        device=str(device),
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
        latencies_ms=lats,
        includes_nms=includes_nms,
    )


# ── DETR 专用封装 ──────────────────────────────────────────────────────

def benchmark_detr_model(
    model: nn.Module,
    imgsz: Union[int, List[int], Tuple[int, int]] = 640,
    device: Optional[Union[str, torch.device]] = None,
    model_name: str = "detr",
    warmup_iters: int = 50,
    measure_iters: int = 200,
) -> BenchmarkResult:
    """DETR 系列模型 benchmark 封装（RT-DETR、CaS-DETR 等）。"""
    return benchmark_model(
        model, imgsz=imgsz, device=device, model_name=model_name,
        warmup_iters=warmup_iters, measure_iters=measure_iters,
        use_fp16=False,
    )


# ── 格式化 ─────────────────────────────────────────────────────────────

def format_benchmark_report(r: BenchmarkResult) -> str:
    """格式化 benchmark 结果为日志字符串。"""
    lines = [
        f"{'='*60}",
        f"  Model Benchmark: {r.model_name or 'N/A'}",
        f"{'='*60}",
        f"  Input size     : {r.imgsz[0]}x{r.imgsz[1]}",
        f"  Device         : {r.device}",
        f"  GFLOPs         : {r.gflops:.2f}",
        f"  Params (total) : {r.params_total / 1e6:.2f} M",
        f"  Active Params  : {r.params_active / 1e6:.2f} M",
        f"  Params (train) : {r.params_trainable / 1e6:.2f} M",
        f"  FPS            : {r.fps:.1f}",
        f"  Latency (med)  : {r.latency_ms:.2f} ms",
        f"  Warmup iters   : {r.warmup_iters}",
        f"  Measure iters  : {r.measure_iters}",
    ]
    if r.latencies_ms:
        arr = np.array(r.latencies_ms)
        lines.append(f"  Latency (mean)  : {float(np.mean(arr)):.2f} ms")
        lines.append(f"  Latency (std)   : {float(np.std(arr)):.2f} ms")
        lines.append(f"  Latency (p5)    : {float(np.percentile(arr, 5)):.2f} ms")
        lines.append(f"  Latency (p95)   : {float(np.percentile(arr, 95)):.2f} ms")
    lines.append(f"{'='*60}")
    return "\n".join(lines)


def benchmark_to_dict(r: BenchmarkResult) -> dict:
    """转换为平面 dict（用于 CSV / JSON 写入）。"""
    return {
        "GFLOPs": round(r.gflops, 2),
        "Params_M": round(r.params_total / 1e6, 2),
        "Active_Params_M": round(r.params_active / 1e6, 2),
        "Params_trainable_M": round(r.params_trainable / 1e6, 2),
        "FPS": round(r.fps, 1),
        "Latency_ms": round(r.latency_ms, 2),
    }


def log_benchmark(
    log_fn,
    r: BenchmarkResult,
    *,
    header: str = "",
) -> None:
    """
    用统一格式将 benchmark 结果写入日志。

    YOLO 和 DETR **必须**调用此函数输出 benchmark，保证格式完全一致，
    方便日志对比和自动化解析。

    Args:
        log_fn: 日志函数，如 ``logger.info``。
        r: ``BenchmarkResult`` 对象。
        header: 可选的标题前缀（如模型名）。
    """
    tag = f" ({header})" if header else ""
    nms_tag = "+NMS" if r.includes_nms else "e2e"
    latency_stats = ""
    if r.latencies_ms:
        arr = np.array(r.latencies_ms)
        latency_stats = (
            f"  mean={float(np.mean(arr)):.2f}ms  std={float(np.std(arr)):.2f}ms  "
            f"p5={float(np.percentile(arr, 5)):.2f}ms  p95={float(np.percentile(arr, 95)):.2f}ms"
        )
    log_fn(
        f"{'='*72}\n"
        f"  Model Benchmark{tag}\n"
        f"{'='*72}\n"
        f"  GFLOPs         : {r.gflops:.2f}\n"
        f"  Params (total) : {r.params_total / 1e6:.2f} M\n"
        f"  Active Params  : {r.params_active / 1e6:.2f} M\n"
        f"  Params (train) : {r.params_trainable / 1e6:.2f} M\n"
        f"  FPS            : {r.fps:.1f}  (batch=1, {r.imgsz[0]}x{r.imgsz[1]}, {r.device}, {nms_tag})\n"
        f"  Latency (med)  : {r.latency_ms:.2f} ms  ({nms_tag})\n"
        f"  Protocol       : warmup={r.warmup_iters}, iters={r.measure_iters}, sync=cuda\n"
        f"{('  Latency stats  :' + latency_stats + chr(10)) if latency_stats else ''}"
        f"{'='*72}"
    )


# ── 内部工具 ───────────────────────────────────────────────────────────

def _unwrap(model: nn.Module) -> nn.Module:
    """去除 DataParallel / DDP 包裹。"""
    if hasattr(model, "module"):
        return model.module
    return model


def _guess_in_channels(model: nn.Module) -> int:
    """推断模型输入通道数（默认 3）。"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            return m.in_channels
    return 3
