"""显存档位自适应 batch / num_workers / prefetch（各实验 train 共用）。

规则：YAML 中 batch 为基准；
- 总显存 ≥ 15 GiB（约 16G 及以上档）：batch × 3
- 更小（约 8G / 12G 档）：× 1

num_workers / prefetch 会随更高显存档位最多再 ×2，并封顶（8 / 4），降低 EMFILE 风险。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class VramBatchAdjustResult:
    batch_size: int
    num_workers: int
    prefetch_factor: int
    total_vram_gb: float
    batch_scale: int
    base_batch_size: int


def resolve_cuda_device_index(device_str: str) -> int:
    """从 misc.device（如 cuda / cuda:0）解析 GPU 序号。"""
    s = (device_str or "cuda").lower().strip()
    if "cuda" not in s:
        return 0
    if ":" in s:
        return int(s.split(":")[-1].strip())
    return 0


def compute_vram_batch_adjustment(
    base_batch_size: int,
    num_workers: int = 4,
    prefetch_factor: int = 1,
    *,
    device_index: Optional[int] = None,
) -> Optional[VramBatchAdjustResult]:
    """CUDA 可用时返回调整后的参数；非 CUDA 返回 None。"""
    if not torch.cuda.is_available():
        return None

    idx = device_index if device_index is not None else torch.cuda.current_device()
    total_vram_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)

    if total_vram_gb >= 15:
        batch_scale = 3
    else:
        batch_scale = 1

    base_bs = max(1, int(base_batch_size))
    new_bs = max(1, base_bs * batch_scale)
    nw_mult = 2 if batch_scale > 1 else 1
    new_nw = max(1, min(int(num_workers) * nw_mult, 8))
    new_pf = max(1, min(int(prefetch_factor) * nw_mult, 4))

    return VramBatchAdjustResult(
        batch_size=new_bs,
        num_workers=new_nw,
        prefetch_factor=new_pf,
        total_vram_gb=total_vram_gb,
        batch_scale=batch_scale,
        base_batch_size=base_bs,
    )


def format_vram_batch_log(r: VramBatchAdjustResult) -> str:
    return (
        f"📊 显存自适应: GPU 总显存≈{r.total_vram_gb:.1f} GiB → batch 基准 {r.base_batch_size} × {r.batch_scale} = {r.batch_size}, "
        f"num_workers={r.num_workers}, prefetch_factor={r.prefetch_factor}"
    )
