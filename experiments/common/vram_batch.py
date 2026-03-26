"""显存档位自适应 num_workers / prefetch（各实验 train 共用）。

规则：YAML 中 batch 为最终实际 batch，不再随显存档位自动放大；
- 所有显存档位：保持配置中的 batch_size 不变

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

    batch_scale = 1

    base_bs = max(1, int(base_batch_size))
    new_bs = base_bs
    new_nw = max(1, int(num_workers))
    new_pf = max(1, int(prefetch_factor))

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
        f"📊 显存自适应: GPU 总显存≈{r.total_vram_gb:.1f} GiB → batch 固定为配置值 {r.batch_size}, "
        f"num_workers={r.num_workers}, prefetch_factor={r.prefetch_factor}"
    )
