"""训练启动时的设备与 DataLoader 参数（各实验 train 共用）。

- **batch_size / num_workers / prefetch_factor**：均使用配置中的值，不做按显存分档下调；
  启动时若 CUDA 可用，会记录 GPU 总显存与上述参数，便于对照日志排查 OOM。
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
    """CUDA 可用时返回配置参数并附带当前 GPU 总显存；非 CUDA 返回 None。"""
    if not torch.cuda.is_available():
        return None

    idx = device_index if device_index is not None else torch.cuda.current_device()
    total_vram_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)

    base_bs = max(1, int(base_batch_size))
    new_bs = base_bs
    batch_scale = 1

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
        f"📊 设备信息: GPU 总显存≈{r.total_vram_gb:.1f} GiB → batch_size={r.batch_size}（配置值）, "
        f"num_workers={r.num_workers}, prefetch_factor={r.prefetch_factor}"
    )
