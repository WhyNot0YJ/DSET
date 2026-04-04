"""
DETR 系列（RT-DETR / CaS-DETR / MoE-RTDETR）共用的训练后评估工具。

将三份训练脚本中 **完全一致** 的评估流程抽出，消除代码重复：
- ``run_detr_benchmark``          → 运行 GFLOPs/FPS 并打印标准报告
- ``format_bench_inline``         → 评估摘要行尾的 benchmark 短字符串
- ``log_detr_eval_summary``       → 打印评估摘要（mAP + bench）
- ``write_detr_eval_csv``         → 写入汇总 CSV（含 benchmark 列）
- ``evaluate_best_model_after_training`` → 完整的 "加载 best → benchmark → val/test 评估" 流程
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

from common.det_eval_metrics import (
    dataset_display_name,
    model_display_name,
    write_eval_csv,
)
from common.model_benchmark import (
    benchmark_detr_model,
    benchmark_to_dict,
    log_benchmark,
)


# ── Benchmark ──────────────────────────────────────────────────────────

def run_detr_benchmark(
    ema_module: torch.nn.Module,
    config: Dict[str, Any],
    experiment_name: str,
    device,
    logger,
) -> Optional[Dict[str, float]]:
    """
    运行 GFLOPs / FPS benchmark 并用 ``log_benchmark`` 打印标准报告。

    Returns:
        ``benchmark_to_dict()`` 结果，失败时返回 ``None``。
    """
    bench_dict = None
    try:
        name = model_display_name(config, experiment_name)
        result = benchmark_detr_model(
            ema_module, imgsz=640, device=device, model_name=name,
        )
        log_benchmark(logger.info, result, header=name)
        bench_dict = benchmark_to_dict(result)
    except Exception as exc:
        logger.warning(f"Model benchmark 失败（不影响评估结果）: {exc}")
    return bench_dict


# ── 评估摘要日志 + CSV ─────────────────────────────────────────────────

def format_bench_inline(bench_dict: Optional[Dict[str, float]]) -> str:
    """将 bench_dict 格式化为评估摘要行尾的短字符串。"""
    if not bench_dict:
        return ""
    return (
        f"  |  Params={bench_dict['Params_M']:.2f}M  "
        f"Active={bench_dict.get('Active_Params_M', bench_dict['Params_M']):.2f}M  "
        f"GFLOPs={bench_dict['GFLOPs']:.2f}  "
        f"FPS={bench_dict['FPS']:.1f}  "
        f"Latency={bench_dict['Latency_ms']:.2f}ms"
    )


def log_detr_eval_summary(
    logger,
    split_label: str,
    metrics: Dict[str, Any],
    bench_dict: Optional[Dict[str, float]] = None,
) -> None:
    """打印标准化 DETR 评估摘要（mAP + E/M/H + S/M/L + benchmark）。"""
    m = metrics
    bench_line = format_bench_inline(bench_dict)
    logger.info(
        f"  best_model [{split_label}]  "
        f"mAP50={m.get('mAP_0.5', 0):.4f}  "
        f"mAP75={m.get('mAP_0.75', 0):.4f}  "
        f"mAP={m.get('mAP_0.5_0.95', 0):.4f}\n"
        f"    E/M/H@0.5: "
        f"{m.get('AP_easy', 0):.4f}/"
        f"{m.get('AP_moderate', 0):.4f}/"
        f"{m.get('AP_hard', 0):.4f}  |  "
        f"S/M/L@0.5: "
        f"{m.get('AP_small_50', 0):.4f}/"
        f"{m.get('AP_medium_50', 0):.4f}/"
        f"{m.get('AP_large_50', 0):.4f}  |  "
        f"S/M/L@0.5:0.95: "
        f"{m.get('AP_small', 0):.4f}/"
        f"{m.get('AP_medium', 0):.4f}/"
        f"{m.get('AP_large', 0):.4f}"
        f"{bench_line}"
    )
    if "gt_boxes_easy" in m:
        logger.info(
            "    KITTI GT 框数: easy=%d moderate=%d hard=%d ignore=%d（与本次评估 GT 一致）",
            int(m["gt_boxes_easy"]),
            int(m["gt_boxes_moderate"]),
            int(m["gt_boxes_hard"]),
            int(m["gt_boxes_ignore"]),
        )


def write_detr_eval_csv(
    log_dir: Path,
    config: Dict[str, Any],
    experiment_name: str,
    split_label: str,
    metrics: Dict[str, Any],
    class_names: List[str],
    bench_dict: Optional[Dict[str, float]] = None,
    *,
    aggregate_at_parent: bool = True,
) -> Path:
    """写入汇总 eval_metrics.csv（含 benchmark 列）。返回 CSV 路径。"""
    if aggregate_at_parent:
        summary_csv = log_dir.parent / "eval_metrics.csv"
    else:
        summary_csv = log_dir / "eval_metrics.csv"
    write_eval_csv(
        summary_csv,
        model=model_display_name(config, experiment_name),
        dataset=dataset_display_name(config),
        eval_split=split_label,
        metrics=metrics,
        class_names=class_names,
        append=summary_csv.exists(),
        benchmark=bench_dict,
    )
    return summary_csv


# ── 训练后 best-model 评估完整流程 ─────────────────────────────────────

def evaluate_best_model_after_training(
    *,
    log_dir: Path,
    device,
    config: Dict[str, Any],
    experiment_name: str,
    logger,
    ema,
    val_loader,
    build_test_loader_fn: Callable[[], Any],
    run_eval_fn: Callable,
) -> None:
    """
    DETR 系列共用的训练后评估流程：

    1. 加载 best / latest checkpoint → 恢复 EMA 权重
    2. 运行 benchmark（仅一次）
    3. 在 val 上评估
    4. 在 test 上评估（若可用）
    5. 恢复原始 EMA 状态

    Args:
        log_dir:   实验日志目录（含 best_model.pth）
        device:    torch device
        config:    训练配置 dict
        experiment_name: 实验名
        logger:    logging.Logger
        ema:       ModelEMA 对象
        val_loader: 验证 DataLoader
        build_test_loader_fn: 返回 test DataLoader 或 None 的可调用对象
        run_eval_fn: ``_run_ema_eval_on_dataloader(loader, split, epoch, bench_dict)``
    """
    best_path = log_dir / "best_model.pth"
    latest_path = log_dir / "latest_checkpoint.pth"

    if best_path.exists():
        checkpoint_path = best_path
    elif latest_path.exists():
        checkpoint_path = latest_path
        logger.warning("未找到best_model.pth，改用latest_checkpoint.pth进行训练结束评估")
    else:
        logger.warning("未找到best_model.pth和latest_checkpoint.pth，跳过训练结束评估")
        return

    original_ema_state = None
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        best_ema_state = checkpoint.get("ema_state_dict")
        best_epoch = checkpoint.get("epoch")

        if ema and best_ema_state is not None:
            original_ema_state = ema.state_dict()
            ema.load_state_dict(best_ema_state)

        ema.module.eval()

        bench_dict = run_detr_benchmark(
            ema.module, config, experiment_name, device, logger,
        )

        run_eval_fn(
            val_loader, split_label="val",
            best_epoch=best_epoch, bench_dict=bench_dict,
        )

        test_loader = build_test_loader_fn()
        if test_loader is not None:
            logger.info("在 test 划分上进行训练结束评估（与 YOLO 一致，仅一次）…")
            run_eval_fn(
                test_loader, split_label="test",
                best_epoch=best_epoch, bench_dict=bench_dict,
            )
        else:
            logger.info("无可用 test 数据或未配置 test，跳过 test 评估。")

    except Exception as e:
        logger.warning(f"训练结束后的best_model评估失败: {e}")
    finally:
        if original_ema_state is not None and ema:
            try:
                ema.load_state_dict(original_ema_state)
            except Exception:
                pass
