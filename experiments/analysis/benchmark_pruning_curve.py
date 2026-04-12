#!/usr/bin/env python3
"""
Benchmark and plot Accuracy vs Efficiency (keep ratio) for CaS-DETR.

Modes:
  - benchmark: run CaS-DETR eval at multiple keep ratios and write JSON
  - plot: read JSON and generate curve figure
  - both: benchmark then plot
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

plt = None


def _default_ratios() -> List[float]:
    return [i / 10.0 for i in range(11)]


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {"inference_ratios": [], "results": {}}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "inference_ratios" not in data or "results" not in data:
        raise ValueError(f"Invalid JSON schema: {path}")
    return data


def _save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _parse_coco_ap_values(stdout_text: str) -> List[float]:
    # Parse lines such as:
    # "Average Precision  (AP) @[ IoU=0.50:0.95 | ... ] = 0.687"
    values = re.findall(r"Average Precision\s+\(AP\).*= ([0-9]*\.[0-9]+)", stdout_text)
    return [float(v) for v in values]


def _run_single_eval(
    root_dir: Path,
    config: str,
    resume: str,
    keep_ratio: float,
    device: str,
    extra_updates: Sequence[str],
) -> str:
    train_py = root_dir / "experiments" / "CaS-DETR" / "train.py"
    update_items = [
        "tuning=null",
        f"HybridEncoder.token_keep_ratio={keep_ratio}",
    ]
    update_items.extend(list(extra_updates))

    cmd = [
        sys.executable,
        str(train_py),
        "-c",
        config,
        "-r",
        resume,
        "--tuning",
        "",
        "--test-only",
        "--device",
        device,
        "-u",
        *update_items,
    ]
    print(f"[benchmark] keep_ratio={keep_ratio:.2f}")
    proc = subprocess.run(
        cmd,
        cwd=str(root_dir),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Eval command failed at keep_ratio={keep_ratio:.2f}")
    return proc.stdout + "\n" + proc.stderr


def benchmark_curve(
    root_dir: Path,
    config: str,
    resume: str,
    ratios: Sequence[float],
    device: str,
    metric_index: int,
    extra_updates: Sequence[str],
    dry_run: bool = False,
    dry_bias: float = 0.0,
) -> List[float]:
    metrics: List[float] = []
    for ratio in ratios:
        if dry_run:
            # Deterministic synthetic curve: rises then saturates.
            r = float(ratio)
            y = 0.60 + 0.09 * (1.0 - (1.0 - r) ** 2) + float(dry_bias)
            metrics.append(float(min(max(y, 0.0), 1.0)))
        else:
            out = _run_single_eval(
                root_dir, config, resume, float(ratio), device, extra_updates
            )
            ap_values = _parse_coco_ap_values(out)
            if not ap_values:
                raise RuntimeError("Could not parse COCO AP values from evaluation output.")
            if metric_index < 0 or metric_index >= len(ap_values):
                raise IndexError(
                    f"metric_index={metric_index} out of range; parsed {len(ap_values)} AP values."
                )
            metrics.append(float(ap_values[metric_index]))
        print(f"  -> metric[{metric_index}]={metrics[-1]:.4f}")
    return metrics


def _setup_style() -> None:
    assert plt is not None
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "figure.dpi": 150,
        }
    )


def plot_results(
    inference_ratios: Sequence[float],
    results: Dict[str, Sequence[float]],
    output_plot: Path,
) -> None:
    global plt
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
    except ImportError as exc:
        raise RuntimeError("plot mode requires matplotlib. Please install matplotlib first.") from exc
    plt = _plt
    if not results:
        raise ValueError("No curves found in results.")

    _setup_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    markers = ["o", "s", "^", "D", "P", "X", "*", "v"]

    for idx, (name, vals) in enumerate(results.items()):
        if len(vals) != len(inference_ratios):
            raise ValueError(
                f"Curve '{name}' length {len(vals)} != inference_ratios {len(inference_ratios)}"
            )
        ax.plot(
            inference_ratios,
            vals,
            marker=markers[idx % len(markers)],
            linewidth=2,
            markersize=6,
            label=name,
        )

    ax.set_xlabel("Inference Keep Ratio")
    ax.set_ylabel("COCO mAP (50-95)")
    ax.set_title("Accuracy vs. Efficiency Trade-off (CaS-DETR)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(min(inference_ratios), max(inference_ratios))
    ax.legend(loc="best", framealpha=0.95)
    plt.tight_layout()

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_plot), bbox_inches="tight", dpi=300)
    png_path = output_plot.with_suffix(".png")
    plt.savefig(str(png_path), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved plot: {output_plot}")
    print(f"Saved plot: {png_path}")


def _check_ratio_compat(existing: Sequence[float], new: Sequence[float]) -> bool:
    if len(existing) != len(new):
        return False
    for a, b in zip(existing, new):
        if abs(float(a) - float(b)) > 1e-9:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark and plot CaS-DETR pruning trade-off curves.")
    parser.add_argument("--mode", choices=["benchmark", "plot", "both"], default="both")
    parser.add_argument(
        "--output_json",
        type=str,
        default=str(Path(__file__).parent / "benchmark_pruning_curve_results.json"),
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default=str(Path(__file__).parent / "pruning_tradeoff.pdf"),
    )
    parser.add_argument("--inference_ratios", nargs="+", type=float, default=_default_ratios())
    parser.add_argument("--config_a", type=str, default=None)
    parser.add_argument("--resume_a", type=str, default=None)
    parser.add_argument("--curve_name_a", type=str, default="CaS_DETR_dair")
    parser.add_argument("--config_b", type=str, default=None)
    parser.add_argument("--resume_b", type=str, default=None)
    parser.add_argument("--curve_name_b", type=str, default="CaS_DETR_ua")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--metric_index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Generate synthetic benchmark values without running eval.")
    parser.add_argument(
        "--extra_update",
        nargs="+",
        default=[
            "HybridEncoder.caip_dynamic_warmup_epochs=0",
            "HybridEncoder.caip_static_keep_eval=True",
        ],
        help="Extra -u overrides passed to train.py to make sweep effective in eval.",
    )
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[2]
    output_json = Path(args.output_json).resolve()
    output_plot = Path(args.output_plot).resolve()
    ratios = [float(x) for x in args.inference_ratios]

    data = _load_json(output_json)
    if "inference_ratios" in data and data["inference_ratios"]:
        if not _check_ratio_compat(data["inference_ratios"], ratios):
            raise ValueError(
                "Existing JSON inference_ratios differ from requested ratios. "
                "Please align ratios or use a different --output_json."
            )
    data["inference_ratios"] = ratios
    data.setdefault("results", {})

    if args.mode in ("benchmark", "both"):
        if not args.config_a or not args.resume_a:
            raise ValueError("--config_a and --resume_a are required for benchmark mode.")
        if not args.config_b or not args.resume_b:
            raise ValueError("--config_b and --resume_b are required for benchmark mode.")

        for key in (args.curve_name_a, args.curve_name_b):
            if (key in data["results"]) and (not args.overwrite):
                raise ValueError(f"Curve '{key}' already exists. Use --overwrite to replace.")

        curve_a = benchmark_curve(
            root_dir=root_dir,
            config=args.config_a,
            resume=args.resume_a,
            ratios=ratios,
            device=args.device,
            metric_index=args.metric_index,
            extra_updates=args.extra_update,
            dry_run=args.dry_run,
            dry_bias=0.0,
        )
        curve_b = benchmark_curve(
            root_dir=root_dir,
            config=args.config_b,
            resume=args.resume_b,
            ratios=ratios,
            device=args.device,
            metric_index=args.metric_index,
            extra_updates=args.extra_update,
            dry_run=args.dry_run,
            dry_bias=0.01,
        )
        data["results"][args.curve_name_a] = curve_a
        data["results"][args.curve_name_b] = curve_b
        _save_json(output_json, data)
        print(f"Saved benchmark JSON: {output_json}")

    if args.mode in ("plot", "both"):
        data = _load_json(output_json)
        plot_results(
            inference_ratios=data["inference_ratios"],
            results=data["results"],
            output_plot=output_plot,
        )


if __name__ == "__main__":
    main()
