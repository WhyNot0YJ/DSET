#!/usr/bin/env python3
"""
Benchmark and plot Accuracy vs Efficiency (keep ratio) for CaS-DETR.

Modes:
  - benchmark: run CaS-DETR eval at multiple keep ratios and write JSON
  - plot: read JSON and generate curve figure
  - both: benchmark then plot

COCO bbox summarize prints six AP lines first; --metric_indices selects which appear in stdout,
in order: 0 all @0.5:0.95, 1 all @0.5, 2 all @0.75, 3 small @0.5:0.95, 4 medium, 5 large.
Optional --maps50 reads AP for small objects at IoU=0.50 from COCOeval precision, same as
common.det_eval_metrics.coco_area_ap_at_iou50, via CAS_BENCH_AP_small_50= in train stdout.
After benchmark, a markdown metric table is printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


def _parse_ap_small_50(stdout_text: str) -> Optional[float]:
    """Parse ``CAS_BENCH_AP_small_50=0.xxx`` printed by CaS-DETR ``det_engine.evaluate``."""
    m = re.search(r"CAS_BENCH_AP_small_50=([0-9]+(?:\.[0-9]+)?)", stdout_text)
    if not m:
        return None
    return float(m.group(1))


def _run_single_eval(
    root_dir: Path,
    config: str,
    resume: str,
    keep_ratio: float,
    device: str,
    extra_updates: Sequence[str],
    *,
    print_ap_small_50: bool = False,
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
    env = os.environ.copy()
    if print_ap_small_50:
        env["CAS_BENCH_PRINT_AP_SMALL_50"] = "1"
    print(f"[benchmark] keep_ratio={keep_ratio:.2f}")
    proc = subprocess.run(
        cmd,
        cwd=str(root_dir),
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Eval command failed at keep_ratio={keep_ratio:.2f}")
    return proc.stdout + "\n" + proc.stderr


def benchmark_curve_multi(
    root_dir: Path,
    config: str,
    resume: str,
    ratios: Sequence[float],
    device: str,
    metric_indices: Sequence[int],
    extra_updates: Sequence[str],
    dry_run: bool = False,
    dry_bias: float = 0.0,
    *,
    print_ap_small_50: bool = False,
    metric_labels: Optional[Sequence[str]] = None,
    maps50_label: str = "mAPs50",
) -> Tuple[Dict[int, List[float]], List[float]]:
    out_metrics: Dict[int, List[float]] = {int(mi): [] for mi in metric_indices}
    out_maps50: List[float] = []
    labels_ok = metric_labels is not None and len(metric_labels) == len(metric_indices)
    for ratio in ratios:
        if dry_run:
            # Deterministic synthetic curve: rises then saturates.
            r = float(ratio)
            y = 0.60 + 0.09 * (1.0 - (1.0 - r) ** 2) + float(dry_bias)
            y = float(min(max(y, 0.0), 1.0))
            for mi in metric_indices:
                # Make different metrics slightly separated in dry run.
                out_metrics[int(mi)].append(float(min(max(y + 0.003 * int(mi), 0.0), 1.0)))
            if print_ap_small_50:
                out_maps50.append(float(min(max(y - 0.02 + dry_bias, 0.0), 1.0)))
        else:
            out = _run_single_eval(
                root_dir,
                config,
                resume,
                float(ratio),
                device,
                extra_updates,
                print_ap_small_50=print_ap_small_50,
            )
            ap_values = _parse_coco_ap_values(out)
            if not ap_values:
                raise RuntimeError("Could not parse COCO AP values from evaluation output.")
            for mi in metric_indices:
                if mi < 0 or mi >= len(ap_values):
                    raise IndexError(
                        f"metric_index={mi} out of range; parsed {len(ap_values)} AP values."
                    )
                out_metrics[int(mi)].append(float(ap_values[mi]))
            if print_ap_small_50:
                v = _parse_ap_small_50(out)
                if v is None:
                    raise RuntimeError(
                        "Could not parse CAS_BENCH_AP_small_50 from eval output. "
                        "Use an up-to-date CaS-DETR det_engine that prints it when "
                        "CAS_BENCH_PRINT_AP_SMALL_50=1."
                    )
                out_maps50.append(float(v))
        parts: List[str] = []
        for j, mi in enumerate(metric_indices):
            val = out_metrics[int(mi)][-1]
            if labels_ok:
                parts.append(f"{metric_labels[j]}={val:.4f}")
            else:
                parts.append(f"metric[{mi}]={val:.4f}")
        if print_ap_small_50:
            parts.append(f"{maps50_label}={out_maps50[-1]:.4f}")
        print("  -> " + ", ".join(parts))
    return out_metrics, out_maps50


def _setup_style() -> None:
    assert plt is not None
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
            "font.size": 12,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "figure.dpi": 150,
        }
    )


def _pretty_curve_name(name: str) -> str:
    alias = {
        "mAP_50-95": r"Overall ($mAP$)",
        "mAP_S_50-95": r"Small ($AP_S$)",
        "Overall mAP": r"Overall ($mAP$)",
        "Small Object mAP (mAP_S)": r"Small ($AP_S$)",
        "Overall (mAP)": r"Overall ($mAP$)",
        "Small (AP_S)": r"Small ($AP_S$)",
        "mAP50": r"$mAP^{50}$",
        "mAPs50": r"$AP_S^{50}$",
        "mAPs50 - UA": r"$AP_S^{50}$ - UA",
        "mAPs50 - DAIR": r"$AP_S^{50}$ - DAIR",
    }
    if name in alias:
        return alias[name]
    # Fallback: replace underscores for display only.
    return name.replace("_", " ")


def _markdown_metric_table(
    inference_ratios: Sequence[float],
    results: Dict[str, Sequence[float]],
) -> str:
    """Build a GitHub-flavored markdown table: one row per keep ratio, one column per curve."""
    if not results:
        return ""
    cols = list(results.keys())
    lines: List[str] = []
    header = "| " + " | ".join([r"$r$", *[ _pretty_curve_name(c) for c in cols]]) + " |"
    sep = "| " + " | ".join(["---"] * (1 + len(cols))) + " |"
    lines.append(header)
    lines.append(sep)
    n = len(inference_ratios)
    for i in range(n):
        row_vals: List[str] = [f"{float(inference_ratios[i]):.2f}"]
        for c in cols:
            seq = results[c]
            if len(seq) != n:
                row_vals.append("—")
            else:
                row_vals.append(f"{float(seq[i]):.4f}")
        lines.append("| " + " | ".join(row_vals) + " |")
    return "\n".join(lines)


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
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

    for idx, (name, vals) in enumerate(results.items()):
        if len(vals) != len(inference_ratios):
            raise ValueError(
                f"Curve '{name}' length {len(vals)} != inference_ratios {len(inference_ratios)}"
            )
        ax.plot(
            inference_ratios,
            vals,
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            linewidth=2,
            markersize=6,
            label=_pretty_curve_name(name),
        )

    ax.set_xlabel(r"Token Keep Ratio $r$")
    ax.set_ylabel("Average Precision (AP)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="lower right", framealpha=0.95)
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
    parser.add_argument("--curve_name_a", type=str, default="A")
    parser.add_argument("--config_b", type=str, default=None)
    parser.add_argument("--resume_b", type=str, default=None)
    parser.add_argument("--curve_name_b", type=str, default="B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--metric_index", type=int, default=0)
    parser.add_argument(
        "--metric_indices",
        nargs="+",
        type=int,
        default=None,
        help=(
            "COCO bbox AP line indices from eval stdout, in print order: "
            "0=mAP@[0.5:0.95] all, 1=mAP50 all, 3=AP_S@[0.5:0.95] small. "
            "Example: 0 1 3 for mAP, mAP50, AP_S."
        ),
    )
    parser.add_argument(
        "--curve_names_a",
        nargs="+",
        default=None,
        help="Optional curve names for model A, length must match metric indices.",
    )
    parser.add_argument(
        "--curve_names_b",
        nargs="+",
        default=None,
        help="Optional curve names for model B, length must match metric indices.",
    )
    parser.add_argument(
        "--maps50",
        action="store_true",
        help="Also record small-object AP at IoU=0.50 from COCOeval precision, same as AP_small_50 in cas_style_map_metrics.",
    )
    parser.add_argument(
        "--maps50_curve_name",
        type=str,
        default="mAPs50",
        help="JSON and plot legend key for model A maps50 series.",
    )
    parser.add_argument(
        "--maps50_curve_name_b",
        type=str,
        default=None,
        help="Legend key for model B maps50; default mAPs50 - UA when model B is enabled.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Generate synthetic benchmark values without running eval.")
    parser.add_argument("--eval_split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--extra_update",
        nargs="+",
        default=[
            "HybridEncoder.caip_dynamic_warmup_epochs=0",
            "HybridEncoder.caip_static_keep_eval=True",
        ],
        help="Extra -u overrides passed to train.py to make sweep effective in eval.",
    )
    parser.add_argument(
        "--extra_update_a",
        nargs="+",
        default=[],
        help="Model-A specific -u overrides (in addition to --extra_update).",
    )
    parser.add_argument(
        "--extra_update_b",
        nargs="+",
        default=[],
        help="Model-B specific -u overrides (in addition to --extra_update).",
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
        metric_indices = list(args.metric_indices) if args.metric_indices else [int(args.metric_index)]

        curve_names_a = list(args.curve_names_a) if args.curve_names_a else [
            f"{args.curve_name_a}_m{mi}" for mi in metric_indices
        ]
        if len(curve_names_a) != len(metric_indices):
            raise ValueError("--curve_names_a length must match metric count.")

        use_model_b = bool(args.config_b and args.resume_b)
        if args.curve_names_b and (not use_model_b):
            raise ValueError("--curve_names_b provided but model B is not configured.")
        curve_names_b = list(args.curve_names_b) if args.curve_names_b else [
            f"{args.curve_name_b}_m{mi}" for mi in metric_indices
        ]
        if use_model_b and len(curve_names_b) != len(metric_indices):
            raise ValueError("--curve_names_b length must match metric count.")

        keys_to_check = list(curve_names_a)
        if use_model_b:
            keys_to_check.extend(curve_names_b)
        name_maps50_b = args.maps50_curve_name_b or "mAPs50 - UA"
        if args.maps50:
            keys_to_check.append(args.maps50_curve_name)
            if use_model_b:
                keys_to_check.append(name_maps50_b)
        for key in keys_to_check:
            if (key in data["results"]) and (not args.overwrite):
                raise ValueError(f"Curve '{key}' already exists. Use --overwrite to replace.")

        updates_common = list(args.extra_update)
        updates_a = updates_common + list(args.extra_update_a)
        updates_b = updates_common + list(args.extra_update_b)

        curves_a, maps50_a = benchmark_curve_multi(
            root_dir=root_dir,
            config=args.config_a,
            resume=args.resume_a,
            ratios=ratios,
            device=args.device,
            metric_indices=metric_indices,
            extra_updates=updates_a,
            dry_run=args.dry_run,
            dry_bias=0.0,
            print_ap_small_50=args.maps50,
            metric_labels=curve_names_a,
            maps50_label=args.maps50_curve_name,
        )
        for mi, cname in zip(metric_indices, curve_names_a):
            data["results"][cname] = curves_a[int(mi)]
        if args.maps50:
            data["results"][args.maps50_curve_name] = maps50_a

        if use_model_b:
            curves_b, maps50_b = benchmark_curve_multi(
                root_dir=root_dir,
                config=args.config_b,
                resume=args.resume_b,
                ratios=ratios,
                device=args.device,
                metric_indices=metric_indices,
                extra_updates=updates_b,
                dry_run=args.dry_run,
                dry_bias=0.01,
                print_ap_small_50=args.maps50,
                metric_labels=curve_names_b,
                maps50_label=name_maps50_b,
            )
            for mi, cname in zip(metric_indices, curve_names_b):
                data["results"][cname] = curves_b[int(mi)]
            if args.maps50:
                data["results"][name_maps50_b] = maps50_b
        _save_json(output_json, data)
        print(f"Saved benchmark JSON: {output_json}")
        table = _markdown_metric_table(data["inference_ratios"], data["results"])
        if table:
            print("\nMetric table:\n")
            print(table)
            print()

    if args.mode in ("plot", "both"):
        data = _load_json(output_json)
        if args.mode == "plot" and data.get("results"):
            table = _markdown_metric_table(data["inference_ratios"], data["results"])
            if table:
                print("\nMetric table:\n")
                print(table)
                print()
        plot_results(
            inference_ratios=data["inference_ratios"],
            results=data["results"],
            output_plot=output_plot,
        )


if __name__ == "__main__":
    main()
