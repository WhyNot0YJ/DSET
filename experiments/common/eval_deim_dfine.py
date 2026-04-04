#!/usr/bin/env python3
"""
Post-training evaluation for DEIM / D-FINE models.
Produces CaS_DETR-compatible eval_metrics.csv with the same metric columns.

Usage (from experiments/ directory):
  python3 common/eval_deim_dfine.py \\
      --framework deim \\
      --config DEIM/configs/deim_dfine/deim_hgnetv2_s_dairv2x.yml \\
      --resume DEIM/outputs/deim_hgnetv2_s_dairv2x/best_stg2.pth \\
      --model-name deim_hgnetv2_s \\
      --dataset-name DAIR-V2X

After loading weights, runs ``run_detr_benchmark`` for GFLOPs, Params, FPS, and latency,
then val and optional test metrics and ``eval_metrics.csv``.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from collections import Counter
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

from common.det_eval_metrics import (
    kitti_difficulty_from_coco_ann,
    coco_gt_with_difficulty_iscrowd,
    coco_ap_at_iou50_all,
    coco_area_ap_at_iou50,
    extract_per_category_ap_from_coco_eval,
    run_coco_bbox_eval,
    write_eval_csv,
)
from common.detr_eval_utils import log_detr_eval_summary, run_detr_benchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger(__name__)


def _resolve_resume_path(resume: Optional[str]) -> Optional[str]:
    """Resolve checkpoint path before ``os.chdir`` into DEIM or D-FINE.

    Relative paths such as ``DEIM/outputs/...`` are interpreted from ``experiments/``,
    not from the framework subdirectory after ``chdir``.
    """
    if not resume:
        return resume
    p = Path(resume)
    if p.is_absolute():
        return str(p)
    cand = (EXPERIMENTS_DIR / resume).resolve()
    if cand.is_file():
        return str(cand)
    cand = (Path.cwd() / resume).resolve()
    if cand.is_file():
        return str(cand)
    return str((EXPERIMENTS_DIR / resume).resolve())


# ---------------------------------------------------------------------------
# Framework helpers
# ---------------------------------------------------------------------------

def _setup_deim(config_path: str, resume: str):
    fw_dir = EXPERIMENTS_DIR / "DEIM"
    sys.path.insert(0, str(fw_dir))
    saved_cwd = os.getcwd()
    os.chdir(fw_dir)

    from engine.core import YAMLConfig
    from engine.solver import TASKS

    cfg = YAMLConfig(config_path, resume=resume)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    return solver, cfg, saved_cwd


def _setup_dfine(config_path: str, resume: str):
    fw_dir = EXPERIMENTS_DIR / "D-FINE"
    sys.path.insert(0, str(fw_dir))
    saved_cwd = os.getcwd()
    os.chdir(fw_dir)

    from src.core import YAMLConfig
    from src.solver import TASKS

    cfg = YAMLConfig(config_path, resume=resume)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    return solver, cfg, saved_cwd


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _dataset_label2category_map(data_loader):
    """Walk wrappers (e.g. Subset) to find CocoDetection.label2category."""
    ds = data_loader.dataset
    for _ in range(8):
        m = getattr(ds, "label2category", None)
        if m is not None:
            return m
        ds = getattr(ds, "dataset", None)
        if ds is None:
            break
    return None


def _resolve_test_ann_file(ann_file: str) -> Optional[str]:
    """Infer test annotation path from the configured val annotation path."""
    if not ann_file:
        return None
    ann_path = Path(ann_file)
    candidates = []
    name = ann_path.name
    if "instances_val" in name:
        candidates.append(ann_path.with_name(name.replace("instances_val", "instances_test")))
    if "instances_train" in name:
        candidates.append(ann_path.with_name(name.replace("instances_train", "instances_test")))
    if name != "instances_test.json":
        candidates.append(ann_path.with_name("instances_test.json"))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _resolve_test_img_folder(img_folder: str) -> str:
    """Infer test image folder while keeping dataset roots that already contain all images."""
    if not img_folder:
        return img_folder
    img_path = Path(img_folder)
    if img_path.name in {"val", "train"}:
        test_dir = img_path.with_name("test")
        if test_dir.exists():
            return str(test_dir)
    return img_folder


def _build_test_dataloader(cfg) -> Tuple[Optional[Any], Optional[str]]:
    """Build an eval-only test dataloader by cloning val_dataloader config."""
    val_cfg = cfg.yaml_cfg.get("val_dataloader", {})
    dataset_cfg = val_cfg.get("dataset", {})
    test_ann = _resolve_test_ann_file(dataset_cfg.get("ann_file", ""))
    if not test_ann:
        return None, None

    test_loader_cfg = deepcopy(val_cfg)
    test_loader_cfg["dataset"] = deepcopy(dataset_cfg)
    test_loader_cfg["dataset"]["ann_file"] = test_ann
    test_loader_cfg["dataset"]["img_folder"] = _resolve_test_img_folder(
        test_loader_cfg["dataset"].get("img_folder", "")
    )

    cfg.yaml_cfg["test_dataloader"] = test_loader_cfg
    loader = cfg.build_dataloader("test_dataloader")
    return loader, test_ann


@torch.no_grad()
def collect_predictions(
    model,
    postprocessor,
    data_loader,
    device,
    *,
    remap_mscoco_category: bool = False,
    label2category=None,
) -> List[Dict]:
    """Run inference and return predictions in COCO detection format."""
    model.eval()
    all_preds: List[Dict] = []

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
            for t in targets
        ]

        outputs = model(samples)
        orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessor(outputs, orig_sizes)

        for target, result in zip(targets, results):
            img_id = int(target["image_id"].flatten()[0].item())
            boxes = result["boxes"].cpu()
            scores = result["scores"].cpu()
            labels = result["labels"].cpu()

            # xyxy -> xywh
            xywh = boxes.clone()
            xywh[:, 2] -= xywh[:, 0]
            xywh[:, 3] -= xywh[:, 1]

            for j in range(len(scores)):
                lid = int(labels[j].item())
                # PostProcessor with remap_mscoco_category=True already emits COCO category ids.
                # Otherwise labels are train indices 0..N-1 — map via dataset (same as CocoEvaluatorTrainLabelMapping / CaS +1 for contiguous 1..N).
                if remap_mscoco_category:
                    cat_id = lid
                elif label2category is not None:
                    cat_id = int(label2category[lid])
                else:
                    cat_id = lid

                all_preds.append({
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": xywh[j].tolist(),
                    "score": scores[j].item(),
                })
    return all_preds


# ---------------------------------------------------------------------------
# CaS-compatible metric computation
# ---------------------------------------------------------------------------

def _build_coco_gt_dict(ann_file: str) -> Dict[str, Any]:
    """Read COCO annotation JSON and return as dict (for pycocotools)."""
    with open(ann_file, "r", encoding="utf-8") as f:
        gt = json.load(f)
    gt.setdefault("info", {"description": "eval", "version": "1.0", "year": 2025})
    return gt


def _is_dair_dataset(dataset_name: str) -> bool:
    low = dataset_name.lower()
    return "dair" in low or "dairv2x" in low


def _compute_difficulty_aps(
    coco_gt: Dict[str, Any],
    predictions: List[Dict],
    dair_categorical: bool,
) -> Dict[str, float]:
    """KITTI-style E/M/H AP@0.5 (same logic as CaS_DETR)."""
    anns = coco_gt.get("annotations", [])
    if not anns:
        return {"AP_easy": 0.0, "AP_moderate": 0.0, "AP_hard": 0.0}

    difficulties = [
        kitti_difficulty_from_coco_ann(a, dair_categorical_trunc=dair_categorical)
        for a in anns
    ]

    result = {}
    for level in ("easy", "moderate", "hard"):
        gt_mod = coco_gt_with_difficulty_iscrowd(coco_gt, difficulties, level)
        ce = run_coco_bbox_eval(gt_mod, predictions)
        result[f"AP_{level}"] = coco_ap_at_iou50_all(ce)

    return result


def compute_cas_metrics(
    ann_file: str,
    predictions: List[Dict],
    dataset_name: str,
) -> Tuple[Dict[str, Any], List[str]]:
    """Full CaS_DETR-compatible metrics from GT + predictions."""
    coco_gt = _build_coco_gt_dict(ann_file)
    categories = sorted(coco_gt.get("categories", []), key=lambda c: c["id"])
    class_names = [str(c["name"]) for c in categories]

    ce = run_coco_bbox_eval(coco_gt, predictions)
    if ce is None:
        empty = {k: 0.0 for k in [
            "mAP_0.5", "mAP_0.75", "mAP_0.5_0.95",
            "AP_small", "AP_medium", "AP_large",
            "AP_small_50", "AP_medium_50", "AP_large_50",
            "AP_easy", "AP_moderate", "AP_hard",
        ]}
        return empty, class_names

    stats = ce.stats
    s50, m50, l50 = coco_area_ap_at_iou50(ce)

    metrics: Dict[str, Any] = {
        "mAP_0.5": float(stats[1]),
        "mAP_0.75": float(stats[2]),
        "mAP_0.5_0.95": float(stats[0]),
        "AP_small": float(stats[3]) if len(stats) > 3 else 0.0,
        "AP_medium": float(stats[4]) if len(stats) > 4 else 0.0,
        "AP_large": float(stats[5]) if len(stats) > 5 else 0.0,
        "AP_small_50": s50,
        "AP_medium_50": m50,
        "AP_large_50": l50,
    }

    per50, per5095 = extract_per_category_ap_from_coco_eval(ce, categories)
    for name, v in per50.items():
        metrics[f"AP50_{name}"] = v
    for name, v in per5095.items():
        metrics[f"AP5095_{name}"] = v

    dair_cat = _is_dair_dataset(dataset_name)
    diff = _compute_difficulty_aps(coco_gt, predictions, dair_categorical=dair_cat)
    metrics.update(diff)

    return metrics, class_names


def _config_stub_for_benchmark(yaml_cfg: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    """Minimal config dict for ``run_detr_benchmark`` / ``model_display_name``."""
    ds = yaml_cfg.get("train_dataloader", {}).get("dataset", {})
    return {
        "data": {
            "data_root": str(ds.get("data_root", "")),
            "dataset_class": str(ds.get("type", "")),
        },
        "model": {},
        "_config_path": config_path,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _find_best_checkpoint(output_dir: str) -> Optional[str]:
    """Search for best checkpoint in common save locations."""
    d = Path(output_dir)
    for name in ("best_stg2.pth", "best.pth", "best_stg1.pth", "last.pth"):
        p = d / name
        if p.exists():
            return str(p)
    return None


def main():
    parser = argparse.ArgumentParser(description="CaS-compatible eval for DEIM / D-FINE")
    parser.add_argument("--framework", required=True, choices=["deim", "dfine"],
                        help="Which framework (deim or dfine)")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--resume", default=None,
                        help="Checkpoint path. If omitted, auto-detect from output_dir in config.")
    parser.add_argument("--model-name", default=None,
                        help="Model display name for CSV (default: config file stem)")
    parser.add_argument("--dataset-name", default=None,
                        help="Dataset display name for CSV (auto-detect from config paths)")
    parser.add_argument("--output-csv", default=None,
                        help="CSV path (default: <output_dir>/eval_metrics.csv)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", default="val,test",
                        help="Comma-separated eval splits to run (default: val,test)")
    args = parser.parse_args()

    config_path = str(Path(args.config).resolve())
    model_name = args.model_name or Path(args.config).stem

    # Auto-detect dataset name from config path
    dataset_name = args.dataset_name
    if dataset_name is None:
        low = config_path.lower()
        if "dairv2x" in low or "dair-v2x" in low or "dair_v2x" in low:
            dataset_name = "DAIR-V2X"
        elif "uadetrac" in low or "ua-detrac" in low or "ua_detrac" in low:
            dataset_name = "UA-DETRAC"
        else:
            dataset_name = "unknown"

    LOG.info("Framework: %s | Config: %s | Dataset: %s", args.framework, config_path, dataset_name)

    if args.resume:
        args.resume = _resolve_resume_path(args.resume)
        LOG.info("Resolved --resume to %s", args.resume)

    # Setup framework
    if args.framework == "deim":
        solver, cfg, saved_cwd = _setup_deim(config_path, args.resume or "")
    else:
        solver, cfg, saved_cwd = _setup_dfine(config_path, args.resume or "")

    # Resolve checkpoint, then solver.eval(): runs _setup() so solver has model/ema and loads weights.
    # Without eval(), DetSolver never runs BaseSolver._setup(), so attributes like solver.ema do not exist.
    resume_path = args.resume
    if not resume_path:
        output_dir = cfg.yaml_cfg.get("output_dir", "./outputs")
        ckpt = _find_best_checkpoint(output_dir)
        if ckpt is None:
            LOG.error("No checkpoint found in %s. Use --resume to specify.", output_dir)
            sys.exit(1)
        LOG.info("Auto-detected checkpoint: %s", ckpt)
        resume_path = ckpt
    if resume_path:
        resume_path = _resolve_resume_path(resume_path) or resume_path
    cfg.resume = resume_path

    solver.eval()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = solver.ema.module if solver.ema else solver.model
    model.to(device)
    model.eval()

    yaml_cfg = getattr(cfg, "yaml_cfg", {}) or {}
    remap_mscoco = bool(yaml_cfg.get("remap_mscoco_category", False))

    cfg_stub = _config_stub_for_benchmark(yaml_cfg, config_path)
    bench_dict = run_detr_benchmark(model, cfg_stub, args.framework, device, LOG)

    # Write CSV
    output_dir = Path(cfg.yaml_cfg.get("output_dir", "./outputs"))
    csv_path = Path(args.output_csv) if args.output_csv else output_dir / "eval_metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    split_names = [s.strip() for s in args.splits.split(",") if s.strip()]
    append_csv = csv_path.exists()
    wrote_any = False

    for split_name in split_names:
        if split_name == "val":
            data_loader = solver.val_dataloader
            split_cfg = cfg.yaml_cfg.get("val_dataloader", {}).get("dataset", {})
            ann_file = split_cfg.get("ann_file", "")
        elif split_name == "test":
            data_loader, ann_file = _build_test_dataloader(cfg)
            if data_loader is None or not ann_file:
                LOG.info("No test split found for config, skipping test evaluation.")
                continue
        else:
            LOG.warning("Unknown split '%s', skipping.", split_name)
            continue

        if not ann_file or not Path(ann_file).exists():
            LOG.warning("Cannot find %s annotation file: %s, skipping.", split_name, ann_file)
            continue

        l2c = _dataset_label2category_map(data_loader)
        LOG.info("Running inference on %s set ...", split_name)
        preds = collect_predictions(
            model,
            solver.postprocessor,
            data_loader,
            device,
            remap_mscoco_category=remap_mscoco,
            label2category=l2c,
        )
        LOG.info("Collected %d predictions for %s", len(preds), split_name)
        LOG.info("Computing CaS-compatible metrics from %s ...", ann_file)
        metrics, class_names = compute_cas_metrics(ann_file, preds, dataset_name)
        log_detr_eval_summary(LOG, split_name, metrics, bench_dict)

        write_eval_csv(
            csv_path,
            model=model_name,
            dataset=dataset_name,
            eval_split=split_name,
            metrics=metrics,
            class_names=class_names,
            append=append_csv,
            benchmark=bench_dict,
        )
        append_csv = True
        wrote_any = True

    if wrote_any:
        LOG.info("Wrote %s", csv_path)
    else:
        LOG.warning("No eval split was written to CSV.")

    os.chdir(saved_cwd)


if __name__ == "__main__":
    main()
