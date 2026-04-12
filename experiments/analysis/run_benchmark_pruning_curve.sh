#!/usr/bin/env bash
set -euo pipefail

# Quick launcher for benchmark_pruning_curve.py
#
# Default behavior:
#   MODE=plot, using existing JSON to generate plot quickly.
#
# Examples:
#   ./experiments/analysis/run_benchmark_pruning_curve.sh
#   MODE=both OVERWRITE=1 ./experiments/analysis/run_benchmark_pruning_curve.sh
#   MODE=both DRY_RUN=1 ./experiments/analysis/run_benchmark_pruning_curve.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

MODE="${MODE:-plot}"                     # plot | benchmark | both
EVAL_SPLIT="${EVAL_SPLIT:-test}"         # val | test
DEVICE="${DEVICE:-cuda}"
METRIC_INDEX="${METRIC_INDEX:-0}"        # 0 => mAP@[0.5:0.95]
METRIC_INDICES="${METRIC_INDICES:-0 1 3}"  # 0 mAP@[0.5:0.95], 1 mAP50, 3 AP_S small
OVERWRITE="${OVERWRITE:-0}"              # 1 to replace existing curve keys
DRY_RUN="${DRY_RUN:-0}"                  # 1 to generate synthetic benchmark data
ENABLE_MODEL_B="${ENABLE_MODEL_B:-0}"    # 0 => only model A (DAIR), 1 => include model B (UA)

OUTPUT_JSON="${OUTPUT_JSON:-$SCRIPT_DIR/benchmark_pruning_curve_results.json}"
OUTPUT_PLOT="${OUTPUT_PLOT:-$SCRIPT_DIR/pruning_tradeoff.pdf}"
INFERENCE_RATIOS="${INFERENCE_RATIOS:-0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"

# Model A (DAIR)
CONFIG_A="${CONFIG_A:-experiments/CaS-DETR/configs/dataset/ablation/cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x.yml}"
RESUME_A="${RESUME_A:-experiments/CaS-DETR/outputs/ablation/cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x/best_stg2.pth}"

# Model B (UA)
CONFIG_B="${CONFIG_B:-experiments/CaS-DETR/configs/dataset/ablation/cas_deim_moe4_cass_caip_base05_a10_hgnetv2_s_uadetrac.yml}"
RESUME_B="${RESUME_B:-experiments/CaS-DETR/outputs/ablation/base05_a10/cas_deim_moe4_cass_caip_base05_a10_hgnetv2_s_uadetrac/best_stg2.pth}"

# Legend keys: order matches METRIC_INDICES default 0 1 3.
if [[ "$ENABLE_MODEL_B" == "1" ]]; then
  CURVE_A_1="${CURVE_A_1:-Overall (mAP) - DAIR}"
  CURVE_A_2="${CURVE_A_2:-mAP50 - DAIR}"
  CURVE_A_3="${CURVE_A_3:-Small (AP_S) - DAIR}"
else
  CURVE_A_1="${CURVE_A_1:-Overall (mAP)}"
  CURVE_A_2="${CURVE_A_2:-mAP50}"
  CURVE_A_3="${CURVE_A_3:-Small (AP_S)}"
fi
CURVE_B_1="${CURVE_B_1:-Overall (mAP) - UA}"
CURVE_B_2="${CURVE_B_2:-mAP50 - UA}"
CURVE_B_3="${CURVE_B_3:-Small (AP_S) - UA}"

# Test split overrides for val_dataloader (used when EVAL_SPLIT=test):
# DAIR uses shared img root + instances_test.json
TEST_IMG_A="${TEST_IMG_A:-/root/autodl-fs/datasets/DAIR-V2X}"
TEST_ANN_A="${TEST_ANN_A:-/root/autodl-fs/datasets/DAIR-V2X/annotations/instances_test.json}"
# UA-DETRAC uses dedicated /test folder + instances_test.json
TEST_IMG_B="${TEST_IMG_B:-/root/autodl-fs/datasets/UA-DETRAC_COCO/test}"
TEST_ANN_B="${TEST_ANN_B:-/root/autodl-fs/datasets/UA-DETRAC_COCO/annotations/instances_test.json}"

CMD=(
  python3 experiments/analysis/benchmark_pruning_curve.py
  --mode "$MODE"
  --output_json "$OUTPUT_JSON"
  --output_plot "$OUTPUT_PLOT"
  --inference_ratios $INFERENCE_RATIOS
  --device "$DEVICE"
  --eval_split "$EVAL_SPLIT"
  --metric_index "$METRIC_INDEX"
  --metric_indices $METRIC_INDICES
  --curve_names_a "$CURVE_A_1" "$CURVE_A_2" "$CURVE_A_3"
)

if [[ "$ENABLE_MODEL_B" == "1" ]]; then
  CMD+=(--curve_names_b "$CURVE_B_1" "$CURVE_B_2" "$CURVE_B_3")
fi

if [[ "$OVERWRITE" == "1" ]]; then
  CMD+=(--overwrite)
fi

if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=(--dry_run)
fi

if [[ "$MODE" == "benchmark" || "$MODE" == "both" ]]; then
  for f in "$CONFIG_A"; do
    [[ -f "$f" ]] || { echo "Missing config: $f"; exit 1; }
  done
  if [[ "$DRY_RUN" != "1" ]]; then
    for f in "$RESUME_A"; do
      [[ -f "$f" ]] || { echo "Missing checkpoint: $f"; exit 1; }
    done
  fi
  CMD+=(
    --config_a "$CONFIG_A"
    --resume_a "$RESUME_A"
  )
  if [[ "$ENABLE_MODEL_B" == "1" ]]; then
    [[ -f "$CONFIG_B" ]] || { echo "Missing config: $CONFIG_B"; exit 1; }
    if [[ "$DRY_RUN" != "1" ]]; then
      [[ -f "$RESUME_B" ]] || { echo "Missing checkpoint: $RESUME_B"; exit 1; }
    fi
    CMD+=(
      --config_b "$CONFIG_B"
      --resume_b "$RESUME_B"
    )
  fi
  if [[ "$EVAL_SPLIT" == "test" ]]; then
    for f in "$TEST_ANN_A"; do
      [[ -f "$f" ]] || { echo "Missing test annotation: $f"; exit 1; }
    done
    CMD+=(
      --extra_update_a
      "val_dataloader.dataset.img_folder=$TEST_IMG_A"
      "val_dataloader.dataset.ann_file=$TEST_ANN_A"
    )
    if [[ "$ENABLE_MODEL_B" == "1" ]]; then
      [[ -f "$TEST_ANN_B" ]] || { echo "Missing test annotation: $TEST_ANN_B"; exit 1; }
      CMD+=(
        --extra_update_b
        "val_dataloader.dataset.img_folder=$TEST_IMG_B"
        "val_dataloader.dataset.ann_file=$TEST_ANN_B"
      )
    fi
  fi
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
echo "Done. JSON: $OUTPUT_JSON"
echo "Done. Plot: $OUTPUT_PLOT"
