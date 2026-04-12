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
DEVICE="${DEVICE:-cuda}"
METRIC_INDEX="${METRIC_INDEX:-0}"        # 0 => mAP@[0.5:0.95]
OVERWRITE="${OVERWRITE:-0}"              # 1 to replace existing curve keys
DRY_RUN="${DRY_RUN:-0}"                  # 1 to generate synthetic benchmark data

OUTPUT_JSON="${OUTPUT_JSON:-$SCRIPT_DIR/benchmark_pruning_curve_results.json}"
OUTPUT_PLOT="${OUTPUT_PLOT:-$SCRIPT_DIR/pruning_tradeoff.pdf}"
INFERENCE_RATIOS="${INFERENCE_RATIOS:-0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"

# Model A (DAIR)
CONFIG_A="${CONFIG_A:-experiments/CaS-DETR/configs/dataset/ablation/cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x.yml}"
RESUME_A="${RESUME_A:-experiments/CaS-DETR/outputs/ablation/cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x/best_stg2.pth}"
CURVE_NAME_A="${CURVE_NAME_A:-CaS_DETR_dair}"

# Model B (UA)
CONFIG_B="${CONFIG_B:-experiments/CaS-DETR/configs/dataset/ablation/cas_deim_moe4_cass_caip_base05_a10_hgnetv2_s_uadetrac.yml}"
RESUME_B="${RESUME_B:-experiments/CaS-DETR/outputs/ablation/base05_a10/cas_deim_moe4_cass_caip_base05_a10_hgnetv2_s_uadetrac/best_stg2.pth}"
CURVE_NAME_B="${CURVE_NAME_B:-CaS_DETR_ua}"

CMD=(
  python3 experiments/analysis/benchmark_pruning_curve.py
  --mode "$MODE"
  --output_json "$OUTPUT_JSON"
  --output_plot "$OUTPUT_PLOT"
  --inference_ratios $INFERENCE_RATIOS
  --device "$DEVICE"
  --metric_index "$METRIC_INDEX"
  --curve_name_a "$CURVE_NAME_A"
  --curve_name_b "$CURVE_NAME_B"
)

if [[ "$OVERWRITE" == "1" ]]; then
  CMD+=(--overwrite)
fi

if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=(--dry_run)
fi

if [[ "$MODE" == "benchmark" || "$MODE" == "both" ]]; then
  for f in "$CONFIG_A" "$CONFIG_B"; do
    [[ -f "$f" ]] || { echo "Missing config: $f"; exit 1; }
  done
  if [[ "$DRY_RUN" != "1" ]]; then
    for f in "$RESUME_A" "$RESUME_B"; do
      [[ -f "$f" ]] || { echo "Missing checkpoint: $f"; exit 1; }
    done
  fi
  CMD+=(
    --config_a "$CONFIG_A"
    --resume_a "$RESUME_A"
    --config_b "$CONFIG_B"
    --resume_b "$RESUME_B"
  )
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
echo "Done. JSON: $OUTPUT_JSON"
echo "Done. Plot: $OUTPUT_PLOT"
