#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COCO_JSON=${1:-"$PROJECT_ROOT/a9_coco_raw/instances_train.json"}
IMAGES_DIR=${2:-"$PROJECT_ROOT/datasets/A9_coco/images/train"}
NUM=${3:-3}
OUT_DIR=${4:-"$PROJECT_ROOT/outputs/vis_samples"}
mkdir -p "$OUT_DIR"
cd "$PROJECT_ROOT"
python3 scripts/visualize_coco.py --coco_json "$COCO_JSON" --images_dir "$IMAGES_DIR" --num_images "$NUM" --no_show --output "$OUT_DIR"
echo "Visualizations saved to $OUT_DIR"
