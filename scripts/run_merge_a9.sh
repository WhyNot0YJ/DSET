#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INPUT_DIR=${1:-"$PROJECT_ROOT/a9_coco_raw"}
OUTPUT_JSON=${2:-"$PROJECT_ROOT/a9_coco_raw/instances_train.json"}
cd "$PROJECT_ROOT"
python3 scripts/merge_a9_coco.py --input_dir "$INPUT_DIR" --output "$OUTPUT_JSON"
TARGET_DIR="$PROJECT_ROOT/datasets/A9_coco/annotations"
if [ -d "$TARGET_DIR" ]; then
    if cp "$OUTPUT_JSON" "$TARGET_DIR/instances_train.json" 2>/dev/null; then
        echo "Copied to $TARGET_DIR/instances_train.json"
    else
        echo "[warn] Could not copy to $TARGET_DIR (permission denied?). Please copy manually." >&2
    fi
fi
