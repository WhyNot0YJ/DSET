#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
A9_ROOT=${1:-"$PROJECT_ROOT/datasets/A9"}
OUTPUT_DIR=${2:-"$PROJECT_ROOT/a9_coco_raw"}
mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"
python3 scripts/convert_a9_to_coco.py --a9_root "$A9_ROOT" --output_dir "$OUTPUT_DIR"
