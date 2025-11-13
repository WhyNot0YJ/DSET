#!/usr/bin/env bash
set -euo pipefail

# 在仓库根目录下为 RT-DETR 的默认相对路径创建软链接：
# external/RT-DETR/rtdetrv2_pytorch/configs/dataset/coco_detection.yml 使用 ./dataset/coco/

ROOT_DIR="${1:-/workspace}"
DATASETS_DIR="$ROOT_DIR/experiments/datasets"
COCO_REAL="$DATASETS_DIR/coco"
COCO_LINK="/workspace/dataset/coco"

mkdir -p "/workspace/dataset"

if [ -d "$COCO_REAL" ]; then
  if [ -L "$COCO_LINK" ] || [ -d "$COCO_LINK" ]; then
    echo "[INFO] 已存在：$COCO_LINK"
  else
    ln -s "$COCO_REAL" "$COCO_LINK"
    echo "[OK] 链接创建：$COCO_LINK -> $COCO_REAL"
  fi
else
  echo "[ERROR] 未找到数据集目录：$COCO_REAL，请先运行 scripts/prepare_coco2017.sh"
  exit 1
fi


