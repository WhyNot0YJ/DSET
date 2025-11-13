#!/usr/bin/env bash
set -euo pipefail

# COCO2017 下载与整理脚本
# 目录结构：
#   /workspace/datasets/coco/
#     ├── images/{train2017,val2017,test2017}
#     ├── annotations/{instances_train2017.json,instances_val2017.json,image_info_test-dev2017.json}
#     ├── train2017.txt / val2017.txt / test-dev2017.txt （ultralytics 索引）

ROOT_DIR="${1:-/workspace}"
DATASETS_DIR="$ROOT_DIR/experiments/datasets"
COCO_DIR="$DATASETS_DIR/coco"
IMG_DIR="$COCO_DIR/images"
ANN_DIR="$COCO_DIR/annotations"

mkdir -p "$IMG_DIR" "$ANN_DIR"

echo "[INFO] Root: $ROOT_DIR"
echo "[INFO] Datasets: $DATASETS_DIR"
echo "[INFO] COCO: $COCO_DIR"

cd "$COCO_DIR"

# 下载图片 zip（如已存在则跳过）
download_if_absent() {
  local url="$1"; shift
  local out="$1"; shift
  if [ -f "$out" ]; then
    echo "[INFO] 已存在：$out，跳过下载"
  else
    echo "[INFO] 下载：$url -> $out"
    curl -L "$url" -o "$out"
  fi
}

download_if_absent "http://images.cocodataset.org/zips/train2017.zip" "train2017.zip"
download_if_absent "http://images.cocodataset.org/zips/val2017.zip"   "val2017.zip"
download_if_absent "http://images.cocodataset.org/zips/test2017.zip"  "test2017.zip"
download_if_absent "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "annotations_trainval2017.zip"
download_if_absent "http://images.cocodataset.org/annotations/image_info_test2017.zip"      "image_info_test2017.zip"

# 解压（存在则跳过）
unzip_if_needed() {
  local zip="$1"; shift
  local target_dir="$1"; shift
  if [ -d "$target_dir" ]; then
    echo "[INFO] 已存在目录：$target_dir，跳过解压"
  else
    echo "[INFO] 解压：$zip -> $target_dir"
    unzip -q "$zip" -d .
  fi
}

unzip_if_needed train2017.zip train2017
unzip_if_needed val2017.zip   val2017
unzip_if_needed test2017.zip  test2017
unzip_if_needed annotations_trainval2017.zip annotations
unzip_if_needed image_info_test2017.zip      annotations

# 规范图片位置到 images/
mkdir -p "$IMG_DIR/train2017" "$IMG_DIR/val2017" "$IMG_DIR/test2017"
if [ -d "train2017" ] && [ ! -d "$IMG_DIR/train2017/copy_ok" ]; then
  echo "[INFO] 移动 train2017 -> images/train2017"
  rsync -a --remove-source-files train2017/ "$IMG_DIR/train2017/"
  touch "$IMG_DIR/train2017/copy_ok"
fi
if [ -d "val2017" ] && [ ! -d "$IMG_DIR/val2017/copy_ok" ]; then
  echo "[INFO] 移动 val2017 -> images/val2017"
  rsync -a --remove-source-files val2017/ "$IMG_DIR/val2017/"
  touch "$IMG_DIR/val2017/copy_ok"
fi
if [ -d "test2017" ] && [ ! -d "$IMG_DIR/test2017/copy_ok" ]; then
  echo "[INFO] 移动 test2017 -> images/test2017"
  rsync -a --remove-source-files test2017/ "$IMG_DIR/test2017/"
  touch "$IMG_DIR/test2017/copy_ok"
fi

# annotations 保持在 $ANN_DIR
if [ -d "annotations" ]; then
  echo "[INFO] 同步 annotations -> $ANN_DIR"
  rsync -a annotations/ "$ANN_DIR/"
fi

# 生成 Ultralytics 索引文件（相对路径相对于 path 字段）
make_list() {
  local split="$1"
  local list_file="$COCO_DIR/${split}2017.txt"
  local base="$IMG_DIR/${split}2017"
  if [ -f "$list_file" ]; then
    echo "[INFO] 已存在：$list_file"
  else
    echo "[INFO] 生成：$list_file"
    find "$base" -type f -name "*.jpg" | LC_ALL=C sort > "$list_file"
  fi
}

make_list train
make_list val
make_list test-dev

echo "[OK] COCO2017 就绪：$COCO_DIR"
echo "images: $IMG_DIR"
echo "annotations: $ANN_DIR"


