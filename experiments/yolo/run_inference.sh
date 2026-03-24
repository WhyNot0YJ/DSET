#!/bin/bash
set -e

VERSION="8"
DATASET="dairv2x"
DATASET_REGISTRY="configs/datasets.yaml"
CHECKPOINT="logs/*/weights/best.pt"
IMAGE_DIR=""
OUTPUT_DIR=""
CONF="0.5"
IMGSZ="640"
DEVICE="cuda"
MAX_IMAGES="50"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)
            VERSION="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --dataset_registry)
            DATASET_REGISTRY="$2"
            shift 2
            ;;
        --image_dir)
            IMAGE_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --conf)
            CONF="$2"
            shift 2
            ;;
        --imgsz)
            IMGSZ="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --max_images)
            MAX_IMAGES="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

VERSION="${VERSION#v}"

echo "🚀 推理 YOLOv${VERSION}"
echo "🗂️  数据集 ${DATASET}"

CMD_ARGS=(
    --version "${VERSION}"
    --dataset "${DATASET}"
    --dataset_registry "${DATASET_REGISTRY}"
    --checkpoint "${CHECKPOINT}"
    --conf "${CONF}"
    --imgsz "${IMGSZ}"
    --device "${DEVICE}"
    --max_images "${MAX_IMAGES}"
)

if [[ -n "${IMAGE_DIR}" ]]; then
    CMD_ARGS+=(--image_dir "${IMAGE_DIR}")
fi

if [[ -n "${OUTPUT_DIR}" ]]; then
    CMD_ARGS+=(--output_dir "${OUTPUT_DIR}")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD_ARGS+=("${EXTRA_ARGS[@]}")
fi

python3 batch_inference.py "${CMD_ARGS[@]}"
