#!/bin/bash
set -e

VERSION="8"
CONFIG_FILE=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)
            VERSION="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

VERSION="${VERSION#v}"
if [[ -z "$CONFIG_FILE" ]]; then
    CONFIG_FILE="configs/yolov${VERSION}n_dairv2x.yaml"
fi

echo "🚀 训练 YOLOv${VERSION}"
echo "配置文件: ${CONFIG_FILE}"

python3 train.py \
    --version "${VERSION}" \
    --config "${CONFIG_FILE}" \
    "${EXTRA_ARGS[@]}"
