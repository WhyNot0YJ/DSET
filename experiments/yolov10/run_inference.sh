#!/bin/bash

# YOLOv10 批量推理脚本
# 使用方法: ./run_inference.sh [参数覆盖...]
# 
# 示例:
#   ./run_inference.sh
#   ./run_inference.sh --conf 0.1 --max_images 10
#   ./run_inference.sh --image_dir datasets/DAIR-V2X/image --checkpoint logs/yolo_v10n_20240101_120000/weights/best.pt

# 默认参数
DEFAULT_CHECKPOINT=${CHECKPOINT:-logs/*/weights/best.pt}
DEFAULT_IMAGE_DIR=${IMAGE_DIR:-/root/autodl-tmp/datasets/DAIR-V2X/image}
DEFAULT_OUTPUT_DIR=${OUTPUT_DIR:-/root/autodl-tmp/datasets/DAIR-V2X/image_results}
DEFAULT_CONF=${CONF:-0.5}
DEFAULT_DEVICE=${DEVICE:-cuda}
DEFAULT_MAX_IMAGES=${MAX_IMAGES:-50}

# 解析命令行参数
CHECKPOINT_FILE=$DEFAULT_CHECKPOINT
IMAGE_DIR=$DEFAULT_IMAGE_DIR
OUTPUT_DIR=$DEFAULT_OUTPUT_DIR
CONF_THRESHOLD=$DEFAULT_CONF
DEVICE=$DEFAULT_DEVICE
MAX_IMAGES=$DEFAULT_MAX_IMAGES

# 解析参数
ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT_FILE="$2"
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
            CONF_THRESHOLD="$2"
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
            ARGS+=("$1")
            shift
            ;;
    esac
done

# 如果checkpoint包含通配符，查找最新的
if [[ "$CHECKPOINT_FILE" == *"*"* ]]; then
    CHECKPOINT_FILE=$(find logs -name "best.pt" -type f 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
    if [ -z "$CHECKPOINT_FILE" ]; then
        echo "❌ 错误: 未找到模型检查点"
        echo "请指定 --checkpoint 参数"
        exit 1
    fi
fi

echo "🚀 开始批量推理 YOLOv10"
echo "============================================================"
echo "模型检查点: $CHECKPOINT_FILE"
echo "输入图像目录: $IMAGE_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "置信度阈值: $CONF_THRESHOLD"
echo "设备: $DEVICE"
echo "最大处理图像数: $MAX_IMAGES"
echo "============================================================"

# 检查检查点是否存在
if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo "❌ 错误: 模型检查点不存在: $CHECKPOINT_FILE"
    echo "可用的检查点:"
    find logs -name "*.pt" -type f 2>/dev/null | head -5 || echo "  未找到检查点"
    exit 1
fi

# 检查图像目录是否存在
if [ ! -d "$IMAGE_DIR" ]; then
    echo "❌ 错误: 图像目录不存在: $IMAGE_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行批量推理
python batch_inference.py \
    --checkpoint "$CHECKPOINT_FILE" \
    --image_dir "$IMAGE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --conf "$CONF_THRESHOLD" \
    --device "$DEVICE" \
    --max_images "$MAX_IMAGES" \
    "${ARGS[@]}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ 批量推理完成！"
    echo "结果保存在: $OUTPUT_DIR"
else
    echo ""
    echo "❌ 批量推理失败，退出码: $EXIT_CODE"
    exit $EXIT_CODE
fi

