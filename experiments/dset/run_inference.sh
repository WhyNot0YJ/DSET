#!/bin/bash

# DSET (Dual-Sparse Expert Transformer) 批量推理脚本
# 使用方法: ./run_inference.sh [参数覆盖...]
# 
# 示例:
#   ./run_inference.sh
#   ./run_inference.sh --conf 0.1 --max_images 10
#   ./run_inference.sh --image_dir datasets/DAIR-V2X/image --config configs/dset6_r18.yaml

# 默认参数
DEFAULT_CONFIG=${CONFIG:-configs/dset6_r18.yaml}
DEFAULT_CHECKPOINT=${CHECKPOINT:-logs/best_model.pth}
DEFAULT_IMAGE_DIR=${IMAGE_DIR:-/root/autodl-fs/datasets/DAIR-V2X/image}
DEFAULT_OUTPUT_DIR=${OUTPUT_DIR:-/root/autodl-fs/datasets/DAIR-V2X/image_results}
DEFAULT_CONF=${CONF:-0.5}
DEFAULT_DEVICE=${DEVICE:-cuda}
DEFAULT_MAX_IMAGES=${MAX_IMAGES:-50}
DEFAULT_TARGET_SIZE=${TARGET_SIZE:-1280}

# 解析命令行参数
CONFIG_FILE=$DEFAULT_CONFIG
CHECKPOINT_FILE=$DEFAULT_CHECKPOINT
IMAGE_DIR=$DEFAULT_IMAGE_DIR
OUTPUT_DIR=$DEFAULT_OUTPUT_DIR
CONF_THRESHOLD=$DEFAULT_CONF
DEVICE=$DEFAULT_DEVICE
MAX_IMAGES=$DEFAULT_MAX_IMAGES
TARGET_SIZE=$DEFAULT_TARGET_SIZE

# 解析参数
ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
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
        --target_size)
            TARGET_SIZE="$2"
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

echo "🚀 开始批量推理 DSET"
echo "="*60
echo "配置文件: $CONFIG_FILE"
echo "模型检查点: $CHECKPOINT_FILE"
echo "输入图像目录: $IMAGE_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "置信度阈值: $CONF_THRESHOLD"
echo "设备: $DEVICE"
echo "最大处理图像数: $MAX_IMAGES"
echo "推理尺寸: $TARGET_SIZE"
echo "="*60

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误: 配置文件不存在: $CONFIG_FILE"
    echo "可用的配置文件:"
    ls -la configs/*.yaml 2>/dev/null || echo "  未找到配置文件"
    exit 1
fi

# 检查检查点是否存在
if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo "❌ 错误: 模型检查点不存在: $CHECKPOINT_FILE"
    echo "可用的检查点:"
    find logs -name "*.pth" -type f 2>/dev/null | head -5 || echo "  未找到检查点"
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
    --image_dir "$IMAGE_DIR" \
    --config "$CONFIG_FILE" \
    --checkpoint "$CHECKPOINT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --conf "$CONF_THRESHOLD" \
    --device "$DEVICE" \
    --max_images "$MAX_IMAGES" \
    --target_size "$TARGET_SIZE" \
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

