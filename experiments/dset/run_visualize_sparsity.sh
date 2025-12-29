#!/bin/bash

# DSET Sparsity Visualization 可视化脚本
# 使用方法: ./run_visualize_sparsity.sh [参数覆盖...]
# 
# 示例:
#   ./run_visualize_sparsity.sh --image 000562.jpg --config configs/dset6_r18_ratio0.3.yaml --checkpoint logs/dset6_r18_20251224_153734/best_model.pth
#   ./run_visualize_sparsity.sh --image 000562.jpg --mode heatmap
#   ./run_visualize_sparsity.sh --image 000562.jpg --mode teaser --output_dir ./visualizations

# 默认参数
DEFAULT_CONFIG=${CONFIG:-configs/dset6_r18_ratio0.3.yaml}
DEFAULT_CHECKPOINT=${CHECKPOINT:-logs/dset6_r18_20251227_022803/best_model.pth}
DEFAULT_IMAGE=${IMAGE:-000562.jpg}
DEFAULT_MODE=${MODE:-heatmap}
DEFAULT_OUTPUT_DIR=${OUTPUT_DIR:-}
DEFAULT_DEVICE=${DEVICE:-cuda}
DEFAULT_TARGET_SIZE=${TARGET_SIZE:-1280}

# 解析命令行参数
CONFIG_FILE=$DEFAULT_CONFIG
CHECKPOINT_FILE=$DEFAULT_CHECKPOINT
IMAGE_FILE=$DEFAULT_IMAGE
MODE=$DEFAULT_MODE
OUTPUT_DIR=$DEFAULT_OUTPUT_DIR
DEVICE=$DEFAULT_DEVICE
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
        --image)
            IMAGE_FILE="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
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

echo "🎨 DSET Sparsity Visualization"
echo "============================================================"
echo "输入图像: $IMAGE_FILE"
echo "配置文件: $CONFIG_FILE"
echo "模型检查点: $CHECKPOINT_FILE"
echo "可视化模式: $MODE"
if [ -n "$OUTPUT_DIR" ]; then
    echo "输出目录: $OUTPUT_DIR"
else
    echo "输出目录: 图像所在目录"
fi
echo "设备: $DEVICE"
echo "推理尺寸: $TARGET_SIZE"
echo "============================================================"

# 检查图像文件是否存在
if [ ! -f "$IMAGE_FILE" ]; then
    echo "❌ 错误: 图像文件不存在: $IMAGE_FILE"
    echo "可用的图像文件:"
    ls -la *.jpg 2>/dev/null | head -5 || echo "  未找到图像文件"
    exit 1
fi

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
    echo "尝试查找可用的检查点:"
    find logs -name "best*.pth" -o -name "best_model.pth" -type f 2>/dev/null | head -5 || echo "  未找到检查点"
    exit 1
fi

# 验证模式参数
if [ "$MODE" != "teaser" ] && [ "$MODE" != "heatmap" ]; then
    echo "❌ 错误: 无效的可视化模式: $MODE"
    echo "有效的模式: 'teaser' 或 'heatmap'"
    exit 1
fi

# 创建输出目录（如果指定）
if [ -n "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# 构建命令参数
VISUALIZE_ARGS=(
    --image "$IMAGE_FILE"
    --config "$CONFIG_FILE"
    --checkpoint "$CHECKPOINT_FILE"
    --mode "$MODE"
    --device "$DEVICE"
    --target_size "$TARGET_SIZE"
)

if [ -n "$OUTPUT_DIR" ]; then
    VISUALIZE_ARGS+=(--output_dir "$OUTPUT_DIR")
fi

VISUALIZE_ARGS+=("${ARGS[@]}")

# 运行可视化脚本
python visualize_sparsity.py "${VISUALIZE_ARGS[@]}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ 可视化完成！"
    if [ -n "$OUTPUT_DIR" ]; then
        echo "结果保存在: $OUTPUT_DIR"
    else
        echo "结果保存在图像所在目录"
    fi
else
    echo ""
    echo "❌ 可视化失败，退出码: $EXIT_CODE"
    exit $EXIT_CODE
fi

