#!/bin/bash

# YOLOv10 评估脚本 - 限制最大检测框数量为 100
# 使用方法: ./run_eval_maxdet100.sh [参数覆盖...]
# 
# 示例:
#   ./run_eval_maxdet100.sh --checkpoint logs/yolo_v10s_20251202_202549/weights/best.pt --data_yaml /root/autodl-tmp/datasets/DAIR-V2X_YOLO/dairv2x.yaml
#   ./run_eval_maxdet100.sh --checkpoint logs/xxx/weights/best.pt --data_yaml /path/to/data.yaml --max_det 100 --output results_maxdet100.json

# 默认参数
DEFAULT_CHECKPOINT=${CHECKPOINT:-logs/yolo_v10s_20251202_112836/best_model.pth}
DEFAULT_DATA_YAML=${DATA_YAML:-/root/autodl-tmp/datasets/DAIR-V2X_YOLO/dairv2x.yaml}
DEFAULT_MAX_DET=${MAX_DET:-100}
DEFAULT_CONF=${CONF:-0.001}
DEFAULT_IOU=${IOU:-0.6}
DEFAULT_IMGSZ=${IMGSZ:-1280}
DEFAULT_DEVICE=${DEVICE:-cuda}
DEFAULT_SPLIT=${SPLIT:-val}
DEFAULT_OUTPUT=${OUTPUT:-results_yolo_v10s_20251202_112836_maxdet100.json}

# 解析命令行参数
CHECKPOINT_FILE=$DEFAULT_CHECKPOINT
DATA_YAML=$DEFAULT_DATA_YAML
MAX_DET=$DEFAULT_MAX_DET
CONF_THRESHOLD=$DEFAULT_CONF
IOU_THRESHOLD=$DEFAULT_IOU
IMGSZ=$DEFAULT_IMGSZ
DEVICE=$DEFAULT_DEVICE
SPLIT=$DEFAULT_SPLIT
OUTPUT=$DEFAULT_OUTPUT

# 解析参数
ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT_FILE="$2"
            shift 2
            ;;
        --data_yaml)
            DATA_YAML="$2"
            shift 2
            ;;
        --max_det)
            MAX_DET="$2"
            shift 2
            ;;
        --conf)
            CONF_THRESHOLD="$2"
            shift 2
            ;;
        --iou)
            IOU_THRESHOLD="$2"
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
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
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

# 检查checkpoint文件是否存在
if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo "❌ 错误: 模型检查点文件不存在: $CHECKPOINT_FILE"
    echo "请检查路径是否正确"
    exit 1
fi

# 如果是 .pth 文件，检查是否有对应的 .pt 文件，如果没有则提示转换
if [[ "$CHECKPOINT_FILE" == *.pth ]]; then
    PT_FILE="${CHECKPOINT_FILE%.pth}.pt"
    if [ ! -f "$PT_FILE" ]; then
        echo "ℹ️  检测到 .pth 文件，脚本将自动转换为 .pt 格式"
    else
        echo "ℹ️  找到对应的 .pt 文件: $PT_FILE"
    fi
fi

echo "🚀 开始评估 YOLOv10 (max_det=$MAX_DET)"
echo "============================================================"
echo "模型检查点: $CHECKPOINT_FILE"
echo "数据集配置: $DATA_YAML"
echo "最大检测框数: $MAX_DET"
echo "置信度阈值: $CONF_THRESHOLD"
echo "IoU阈值: $IOU_THRESHOLD"
echo "图像尺寸: $IMGSZ"
echo "设备: $DEVICE"
echo "数据集分割: $SPLIT"
echo "结果输出: $OUTPUT"
echo "============================================================"

# 运行评估脚本
python3 eval_maxdet100.py \
    --checkpoint "$CHECKPOINT_FILE" \
    --data_yaml "$DATA_YAML" \
    --max_det "$MAX_DET" \
    --conf "$CONF_THRESHOLD" \
    --iou "$IOU_THRESHOLD" \
    --imgsz "$IMGSZ" \
    --device "$DEVICE" \
    --split "$SPLIT" \
    --output "$OUTPUT" \
    "${ARGS[@]}"

echo ""
echo "✅ 评估完成！"

