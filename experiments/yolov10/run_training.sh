#!/bin/bash

# YOLOv10 训练脚本
# 使用方法: ./run_training.sh [config_file] [override_params...]
# 
# 示例:
#   ./run_training.sh configs/yolov10s_dairv2x.yaml
#   ./run_training.sh configs/yolov10s_dairv2x.yaml --epochs 100
#   ./run_training.sh configs/yolov10s_dairv2x.yaml --batch_size 48
#   ./run_training.sh configs/yolov10s_dairv2x.yaml --resume  # 自动从最新检查点恢复
#   ./run_training.sh configs/yolov10s_dairv2x.yaml --resume_from_checkpoint logs/yolo_v10s_20240101_120000/weights/best.pt

# 默认配置文件
CONFIG_FILE=${1:-configs/yolov10s_dairv2x.yaml}

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误: 配置文件不存在: $CONFIG_FILE"
    echo "可用的配置文件:"
    ls -la configs/*.yaml 2>/dev/null || echo "  未找到配置文件"
    exit 1
fi

echo "🚀 开始训练 YOLOv10"
echo "配置文件: $CONFIG_FILE"

# 检查是否有 --resume 参数（自动恢复）
AUTO_RESUME=false
ARGS=()
for arg in "${@:2}"; do
    if [ "$arg" == "--resume" ] || [ "$arg" == "-r" ]; then
        AUTO_RESUME=true
    else
        ARGS+=("$arg")
    fi
done

# 如果启用自动恢复，查找最新的检查点
if [ "$AUTO_RESUME" = true ]; then
    # 查找所有 best.pt 文件，按修改时间排序
    LATEST_CHECKPOINT=$(find logs -name "best.pt" -type f 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
    
    if [ -n "$LATEST_CHECKPOINT" ] && [ -f "$LATEST_CHECKPOINT" ]; then
        echo "📦 找到最新检查点: $LATEST_CHECKPOINT"
        ARGS+=("--resume_from_checkpoint" "$LATEST_CHECKPOINT")
    else
        echo "⚠️  未找到检查点，将从头开始训练"
    fi
fi

# 创建日志目录
mkdir -p logs

# 开始训练
python3 train.py \
    --config "$CONFIG_FILE" \
    "${ARGS[@]}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ 训练完成！"
else
    echo ""
    echo "❌ 训练失败，退出码: $EXIT_CODE"
    exit $EXIT_CODE
fi

