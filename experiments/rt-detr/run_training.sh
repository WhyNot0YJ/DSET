#!/bin/bash

# RT-DETR 训练脚本
# 使用方法: ./run_training.sh [config_file] [override_params...]
# 
# 示例:
#   ./run_training.sh configs/rtdetr_r34.yaml
#   ./run_training.sh configs/rtdetr_r18.yaml --epochs 100
#   ./run_training.sh configs/rtdetr_r34.yaml --batch_size 48
#   ./run_training.sh configs/rtdetr_r34.yaml --resume  # 自动从最新检查点恢复
#   ./run_training.sh configs/rtdetr_r34.yaml --resume_from_checkpoint logs/rtdetr_r34_20240101_120000/latest_checkpoint.pth

# 默认配置文件 (PResNet34 - 平衡速度和精度，适合路测)
CONFIG_FILE=${1:-configs/rtdetr_r34.yaml}

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误: 配置文件不存在: $CONFIG_FILE"
    echo "可用的配置文件:"
    ls -la configs/*.yaml 2>/dev/null || echo "  未找到配置文件"
    exit 1
fi

echo "🚀 开始训练 RT-DETR"
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
    # 查找所有 latest_checkpoint.pth 文件，按修改时间排序（使用ls -t）
    LATEST_CHECKPOINT=$(find logs -name "latest_checkpoint.pth" -type f 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
    
    if [ -n "$LATEST_CHECKPOINT" ] && [ -f "$LATEST_CHECKPOINT" ]; then
        echo "📦 找到最新检查点: $LATEST_CHECKPOINT"
        ARGS+=("--resume_from_checkpoint" "$LATEST_CHECKPOINT")
    else
        echo "⚠️  未找到检查点，将从头开始训练"
    fi
fi

# 创建日志目录
mkdir -p logs

# 开始训练 - 直接传递配置文件给Python脚本
python3 train.py \
    --config $CONFIG_FILE \
    "${ARGS[@]}"  # 传递处理后的参数

echo "✅ 训练完成！"