#!/bin/bash
# DSET训练启动脚本

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 训练配置（使用最推荐的dset6_r34配置 - 平衡速度和精度，适合路测）
CONFIG="${1:-configs/dset6_r34.yaml}"

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

# 启动训练
echo "=========================================="
echo "启动DSET训练"
echo "配置文件: $CONFIG"
echo "=========================================="
echo ""
echo "提示: 可以通过参数指定其他配置，例如："
echo "  ./run_training.sh configs/dset3_r34.yaml"
echo "  ./run_training.sh configs/dset2_r18.yaml"
echo "  ./run_training.sh configs/dset6_r34.yaml --resume  # 自动从最新检查点恢复"
echo ""
echo "=========================================="

python train.py --config $CONFIG "${ARGS[@]}"

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
