#!/bin/bash
# DSET训练启动脚本

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 训练配置（使用最推荐的dset6_r34配置 - 平衡速度和精度，适合路测）
CONFIG="${1:-configs/dset6_r34.yaml}"

# 检查是否有 --resume 参数（支持自动查找或指定路径）
RESUME_CHECKPOINT=""
ARGS=()
i=2
while [ $i -le $# ]; do
    arg="${!i}"
    
    # 检查 --resume=path 格式
    if [[ "$arg" == --resume=* ]]; then
        RESUME_CHECKPOINT="${arg#*=}"
    # 检查 --resume 或 -r 参数
    elif [ "$arg" == "--resume" ] || [ "$arg" == "-r" ]; then
        # 检查下一个参数是否存在且不是另一个选项
        next_i=$((i+1))
        if [ $next_i -le $# ] && [[ ! "${!next_i}" == -* ]]; then
            RESUME_CHECKPOINT="${!next_i}"
            i=$next_i  # 跳过下一个参数，因为它已经被用作检查点路径
        else
            # 没有指定路径，自动查找最新的检查点
            RESUME_CHECKPOINT="AUTO"
        fi
    else
        ARGS+=("$arg")
    fi
    i=$((i+1))
done

# 如果指定了恢复检查点，处理路径
if [ -n "$RESUME_CHECKPOINT" ]; then
    if [ "$RESUME_CHECKPOINT" == "AUTO" ]; then
        # 自动查找最新的检查点
        # 查找所有 latest_checkpoint.pth 文件，按修改时间排序
        if [ -d "logs" ]; then
            # 尝试使用 find + ls -t（适用于大多数 Unix 系统）
            LATEST_CHECKPOINT=$(find logs -name "latest_checkpoint.pth" -type f 2>/dev/null | \
                while IFS= read -r file; do
                    if [ -f "$file" ]; then
                        echo "$file"
                    fi
                done | xargs ls -t 2>/dev/null | head -1)
        fi
        
        if [ -n "$LATEST_CHECKPOINT" ] && [ -f "$LATEST_CHECKPOINT" ]; then
            echo "📦 自动找到最新检查点: $LATEST_CHECKPOINT"
            ARGS+=("--resume_from_checkpoint" "$LATEST_CHECKPOINT")
        else
            echo "⚠️  未找到检查点，将从头开始训练"
        fi
    else
        # 使用指定的检查点路径
        if [ -f "$RESUME_CHECKPOINT" ]; then
            echo "📦 使用指定检查点: $RESUME_CHECKPOINT"
            ARGS+=("--resume_from_checkpoint" "$RESUME_CHECKPOINT")
        else
            echo "❌ 错误: 指定的检查点文件不存在: $RESUME_CHECKPOINT"
            exit 1
        fi
    fi
fi

# 启动训练
echo "=========================================="
echo "启动DSET训练"
echo "配置文件: $CONFIG"
echo "=========================================="
echo ""
echo "提示: 可以通过参数指定其他配置，例如："
echo "  ./run_training.sh configs/dset6_r34.yaml"
echo "  ./run_training.sh configs/dset4_r18.yaml"
echo "  ./run_training.sh configs/dset6_r34.yaml --resume  # 自动从最新检查点恢复"
echo "  ./run_training.sh configs/dset6_r34.yaml --resume logs/xxx/latest_checkpoint.pth  # 指定检查点路径"
echo "  ./run_training.sh configs/dset6_r34.yaml --resume=logs/xxx/latest_checkpoint.pth  # 另一种指定方式"
echo ""
echo "=========================================="

python train.py --config $CONFIG "${ARGS[@]}"

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
