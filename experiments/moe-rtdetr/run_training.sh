#!/bin/bash

# MOE RT-DETR 训练脚本
# 使用方法: ./run_training.sh [config_file] [override_params...]
# 
# 示例:
#   ./run_training.sh configs/moe6_presnet50.yaml
#   ./run_training.sh configs/moe6_presnet18.yaml --epochs 100
#   ./run_training.sh configs/moe3_presnet50.yaml --top_k 2

# 默认配置文件 (6专家 + PResNet50)
CONFIG_FILE=${1:-configs/moe6_presnet50.yaml}

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误: 配置文件不存在: $CONFIG_FILE"
    echo "可用的配置文件:"
    ls -la configs/*.yaml 2>/dev/null || echo "  未找到配置文件"
    exit 1
fi

echo "🚀 开始训练 MOE RT-DETR"
echo "配置文件: $CONFIG_FILE"



# 创建日志目录
mkdir -p logs

# 开始训练 - 直接传递配置文件给Python脚本
python train.py \
    --config $CONFIG_FILE \
    "${@:2}"  # 传递除第一个参数外的所有参数

echo "✅ 训练完成！"
