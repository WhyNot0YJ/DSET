#!/bin/bash
# DSET训练启动脚本

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 训练配置（使用最推荐的dset6_presnet50配置）
CONFIG="${1:-configs/dset6_presnet50.yaml}"

# 启动训练
echo "=========================================="
echo "启动DSET训练"
echo "配置文件: $CONFIG"
echo "=========================================="
echo ""
echo "提示: 可以通过参数指定其他配置，例如："
echo "  ./run_training.sh configs/dset3_presnet50.yaml"
echo "  ./run_training.sh configs/dset2_presnet18.yaml"
echo ""
echo "=========================================="

python train.py --config $CONFIG

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
