#!/bin/bash

# MOE RT-DETR训练启动脚本

echo "=== MOE RT-DETR DAIR-V2X训练 ==="

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

# 创建日志目录
mkdir -p logs

# 训练配置A：6个专家（按类别）
echo "开始训练配置A：6个专家（按类别）"
python train_moe_rtdetr_dair_v2x.py \
    --config A \
    --data_root datasets/DAIR-V2X \
    --epochs 100 \
    --batch_size 16 \
    --pretrained_lr 1e-5 \
    --new_lr 1e-4 \
    --top_k 2 \
    --pretrained_weights torch_hub \
    2>&1 | tee logs/moe_rtdetr_config_a.log

if [ $? -eq 0 ]; then
    echo "配置A训练完成！"
else
    echo "配置A训练失败！"
    exit 1
fi

# 训练配置B：3个专家（按复杂度）
echo "开始训练配置B：3个专家（按复杂度）"
python train_moe_rtdetr_dair_v2x.py \
    --config B \
    --data_root datasets/DAIR-V2X \
    --epochs 100 \
    --batch_size 16 \
    --pretrained_lr 1e-5 \
    --new_lr 1e-4 \
    --top_k 2 \
    --pretrained_weights torch_hub \
    2>&1 | tee logs/moe_rtdetr_config_b.log

if [ $? -eq 0 ]; then
    echo "配置B训练完成！"
else
    echo "配置B训练失败！"
    exit 1
fi

# 训练配置C：3个专家（按尺寸）
echo "开始训练配置C：3个专家（按尺寸）"
python train_moe_rtdetr_dair_v2x.py \
    --config C \
    --data_root datasets/DAIR-V2X \
    --epochs 100 \
    --batch_size 16 \
    --pretrained_lr 1e-5 \
    --new_lr 1e-4 \
    --top_k 2 \
    --pretrained_weights torch_hub \
    2>&1 | tee logs/moe_rtdetr_config_c.log

if [ $? -eq 0 ]; then
    echo "配置C训练完成！"
else
    echo "配置C训练失败！"
    exit 1
fi

echo "=== 所有配置训练完成！ ==="
echo "日志文件保存在 logs/ 目录下"
echo "模型检查点保存在 logs/moe_rtdetr_*/ 目录下"
