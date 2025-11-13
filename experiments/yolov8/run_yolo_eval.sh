#!/bin/bash

# YOLO 格式数据集评估脚本
# 使用方法: ./run_yolo_eval.sh [模型路径] [实验名称]

MODEL=${1:-"yolov8n.pt"}
EXP_NAME=${2:-"yolo_val_$(date +%Y%m%d_%H%M%S)"}

echo "=========================================="
echo "YOLO 格式数据集评估"
echo "=========================================="
echo "模型: $MODEL"
echo "实验名: $EXP_NAME"
echo "=========================================="

# 完整验证模式（包含所有图表和指标）
echo "运行完整验证模式..."
python val_yolo_dataset.py \
    --model "$MODEL" \
    --data "/workspace/experiments/datasets/coco_yolo/coco_yolo.yaml" \
    --imgsz 640 \
    --conf 0.25 \
    --exp "$EXP_NAME"

echo ""
echo "=========================================="
echo "评估完成！"
echo "结果保存在: ./runs/detect/$EXP_NAME/"
echo "=========================================="
