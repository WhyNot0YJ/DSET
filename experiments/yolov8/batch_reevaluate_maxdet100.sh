#!/bin/bash
# 批量重新评估所有YOLOv8模型，使用max_det=100进行公平对比

# 数据配置文件
DATA_YAML="/root/autodl-fs/datasets/DAIR-V2X_YOLO/dairv2x.yaml"

# 模型目录
LOG_DIR="logs"

# 输出目录
OUTPUT_DIR="logs/reevaluation_maxdet100"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "批量重新评估YOLOv8模型 (max_det=100)"
echo "=========================================="
echo ""

# 查找所有best.pt模型
models=(
    "logs/yolo_v8l_20251127_155744/weights/best.pt"
    "logs/yolo_v8m_20251127_170938/weights/best.pt"
    "logs/yolo_v8s_20251127_181059/weights/best.pt"
)

for model_path in "${models[@]}"; do
    if [ ! -f "$model_path" ]; then
        echo "⚠️  模型不存在: $model_path"
        continue
    fi
    
    # 提取模型名称
    model_name=$(basename $(dirname $(dirname "$model_path")))
    echo "评估模型: $model_name"
    echo "模型路径: $model_path"
    
    # 运行评估
    python3 reevaluate_with_maxdet100.py \
        --model "$model_path" \
        --data "$DATA_YAML" \
        --max_det 100 \
        --device cuda \
        > "$OUTPUT_DIR/${model_name}_maxdet100.log" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✓ $model_name 评估完成"
    else
        echo "✗ $model_name 评估失败"
    fi
    echo ""
done

echo "=========================================="
echo "所有模型评估完成"
echo "结果保存在: $OUTPUT_DIR"
echo "=========================================="

