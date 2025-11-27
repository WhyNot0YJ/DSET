#!/bin/bash
# 批量重新评估所有YOLOv8模型，使用max_det=100进行公平对比

# 数据配置文件
DATA_YAML="/root/autodl-fs/datasets/DAIR-V2X_YOLO/dairv2x.yaml"

# 检查数据文件是否存在
if [ ! -f "$DATA_YAML" ]; then
    echo "错误: 数据配置文件不存在: $DATA_YAML"
    echo "请检查路径是否正确"
    exit 1
fi

# 输出目录
OUTPUT_DIR="logs/reevaluation_maxdet100"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "批量重新评估YOLOv8模型 (max_det=100)"
echo "=========================================="
echo "数据配置: $DATA_YAML"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 模型列表
models=(
    "best_model_yolov8l.pth"
    "best_model_yolov8m.pth"
    "best_model_yolov8s.pth"
)

# 检查模型文件是否存在
for model_file in "${models[@]}"; do
    if [ ! -f "$model_file" ]; then
        echo "⚠️  模型文件不存在: $model_file"
    fi
done

echo "开始评估..."
echo ""

# 评估每个模型
for model_file in "${models[@]}"; do
    if [ ! -f "$model_file" ]; then
        continue
    fi
    
    # 提取模型名称（去掉.pth后缀）
    model_name=$(basename "$model_file" .pth)
    
    echo "=========================================="
    echo "评估模型: $model_name"
    echo "模型文件: $model_file"
    echo "=========================================="
    
    # 运行评估
    python3 reevaluate_with_maxdet100.py \
        --model "$model_file" \
        --data "$DATA_YAML" \
        --max_det 100 \
        --device cuda \
        2>&1 | tee "$OUTPUT_DIR/${model_name}_maxdet100.log"
    
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✓ $model_name 评估完成"
    else
        echo ""
        echo "✗ $model_name 评估失败 (退出码: $exit_code)"
    fi
    echo ""
done

echo "=========================================="
echo "所有模型评估完成"
echo "结果保存在: $OUTPUT_DIR"
echo "=========================================="
echo ""
echo "查看结果:"
echo "  cat $OUTPUT_DIR/*_maxdet100.log | grep -A 10 '评估结果'"

