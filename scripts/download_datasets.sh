#!/bin/bash

# V2X数据集下载脚本
# 作者: AI Assistant
# 日期: $(date)

set -e

# 设置数据集目录
DATASET_DIR="/home/yujie/proj/task-selective-det/datasets"
RCOOPER_DIR="$DATASET_DIR/RCooper"
DAIR_V2X_DIR="$DATASET_DIR/DAIR-V2X"

echo "=== V2X数据集下载脚本 ==="
echo "数据集目录: $DATASET_DIR"

# 创建目录
mkdir -p "$RCOOPER_DIR"
mkdir -p "$DAIR_V2X_DIR"

echo ""
echo "=== 1. 下载RCooper数据集 ==="
echo "请手动访问以下链接下载RCooper数据集："
echo "GitHub: https://github.com/AIR-THU/DAIR-RCooper"
echo "论文: https://arxiv.org/abs/2403.10145"
echo ""
echo "下载后请将数据解压到: $RCOOPER_DIR"
echo ""

echo "=== 2. 下载DAIR-V2X数据集 ==="
echo "方式1 - 使用OpenDataLab:"
echo "pip install openxlab"
echo "openxlab dataset get --dataset-repo DAIR-V2X/DAIR-V2X"
echo ""
echo "方式2 - 访问官网:"
echo "官网: https://thudair.baai.ac.cn/"
echo "OpenDataLab: https://opendatalab.com/DAIR-V2X"
echo ""
echo "下载后请将数据解压到: $DAIR_V2X_DIR"
echo ""

echo "=== 3. 数据集结构检查 ==="
echo "下载完成后，请检查以下目录结构："
echo ""
echo "RCooper数据集结构："
echo "$RCOOPER_DIR/"
echo "├── rsu/          # 路侧单元数据"
echo "├── vehicle/      # 车载数据"
echo "├── annotations/  # 标注文件"
echo "└── calibration/  # 标定信息"
echo ""
echo "DAIR-V2X数据集结构："
echo "$DAIR_V2X_DIR/"
echo "├── infrastructure-side/  # 路侧数据"
echo "├── vehicle-side/         # 车载数据"
echo "├── co-operative/         # 协同数据"
echo "└── annotations/          # 标注文件"
echo ""

echo "=== 下载完成后的下一步 ==="
echo "1. 检查数据集完整性"
echo "2. 运行数据预处理脚本"
echo "3. 生成语言指令标注"
echo "4. 开始MoE模型训练"
echo ""

echo "脚本执行完成！请按照上述说明手动下载数据集。"
