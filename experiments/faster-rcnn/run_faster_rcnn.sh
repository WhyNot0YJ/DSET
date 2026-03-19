#!/bin/bash
# run_faster_rcnn.sh
# 启动 Faster R-CNN (MMDetection) 实验的脚本

# 获取当前脚本的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 如果由于尚未在 AutoDL 上克隆 mmdetection，自动克隆
if [ ! -d "$SCRIPT_DIR/mmdetection" ]; then
    echo "mmdetection directory not found. Cloning the mmdetection repository using a proxy..."
    cd "$SCRIPT_DIR"
    # 使用 GitHub 代理加速克隆，防止 AutoDL 连不上 GitHub 报 443 错误
    git clone https://ghproxy.net/https://github.com/open-mmlab/mmdetection.git
    
    if [ ! -d "$SCRIPT_DIR/mmdetection" ]; then
        echo "Error: Failed to clone mmdetection. Please check your network."
        exit 1
    fi

cd "$SCRIPT_DIR/mmdetection"

# 首先安装 mmdet 依赖 (如果尚未安装)
pip install -r requirements/build.txt
pip install -v -e .

# 运行基于 DSET 配置的 Faster RCNN 训练
python tools/train.py \
    configs/faster_rcnn/faster-rcnn_r50_fpn_dairv2x.py \
    --work-dir ../../logs/faster_rcnn_dairv2x
