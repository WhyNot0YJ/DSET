#!/bin/bash
# run_sparse_detr_r50.sh
# 启动 Sparse DETR (R50) 实验的脚本 (带 COCO 预训练)

# 获取当前脚本的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/SparseDETR"

echo "====================================================================="
echo "启动 Sparse DETR (R50) 训练..."
echo "配置对齐: Epochs: 200, Batch Size: 12, LR: 1e-4, 稀疏度 rho: 0.3..."
echo "====================================================================="

# 您需要自行前往官网下载对应 R50 + 该 rho 参数的 pre-trained COCO pth
# (例如：https://twg.kakaocdn.net/brainrepo/sparse_detr/sparse_detr_r50_30.pth)
# 并命名到 `experiments/pretrained/sparse_detr_r50_coco.pth` 中

python main.py \
    --dataset_file coco \
    --coco_path /root/autodl-tmp/datasets/DAIR-V2X \
    --output_dir ../../logs/sparse_detr_r50_dairv2x \
    --batch_size 12 \
    --epochs 200 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --lr_drop 190 \
    --weight_decay 1e-4 \
    --num_workers 16 \
    --backbone resnet50 \
    --rho 0.3 \
    --resume ../../pretrained/sparse_detr_r50_coco.pth
