#!/bin/bash
# run_conditional_detr.sh
# 启动 Conditional DETR (R18) 实验的脚本

# 获取当前脚本的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/ConditionalDETR"

# 模型配置: ResNet-18, COCO预训练权重初始化，对标 DSET 的 DAIR-V2X (200 epoch, bs 12)
# 请确保将社区下载的 COCO R18 预训练权重重命名并放入相应位置，再将 strict=False 加载其可用参数
python main.py \
    --dataset_file coco \
    --coco_path /root/autodl-tmp/datasets/DAIR-V2X \
    --output_dir ../../logs/conditional_detr_r18_dairv2x \
    --batch_size 12 \
    --epochs 200 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --lr_drop 190 \
    --weight_decay 1e-4 \
    --num_workers 16 \
    --backbone resnet18 \
    --resume ../../pretrained/conditional_detr_r18_coco.pth
