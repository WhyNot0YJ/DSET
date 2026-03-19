#!/bin/bash
# run_deformable_detr_r50.sh
# 启动 Deformable DETR (R50 版带 COCO 预训练) 的 DAIR-V2X 对标实验

# 切换到项目根目录执行（因为 Python 脚本内使用的是基于项目根目录的相对路径）
cd /home/yujie/proj/DSET

echo "====================================================================="
echo "启动 Deformable DETR (R50) 训练..."
echo "配置对齐: Epochs: 200, Batch Size: 12, LR: 1e-4, CosineAnnealing..."
echo "====================================================================="

# 确保您已经下载了权重到 experiments/pretrained/deformable_detr_r50_coco.pth
python experiments/deformable-detr/train_deformable_r50.py

# 如果中断后需要恢复训练，您可以取消下面这行的注释并使用它：
# python experiments/deformable-detr/train_deformable_r50.py --resume
