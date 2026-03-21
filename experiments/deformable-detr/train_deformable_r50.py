#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import argparse
from mmengine.config import Config, DictAction

def parse_args():
    parser = argparse.ArgumentParser(description='Train Deformable DETR R50 on DAIR-V2X')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume', action='store_true', help='resume from the latest checkpoint')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # 1. 基础配置路径，这里直接用 MMDetection 自带的 R50 版本做底版：
    base_config = 'experiments/faster-rcnn/mmdetection/configs/deformable_detr/deformable-detr_r50_16xb2-50e_coco.py'
    
    # 读取配置
    cfg = Config.fromfile(base_config)
    
    # 这里通过指定带有 COCO 的预训练权重，保证公平起跑：
    # 待您下载官方 COCO Deformable DETR (R50) 的 pth 文件至这个途径下
    cfg.load_from = 'experiments/pretrained/deformable_detr_r50_coco.pth'

    # 设置您的工作路径
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = 'experiments/logs/deformable_detr_r50_dairv2x'

    os.makedirs(cfg.work_dir, exist_ok=True)

    # ------------------ 修改为 DAIR-V2X 的配置 ------------------
    data_root = '/root/autodl-tmp/datasets/DAIR-V2X/'
    classes = ('Car', 'Truck', 'Van', 'Bus', 'Pedestrian', 'Cyclist', 'Motorcyclist', 'Trafficcone')
    num_classes = len(classes)

    # 1. 修改模型分类头的类别数
    cfg.model.bbox_head.num_classes = num_classes

    # 2. 修改 Dataloader
    cfg.train_dataloader.dataset.data_root = data_root
    cfg.train_dataloader.dataset.ann_file = 'annotations/instances_train2017.json'
    cfg.train_dataloader.dataset.data_prefix.img = 'images/train2017/'
    cfg.train_dataloader.dataset.metainfo = dict(classes=classes)
    cfg.train_dataloader.batch_size = 12
    cfg.train_dataloader.num_workers = 16

    cfg.val_dataloader.dataset.data_root = data_root
    cfg.val_dataloader.dataset.ann_file = 'annotations/instances_val2017.json'
    cfg.val_dataloader.dataset.data_prefix.img = 'images/val2017/'
    cfg.val_dataloader.dataset.metainfo = dict(classes=classes)
    cfg.val_dataloader.batch_size = 4
    cfg.val_dataloader.num_workers = 4

    cfg.test_dataloader = cfg.val_dataloader

    # 3. 修改 Evaluator
    cfg.val_evaluator.ann_file = data_root + 'annotations/instances_val2017.json'
    cfg.test_evaluator = cfg.val_evaluator

    # 4. 训练周期：200 Epochs (对标 CaS_DETR)
    cfg.train_cfg.max_epochs = 200
    cfg.train_cfg.val_interval = 10

    # 5. 优化器、调度器：沿用 CaS_DETR 设定的 CosineAnnealing 调度和学习率缩放
    cfg.optim_wrapper.optimizer.lr = 1e-4
    cfg.optim_wrapper.optimizer.weight_decay = 1e-4

    cfg.param_scheduler = [
        dict(
            type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=3),
        dict(
            type='CosineAnnealingLR',
            by_epoch=True,
            begin=3,
            end=200,
            T_max=197,
            eta_min=1e-6)
    ]

    # 保存新的运行配置
    dump_config_path = os.path.join(cfg.work_dir, 'config_r50.py')
    cfg.dump(dump_config_path)

    print(f"配置文件已生成: {dump_config_path}")
    print(f"正在从 MMDetection 启动 Deformable DETR R50 训练...")

    # 执行命令调用 MMDetection 工具：
    cmd = f"python experiments/faster-rcnn/mmdetection/tools/train.py {dump_config_path}"
    if args.resume:
        cmd += " --resume"

    os.system(cmd)


if __name__ == '__main__':
    main()
