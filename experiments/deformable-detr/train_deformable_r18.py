import os
import sys
import subprocess

# --- Auto-Installation Block ---
# 自动安装依赖模块 (如果缺失)
# Checks and installs openmim, mmengine, mmcv, mmdet if not present
def check_and_install_dependencies():
    try:
        import mmengine
        import mmdet
        print("Dependencies found.")
    except ImportError:
        print("Dependencies missing. Installing...")
        try:
            import mim
        except ImportError:
            print("Installing openmim...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "openmim"])
            import mim

        print("Installing mmengine, mmcv, mmdet via mim...")
        # Install compatible versions. Adjust versions if needed.
        subprocess.check_call(["mim", "install", "mmengine", "mmcv>=2.0.0", "mmdet>=3.0.0"])

# Check dependencies before main imports if running in a fresh environment
check_and_install_dependencies()

from mmengine.config import Config
from mmengine.runner import Runner

def main():
    # ==================================================================
    # 1. Base Config (基础配置)
    # ==================================================================
    # Load the standard deformable-detr R50 config from mmdet model zoo
    # 使用 mmdet:: 前缀从安装包中加载默认配置
    try:
        config_path = 'mmdet::deformable_detr/deformable-detr_r50_16xb2-50e_coco.py'
        cfg = Config.fromfile(config_path)
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Ensure mmdet is installed properly. You can try running: mim install mmdet")
        return

    # ==================================================================
    # 2. Model Modifications (模型修改: R50 -> R18)
    # ==================================================================
    # Change Backbone to ResNet-18
    # 修改主干网络深度为 18
    cfg.model.backbone.depth = 18
    
    # Load ImageNet pretrained weights for R18
    # 加载 ResNet-18 的 ImageNet 预训练权重
    cfg.model.backbone.init_cfg = dict(type='Pretrained', checkpoint='torchvision://resnet18')
    
    # Fix Channel Mismatch: ResNet-18 outputs [64, 128, 256, 512]
    # Deformable DETR typically uses the last 3 stages (indices 1, 2, 3) -> [128, 256, 512]
    # 更新 Neck 输入通道数以匹配 ResNet-18 的输出
    cfg.model.neck.in_channels = [128, 256, 512]
    
    # Set Number of Classes
    # 修改分类头类别数为 8
    cfg.model.bbox_head.num_classes = 8

    # ==================================================================
    # 3. Dataset Overrides (数据集配置)
    # ==================================================================
    # Dataset Root
    # 设置数据根目录
    data_root = '/root/autodl-tmp/datasets/DAIR-V2X/'
    cfg.data_root = data_root

    # Class Names
    # 定义类别名称
    class_names = ('Car', 'Truck', 'Van', 'Bus', 'Pedestrian', 'Cyclist', 'Motorcyclist', 'Trafficcone')
    metainfo = dict(classes=class_names)

    # ------------------------------------------------------------------
    # Data Augmentation Pipeline (Strict Alignment with DSET)
    # ------------------------------------------------------------------
    # Train Pipeline
    # 1. RandomPhotometricDistort (MMDet default is usually stronger, keeping default is fine/fair)
    # 2. RandomIoUCrop (mapped to MinIoURandomCrop)
    #    - DSET Config: crop_min: 0.1
    # 3. RandomHorizontalFlip (p=0.5)
    # 4. RandomResize (scales 480-800, step 32)
    #    - DSET Config: max_size: 1280
    train_pipeline = [
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='PhotoMetricDistortion'),  
        # [Fix 1] Change 0.3 to 0.1 to match DSET's 'crop_min: 0.1'
        dict(type='MinIoURandomCrop', min_crop_size=0.1),  
        dict(type='RandomFlip', prob=0.5),   
        dict(
            type='RandomResize',
            # [Fix 2] Change 1333 to 1280 to match DSET's 'max_size: 1280'
            scale=[(1280, x) for x in range(480, 801, 32)],
            keep_ratio=True),  
        dict(type='PackDetInputs')
    ]

    # Val/Test Pipeline
    # 1. Resize (short=720, max=1280)
    test_pipeline = [
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='Resize', scale=(1280, 720), keep_ratio=True),  # Match Resize(720, max=1280)
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                       'scale_factor'))
    ]

    # Train Dataloader
    # 配置训练数据加载器
    cfg.train_dataloader.dataset.data_root = data_root
    cfg.train_dataloader.dataset.ann_file = 'annotations/instances_train.json'
    cfg.train_dataloader.dataset.data_prefix = dict(img='train2017/')
    cfg.train_dataloader.dataset.metainfo = metainfo
    cfg.train_dataloader.dataset.pipeline = train_pipeline  # Apply new pipeline

    # Modify Batch Size and Num Workers
    # 设置 Batch Size 为 16，Num Workers 为 16
    cfg.train_dataloader.batch_size = 16
    cfg.train_dataloader.num_workers = 16

    # Val Dataloader
    # 配置验证数据加载器
    cfg.val_dataloader.dataset.data_root = data_root
    cfg.val_dataloader.dataset.ann_file = 'annotations/instances_val.json'
    cfg.val_dataloader.dataset.data_prefix = dict(img='val2017/')
    cfg.val_dataloader.dataset.metainfo = metainfo
    cfg.val_dataloader.dataset.pipeline = test_pipeline  # Apply new pipeline
    
    # Modify Batch Size and Num Workers for Val
    # 设置验证集 Batch Size 为 16
    cfg.val_dataloader.batch_size = 16
    cfg.val_dataloader.num_workers = 16

    # Test Dataloader (Same as Val)
    # 配置测试数据加载器
    cfg.test_dataloader.dataset.data_root = data_root
    cfg.test_dataloader.dataset.ann_file = 'annotations/instances_val.json'
    cfg.test_dataloader.dataset.data_prefix = dict(img='val2017/')
    cfg.test_dataloader.dataset.metainfo = metainfo
    cfg.test_dataloader.dataset.pipeline = test_pipeline  # Apply new pipeline
    cfg.test_dataloader.batch_size = 16
    cfg.test_dataloader.num_workers = 16

    # Evaluators
    # 配置评估器指向正确的标注文件
    cfg.val_evaluator.ann_file = os.path.join(data_root, 'annotations/instances_val.json')
    cfg.test_evaluator.ann_file = os.path.join(data_root, 'annotations/instances_val.json')

    # ==================================================================
    # 4. Training Schedule (训练计划)
    # ==================================================================
    # Set max epochs
    # 设置最大训练轮数为 200
    cfg.train_cfg.max_epochs = 200
    
    # Configure LR Scheduler (MultiStepLR for 200 epochs)
    # 重写学习率衰减策略
    # 1. Linear Warmup: 前 500 次迭代线性预热
    # 2. MultiStepLR: 总共 200 Epochs，在第 160 Epoch 衰减学习率
    cfg.param_scheduler = [
        dict(
            type='LinearLR',
            start_factor=0.001,
            by_epoch=False,
            begin=0,
            end=500),
        dict(
            type='MultiStepLR',
            begin=0,
            end=200,
            by_epoch=True,
            milestones=[160],
            gamma=0.1)
    ]
    
    # Auto Scale LR
    # 启用自动缩放学习率，基于 batch_size=16
    cfg.auto_scale_lr = dict(enable=True, base_batch_size=16)
    
    # ==================================================================
    # 5. Work Directory (输出目录)
    # ==================================================================
    # 设置工作目录
    cfg.work_dir = 'experiments/Deformable_DETR/work_dirs/r18_baseline'

    # ==================================================================
    # 6. Execution (执行训练)
    # ==================================================================
    print(f"Starting training...")
    print(f"Config path: {config_path}")
    print(f"Work dir: {cfg.work_dir}")
    print(f"Max Epochs: {cfg.train_cfg.max_epochs}")
    print(f"Batch Size: {cfg.train_dataloader.batch_size}")
    
    # Build Runner
    # 构建并启动 Runner
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()
