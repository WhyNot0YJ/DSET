import os
import sys
import subprocess
import argparse
import torch

# Auto-Installation Block: Checks and installs openmim, mmengine, mmcv, mmdet if not present
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

        # Set CUDA arch to 9.0 for RTX 5090 (CUDA 12.1 compatible, prevents compute_100 error)
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
        print("✓ Set TORCH_CUDA_ARCH_LIST=9.0 for RTX 5090 (CUDA 12.1 compatible)")

        print("Installing mmengine, mmcv, mmdet via mim (Force Arch 9.0)...")
        subprocess.check_call(["mim", "install", "mmengine", "mmcv>=2.0.0", "mmdet>=3.0.0", "-v"])

# Check dependencies before main imports if running in a fresh environment
check_and_install_dependencies()

from mmengine.config import Config
from mmengine.runner import Runner

def setup_gpu_optimizations():
    """Configure GPU optimizations for RTX 5090 with CUDA 12.8"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"✓ GPU 优化已启用:")
        print(f"  - Device: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA Version: {torch.version.cuda}")
        print(f"  - PyTorch Version: {torch.__version__}")
        print(f"  - cudnn.benchmark: {torch.backends.cudnn.benchmark}")
        print(f"  - TF32 (matmul): {torch.backends.cuda.matmul.allow_tf32}")
        print(f"  - TF32 (cudnn): {torch.backends.cudnn.allow_tf32}")
    else:
        print("⚠ GPU 不可用，将使用 CPU 训练（速度会很慢）")

def main():
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Deformable DETR R18 Training')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override max_epochs (for test mode, use --epochs 2)')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Override data root path (default: /root/autodl-tmp/datasets/DAIR-V2X/)')
    parser.add_argument('--work_dir', type=str, default=None,
                        help='Override work directory')
    args = parser.parse_args()
    setup_gpu_optimizations()
    
    # Load Base Config
    try:
        config_path = '/root/miniconda3/lib/python3.10/site-packages/mmdet/.mim/configs/deformable_detr/deformable-detr_r50_16xb2-50e_coco.py'
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        cfg = Config.fromfile(config_path)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # Model Modifications: R50 -> R18
    cfg.model.backbone.depth = 18
    cfg.model.backbone.init_cfg = dict(type='Pretrained', checkpoint='torchvision://resnet18')
    # ResNet-18 outputs [64, 128, 256, 512], use last 3 stages -> [128, 256, 512]
    cfg.model.neck.in_channels = [128, 256, 512]
    cfg.model.bbox_head.num_classes = 8

    # Dataset Configuration
    if args.data_root:
        data_root = args.data_root
    else:
        possible_paths = [
            '/root/autodl-tmp/datasets/DAIR-V2X/',
            '/home/yujie/proj/task-selective-det/data/DAIR-V2X/',
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/DAIR-V2X/'),
        ]
        data_root = None
        for path in possible_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, 'annotations')):
                data_root = path
                break
        if data_root is None:
            data_root = '/root/autodl-tmp/datasets/DAIR-V2X/'
            print(f"⚠ Warning: Using default data root: {data_root}")
            print(f"   If data is elsewhere, use --data_root to specify")
    
    cfg.data_root = data_root
    print(f"✓ Data root: {data_root}")

    class_names = ('Car', 'Truck', 'Van', 'Bus', 'Pedestrian', 'Cyclist', 'Motorcyclist', 'Trafficcone')
    metainfo = dict(classes=class_names)

    # Data Augmentation Pipeline (aligned with DSET)
    train_pipeline = [
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='PhotoMetricDistortion'),  
        dict(type='MinIoURandomCrop', min_crop_size=0.1),  
        dict(type='RandomFlip', prob=0.5),   
        dict(
            type='RandomChoiceResize',
            scales=[(1280, x) for x in range(480, 801, 32)],
            keep_ratio=True),  
        dict(type='PackDetInputs')
    ]

    test_pipeline = [
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='Resize', scale=(1280, 720), keep_ratio=True),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                       'scale_factor'))
    ]

    # Train Dataloader
    cfg.train_dataloader.dataset.data_root = data_root
    cfg.train_dataloader.dataset.ann_file = 'annotations/instances_train.json'
    cfg.train_dataloader.dataset.data_prefix = dict(img='')
    cfg.train_dataloader.dataset.metainfo = metainfo
    cfg.train_dataloader.dataset.pipeline = train_pipeline
    cfg.train_dataloader.batch_size = 8
    cfg.train_dataloader.num_workers = 16

    # Val Dataloader
    cfg.val_dataloader.dataset.data_root = data_root
    cfg.val_dataloader.dataset.ann_file = 'annotations/instances_val.json'
    cfg.val_dataloader.dataset.data_prefix = dict(img='')
    cfg.val_dataloader.dataset.metainfo = metainfo
    cfg.val_dataloader.dataset.pipeline = test_pipeline
    cfg.val_dataloader.batch_size = 16
    cfg.val_dataloader.num_workers = 16

    # Test Dataloader
    cfg.test_dataloader.dataset.data_root = data_root
    cfg.test_dataloader.dataset.ann_file = 'annotations/instances_val.json'
    cfg.test_dataloader.dataset.data_prefix = dict(img='')
    cfg.test_dataloader.dataset.metainfo = metainfo
    cfg.test_dataloader.dataset.pipeline = test_pipeline
    cfg.test_dataloader.batch_size = 16
    cfg.test_dataloader.num_workers = 16

    # Evaluators
    cfg.val_evaluator.ann_file = os.path.join(data_root, 'annotations/instances_val.json')
    cfg.test_evaluator.ann_file = os.path.join(data_root, 'annotations/instances_val.json')

    # Training Schedule
    if args.epochs is not None:
        max_epochs = args.epochs
        print(f"✓ Overriding max_epochs to {max_epochs} (from --epochs argument)")
    else:
        max_epochs = 200
    cfg.train_cfg.max_epochs = max_epochs
    
    # LR Scheduler: Linear Warmup (500 iters) + MultiStepLR (decay at epoch 160)
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
    
    cfg.auto_scale_lr = dict(enable=True, base_batch_size=16)
    
    # Work Directory
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = 'work_dirs/r18_baseline'
    print(f"✓ Work directory: {cfg.work_dir}")

    # Enable AMP (Mixed Precision Training)
    cfg.optim_wrapper.type = 'AmpOptimWrapper'
    cfg.optim_wrapper.loss_scale = 'dynamic'
    
    # ==================================================================
    # Early Stopping & Best Model Saving Configuration
    # ==================================================================
    
    # 1. 配置 CheckpointHook 保存【最佳模型】
    # (如果不加这一步，早停后你拿到的是最后一次迭代的模型，而不是分数最高的模型)
    if 'default_hooks' not in cfg:
        cfg.default_hooks = {}
    
    # 确保 checkpoint hook 存在并设置 save_best
    if hasattr(cfg.default_hooks, 'checkpoint'):
        cfg.default_hooks.checkpoint.save_best = 'coco/bbox_mAP'
        cfg.default_hooks.checkpoint.rule = 'greater'
        cfg.default_hooks.checkpoint.interval = 1  # 验证间隔
        # cfg.default_hooks.checkpoint.max_keep_ckpts = 3  # 可选：只保留最好的3个
    else:
        cfg.default_hooks.checkpoint = dict(
            type='CheckpointHook', 
            interval=1, 
            save_best='coco/bbox_mAP',
            rule='greater'
        )
    
    # 2. 配置 EarlyStoppingHook (放在 custom_hooks 中，符合 MMEngine 规范)
    if 'custom_hooks' not in cfg:
        cfg.custom_hooks = []
        
    cfg.custom_hooks.append(dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',  # 监控的指标
        patience=20,  # 容忍多少个epoch没有改善
        min_delta=0.0001,  # 最小改善阈值
        rule='greater'  # 'greater'表示越大越好，'less'表示越小越好
    ))
    
    print(f"✓ Early Stopping & Save Best: 已启用 (patience=20, monitor=coco/bbox_mAP)")
    
    print(f"\n{'='*60}")
    print(f"Starting Deformable DETR R18 Training")
    print(f"{'='*60}")
    print(f"Config path: {config_path}")
    print(f"Work dir: {cfg.work_dir}")
    print(f"Max Epochs: {cfg.train_cfg.max_epochs}")
    print(f"Batch Size: {cfg.train_dataloader.batch_size}")
    print(f"Data Root: {data_root}")
    print(f"{'='*60}\n")
    
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()
