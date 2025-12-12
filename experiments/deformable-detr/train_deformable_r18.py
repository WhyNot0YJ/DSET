import os
import sys
import subprocess
import argparse
import torch

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

        # [FIX] Environment has CUDA 12.1 which supports max arch 9.0 (Hopper).
        # RTX 5090 is backward compatible with 9.0.
        # Do NOT use 10.0 as it crashes nvcc 12.1.
        # This prevents the "unsupported gpu architecture 'compute_100'" error
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
        print("✓ Set TORCH_CUDA_ARCH_LIST=9.0 for RTX 5090 (CUDA 12.1 compatible)")

        print("Installing mmengine, mmcv, mmdet via mim (Force Arch 9.0)...")
        # Install compatible versions. Adjust versions if needed.
        # Note: For CUDA 12.1 / RTX 5090, we use arch 9.0 (Hopper) for backward compatibility
        # -v flag for verbose output to see compilation progress
        subprocess.check_call(["mim", "install", "mmengine", "mmcv>=2.0.0", "mmdet>=3.0.0", "-v"])

# Check dependencies before main imports if running in a fresh environment
check_and_install_dependencies()

from mmengine.config import Config
from mmengine.runner import Runner

def setup_gpu_optimizations():
    """
    配置 GPU 优化设置（RTX 5090 + CUDA 12.8 兼容）
    Configure GPU optimizations for RTX 5090 with CUDA 12.8
    """
    if torch.cuda.is_available():
        # 启用 cudnn benchmark 以加速卷积操作（输入尺寸固定时）
        torch.backends.cudnn.benchmark = True
        
        # 启用 TensorFloat-32 (TF32) - RTX 5090 支持，可加速某些操作
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # 显示 GPU 信息
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
    # ==================================================================
    # 0. Parse Arguments (命令行参数)
    # ==================================================================
    parser = argparse.ArgumentParser(description='Deformable DETR R18 Training')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override max_epochs (for test mode, use --epochs 2)')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Override data root path (default: /root/autodl-tmp/datasets/DAIR-V2X/)')
    parser.add_argument('--work_dir', type=str, default=None,
                        help='Override work directory')
    args = parser.parse_args()
    
    # ==================================================================
    # 0.5. Setup GPU Optimizations (GPU 优化设置)
    # ==================================================================
    setup_gpu_optimizations()
    
    # ==================================================================
    # 1. Base Config (基础配置)
    # ==================================================================
    # Load the standard deformable-detr R50 config from mmdet model zoo
    # 使用绝对路径加载配置文件
    try:
        config_path = '/root/miniconda3/lib/python3.10/site-packages/mmdet/.mim/configs/deformable_detr/deformable-detr_r50_16xb2-50e_coco.py'
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        cfg = Config.fromfile(config_path)
    except Exception as e:
        print(f"Error loading config: {e}")
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
    # 设置数据根目录（支持命令行参数覆盖）
    if args.data_root:
        data_root = args.data_root
    else:
        # 尝试自动检测数据路径
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
            # 使用默认路径（如果不存在会在运行时报错）
            data_root = '/root/autodl-tmp/datasets/DAIR-V2X/'
            print(f"⚠ Warning: Using default data root: {data_root}")
            print(f"   If data is elsewhere, use --data_root to specify")
    
    cfg.data_root = data_root
    print(f"✓ Data root: {data_root}")

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
    cfg.train_dataloader.dataset.data_prefix = dict(img='')
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
    cfg.val_dataloader.dataset.data_prefix = dict(img='')
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
    # Set max epochs (支持命令行参数覆盖，用于测试模式)
    # 设置最大训练轮数（默认 200，可通过 --epochs 覆盖）
    if args.epochs is not None:
        max_epochs = args.epochs
        print(f"✓ Overriding max_epochs to {max_epochs} (from --epochs argument)")
    else:
        max_epochs = 200
    cfg.train_cfg.max_epochs = max_epochs
    
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
    # 设置工作目录（支持命令行参数覆盖）
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        # 使用相对路径（从 experiments/deformable-detr/ 目录运行）
        cfg.work_dir = 'work_dirs/r18_baseline'
    print(f"✓ Work directory: {cfg.work_dir}")

    # ==================================================================
    # 6. Execution (执行训练)
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"Starting Deformable DETR R18 Training")
    print(f"{'='*60}")
    print(f"Config path: {config_path}")
    print(f"Work dir: {cfg.work_dir}")
    print(f"Max Epochs: {cfg.train_cfg.max_epochs}")
    print(f"Batch Size: {cfg.train_dataloader.batch_size}")
    print(f"Data Root: {data_root}")
    print(f"{'='*60}\n")
    
    # Build Runner
    # 构建并启动 Runner
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()
