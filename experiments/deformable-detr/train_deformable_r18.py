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
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint (path to checkpoint file, or "auto" to resume from latest)')
    parser.add_argument('--disable_early_stop', action='store_true',
                        help='Disable early stopping (useful when resuming training)')
    parser.add_argument('--early_stop_patience', type=int, default=None,
                        help='Override early stopping patience (default: 20)')
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
    # 🔥 核心修改：强制使用 100 Queries（与 RT-DETR 对齐，确保公平对比）
    cfg.model.num_queries = 100

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
    # 🔥 关键点：proposal_nums 决定了"计算"哪些 AR
    cfg.val_evaluator.proposal_nums = (1, 10, 100)
    cfg.test_evaluator.proposal_nums = (1, 10, 100)
    # 🔥 关键点：metric_items 决定了"在进度条最后打印"哪些 Key
    # 这里不能写 AR，否则会报错。删掉 AR 后，完整的 AR 数据依然会在详细日志中打印出来。
    cfg.val_evaluator.metric_items = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
    cfg.test_evaluator.metric_items = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']

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
    
    # 确保 work_dir 是绝对路径（便于 checkpoint 查找）
    if not os.path.isabs(cfg.work_dir):
        # 如果 work_dir 是相对路径，尝试从当前工作目录或脚本所在目录解析
        # 默认假设在项目根目录运行
        cfg.work_dir = os.path.abspath(cfg.work_dir)
    
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
        cfg.default_hooks.checkpoint.max_keep_ckpts = 1  # 🔥 只保留最新 1 个，节省硬盘空间
    else:
        cfg.default_hooks.checkpoint = dict(
            type='CheckpointHook', 
            interval=1, 
            save_best='coco/bbox_mAP',
            rule='greater',
            max_keep_ckpts=1  # 🔥 只保留最新 1 个，节省硬盘空间
        )
    
    # 2. 配置 EarlyStoppingHook (放在 custom_hooks 中，符合 MMEngine 规范)
    if 'custom_hooks' not in cfg:
        cfg.custom_hooks = []
    
    # Early Stopping 配置（可通过参数禁用或调整）
    if not args.disable_early_stop:
        early_stop_patience = args.early_stop_patience if args.early_stop_patience is not None else 20
        cfg.custom_hooks.append(dict(
            type='EarlyStoppingHook',
            monitor='coco/bbox_mAP',  # 监控的指标
            patience=early_stop_patience,  # 容忍多少个epoch没有改善
            min_delta=0.0001,  # 最小改善阈值
            rule='greater'  # 'greater'表示越大越好，'less'表示越小越好
        ))
        print(f"✓ Early Stopping & Save Best: 已启用 (patience={early_stop_patience}, monitor=coco/bbox_mAP)")
    else:
        print(f"⚠ Early Stopping: 已禁用 (--disable_early_stop)")
    
    # Resume training configuration
    resume_from = None
    if args.resume:
        if args.resume.lower() == 'auto':
            # 自动查找最新的 checkpoint（按优先级）
            import glob
            work_dir = cfg.work_dir
            
            # 优先级1: 查找 latest.pth（MMEngine 通常保存这个）
            latest_checkpoint = os.path.join(work_dir, 'latest.pth')
            if os.path.exists(latest_checkpoint):
                resume_from = latest_checkpoint
                # 读取 checkpoint 查看 epoch 信息
                try:
                    import torch
                    ckpt = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)
                    epoch = ckpt.get('meta', {}).get('epoch', ckpt.get('epoch', 'unknown'))
                    print(f"📦 自动找到 latest checkpoint: {resume_from} (Epoch: {epoch})")
                except:
                    print(f"📦 自动找到 latest checkpoint: {resume_from}")
            else:
                # 优先级2: 查找 epoch_*.pth
                checkpoint_pattern = os.path.join(work_dir, 'epoch_*.pth')
                checkpoints = glob.glob(checkpoint_pattern)
                if checkpoints:
                    # 按文件名中的 epoch 数字排序
                    checkpoints.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
                    resume_from = checkpoints[-1]
                    try:
                        import torch
                        ckpt = torch.load(resume_from, map_location='cpu', weights_only=False)
                        epoch = ckpt.get('meta', {}).get('epoch', ckpt.get('epoch', 'unknown'))
                        print(f"📦 自动找到最新 epoch checkpoint: {resume_from} (Epoch: {epoch})")
                    except:
                        print(f"📦 自动找到最新 epoch checkpoint: {resume_from}")
                else:
                    # 优先级3: 查找 best_*.pth
                    best_pattern = os.path.join(work_dir, 'best_*.pth')
                    best_checkpoints = glob.glob(best_pattern)
                    if best_checkpoints:
                        resume_from = best_checkpoints[0]  # 通常只有一个 best
                        try:
                            import torch
                            ckpt = torch.load(resume_from, map_location='cpu', weights_only=False)
                            epoch = ckpt.get('meta', {}).get('epoch', ckpt.get('epoch', 'unknown'))
                            print(f"📦 自动找到 best checkpoint: {resume_from} (Epoch: {epoch})")
                        except:
                            print(f"📦 自动找到 best checkpoint: {resume_from}")
                    else:
                        print(f"⚠ 未找到 checkpoint，将从 epoch 0 开始训练")
                        print(f"   检查目录: {work_dir}")
                        print(f"   尝试查找的文件: latest.pth, epoch_*.pth, best_*.pth")
        else:
            # 使用指定的 checkpoint 路径
            # 尝试多个可能的路径位置（按优先级）
            original_path = args.resume
            possible_paths = []
            
            # 如果已经是绝对路径，直接使用
            if os.path.isabs(original_path):
                possible_paths.append(original_path)
            else:
                # 1. 相对于当前工作目录
                possible_paths.append(os.path.abspath(original_path))
                # 2. 相对于 work_dir
                possible_paths.append(os.path.join(cfg.work_dir, original_path))
                # 3. work_dir 下的文件名（去掉前面的目录部分）
                possible_paths.append(os.path.join(cfg.work_dir, os.path.basename(original_path)))
                # 4. 原始路径（保持原样，让后续处理）
                possible_paths.append(original_path)
            
            resume_from = None
            for path in possible_paths:
                if os.path.exists(path):
                    resume_from = os.path.abspath(path)  # 转换为绝对路径
                    print(f"📦 找到 checkpoint: {resume_from}")
                    break
            
            if resume_from:
                # 读取 checkpoint 查看 epoch 信息
                try:
                    import torch
                    ckpt = torch.load(resume_from, map_location='cpu', weights_only=False)
                    epoch = ckpt.get('meta', {}).get('epoch', ckpt.get('epoch', 'unknown'))
                    print(f"📦 使用指定的 checkpoint: {resume_from} (Epoch: {epoch})")
                except Exception as e:
                    print(f"📦 使用指定的 checkpoint: {resume_from} (无法读取 epoch 信息: {e})")
            else:
                print(f"⚠ Checkpoint 不存在，尝试的路径:")
                for path in possible_paths:
                    exists = "✓" if os.path.exists(path) else "✗"
                    print(f"   {exists} {path}")
                print(f"   将从 epoch 0 开始训练")
    
    print(f"\n{'='*60}")
    print(f"Starting Deformable DETR R18 Training")
    print(f"{'='*60}")
    print(f"Config path: {config_path}")
    print(f"Work dir: {cfg.work_dir}")
    print(f"Max Epochs: {cfg.train_cfg.max_epochs}")
    print(f"Batch Size: {cfg.train_dataloader.batch_size}")
    print(f"Data Root: {data_root}")
    if resume_from:
        print(f"Resume from: {resume_from}")
    print(f"{'='*60}\n")
    
    # 配置 resume（MMEngine 的 resume 机制）
    if resume_from:
        # 确保 resume_from 是绝对路径
        if not os.path.isabs(resume_from):
            # 再次尝试解析路径（以防前面的解析失败）
            possible_paths = [
                os.path.abspath(resume_from),  # 相对于当前工作目录
                os.path.join(cfg.work_dir, resume_from),  # 相对于 work_dir
                os.path.join(cfg.work_dir, os.path.basename(resume_from)),  # work_dir 下的文件名
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    resume_from = os.path.abspath(path)
                    break
        
        # 最终验证路径是否存在
        if os.path.exists(resume_from):
            resume_from = os.path.abspath(resume_from)  # 确保是绝对路径
            
            # MMEngine 的 resume 机制说明：
            # - cfg.resume = True: 自动从 work_dir/latest.pth 恢复
            # - cfg.resume = 'path/to/checkpoint.pth': 从指定路径恢复（字符串形式）
            # - cfg.load_from = 'path': 只加载权重，不恢复训练状态
            # 
            # 对于 epoch_*.pth，应该使用字符串路径形式：
            cfg.resume = resume_from  # ✅ 使用路径字符串，MMEngine 会自动恢复所有状态
            
            print(f"✓ 已配置从 checkpoint 恢复: {resume_from}")
            print(f"   cfg.resume = {cfg.resume}")
            
            # 读取并显示 checkpoint 中的 epoch 信息
            try:
                import torch
                ckpt = torch.load(resume_from, map_location='cpu', weights_only=False)
                # MMEngine checkpoint 格式：meta.epoch 或直接是 epoch
                epoch_info = ckpt.get('meta', {}).get('epoch', ckpt.get('epoch', 'unknown'))
                print(f"   Checkpoint 中的 epoch: {epoch_info}")
                if isinstance(epoch_info, int):
                    print(f"   将从 epoch {epoch_info + 1} 继续训练")
                else:
                    print(f"   ⚠ 无法确定 epoch 信息，但会尝试恢复训练状态")
            except Exception as e:
                print(f"   ⚠ 无法读取 checkpoint 信息: {e}")
                print(f"   但仍会尝试恢复训练")
        else:
            print(f"⚠ 错误: Checkpoint 文件不存在: {resume_from}")
            print(f"   当前工作目录: {os.getcwd()}")
            print(f"   Work dir: {cfg.work_dir}")
            print(f"   尝试的路径:")
            if 'possible_paths' in locals():
                for path in possible_paths:
                    exists = "✓" if os.path.exists(path) else "✗"
                    print(f"     {exists} {path}")
            print(f"   将从 epoch 0 开始训练（不使用 resume）")
            cfg.resume = False
    
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()
