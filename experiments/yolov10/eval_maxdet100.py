#!/usr/bin/env python3
"""
YOLOv10 评估脚本 - 限制最大检测框数量为 100
用于评估在 max_det=100 限制下的模型精度（mAP等指标）
"""

import sys
import argparse
import yaml
import torch
from pathlib import Path
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root.parent) not in sys.path:
    sys.path.insert(0, str(project_root.parent))

# 导入ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    # Fallback: Attempt to use yolov8's ultralytics if local one fails
    yolov8_path = project_root.parent / "yolov8"
    if yolov8_path.exists() and str(yolov8_path) not in sys.path:
        print(f"Warning: Local ultralytics not found, attempting to use {yolov8_path}")
        sys.path.insert(0, str(yolov8_path))
    from ultralytics import YOLO


def load_model(checkpoint_path: str, device: str = "cuda", model_name: str = "yolov10l.pt"):
    """加载YOLO模型，支持 .pth 和 .pt 格式"""
    print(f"📦 加载模型: {checkpoint_path}")
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")
    
    # 如果是 .pth 文件，需要先加载权重到 YOLO 模型，然后保存为 .pt
    if checkpoint_path.suffix == '.pth':
        print(f"🔄 检测到 .pth 文件，转换为 YOLO .pt 格式...")
        pt_path = checkpoint_path.with_suffix('.pt')
        
        # 如果 .pt 文件已存在，删除它以便重新转换（避免使用错误的格式）
        if pt_path.exists():
            print(f"  ⚠️  发现已存在的 .pt 文件，删除以重新转换: {pt_path}")
            try:
                pt_path.unlink()
                print(f"  ✓ 已删除旧文件")
            except Exception as e:
                print(f"  ⚠️  删除失败: {e}，将尝试覆盖")
        else:
            # 转换 .pth 到 YOLO .pt 格式
            try:
                # 1. 加载 checkpoint
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                print(f"  ✓ 已加载 checkpoint")
                
                # 2. 检查 checkpoint 结构并转换为 YOLO 格式
                print("  💾 转换 checkpoint 为 YOLO 格式...")
                
                # 检查 checkpoint 结构
                if isinstance(checkpoint, dict):
                    print(f"  📋 Checkpoint 键: {list(checkpoint.keys())}")
                    
                    # 检查 'model' 键的内容
                    if 'model' in checkpoint:
                        model_obj = checkpoint['model']
                        print(f"  ✓ 找到 'model' 键")
                        print(f"  📦 'model' 类型: {type(model_obj)}")
                        
                        # 检查是否是完整的模型对象（有 forward 方法）
                        if hasattr(model_obj, 'forward'):
                            print("  ✓ 'model' 是完整的模型对象")
                            # 直接保存，YOLO 应该能识别
                            torch.save(checkpoint, str(pt_path))
                        elif isinstance(model_obj, dict):
                            print("  ℹ️  'model' 是 state_dict，需要模型结构来创建完整模型")
                            # 这是 state_dict，需要创建模型实例来加载权重
                            # 尝试从配置文件获取模型信息
                            config_file = checkpoint_path.parent / 'config.yaml'
                            model_type = None
                            if config_file.exists():
                                try:
                                    import yaml
                                    with open(config_file, 'r') as f:
                                        config = yaml.safe_load(f)
                                    model_name_from_config = config.get('model', {}).get('model_name', 'yolov10l.pt')
                                    print(f"  ✓ 从配置文件读取模型类型: {model_name_from_config}")
                                    model_type = model_name_from_config
                                except Exception as e:
                                    print(f"  ⚠️  读取配置文件失败: {e}")
                            
                            # 尝试查找本地预训练权重或已训练的模型
                            temp_model = None
                            
                            # 1. 检查同目录下是否有其他 .pt 文件
                            log_dir = checkpoint_path.parent
                            other_pt_files = [f for f in log_dir.glob('*.pt') if f != pt_path]
                            if other_pt_files:
                                print(f"  ✓ 找到其他 .pt 文件作为参考: {other_pt_files[0]}")
                                try:
                                    temp_model = YOLO(str(other_pt_files[0]), task='detect')
                                    print(f"  ✓ 成功加载参考模型")
                                except Exception as e:
                                    print(f"  ⚠️  加载参考模型失败: {e}")
                            
                            # 2. 如果还没有，尝试使用本地预训练权重
                            if temp_model is None:
                                pretrained_paths = [
                                    'pretrained/yolov10l.pt',
                                    'pretrained/yolov10s.pt',
                                    str(Path(__file__).parent / 'pretrained' / 'yolov10l.pt'),
                                    str(Path(__file__).parent / 'pretrained' / 'yolov10s.pt'),
                                ]
                                for pretrained_path in pretrained_paths:
                                    if Path(pretrained_path).exists():
                                        print(f"  ✓ 使用本地预训练权重: {pretrained_path}")
                                        try:
                                            temp_model = YOLO(pretrained_path, task='detect')
                                            break
                                        except:
                                            continue
                            
                            # 3. 如果有模型实例，加载权重并保存
                            if temp_model is not None:
                                print(f"  🔄 加载权重到模型...")
                                try:
                                    missing_keys, unexpected_keys = temp_model.model.load_state_dict(model_obj, strict=False)
                                    if missing_keys:
                                        print(f"  ⚠️  缺失的键: {len(missing_keys)} 个")
                                    if unexpected_keys:
                                        print(f"  ⚠️  意外的键: {len(unexpected_keys)} 个")
                                    print(f"  ✓ 权重加载完成")
                                    
                                    # 保存完整的模型
                                    temp_model.save(str(pt_path))
                                    print(f"  ✓ 已保存完整模型: {pt_path}")
                                except Exception as e:
                                    print(f"  ⚠️  加载权重失败: {e}")
                                    # 如果失败，直接保存 checkpoint
                                    ckpt = checkpoint.copy()
                                    torch.save(ckpt, str(pt_path))
                            else:
                                # 如果无法创建模型实例，直接保存（可能无法加载）
                                print("  ⚠️  无法创建模型实例，直接保存权重")
                                print("  ⚠️  警告：YOLO 可能无法直接加载此文件")
                                ckpt = checkpoint.copy()
                                torch.save(ckpt, str(pt_path))
                        else:
                            print(f"  ⚠️  'model' 类型未知: {type(model_obj)}")
                            # 直接保存
                            torch.save(checkpoint, str(pt_path))
                    else:
                        # 没有 'model' 键，尝试提取权重
                        print("  ℹ️  没有 'model' 键，尝试提取权重...")
                        if 'model_state_dict' in checkpoint:
                            state_dict = checkpoint['model_state_dict']
                            print("  ✓ 找到 'model_state_dict'")
                        elif 'ema_state_dict' in checkpoint:
                            state_dict = checkpoint['ema_state_dict']
                            print("  ✓ 找到 'ema_state_dict'")
                        elif 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                            print("  ✓ 找到 'state_dict'")
                        else:
                            state_dict = checkpoint
                            print("  ℹ️  使用整个 checkpoint 作为 state_dict")
                        
                        # 保存为 YOLO 格式
                        ckpt = {
                            'epoch': checkpoint.get('epoch', -1),
                            'best_fitness': checkpoint.get('best_fitness', None),
                            'model': state_dict,
                            'optimizer': checkpoint.get('optimizer', None),
                            'ema': checkpoint.get('ema', None),
                        }
                        # 保留其他信息
                        for key in ['names', 'nc', 'hyp', 'task', 'yaml', 'args']:
                            if key in checkpoint:
                                ckpt[key] = checkpoint[key]
                        torch.save(ckpt, str(pt_path))
                else:
                    # checkpoint 直接是对象
                    print(f"  ℹ️  checkpoint 类型: {type(checkpoint)}")
                    ckpt = {
                        'epoch': -1,
                        'best_fitness': None,
                        'model': checkpoint,
                        'optimizer': None,
                        'ema': None,
                    }
                    torch.save(ckpt, str(pt_path))
                
                print(f"  ✓ 已保存为: {pt_path}")
                checkpoint_path = pt_path
            except Exception as e:
                import traceback
                print(f"  ⚠️  转换失败: {e}")
                print(f"  📋 错误详情:")
                traceback.print_exc()
                print(f"  ℹ️  尝试直接加载 .pth 文件（可能失败）...")
                # 如果转换失败，尝试直接加载（可能会失败）
    
    # 加载模型
    # 如果文件是 .pth 格式，YOLO 可能无法直接加载，需要指定 task
    try:
        model = YOLO(str(checkpoint_path), task='detect')
    except Exception as e:
        print(f"  ⚠️  使用 task='detect' 加载失败: {e}")
        # 尝试不使用 task 参数
        try:
            model = YOLO(str(checkpoint_path))
        except Exception as e2:
            raise RuntimeError(f"无法加载模型文件 {checkpoint_path}: {e2}\n"
                             f"提示：如果这是 .pth 文件转换的，可能需要提供模型结构信息")
    
    # 移动到设备
    try:
        model.to(device)
    except Exception as e:
        print(f"  ⚠️  移动到设备失败: {e}")
        print(f"  ℹ️  模型可能已经在正确的设备上，继续...")
    
    model.eval()
    
    print(f"✓ 模型加载完成")
    return model


def evaluate_with_max_det(
    model,
    data_yaml: str,
    max_det: int = 100,
    conf_threshold: float = 0.001,  # 使用低阈值以获取更多候选框，然后由max_det限制
    iou_threshold: float = 0.6,
    device: str = "cuda",
    imgsz: int = 1280,
    split: str = "val"
):
    """
    在验证集上评估模型，限制最大检测框数量
    
    Args:
        model: YOLO模型
        data_yaml: 数据集配置文件路径
        max_det: 最大检测框数量（默认: 100）
        conf_threshold: 置信度阈值
        iou_threshold: NMS IoU阈值
        device: 设备
        imgsz: 图像尺寸
        split: 数据集分割（'val' 或 'test'）
    
    Returns:
        评估结果字典
    """
    print(f"\n{'='*60}")
    print(f"🔍 开始评估 (max_det={max_det})")
    print(f"{'='*60}")
    print(f"数据集配置: {data_yaml}")
    print(f"最大检测框数: {max_det}")
    print(f"置信度阈值: {conf_threshold}")
    print(f"IoU阈值: {iou_threshold}")
    print(f"图像尺寸: {imgsz}")
    print(f"数据集分割: {split}")
    print(f"{'='*60}\n")
    
    # 修改模型的max_det配置（如果模型支持）
    if hasattr(model, 'model') and hasattr(model.model, 'max_det'):
        original_max_det = model.model.max_det
        model.model.max_det = max_det
        print(f"✓ 已设置模型max_det={max_det}")
    
    # 使用YOLO的val方法进行评估
    results = model.val(
        data=data_yaml,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        device=device,
        split=split,
        max_det=max_det,  # 传递max_det参数
        verbose=True
    )
    
    # 提取评估指标
    metrics = {
        'mAP50': float(results.box.map50) if hasattr(results, 'box') and hasattr(results.box, 'map50') else None,
        'mAP50-95': float(results.box.map) if hasattr(results, 'box') and hasattr(results.box, 'map') else None,
        'precision': float(results.box.mp) if hasattr(results, 'box') and hasattr(results.box, 'mp') else None,
        'recall': float(results.box.mr) if hasattr(results, 'box') and hasattr(results.box, 'mr') else None,
        'max_det': max_det
    }
    
    print(f"\n✅ 评估完成 (max_det={max_det})")
    if metrics['mAP50'] is not None:
        print(f"  mAP@0.5: {metrics['mAP50']:.4f}")
    if metrics['mAP50-95'] is not None:
        print(f"  mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
    if metrics['precision'] is not None:
        print(f"  Precision: {metrics['precision']:.4f}")
    if metrics['recall'] is not None:
        print(f"  Recall: {metrics['recall']:.4f}")
    
    return metrics




def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOv10评估脚本 - 限制最大检测框数量为100')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径（支持 .pt 和 .pth 文件）')
    parser.add_argument('--model_name', type=str, default='yolov10l.pt',
                       help='YOLO 模型名称（用于 .pth 转换，默认: yolov10l.pt）')
    parser.add_argument('--data_yaml', type=str, required=True,
                       help='数据集配置文件路径（YAML格式）')
    parser.add_argument('--max_det', type=int, default=100,
                       help='最大检测框数量（默认: 100）')
    parser.add_argument('--conf', type=float, default=0.001,
                       help='置信度阈值（默认: 0.001，用于获取更多候选框）')
    parser.add_argument('--iou', type=float, default=0.6,
                       help='NMS IoU阈值（默认: 0.6）')
    parser.add_argument('--imgsz', type=int, default=1280,
                       help='图像尺寸（默认: 1280）')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备（默认: cuda）')
    parser.add_argument('--split', type=str, default='val',
                       choices=['val', 'test'],
                       help='数据集分割（默认: val）')
    parser.add_argument('--output', type=str, default=None,
                       help='结果保存路径（可选，JSON格式）')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 YOLOv10 评估脚本 - max_det限制")
    print("="*60)
    print(f"模型: {args.checkpoint}")
    print(f"数据集配置: {args.data_yaml}")
    print(f"最大检测框数: {args.max_det}")
    print(f"置信度阈值: {args.conf}")
    print(f"IoU阈值: {args.iou}")
    print(f"图像尺寸: {args.imgsz}")
    print(f"设备: {args.device}")
    print(f"数据集分割: {args.split}")
    print("="*60)
    
    # 加载模型
    model = load_model(args.checkpoint, args.device, args.model_name)
    
    # 评估
    metrics = evaluate_with_max_det(
        model=model,
        data_yaml=args.data_yaml,
        max_det=args.max_det,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        imgsz=args.imgsz,
        split=args.split
    )
    
    # 保存结果
    if args.output and metrics:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        result_dict = {
            'checkpoint': str(args.checkpoint),
            'data_yaml': str(args.data_yaml),
            'max_det': args.max_det,
            'conf_threshold': args.conf,
            'iou_threshold': args.iou,
            'imgsz': args.imgsz,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"\n💾 结果已保存到: {output_path}")


if __name__ == '__main__':
    main()

