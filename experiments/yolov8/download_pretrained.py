#!/usr/bin/env python3
"""
YOLOv8 预训练模型下载脚本
支持下载 YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x 等模型
"""

import sys
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root.parent) not in sys.path:
    sys.path.insert(0, str(project_root.parent))

from ultralytics import YOLO
from ultralytics.utils import SETTINGS


def download_model(model_name: str, download_dir: Path = None, verbose: bool = True):
    """
    下载YOLOv8预训练模型
    
    Args:
        model_name: 模型名称，如 'yolov8s.pt', 'yolov8n.pt' 等
        download_dir: 下载目录（如果为None，使用项目pretrained目录）
        verbose: 是否显示详细信息
    
    Returns:
        str: 模型文件路径
    """
    print(f"🚀 开始下载模型: {model_name}")
    print("=" * 80)
    
    # 确定下载目录
    if download_dir is None:
        # 默认下载到项目pretrained目录
        script_dir = Path(__file__).parent.resolve()
        download_dir = script_dir / 'pretrained'
    download_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        import shutil
        
        # 目标路径
        target_path = download_dir / model_name
        
        # 如果目标文件已存在，直接返回
        if target_path.exists():
            file_size = target_path.stat().st_size / (1024 * 1024)  # MB
            print(f"ℹ️  模型已存在: {target_path}")
            print(f"   文件大小: {file_size:.2f} MB")
            print("=" * 80)
            return str(target_path)
        
        # 临时设置权重目录为pretrained目录
        original_weights_dir = SETTINGS.get('weights_dir')
        SETTINGS['weights_dir'] = download_dir
        
        try:
            # 加载模型（如果不存在会自动下载到SETTINGS['weights_dir']）
            model = YOLO(model_name)
            
            # 检查模型是否下载到了pretrained目录
            if target_path.exists():
                file_size = target_path.stat().st_size / (1024 * 1024)  # MB
                print(f"✅ 模型下载成功！")
                print(f"   模型名称: {model_name}")
                print(f"   保存路径: {target_path}")
                print(f"   文件大小: {file_size:.2f} MB")
                print("=" * 80)
                return str(target_path)
            
            # 如果模型下载到了默认位置，复制到pretrained目录
            if hasattr(model, 'ckpt_path') and model.ckpt_path:
                source_path = Path(model.ckpt_path)
                if source_path.exists() and source_path != target_path:
                    print(f"📋 复制模型从 {source_path} 到 {target_path}")
                    shutil.copy2(source_path, target_path)
                    file_size = target_path.stat().st_size / (1024 * 1024)  # MB
                    print(f"✅ 模型下载并复制成功！")
                    print(f"   模型名称: {model_name}")
                    print(f"   保存路径: {target_path}")
                    print(f"   文件大小: {file_size:.2f} MB")
                    print("=" * 80)
                    return str(target_path)
        finally:
            # 恢复原始权重目录
            if original_weights_dir:
                SETTINGS['weights_dir'] = original_weights_dir
        
        print(f"⚠️  警告: 模型文件未找到")
        print(f"   预期路径: {target_path}")
        return None
            
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("=" * 80)
        return None


def list_available_models():
    """列出可用的YOLOv8模型"""
    models = [
        'yolov8n.pt',  # nano
        'yolov8s.pt',  # small
        'yolov8m.pt',  # medium
        'yolov8l.pt',  # large
        'yolov8x.pt',  # xlarge
    ]
    return models


def check_model_exists(model_name: str, check_dir: Path = None) -> bool:
    """检查模型是否已存在"""
    if check_dir is None:
        # 默认检查项目pretrained目录
        script_dir = Path(__file__).parent.resolve()
        check_dir = script_dir / 'pretrained'
    model_path = check_dir / model_name
    return model_path.exists()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='YOLOv8 预训练模型下载脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载 YOLOv8s 模型
  python download_pretrained.py yolov8s.pt
  
  # 下载 YOLOv8n 模型
  python download_pretrained.py yolov8n.pt
  
  # 列出所有可用模型
  python download_pretrained.py --list
  
  # 下载所有模型
  python download_pretrained.py --all
        """
    )
    
    parser.add_argument(
        'model',
        nargs='?',
        type=str,
        default=None,
        help='要下载的模型名称 (例如: yolov8s.pt, yolov8n.pt)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='列出所有可用的模型'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='下载所有可用的模型'
    )
    
    parser.add_argument(
        '--check',
        action='store_true',
        help='检查模型是否已存在'
    )
    
    parser.add_argument(
        '--weights-dir',
        type=str,
        default=None,
        help='指定模型保存目录（默认: ~/.ultralytics/weights）'
    )
    
    args = parser.parse_args()
    
    # 确定下载目录
    if args.weights_dir:
        download_dir = Path(args.weights_dir)
    else:
        # 默认下载到项目pretrained目录
        script_dir = Path(__file__).parent.resolve()
        download_dir = script_dir / 'pretrained'
    
    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 下载目录: {download_dir}")
    print("=" * 80)
    
    # 列出可用模型
    if args.list:
        print("📋 可用的 YOLOv8 模型:")
        models = list_available_models()
        for model in models:
            exists = check_model_exists(model, download_dir)
            status = "✓ 已下载" if exists else "✗ 未下载"
            print(f"   {model:15s} - {status}")
        return
    
    # 下载所有模型
    if args.all:
        print("📥 开始下载所有模型...")
        models = list_available_models()
        success_count = 0
        for model in models:
            if check_model_exists(model, download_dir):
                print(f"⏭️  跳过 {model} (已存在)")
                continue
            result = download_model(model, download_dir)
            if result:
                success_count += 1
            print()  # 空行分隔
        
        print("=" * 80)
        print(f"✅ 完成！成功下载 {success_count}/{len(models)} 个模型")
        return
    
    # 检查模型是否存在
    if args.check:
        if not args.model:
            print("❌ 错误: 使用 --check 时必须指定模型名称")
            return
        exists = check_model_exists(args.model, download_dir)
        if exists:
            model_path = download_dir / args.model
            file_size = model_path.stat().st_size / (1024 * 1024)  # MB
            print(f"✅ 模型已存在: {model_path}")
            print(f"   文件大小: {file_size:.2f} MB")
        else:
            print(f"❌ 模型不存在: {args.model}")
        return
    
    # 下载指定模型
    if not args.model:
        parser.print_help()
        return
    
    # 确保模型名称以.pt结尾
    if not args.model.endswith('.pt'):
        args.model = args.model + '.pt'
    
    # 检查是否已存在
    if check_model_exists(args.model, download_dir):
        model_path = download_dir / args.model
        file_size = model_path.stat().st_size / (1024 * 1024)  # MB
        print(f"ℹ️  模型已存在: {model_path}")
        print(f"   文件大小: {file_size:.2f} MB")
        print("   如需重新下载，请先删除现有文件")
        return
    
    # 下载模型
    download_model(args.model, download_dir)


if __name__ == '__main__':
    main()

