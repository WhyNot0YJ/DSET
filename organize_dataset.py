import os
import shutil
from pathlib import Path
import argparse

def organize_dataset(target_root):
    root = Path(target_root)
    if not root.exists():
        print(f"❌ 找不到目标路径: {root}，请检查该路径是否存在！")
        return

    # 定义目标文件夹结构
    dirs_to_create = [
        root / "annotations" / "camera",
        root / "metadata"
    ]
    
    # 1. 创建缺失的目标文件夹
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
        print(f"✅ 确保目录存在: {d}")

    # ==========================
    # 2. 迁移 Label 文件夹到 annotations/camera
    # (DAIR-V2X 默认解压时生成的叫 label/)
    # ==========================
    legacy_label_dir = root / "label"
    legacy_camera_label = root / "label" / "camera"
    target_camera_label = root / "annotations" / "camera"
    
    if legacy_camera_label.exists() and legacy_camera_label.is_dir():
        print(f"🚚 正在将 {legacy_camera_label} 移动到 {target_camera_label} ...")
        for item in legacy_camera_label.iterdir():
            if item.is_file():
                shutil.move(str(item), str(target_camera_label / item.name))
        try:
            shutil.rmtree(legacy_label_dir)
        except Exception:
            pass
    elif legacy_label_dir.exists() and legacy_label_dir.is_dir():
        print(f"🚚 正在将 {legacy_label_dir} 移动到 {target_camera_label} ...")
        for item in legacy_label_dir.iterdir():
            if item.is_file():
                shutil.move(str(item), str(target_camera_label / item.name))
        try:
            legacy_label_dir.rmdir()
        except Exception:
            pass

    # ==========================
    # 3. 整理元数据 (data_info.json / split_data.json -> metadata)
    # ==========================
    metadata_files = {
        "data_info.json": root / "metadata" / "data_info.json",
        "split_data.json": root / "metadata" / "split_data.json"
    }
    
    for filename, target_path in metadata_files.items():
        # 如果文件在根目录被找到
        src = root / filename
        if src.exists() and not target_path.exists():
            shutil.move(str(src), str(target_path))
            print(f"🚚 移动文件: {filename} -> metadata/{filename}")

    # 有些人直接将包含 split_data.json 的外层文件夹解压到了根目录
    split_dir = root / "single-infrastructure-split-data"
    if split_dir.exists():
        src = split_dir / "split_data.json"
        if src.exists() and not (root / "metadata" / "split_data.json").exists():
            shutil.move(str(src), str(root / "metadata" / "split_data.json"))
            print(f"🚚 移动内层文件: split_data.json -> metadata/split_data.json")
        try:
            shutil.rmtree(split_dir)
        except Exception:
            pass

    print("\n🎉 结构整理完成！")
    print("\n================== 预期的最终目录结构 ==================")
    print(f"{target_root}/")
    print(" ├── annotations/")
    print(" │   └── camera/  (包含所有原始 xxx.json标签文件)")
    print(" ├── calib/       (如果下载了校准文件)")
    print(" ├── image/       (存放原本的所有 xxx.jpg)")
    print(" ├── metadata/")
    print(" │   ├── data_info.json")
    print(" │   └── split_data.json")
    print(" └── dataset_info.json")
    print("======================================================")
    print("\n🚀 【下一步】请运行项目根目录下的转换脚本生成 COCO 标注文件：")
    print("python dair2coco.py\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="一键重组 DAIR-V2X (单车/路端) 数据集目录至标准格式")
    # 默认使用现有项目中常见的 DAIR-V2X 默认路径，可以被命令行覆盖
    parser.add_argument("--root", type=str, default="/root/autodl-tmp/datasets/DAIR-V2X", 
                        help="DAIR-V2X 解压后的根目录。如 /xxx/datasets/DAIR-V2X")
    args = parser.parse_args()

    print(f"开始检查并重组数据集路径: {args.root}")
    organize_dataset(args.root)
