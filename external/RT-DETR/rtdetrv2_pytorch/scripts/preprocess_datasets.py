#!/usr/bin/env python3
"""
V2X数据集预处理脚本
用于处理RCooper和DAIR-V2X数据集，生成MoE训练所需的数据格式
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

def setup_directories(base_dir: Path) -> Dict[str, Path]:
    """创建数据集目录结构"""
    dirs = {
        'rcooper': base_dir / 'RCooper',
        'dair_v2x': base_dir / 'DAIR-V2X',
        'processed': base_dir / 'processed',
        'language_annotations': base_dir / 'processed' / 'language_annotations',
        'expert_data': base_dir / 'processed' / 'expert_data',
        'splits': base_dir / 'processed' / 'splits'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"创建目录: {dir_path}")
    
    return dirs

def check_dataset_availability(dirs: Dict[str, Path]) -> Dict[str, bool]:
    """检查数据集是否已下载"""
    availability = {}
    
    # 检查RCooper数据集
    rcooper_required = ['rsu', 'vehicle', 'annotations', 'calibration']
    rcooper_available = all((dirs['rcooper'] / subdir).exists() for subdir in rcooper_required)
    availability['rcooper'] = rcooper_available
    
    # 检查DAIR-V2X数据集
    dair_required = ['infrastructure-side', 'vehicle-side', 'co-operative', 'annotations']
    dair_available = all((dirs['dair_v2x'] / subdir).exists() for subdir in dair_required)
    availability['dair_v2x'] = dair_available
    
    return availability

def generate_language_instructions(annotations: List[Dict]) -> List[str]:
    """根据检测标注生成语言指令"""
    instructions = []
    
    for ann in annotations:
        # 提取检测到的类别
        detected_classes = set()
        for obj in ann.get('objects', []):
            detected_classes.add(obj.get('class', ''))
        
        # 生成不同复杂度的指令
        if len(detected_classes) == 1:
            # 单类别指令
            class_name = list(detected_classes)[0]
            instructions.extend([
                f"检测{class_name}",
                f"识别{class_name}",
                f"查找{class_name}"
            ])
        elif len(detected_classes) == 2:
            # 双类别指令
            classes = list(detected_classes)
            instructions.extend([
                f"检测{classes[0]}和{classes[1]}",
                f"识别{classes[0]}和{classes[1]}",
                f"查找{classes[0]}和{classes[1]}"
            ])
        elif len(detected_classes) >= 3:
            # 多类别指令
            classes = list(detected_classes)[:3]  # 取前3个类别
            instructions.extend([
                f"检测{classes[0]}、{classes[1]}和{classes[2]}",
                f"识别交通场景中的目标",
                f"检测所有目标"
            ])
    
    return list(set(instructions))  # 去重

def process_rcooper_dataset(dirs: Dict[str, Path]) -> None:
    """处理RCooper数据集"""
    print("=== 处理RCooper数据集 ===")
    
    rcooper_dir = dirs['rcooper']
    annotations_dir = rcooper_dir / 'annotations'
    
    if not annotations_dir.exists():
        print("警告: RCooper标注目录不存在")
        return
    
    # 处理标注文件
    annotation_files = list(annotations_dir.glob('*.json'))
    print(f"找到 {len(annotation_files)} 个标注文件")
    
    all_instructions = []
    processed_annotations = []
    
    for ann_file in annotation_files:
        try:
            with open(ann_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            
            # 生成语言指令
            instructions = generate_language_instructions(annotations)
            all_instructions.extend(instructions)
            
            # 保存处理后的标注
            processed_ann = {
                'image_id': ann_file.stem,
                'annotations': annotations,
                'language_instructions': instructions
            }
            processed_annotations.append(processed_ann)
            
        except Exception as e:
            print(f"处理文件 {ann_file} 时出错: {e}")
    
    # 保存处理结果
    output_file = dirs['language_annotations'] / 'rcooper_language_annotations.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_annotations, f, ensure_ascii=False, indent=2)
    
    print(f"RCooper数据集处理完成，保存到: {output_file}")
    print(f"生成了 {len(set(all_instructions))} 个不同的语言指令")

def process_dair_v2x_dataset(dirs: Dict[str, Path]) -> None:
    """处理DAIR-V2X数据集"""
    print("=== 处理DAIR-V2X数据集 ===")
    
    dair_dir = dirs['dair_v2x']
    annotations_dir = dair_dir / 'annotations'
    
    if not annotations_dir.exists():
        print("警告: DAIR-V2X标注目录不存在")
        return
    
    # 处理标注文件
    annotation_files = list(annotations_dir.glob('*.json'))
    print(f"找到 {len(annotation_files)} 个标注文件")
    
    all_instructions = []
    processed_annotations = []
    
    for ann_file in annotation_files:
        try:
            with open(ann_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            
            # 生成语言指令
            instructions = generate_language_instructions(annotations)
            all_instructions.extend(instructions)
            
            # 保存处理后的标注
            processed_ann = {
                'image_id': ann_file.stem,
                'annotations': annotations,
                'language_instructions': instructions
            }
            processed_annotations.append(processed_ann)
            
        except Exception as e:
            print(f"处理文件 {ann_file} 时出错: {e}")
    
    # 保存处理结果
    output_file = dirs['language_annotations'] / 'dair_v2x_language_annotations.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_annotations, f, ensure_ascii=False, indent=2)
    
    print(f"DAIR-V2X数据集处理完成，保存到: {output_file}")
    print(f"生成了 {len(set(all_instructions))} 个不同的语言指令")

def create_expert_data_splits(dirs: Dict[str, Path]) -> None:
    """创建专家网络训练数据划分"""
    print("=== 创建专家数据划分 ===")
    
    # 专家类别映射
    expert_mapping = {
        'person_expert': ['person', 'pedestrian'],
        'vehicle_expert': ['car', 'truck', 'bus', 'train'],
        'animal_expert': ['cat', 'dog', 'horse', 'cow', 'sheep'],
        'object_expert': ['bottle', 'cup', 'book', 'laptop', 'cell phone'],
        'traffic_expert': ['traffic light', 'stop sign', 'fire hydrant']
    }
    
    # 处理RCooper数据
    rcooper_file = dirs['language_annotations'] / 'rcooper_language_annotations.json'
    if rcooper_file.exists():
        with open(rcooper_file, 'r', encoding='utf-8') as f:
            rcooper_data = json.load(f)
        
        # 按专家划分数据
        for expert_name, classes in expert_mapping.items():
            expert_data = []
            for item in rcooper_data:
                # 检查是否包含该专家的类别
                annotations = item.get('annotations', [])
                has_expert_class = False
                for ann in annotations:
                    for obj in ann.get('objects', []):
                        if obj.get('class', '') in classes:
                            has_expert_class = True
                            break
                    if has_expert_class:
                        break
                
                if has_expert_class:
                    expert_data.append(item)
            
            # 保存专家数据
            expert_file = dirs['expert_data'] / f'{expert_name}_rcooper.json'
            with open(expert_file, 'w', encoding='utf-8') as f:
                json.dump(expert_data, f, ensure_ascii=False, indent=2)
            
            print(f"专家 {expert_name}: {len(expert_data)} 个样本")
    
    print("专家数据划分完成")

def main():
    parser = argparse.ArgumentParser(description='V2X数据集预处理')
    parser.add_argument('--dataset_dir', type=str, 
                       default='datasets',
                       help='数据集根目录')
    parser.add_argument('--skip_rcooper', action='store_true',
                       help='跳过RCooper数据集处理')
    parser.add_argument('--skip_dair_v2x', action='store_true',
                       help='跳过DAIR-V2X数据集处理')
    
    args = parser.parse_args()
    
    # 设置数据集目录
    base_dir = Path(args.dataset_dir)
    print(f"数据集根目录: {base_dir.absolute()}")
    
    # 创建目录结构
    dirs = setup_directories(base_dir)
    
    # 检查数据集可用性
    availability = check_dataset_availability(dirs)
    print(f"\n数据集可用性检查:")
    print(f"RCooper: {'✓' if availability['rcooper'] else '✗'}")
    print(f"DAIR-V2X: {'✓' if availability['dair_v2x'] else '✗'}")
    
    if not availability['rcooper'] and not availability['dair_v2x']:
        print("\n警告: 没有找到可用的数据集，请先下载数据集")
        return
    
    # 处理数据集
    if availability['rcooper'] and not args.skip_rcooper:
        process_rcooper_dataset(dirs)
    
    if availability['dair_v2x'] and not args.skip_dair_v2x:
        process_dair_v2x_dataset(dirs)
    
    # 创建专家数据划分
    create_expert_data_splits(dirs)
    
    print("\n=== 预处理完成 ===")
    print("下一步:")
    print("1. 检查生成的语言标注文件")
    print("2. 开始MoE模型训练")
    print("3. 运行: python tools/train.py -c configs/rtdetrv2/rtdetrv2_r50vd_moe_6x_coco.yml")

if __name__ == '__main__':
    main()
