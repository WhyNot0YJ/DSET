#!/usr/bin/env python3
"""
统计 camera 目录下所有 JSON 文件中 "type" 字段的值和出现次数
"""

import json
from pathlib import Path
from collections import Counter

def check_types(directory):
    """统计目录中所有JSON文件的type字段"""
    directory = Path(directory)
    
    if not directory.exists():
        print(f"错误: 目录不存在: {directory}")
        return
    
    type_counter = Counter()
    total_files = 0
    total_objects = 0
    files_with_types = {}
    
    print(f"正在扫描目录: {directory}")
    print("=" * 60)
    
    # 遍历所有JSON文件
    for json_file in sorted(directory.glob("*.json")):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            total_files += 1
            file_types = Counter()
            
            # 处理JSON数组
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'type' in item:
                        type_name = item['type']
                        type_counter[type_name] += 1
                        file_types[type_name] += 1
                        total_objects += 1
            # 处理单个JSON对象
            elif isinstance(data, dict):
                if 'type' in data:
                    type_name = data['type']
                    type_counter[type_name] += 1
                    file_types[type_name] += 1
                    total_objects += 1
            
            # 记录每个文件包含的类别
            if file_types:
                files_with_types[json_file.name] = dict(file_types)
                
        except json.JSONDecodeError as e:
            print(f"警告: 无法解析 {json_file.name}: {e}")
        except Exception as e:
            print(f"错误: 处理 {json_file.name} 时出错: {e}")
    
    # 打印统计结果
    print(f"\n统计结果:")
    print(f"  总文件数: {total_files}")
    print(f"  总对象数: {total_objects}")
    print(f"  总类别数: {len(type_counter)}")
    print("\n" + "=" * 60)
    print("各类别出现次数 (按出现次数降序):")
    print("=" * 60)
    
    # 按出现次数排序
    for type_name, count in type_counter.most_common():
        percentage = (count / total_objects * 100) if total_objects > 0 else 0
        print(f"  {type_name:20s}: {count:6d} 次 ({percentage:5.2f}%)")
    
    # 检查特定类别
    print("\n" + "=" * 60)
    print("特殊检查:")
    print("=" * 60)
    
    # 检查 Tricyclist 相关
    tricycl_variants = [t for t in type_counter.keys() if 'tricycl' in t.lower()]
    if tricycl_variants:
        print(f"\n找到 Tricyclist 相关类别:")
        for variant in tricycl_variants:
            print(f"  '{variant}': {type_counter[variant]} 次")
    else:
        print("\n未找到任何包含 'tricycl' 的类别")
    
    # 检查 Trafficcone 相关
    trafficcone_variants = [t for t in type_counter.keys() if 'trafficcone' in t.lower() or 'traffic_cone' in t.lower()]
    if trafficcone_variants:
        print(f"\n找到 Trafficcone 相关类别:")
        for variant in trafficcone_variants:
            print(f"  '{variant}': {type_counter[variant]} 次")
    
    # 显示包含特定类别的文件示例
    print("\n" + "=" * 60)
    print("包含 Tricyclist 的文件示例 (前5个):")
    print("=" * 60)
    tricycl_files = []
    for filename, types in files_with_types.items():
        for type_name in types.keys():
            if 'tricycl' in type_name.lower():
                tricycl_files.append((filename, type_name))
                break
    
    if tricycl_files:
        for filename, type_name in tricycl_files[:5]:
            print(f"  {filename}: 包含 '{type_name}'")
        if len(tricycl_files) > 5:
            print(f"  ... 还有 {len(tricycl_files) - 5} 个文件")
    else:
        print("  未找到包含 Tricyclist 的文件")
    
    return type_counter, files_with_types

if __name__ == "__main__":
    import sys
    
    # 默认使用 camera 目录
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        # 假设脚本在项目根目录运行
        directory = Path(__file__).parent / "camera"
    
    check_types(directory)

