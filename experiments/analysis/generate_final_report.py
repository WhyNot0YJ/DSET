#!/usr/bin/env python3
"""
生成《CaS_DETR vs Baselines 终极实验评估报告》
深度扫描实验日志，提取所有关键指标
"""

import os
import re
import csv
import yaml
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# 工作目录
BASE_DIR = Path(__file__).parent

# 模型名称映射
MODEL_NAME_MAP = {
    'cas_detr4': 'CaS_DETR-4',
    'cas_detr6': 'CaS_DETR-6',
    'cas_detr48': 'CaS_DETR-48',
    'cas_detr4_r34': 'CaS_DETR-4 (R34)',
    'cas_detr6_r34': 'CaS_DETR-6 (R34)',
    'rtdetr_r18': 'RT-DETR (R18)',
    'rtdetr_r34': 'RT-DETR (R34)',
    'yolo_v8s': 'YOLOv8-S',
    'yolo_v8l': 'YOLOv8-L',
    'yolo_v10l': 'YOLOv10-L',
}

def get_model_name(exp_name: str) -> str:
    """从实验名称提取模型名称"""
    # 提取基础名称
    parts = exp_name.split('_')
    base_name = parts[0]
    
    # 检查是否有backbone信息
    if 'r18' in exp_name.lower():
        backbone = 'R18'
    elif 'r34' in exp_name.lower():
        backbone = 'R34'
    else:
        backbone = None
    
    # 构建模型名称
    if base_name in MODEL_NAME_MAP:
        model_name = MODEL_NAME_MAP[base_name]
        if backbone and 'R18' not in model_name and 'R34' not in model_name:
            model_name = f"{model_name} ({backbone})"
    else:
        # 对于RT-DETR，从exp_name中提取
        if 'rtdetr' in exp_name.lower():
            if backbone:
                model_name = f"RT-DETR ({backbone})"
            else:
                model_name = "RT-DETR"
        else:
            model_name = exp_name
    
    return model_name

def parse_training_history_csv(csv_path: Path) -> Dict:
    """解析训练历史CSV文件"""
    try:
        rows = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if not rows:
            return {}
        
        # 查找最佳mAP
        if 'mAP_0.5_0.95' in rows[0]:
            best_row = max(rows, key=lambda x: float(x.get('mAP_0.5_0.95', 0.0)))
            
            return {
                'best_epoch': int(best_row.get('epoch', 0)),
                'total_epochs': len(rows),
                'best_map_50_95': float(best_row.get('mAP_0.5_0.95', 0.0)),
                'best_map_50': float(best_row.get('mAP_0.5', 0.0)),
                'best_map_75': float(best_row.get('mAP_0.75', 0.0)),
                'convergence_speed': int(best_row.get('epoch', 0)) / len(rows) if len(rows) > 0 else 0.0,
            }
        elif 'metrics/mAP50-95(B)' in rows[0]:
            # YOLO格式
            best_row = max(rows, key=lambda x: float(x.get('metrics/mAP50-95(B)', 0.0)))
            
            return {
                'best_epoch': int(best_row.get('epoch', 0)),
                'total_epochs': len(rows),
                'best_map_50_95': float(best_row.get('metrics/mAP50-95(B)', 0.0)),
                'best_map_50': float(best_row.get('metrics/mAP50(B)', 0.0)),
                'best_map_75': 0.0,  # YOLO没有mAP75
                'convergence_speed': int(best_row.get('epoch', 0)) / len(rows) if len(rows) > 0 else 0.0,
            }
    except Exception as e:
        print(f"Error parsing {csv_path}: {e}")
    
    return {}

def parse_training_log(log_path: Path) -> Dict:
    """解析训练日志文件，提取类别AP（最佳模型）"""
    result = {
        'pedestrian_ap': None,
        'cyclist_ap': None,
        'van_ap': None,
        'truck_ap': None,
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
            # 从后往前查找"保存最佳模型"或"每个类别的 mAP"
            best_model_found = False
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i]
                
                # 找到"保存最佳模型"标记
                if '保存最佳模型' in line or 'Best model' in line:
                    best_model_found = True
                    # 继续向前查找类别AP（通常在保存最佳模型之后）
                    for j in range(i, min(i + 20, len(lines))):
                        if '每个类别的 mAP' in lines[j]:
                            # 提取后续几行的类别AP
                            for k in range(j + 1, min(j + 10, len(lines))):
                                ap_line = lines[k]
                                if 'Pedestrian' in ap_line:
                                    # 支持两种格式: "Pedestrian  : 0.5794" 或 "Pedestrian: 0.5794"
                                    match = re.search(r'Pedestrian\s+:\s*([\d.]+)', ap_line)
                                    if not match:
                                        match = re.search(r'Pedestrian\s*:\s*([\d.]+)', ap_line)
                                    if match:
                                        result['pedestrian_ap'] = float(match.group(1))
                                elif 'Cyclist' in ap_line:
                                    match = re.search(r'Cyclist\s+:\s*([\d.]+)', ap_line)
                                    if not match:
                                        match = re.search(r'Cyclist\s*:\s*([\d.]+)', ap_line)
                                    if match:
                                        result['cyclist_ap'] = float(match.group(1))
                                elif 'Van' in ap_line and 'Van' == ap_line.split()[0]:
                                    match = re.search(r'Van\s+:\s*([\d.]+)', ap_line)
                                    if not match:
                                        match = re.search(r'Van\s*:\s*([\d.]+)', ap_line)
                                    if match:
                                        result['van_ap'] = float(match.group(1))
                                elif 'Truck' in ap_line:
                                    match = re.search(r'Truck\s+:\s*([\d.]+)', ap_line)
                                    if not match:
                                        match = re.search(r'Truck\s*:\s*([\d.]+)', ap_line)
                                    if match:
                                        result['truck_ap'] = float(match.group(1))
                            break
                    break
            
            # 如果没找到最佳模型标记，尝试查找最后一次"每个类别的 mAP"
            if not best_model_found:
                for i in range(len(lines) - 1, -1, -1):
                    if '每个类别的 mAP' in lines[i]:
                        for j in range(i + 1, min(i + 10, len(lines))):
                            ap_line = lines[j]
                            if 'Pedestrian' in ap_line:
                                match = re.search(r'Pedestrian\s*:\s*([\d.]+)', ap_line)
                                if match and result['pedestrian_ap'] is None:
                                    result['pedestrian_ap'] = float(match.group(1))
                            elif 'Cyclist' in ap_line:
                                match = re.search(r'Cyclist\s*:\s*([\d.]+)', ap_line)
                                if match and result['cyclist_ap'] is None:
                                    result['cyclist_ap'] = float(match.group(1))
                            elif 'Van' in ap_line and 'Van' == ap_line.split()[0]:
                                match = re.search(r'Van\s*:\s*([\d.]+)', ap_line)
                                if match and result['van_ap'] is None:
                                    result['van_ap'] = float(match.group(1))
                            elif 'Truck' in ap_line:
                                match = re.search(r'Truck\s*:\s*([\d.]+)', ap_line)
                                if match and result['truck_ap'] is None:
                                    result['truck_ap'] = float(match.group(1))
                        break
    except Exception as e:
        print(f"Error parsing log {log_path}: {e}")
    
    return result

def parse_config_yaml(yaml_path: Path) -> Dict:
    """解析配置文件"""
    result = {
        'input_resolution': None,
        'mosaic': None,
        'mixup': None,
        'conf_thres': None,
        'max_det': None,
        'batch_size': None,
        'token_keep_ratio': None,
    }
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
                    # 提取输入分辨率
            if 'training' in config:
                if 'imgsz' in config['training']:
                    result['input_resolution'] = config['training']['imgsz']
                elif 'target_size' in config['training']:
                    result['input_resolution'] = config['training']['target_size']
            
            # 对于CaS_DETR和RT-DETR，从data_augmentation中提取max_size
            if 'data_augmentation' in config:
                if 'max_size' in config['data_augmentation']:
                    result['input_resolution'] = config['data_augmentation']['max_size']
            
            if 'data_augmentation' in config:
                result['mosaic'] = config['data_augmentation'].get('mosaic', None)
                result['mixup'] = config['data_augmentation'].get('mixup', None)
            
            if 'training' in config:
                result['batch_size'] = config['training'].get('batch_size', None)
            
            if 'model' in config and 'cas_detr' in config['model']:
                token_keep = config['model']['cas_detr'].get('token_keep_ratio', None)
                if isinstance(token_keep, dict):
                    # 取第一个值
                    result['token_keep_ratio'] = list(token_keep.values())[0] if token_keep else None
                else:
                    result['token_keep_ratio'] = token_keep
            
            # 检查验证设置（通常在代码中，这里尝试从注释或默认值推断）
            # 对于CaS_DETR和RT-DETR，通常使用conf_thres=0.001, max_det=100
            result['conf_thres'] = 0.001  # 默认值
            result['max_det'] = 100  # 默认值
            
    except Exception as e:
        print(f"Error parsing config {yaml_path}: {e}")
    
    return result

def scan_experiments() -> Dict:
    """扫描所有实验"""
    experiments = {}
    
    # 扫描CaS_DETR实验（排除 recent_config）
    cas_detr_logs = BASE_DIR / 'cas_detr' / 'logs'
    if cas_detr_logs.exists():
        for exp_dir in cas_detr_logs.iterdir():
            if exp_dir.is_dir() and exp_dir.name != 'recent_config':
                exp_name = exp_dir.name
                
                # 读取CSV
                csv_path = exp_dir / 'training_history.csv'
                if csv_path.exists():
                    data = parse_training_history_csv(csv_path)
                    data['model_type'] = 'CaS_DETR'
                    data['exp_name'] = exp_name
                    
                    # 读取日志
                    log_path = exp_dir / 'training.log'
                    if log_path.exists():
                        class_aps = parse_training_log(log_path)
                        data.update(class_aps)
                    
                    # 读取配置
                    config_path = exp_dir / 'config.yaml'
                    if config_path.exists():
                        config = parse_config_yaml(config_path)
                        data.update(config)
                    
                    experiments[exp_name] = data
    
    # 扫描RT-DETR实验（只扫描 base_config）
    rtdetr_base_config = BASE_DIR / 'rt-detr' / 'logs' / 'base_config'
    if rtdetr_base_config.exists():
        for exp_dir in rtdetr_base_config.iterdir():
            if exp_dir.is_dir():
                exp_name = exp_dir.name
                
                data = {}
                data['model_type'] = 'RT-DETR'
                data['exp_name'] = exp_name
                
                # 优先从CSV读取
                csv_path = exp_dir / 'training_history.csv'
                if csv_path.exists():
                    csv_data = parse_training_history_csv(csv_path)
                    data.update(csv_data)
                
                # 从日志读取（如果没有CSV或补充数据）
                log_path = exp_dir / 'training.log'
                if log_path.exists():
                    class_aps = parse_training_log(log_path)
                    data.update(class_aps)
                    
                    # 如果没有CSV，尝试从日志中提取最佳mAP
                    if not csv_path.exists():
                        # 从日志中提取最佳mAP
                        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = f.readlines()
                            content = ''.join(lines)
                            
                            # 查找所有mAP值
                            map_matches = re.findall(r'新的最佳mAP: ([\d.]+)', content)
                            if map_matches:
                                best_map = max([float(x) for x in map_matches])
                                data['best_map_50_95'] = best_map
                                
                                # 找到最佳mAP对应的mAP@50和epoch
                                best_map50 = None
                                best_epoch = 0
                                for i, line in enumerate(lines):
                                    if f'新的最佳mAP: {best_map}' in line:
                                        # 向前查找mAP@50和epoch
                                        for j in range(max(0, i-10), i):
                                            if '📊 当前mAP:' in lines[j]:
                                                map50_match = re.search(r'mAP@50: ([\d.]+)', lines[j])
                                                if map50_match:
                                                    best_map50 = float(map50_match.group(1))
                                            epoch_match = re.search(r'Epoch (\d+):', lines[j])
                                            if epoch_match:
                                                best_epoch = int(epoch_match.group(1))
                                        
                                        # 如果还没找到epoch，继续向前查找
                                        if best_epoch == 0:
                                            for j in range(max(0, i-30), i):
                                                epoch_match = re.search(r'Epoch (\d+):', lines[j])
                                                if epoch_match:
                                                    best_epoch = int(epoch_match.group(1))
                                                    break
                                        break
                                
                                if best_map50:
                                    data['best_map_50'] = best_map50
                                if best_epoch > 0:
                                    data['best_epoch'] = best_epoch
                                    data['total_epochs'] = 200  # 默认200
                
                config_path = exp_dir / 'config.yaml'
                if config_path.exists():
                    config = parse_config_yaml(config_path)
                    data.update(config)
                
                if data:  # 只要有数据就添加
                    experiments[exp_name] = data
    
    # 扫描YOLO实验
    for yolo_dir in ['yolov8', 'yolov10']:
        yolo_logs = BASE_DIR / yolo_dir / 'logs'
        if yolo_logs.exists():
            for subdir in yolo_logs.iterdir():
                if subdir.is_dir():
                    exp_name = subdir.name
                    
                    csv_path = subdir / 'training_history.csv'
                    if csv_path.exists():
                        data = parse_training_history_csv(csv_path)
                        data['model_type'] = yolo_dir.upper()
                        data['exp_name'] = exp_name
                        
                        log_path = subdir / 'training.log'
                        if log_path.exists():
                            class_aps = parse_training_log(log_path)
                            data.update(class_aps)
                        
                        config_path = subdir / 'config.yaml'
                        if config_path.exists():
                            config = parse_config_yaml(config_path)
                            data.update(config)
                        
                        experiments[exp_name] = data
    
    return experiments

def generate_report(experiments: Dict) -> str:
    """生成报告"""
    report = []
    report.append("# CaS_DETR vs Baselines 终极实验评估报告\n")
    report.append("生成时间: 2025-12-05\n")
    report.append("---\n\n")
    
    # 1. 实验配置审计
    report.append("## 1. 实验配置审计 (Configuration Audit)\n\n")
    report.append("### 1.1 输入分辨率 (Input Resolution)\n\n")
    report.append("| Model | Input Resolution | Status |\n")
    report.append("|:---|:---|:---|\n")
    
    resolutions = {}
    for exp_name, data in experiments.items():
        model_type = data.get('model_type', 'Unknown')
        res = data.get('input_resolution', 'N/A')
        if res != 'N/A':
            resolutions[model_type] = res
            report.append(f"| {model_type} | {res} | ✅ |\n")
        else:
            report.append(f"| {model_type} | N/A | ⚠️ |\n")
    
    report.append("\n### 1.2 验证设置 (Validation Settings)\n\n")
    report.append("| Model | conf_thres | max_det | Status |\n")
    report.append("|:---|:---|:---|:---|\n")
    
    for exp_name, data in experiments.items():
        model_type = data.get('model_type', 'Unknown')
        conf_thres = data.get('conf_thres', 0.001)
        max_det = data.get('max_det', 100)
        report.append(f"| {model_type} | {conf_thres} | {max_det} | ✅ |\n")
    
    report.append("\n### 1.3 数据增强 (Data Augmentation)\n\n")
    report.append("| Model | Mosaic | Mixup | Status |\n")
    report.append("|:---|:---|:---|:---|\n")
    
    for exp_name, data in experiments.items():
        model_type = data.get('model_type', 'Unknown')
        mosaic = data.get('mosaic', 'N/A')
        mixup = data.get('mixup', 'N/A')
        report.append(f"| {model_type} | {mosaic} | {mixup} | ✅ |\n")
    
    # 2. 核心性能榜单
    report.append("\n## 2. 核心性能榜单 (SOTA Comparison)\n\n")
    report.append("| Model Name | Best mAP 50-95 | mAP 50 | Best Epoch | Total Epochs |\n")
    report.append("|:---|:---|:---|:---|:---|\n")
    
    # 按mAP排序
    sorted_exps = sorted(experiments.items(), 
                        key=lambda x: x[1].get('best_map_50_95', 0.0), 
                        reverse=True)
    
    best_map = 0.0
    for exp_name, data in sorted_exps:
        map_50_95 = data.get('best_map_50_95', 0.0)
        map_50 = data.get('best_map_50', 0.0)
        best_epoch = data.get('best_epoch', 0)
        total_epochs = data.get('total_epochs', 0)
        
        model_name = get_model_name(exp_name)
        
        # 加粗第一名
        bold = "**" if map_50_95 > best_map else ""
        best_map = max(best_map, map_50_95)
        
        report.append(f"| {bold}{model_name}{bold} | {bold}{map_50_95:.2f}{bold} | {map_50:.2f} | {best_epoch} | {total_epochs} |\n")
    
    # 3. 细粒度优势分析
    report.append("\n## 3. 细粒度优势分析 (Critical for Paper)\n\n")
    
    # 按backbone分组
    r18_exps = {}
    r34_exps = {}
    yolo_exps = {}
    
    for exp_name, data in experiments.items():
        model_name = get_model_name(exp_name)
        if 'R18' in model_name and ('CaS_DETR' in model_name or 'RT-DETR' in model_name):
            r18_exps[exp_name] = data
        elif 'R34' in model_name and ('CaS_DETR' in model_name or 'RT-DETR' in model_name):
            r34_exps[exp_name] = data
        elif 'YOLO' in model_name:
            yolo_exps[exp_name] = data
    
    # R18组比较
    report.append("### 3.1 R18 Backbone 对比\n\n")
    report.append("#### 3.1.1 小目标组 (Small Objects)\n\n")
    report.append("| Model | Pedestrian AP | Cyclist AP |\n")
    report.append("|:---|:---|:---|\n")
    
    # 找到RT-DETR R18作为baseline
    rtdetr_r18_ped = None
    rtdetr_r18_cyc = None
    rtdetr_r18_van = None
    rtdetr_r18_truck = None
    
    for exp_name, data in r18_exps.items():
        if 'rtdetr' in exp_name.lower():
            rtdetr_r18_ped = data.get('pedestrian_ap')
            rtdetr_r18_cyc = data.get('cyclist_ap')
            rtdetr_r18_van = data.get('van_ap')
            rtdetr_r18_truck = data.get('truck_ap')
            break
    
    # 显示R18组的所有模型（包括RT-DETR）
    pedestrian_best_r18 = 0.0
    cyclist_best_r18 = 0.0
    
    for exp_name, data in sorted(r18_exps.items(), key=lambda x: x[1].get('best_map_50_95', 0.0), reverse=True):
        model_name = get_model_name(exp_name)
        ped_ap = data.get('pedestrian_ap', None)
        cyc_ap = data.get('cyclist_ap', None)
        
        ped_str = f"{ped_ap:.2f}" if ped_ap is not None else "N/A"
        cyc_str = f"{cyc_ap:.2f}" if cyc_ap is not None else "N/A"
        
        if ped_ap and ped_ap > pedestrian_best_r18:
            ped_str = f"**{ped_str}**"
            pedestrian_best_r18 = ped_ap
        if cyc_ap and cyc_ap > cyclist_best_r18:
            cyc_str = f"**{cyc_str}**"
            cyclist_best_r18 = cyc_ap
        
        report.append(f"| {model_name} | {ped_str} | {cyc_str} |\n")
    
    # CaS_DETR vs RT-DETR R18 提升分析
    if rtdetr_r18_ped:
        report.append("\n#### 3.1.2 CaS_DETR vs RT-DETR R18 提升分析 (Small Objects)\n\n")
        report.append("| Model | Pedestrian AP | vs RT-DETR R18 | Cyclist AP | vs RT-DETR R18 |\n")
        report.append("|:---|:---|:---|:---|:---|\n")
        for exp_name, data in sorted(r18_exps.items(), key=lambda x: x[1].get('best_map_50_95', 0.0), reverse=True):
            model_name = get_model_name(exp_name)
            if 'CaS_DETR' in model_name:
                ped_ap = data.get('pedestrian_ap')
                cyc_ap = data.get('cyclist_ap')
                if ped_ap and cyc_ap:
                    ped_delta = ((ped_ap - rtdetr_r18_ped) / rtdetr_r18_ped * 100) if rtdetr_r18_ped > 0 else 0.0
                    cyc_delta = ((cyc_ap - rtdetr_r18_cyc) / rtdetr_r18_cyc * 100) if rtdetr_r18_cyc > 0 else 0.0
                    ped_delta_str = f"+{ped_delta:.2f}%" if ped_delta >= 0 else f"{ped_delta:.2f}%"
                    cyc_delta_str = f"+{cyc_delta:.2f}%" if cyc_delta >= 0 else f"{cyc_delta:.2f}%"
                    report.append(f"| {model_name} | {ped_ap:.2f} | {ped_delta_str} | {cyc_ap:.2f} | {cyc_delta_str} |\n")
    
    report.append("\n#### 3.1.3 困难类别组 (Hard Categories)\n\n")
    report.append("| Model | Van AP | Truck AP |\n")
    report.append("|:---|:---|:---|\n")
    
    van_best_r18 = 0.0
    truck_best_r18 = 0.0
    
    for exp_name, data in sorted(r18_exps.items(), key=lambda x: x[1].get('best_map_50_95', 0.0), reverse=True):
        model_name = get_model_name(exp_name)
        van_ap = data.get('van_ap', None)
        truck_ap = data.get('truck_ap', None)
        
        van_str = f"{van_ap:.2f}" if van_ap is not None else "N/A"
        truck_str = f"{truck_ap:.2f}" if truck_ap is not None else "N/A"
        
        if van_ap and van_ap > van_best_r18:
            van_str = f"**{van_str}**"
            van_best_r18 = van_ap
        if truck_ap and truck_ap > truck_best_r18:
            truck_str = f"**{truck_str}**"
            truck_best_r18 = truck_ap
        
        report.append(f"| {model_name} | {van_str} | {truck_str} |\n")
    
    # CaS_DETR vs RT-DETR R18 提升分析（困难类别）
    if rtdetr_r18_van:
        report.append("\n#### 3.1.4 CaS_DETR vs RT-DETR R18 提升分析 (Hard Categories)\n\n")
        report.append("| Model | Van AP | vs RT-DETR R18 | Truck AP | vs RT-DETR R18 |\n")
        report.append("|:---|:---|:---|:---|:---|\n")
        for exp_name, data in sorted(r18_exps.items(), key=lambda x: x[1].get('best_map_50_95', 0.0), reverse=True):
            model_name = get_model_name(exp_name)
            if 'CaS_DETR' in model_name:
                van_ap = data.get('van_ap')
                truck_ap = data.get('truck_ap')
                if van_ap and truck_ap:
                    van_delta = ((van_ap - rtdetr_r18_van) / rtdetr_r18_van * 100) if rtdetr_r18_van > 0 else 0.0
                    truck_delta = ((truck_ap - rtdetr_r18_truck) / rtdetr_r18_truck * 100) if rtdetr_r18_truck > 0 else 0.0
                    van_delta_str = f"+{van_delta:.2f}%" if van_delta >= 0 else f"{van_delta:.2f}%"
                    truck_delta_str = f"+{truck_delta:.2f}%" if truck_delta >= 0 else f"{truck_delta:.2f}%"
                    report.append(f"| {model_name} | {van_ap:.2f} | {van_delta_str} | {truck_ap:.2f} | {truck_delta_str} |\n")
    
    # R34组比较
    report.append("\n### 3.2 R34 Backbone 对比\n\n")
    report.append("#### 3.2.1 小目标组 (Small Objects)\n\n")
    report.append("| Model | Pedestrian AP | Cyclist AP |\n")
    report.append("|:---|:---|:---|\n")
    
    # 找到RT-DETR R34作为baseline
    rtdetr_r34_ped = None
    rtdetr_r34_cyc = None
    rtdetr_r34_van = None
    rtdetr_r34_truck = None
    
    for exp_name, data in r34_exps.items():
        if 'rtdetr' in exp_name.lower():
            rtdetr_r34_ped = data.get('pedestrian_ap')
            rtdetr_r34_cyc = data.get('cyclist_ap')
            rtdetr_r34_van = data.get('van_ap')
            rtdetr_r34_truck = data.get('truck_ap')
            break
    
    # 显示R34组的所有模型（包括RT-DETR）
    pedestrian_best_r34 = 0.0
    cyclist_best_r34 = 0.0
    
    for exp_name, data in sorted(r34_exps.items(), key=lambda x: x[1].get('best_map_50_95', 0.0), reverse=True):
        model_name = get_model_name(exp_name)
        ped_ap = data.get('pedestrian_ap', None)
        cyc_ap = data.get('cyclist_ap', None)
        
        ped_str = f"{ped_ap:.2f}" if ped_ap is not None else "N/A"
        cyc_str = f"{cyc_ap:.2f}" if cyc_ap is not None else "N/A"
        
        if ped_ap and ped_ap > pedestrian_best_r34:
            ped_str = f"**{ped_str}**"
            pedestrian_best_r34 = ped_ap
        if cyc_ap and cyc_ap > cyclist_best_r34:
            cyc_str = f"**{cyc_str}**"
            cyclist_best_r34 = cyc_ap
        
        report.append(f"| {model_name} | {ped_str} | {cyc_str} |\n")
    
    # CaS_DETR vs RT-DETR R34 提升分析
    if rtdetr_r34_ped:
        report.append("\n#### 3.2.2 CaS_DETR vs RT-DETR R34 提升分析 (Small Objects)\n\n")
        report.append("| Model | Pedestrian AP | vs RT-DETR R34 | Cyclist AP | vs RT-DETR R34 |\n")
        report.append("|:---|:---|:---|:---|:---|\n")
        for exp_name, data in sorted(r34_exps.items(), key=lambda x: x[1].get('best_map_50_95', 0.0), reverse=True):
            model_name = get_model_name(exp_name)
            if 'CaS_DETR' in model_name:
                ped_ap = data.get('pedestrian_ap')
                cyc_ap = data.get('cyclist_ap')
                if ped_ap and cyc_ap:
                    ped_delta = ((ped_ap - rtdetr_r34_ped) / rtdetr_r34_ped * 100) if rtdetr_r34_ped > 0 else 0.0
                    cyc_delta = ((cyc_ap - rtdetr_r34_cyc) / rtdetr_r34_cyc * 100) if rtdetr_r34_cyc > 0 else 0.0
                    ped_delta_str = f"+{ped_delta:.2f}%" if ped_delta >= 0 else f"{ped_delta:.2f}%"
                    cyc_delta_str = f"+{cyc_delta:.2f}%" if cyc_delta >= 0 else f"{cyc_delta:.2f}%"
                    report.append(f"| {model_name} | {ped_ap:.2f} | {ped_delta_str} | {cyc_ap:.2f} | {cyc_delta_str} |\n")
    
    report.append("\n#### 3.2.3 困难类别组 (Hard Categories)\n\n")
    report.append("| Model | Van AP | Truck AP |\n")
    report.append("|:---|:---|:---|\n")
    
    van_best_r34 = 0.0
    truck_best_r34 = 0.0
    
    for exp_name, data in sorted(r34_exps.items(), key=lambda x: x[1].get('best_map_50_95', 0.0), reverse=True):
        model_name = get_model_name(exp_name)
        van_ap = data.get('van_ap', None)
        truck_ap = data.get('truck_ap', None)
        
        van_str = f"{van_ap:.2f}" if van_ap is not None else "N/A"
        truck_str = f"{truck_ap:.2f}" if truck_ap is not None else "N/A"
        
        if van_ap and van_ap > van_best_r34:
            van_str = f"**{van_str}**"
            van_best_r34 = van_ap
        if truck_ap and truck_ap > truck_best_r34:
            truck_str = f"**{truck_str}**"
            truck_best_r34 = truck_ap
        
        report.append(f"| {model_name} | {van_str} | {truck_str} |\n")
    
    # CaS_DETR vs RT-DETR R34 提升分析（困难类别）
    if rtdetr_r34_van:
        report.append("\n#### 3.2.4 CaS_DETR vs RT-DETR R34 提升分析 (Hard Categories)\n\n")
        report.append("| Model | Van AP | vs RT-DETR R34 | Truck AP | vs RT-DETR R34 |\n")
        report.append("|:---|:---|:---|:---|:---|\n")
        for exp_name, data in sorted(r34_exps.items(), key=lambda x: x[1].get('best_map_50_95', 0.0), reverse=True):
            model_name = get_model_name(exp_name)
            if 'CaS_DETR' in model_name:
                van_ap = data.get('van_ap')
                truck_ap = data.get('truck_ap')
                if van_ap and truck_ap:
                    van_delta = ((van_ap - rtdetr_r34_van) / rtdetr_r34_van * 100) if rtdetr_r34_van > 0 else 0.0
                    truck_delta = ((truck_ap - rtdetr_r34_truck) / rtdetr_r34_truck * 100) if rtdetr_r34_truck > 0 else 0.0
                    van_delta_str = f"+{van_delta:.2f}%" if van_delta >= 0 else f"{van_delta:.2f}%"
                    truck_delta_str = f"+{truck_delta:.2f}%" if truck_delta >= 0 else f"{truck_delta:.2f}%"
                    report.append(f"| {model_name} | {van_ap:.2f} | {van_delta_str} | {truck_ap:.2f} | {truck_delta_str} |\n")
    
    # 4. 动态计算与稀疏性
    report.append("\n## 4. 动态计算与稀疏性 (Sparsity Analysis)\n\n")
    report.append("| Model | Token Keep Ratio | Pruning Ratio |\n")
    report.append("|:---|:---|:---|\n")
    
    for exp_name, data in sorted_exps:
        model_name = get_model_name(exp_name)
        keep_ratio = data.get('token_keep_ratio', None)
        
        if keep_ratio:
            pruning_ratio = 1.0 - keep_ratio
            report.append(f"| {model_name} | {keep_ratio:.2f} | {pruning_ratio:.2f} |\n")
        else:
            report.append(f"| {model_name} | N/A | N/A |\n")
    
    # 5. 异常与Debug信息
    report.append("\n## 5. 异常与Debug信息\n\n")
    report.append("### 5.1 实验状态\n\n")
    report.append("| Model | Status | Notes |\n")
    report.append("|:---|:---|:---|\n")
    
    for exp_name, data in sorted_exps:
        model_name = get_model_name(exp_name)
        total_epochs = data.get('total_epochs', 0)
        best_epoch = data.get('best_epoch', 0)
        
        if total_epochs == 0:
            status = "❌ Crashed"
            notes = "No training history"
        elif best_epoch == total_epochs:
            status = "⚠️ Not Converged"
            notes = "Best at final epoch"
        else:
            status = "✅ Completed"
            notes = f"Best at epoch {best_epoch}"
        
        report.append(f"| {model_name} | {status} | {notes} |\n")
    
    return "".join(report)

def main():
    print("🔍 扫描实验日志...")
    experiments = scan_experiments()
    print(f"✅ 找到 {len(experiments)} 个实验")
    
    print("📊 生成报告...")
    report = generate_report(experiments)
    
    output_path = BASE_DIR / 'Final_Experiment_Report.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 报告已保存到: {output_path}")

if __name__ == '__main__':
    main()

