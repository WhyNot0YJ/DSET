import os
import re

projects = ['cas_detr', 'rt-detr', 'moe-rtdetr']

# Clean train.py leftover vars mappings
for proj in projects:
    file_path = f'experiments/{proj}/train.py'
    if not os.path.exists(file_path):
        continue
        
    with open(file_path, 'r') as f:
        content = f.read()

    # Remove var readings for scales
    content = re.sub(r'\s*# 读取多尺度训练配置\n', '\n', content)
    content = re.sub(r'\s*train_scales_min = aug_config\.get\(\'scales_min\', 480\)\n', '\n', content)
    content = re.sub(r'\s*train_scales_max = aug_config\.get\(\'scales_max\', 800\)\n', '\n', content)
    content = re.sub(r'\s*train_scales_step = aug_config\.get\(\'scales_step\', 32\)\n', '\n', content)
    content = re.sub(r'\s*train_max_size = aug_config\.get\(\'max_size\', 1333\)?\n', '\n', content)
    
    # Remove max_size=train_max_size if present in dataset kwargs
    # Handled via literal match in case step 1 missed it
    content = re.sub(r'\s*train_max_size=train_max_size,?\n', '\n', content)
    content = re.sub(r'\s*train_scales_min=train_scales_min,?\n', '\n', content)
    content = re.sub(r'\s*train_scales_max=train_scales_max,?\n', '\n', content)
    content = re.sub(r'\s*train_scales_step=train_scales_step,?\n', '\n', content)
    
    with open(file_path, 'w') as f:
        f.write(content)
