import os
import glob
import re

projects = ['cas_detr', 'rt-detr', 'moe-rtdetr']

# 1. Clean configs
for proj in projects:
    configs_path = f'experiments/{proj}/configs/*.yaml'
    for file_path in glob.glob(configs_path):
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Remove scale related config lines
        content = re.sub(r'\s*scales_min:.*?\n', '\n', content)
        content = re.sub(r'\s*scales_max:.*?\n', '\n', content)
        content = re.sub(r'\s*scales_step:.*?\n', '\n', content)
        content = re.sub(r'\s*max_size:.*?\n', '\n', content)
        
        with open(file_path, 'w') as f:
            f.write(content)

# 2. Clean dataset class init arguments and mosaic helpers
for proj in projects:
    file_path = f'experiments/{proj}/src/data/dataset/dairv2x_detection.py'
    if not os.path.exists(file_path):
        continue
        
    with open(file_path, 'r') as f:
        content = f.read()

    # Remove train scale parameters from __init__ docstring
    content = re.sub(r'\s*train_scales_min:.*?training\n', '\n', content)
    content = re.sub(r'\s*train_scales_max:.*?\n', '\n', content)
    content = re.sub(r'\s*train_scales_step:.*?\n', '\n', content)
    content = re.sub(r'\s*train_max_size:.*?\n', '\n', content)

    # Remove train scale parameters from __init__ kwargs
    content = re.sub(r'\s*train_scales_min: int = \d+,?\n', '\n', content)
    content = re.sub(r'\s*train_scales_max: int = \d+,?\n', '\n', content)
    content = re.sub(r'\s*train_scales_step: int = \d+,?\n', '\n', content)
    content = re.sub(r'\s*train_max_size: int = \d+,?\n', '\n', content)
    
    # Remove scales list creation
    content = re.sub(r'\s*scales = list\(range\(train_scales_min, train_scales_max \+ 1, train_scales_step\)\)\n', '\n', content)
    
    # Handle mosaic helper changing train_scales_max -> 640
    content = content.replace("Mosaic(size=train_scales_max)", "Mosaic(size=640)")
    
    with open(file_path, 'w') as f:
        f.write(content)

# 3. Clean train.py config mappings
for proj in projects:
    file_path = f'experiments/{proj}/train.py'
    if not os.path.exists(file_path):
        continue
        
    with open(file_path, 'r') as f:
        content = f.read()

    # Remove dataset init mappings for scales
    content = re.sub(r'\s*train_scales_min=aug_config\.get\(\'scales_min\', 480\),?\n', '\n', content)
    content = re.sub(r'\s*train_scales_max=aug_config\.get\(\'scales_max\', 800\),?\n', '\n', content)
    content = re.sub(r'\s*train_scales_step=aug_config\.get\(\'scales_step\', 32\),?\n', '\n', content)
    content = re.sub(r'\s*train_max_size=aug_config\.get\(\'max_size\', 1333\),?\n', '\n', content)
    
    with open(file_path, 'w') as f:
        f.write(content)

print("Cleaned up scale parameters from configs, dataset definition and train.py mappings.")
